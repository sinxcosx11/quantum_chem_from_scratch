"""
╔══════════════════════════════════════════════════════════════════════╗
║   H₂ MOLECULE  ·  HF / STO-3G  ·  Geometry Optimization            ║
║   From-Scratch Python  |  GPU: PyTorch (CUDA)                       ║
╚══════════════════════════════════════════════════════════════════════╝

Builds on the integral engine from hartree_fock_atoms.py and adds:
  1. Two-center integrals   (two H nuclei at different positions)
  2. Nuclear repulsion      V_nuc = Z_A Z_B / R_AB
  3. Closed-shell RHF       (H₂ = singlet, 2 electrons)
  4. E(R) potential energy curve scan
  5. Golden-section geometry optimization
  6. Vibrational frequency  (numerical ∂²E/∂R²)

Theory — RHF (Restricted Hartree-Fock):
  P_μν = 2 · C_μ1 C_ν1    (factor of 2: alpha + beta share the same spatial orbital)
  F_μν = H_μν + Σ_λσ P_λσ [(μν|λσ) - 1/2 (μλ|νσ)]
  E    = 1/2 Tr[P(H+F)] + V_nuc

Why RHF instead of UHF for H₂?
  H₂ is a closed-shell singlet (S=0): two electrons in the same 1sσ orbital.
  RHF is the correct formalism here; UHF would introduce artificial spin
  contamination ("UHF instability") at stretched geometries.
"""

import numpy as np
from scipy.special import erf
from scipy.linalg import eigh as scipy_eigh
import time, warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════ 0. GPU/CPU ═══════════════════
try:
    import torch
    DEVICE    = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    USE_TORCH = True
    label     = f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU (torch)"
    print(f"✅ {label}")

    def gpu_eigh(M):
        """Symmetric eigendecomposition on GPU."""
        t = torch.tensor(M, dtype=torch.float64, device=DEVICE)
        v, U = torch.linalg.eigh(t)
        return v.cpu().numpy(), U.cpu().numpy()

    def gpu_einsum(s, *arrs):
        """Einstein summation on GPU."""
        ts = [torch.tensor(a, dtype=torch.float64, device=DEVICE) for a in arrs]
        return torch.einsum(s, *ts).cpu().numpy()

except ImportError:
    USE_TORCH = False
    print("ℹ️  NumPy/SciPy mode (PyTorch not installed)")
    def gpu_eigh(M):       return scipy_eigh(M)
    def gpu_einsum(s, *a): return np.einsum(s, *a)


# ═══════════════════════════ 1. BASIS SET ═════════════════
# STO-3G for hydrogen: (exponent α, contraction coefficient d)
H_PRIMS = [(3.4252509, 0.1543290),
           (0.6239137, 0.5353281),
           (0.1688554, 0.4446345)]


class CGO:
    """
    Contracted Gaussian Orbital (s-type).
    χ(r) = Σ_k c_k · N(α_k) · exp(-α_k |r - R|²),  N(α) = (2α/π)^(3/4)
    Normalization is pre-folded into the contraction coefficients.
    """
    def __init__(self, prims, center):
        self.center = np.array(center, dtype=float)
        self.exps, self.coeffs = [], []
        for a, d in prims:
            self.exps.append(a)
            self.coeffs.append(d * (2*a/np.pi)**0.75)
        self.exps   = np.array(self.exps)
        self.coeffs = np.array(self.coeffs)


# ═══════════════════════════ 2. INTEGRALS ═════════════════

def boys_F0(t):
    """
    Boys function F₀(t) = ∫₀¹ exp(-t u²) du
    Evaluates as ½√(π/t)·erf(√t) for t>0, or 1-t/3 near zero.
    """
    t = float(t)
    return (1.0 - t/3.0) if t < 1e-10 else 0.5*np.sqrt(np.pi/t)*erf(np.sqrt(t))


def gp(a1, R1, a2, R2):
    """
    Gaussian Product Theorem: returns (γ, R_P, K).
    γ = a1+a2,  R_P = (a1 R1 + a2 R2)/γ,  K = exp(-a1 a2/γ |R1-R2|²)
    """
    g  = a1 + a2
    RP = (a1*R1 + a2*R2) / g
    K  = np.exp(-a1*a2/g * np.sum((R1-R2)**2))
    return g, RP, K


def pS(a1, R1, a2, R2):
    """Overlap primitive: (π/γ)^(3/2) · K"""
    g, _, K = gp(a1, R1, a2, R2)
    return K*(np.pi/g)**1.5


def pT(a1, R1, a2, R2):
    """Kinetic energy primitive: (a1·a2/γ)·[3 - 2(a1·a2/γ)|R1-R2|²]·S"""
    g, _, K = gp(a1, R1, a2, R2)
    mu = a1*a2/g;  r2 = np.sum((R1-R2)**2)
    return mu*(3-2*mu*r2)*K*(np.pi/g)**1.5


def pV(a1, R1, a2, R2, RC, Z):
    """Nuclear attraction primitive: -Z·(2π/γ)·K·F₀(γ|R_P-R_C|²)"""
    g, RP, K = gp(a1, R1, a2, R2)
    t = g*np.sum((RP-RC)**2)
    return -Z*(2*np.pi/g)*K*boys_F0(t)


def pERI(a1, R1, a2, R2, a3, R3, a4, R4):
    """
    Two-electron repulsion primitive:
    (μν|λσ) = 2π^(5/2) / (γ₁₂·γ₃₄·√(γ₁₂+γ₃₄)) · K₁₂·K₃₄·F₀(ρ·|R_P-R_Q|²)
    where ρ = γ₁₂·γ₃₄/(γ₁₂+γ₃₄).
    """
    g12, RP, K12 = gp(a1, R1, a2, R2)
    g34, RQ, K34 = gp(a3, R3, a4, R4)
    rho = g12*g34/(g12+g34)
    t   = rho*np.sum((RP-RQ)**2)
    return 2*np.pi**2.5/(g12*g34*np.sqrt(g12+g34))*K12*K34*boys_F0(t)


def c2(fn, g1, g2, *ex):
    """Contracted 2-center integral: Σ_ij c_i c_j · fn(a_i, R1, a_j, R2, *ex)"""
    return sum(c1*c2_*fn(a1, g1.center, a2, g2.center, *ex)
               for a1, c1  in zip(g1.exps, g1.coeffs)
               for a2, c2_ in zip(g2.exps, g2.coeffs))


def c4(g1, g2, g3, g4):
    """Contracted ERI: Σ_ijkl c_i c_j c_k c_l · (ij|kl)"""
    eri = 0.0
    for a1, c1  in zip(g1.exps, g1.coeffs):
      for a2, c2_ in zip(g2.exps, g2.coeffs):
        for a3, c3  in zip(g3.exps, g3.coeffs):
          for a4, c4_ in zip(g4.exps, g4.coeffs):
            eri += c1*c2_*c3*c4_*pERI(a1, g1.center, a2, g2.center,
                                        a3, g3.center, a4, g4.center)
    return eri


def build_integrals(basis, Z_list, R_list):
    """
    Build S (overlap), H_core = T+V (one-electron), and ERI (two-electron) matrices.
    For H₂: 2 basis functions → 2×2 matrices, 2⁴=16 ERI elements.
    Each V[i,j] accumulates contributions from both nuclei A and B.
    """
    N   = len(basis)
    S   = np.zeros((N, N));  Hc = np.zeros((N, N));  ERI = np.zeros((N, N, N, N))
    RC  = [np.array(r, float) for r in R_list]
    for i in range(N):
        for j in range(N):
            S[i, j]  = c2(pS, basis[i], basis[j])
            Hc[i, j] = c2(pT, basis[i], basis[j])
            for Z, RC_ in zip(Z_list, RC):
                Hc[i, j] += c2(pV, basis[i], basis[j], RC_, Z)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    ERI[i, j, k, l] = c4(basis[i], basis[j], basis[k], basis[l])
    return S, Hc, ERI


# ═══════════════════════════ 3. CLOSED-SHELL RHF SCF ══════
"""
H₂: 2 electrons, singlet → closed-shell RHF (Restricted).
The two electrons occupy the same 1sσ bonding orbital.

Key RHF equations:
  P_μν = 2 · C_μ1 C_ν1       (factor of 2 = α + β in the same orbital)
  F_μν = H_μν + J_μν - K_μν/2
  J_μν = Σ_λσ P_λσ (μν|λσ)   (Coulomb: electron sees average charge cloud)
  K_μν = Σ_λσ P_λσ (μλ|νσ)   (Exchange: quantum correction, same-spin avoidance)
  E    = ½ Tr[P(H+F)]

Note the ½ in the energy: without it, electron-electron repulsion would be
double-counted because J is already embedded in F.
"""

def löwdin(S):
    """Löwdin orthogonalization: X = S^(-1/2). Transforms F C = S C ε to F' C' = C' ε."""
    v, U = gpu_eigh(S)
    return U @ np.diag(v**-0.5) @ U.T


def rhf_scf(H, S, ERI, n_occ=1, tol=1e-10):
    """
    Closed-shell RHF SCF solver.

      n_occ : number of doubly-occupied orbitals (1 for H₂)
      tol   : convergence threshold on the energy change |ΔE|

    Returns: (E_elec, C, orbital_energies, density_matrix)
    """
    X = löwdin(S)
    _, v = gpu_eigh(X.T @ H @ X);  C = X @ v
    P    = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T   # initial guess from core H
    E_p  = 0.0

    for it in range(100):
        J = gpu_einsum('ls,mnls->mn', P, ERI)
        K = gpu_einsum('ls,mlns->mn', P, ERI)
        F = H + J - 0.5*K                          # RHF Fock matrix

        eps, C_new = gpu_eigh(X.T @ F @ X);  C_new = X @ C_new
        P_new      = 2.0 * C_new[:, :n_occ] @ C_new[:, :n_occ].T
        E          = 0.5 * np.sum(P_new * (H + F))

        if abs(E - E_p) < tol and it > 0:
            break
        P, E_p = P_new, E

    return E, C_new, eps, P_new


# ═══════════════════════════ 4. H₂ TOTAL ENERGY ══════════

def h2_energy(R):
    """
    Compute the total HF energy of H₂ at internuclear distance R (Bohr).

    Nuclear geometry: H_A at origin, H_B at (R, 0, 0).
    Total energy = E_elec(R) + V_nuc(R)
    where V_nuc = Z_A · Z_B / R = 1/R  (atomic units).
    """
    RA = np.array([0.0, 0.0, 0.0])
    RB = np.array([R,   0.0, 0.0])
    basis = [CGO(H_PRIMS, RA), CGO(H_PRIMS, RB)]
    S, Hc, ERI = build_integrals(basis, [1, 1], [RA, RB])
    E_elec, C, eps, P = rhf_scf(Hc, S, ERI)
    return E_elec + 1.0/R, E_elec, eps


# ═══════════════════════════ 5. GEOMETRY OPTIMIZATION ════

def golden_section(f, a, b, tol=1e-7):
    """
    Golden-section search: find the minimum of f in the bracket [a, b].

    At each step the interval is reduced by φ = (√5-1)/2 ≈ 0.618.
    No derivative information required — purely function evaluations.
    Convergence is guaranteed for unimodal functions.
    """
    phi = (np.sqrt(5) - 1) / 2
    c, d = b - phi*(b-a), a + phi*(b-a)
    fc, fd = f(c), f(d)
    while abs(b-a) > tol:
        if fc < fd:
            b, d, fd = d, c, fc;  c  = b - phi*(b-a);  fc = f(c)
        else:
            a, c, fc = c, d, fd;  d  = a + phi*(b-a);  fd = f(d)
    x = (a+b)/2;  return x, f(x)


# ═══════════════════════════ 6. VIBRATIONAL FREQUENCY ════

def vib_freq(R_eq, E_eq, h=1e-3):
    """
    Harmonic vibrational frequency from the numerical second derivative:
        k  = ∂²E/∂R²  ≈  [E(R+h) - 2E(R) + E(R-h)] / h²   (force constant)
        ν̃  = (1/2πc) √(k/μ)                                  (wavenumber, cm⁻¹)
        μ  = m_H / 2                                          (reduced mass)

    Unit conversions:
        k [N/m] = k_au × E_h/a₀²  =  k_au × 4.35974e-18 / (5.29177e-11)²
        m_H     = 1.6735e-27 kg  (proton mass)
    """
    Ep, *_ = h2_energy(R_eq + h)
    Em, *_ = h2_energy(R_eq - h)
    k_au   = (Ep - 2*E_eq + Em) / h**2                          # Eh/Bohr²
    k_SI   = k_au * 4.35974e-18 / (5.29177e-11)**2              # N/m
    mu     = 1.6735e-27 / 2.0                                    # kg
    nu     = np.sqrt(k_SI / mu) / (2 * np.pi * 2.99792e10)      # cm⁻¹
    return k_au, k_SI, nu


# ═══════════════════════════ 7. PLOTS ════════════════════

def plot_h2(R_vals, E_vals, R_eq, E_eq):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not found — skipping plots"); return

    E_inf   = 2 * (-0.46658185)          # dissociation limit: 2×E(H)
    De_meV  = (E_eq - E_inf) * 27211.4   # binding energy in meV
    k_au, k_SI, nu = vib_freq(R_eq, E_eq)
    R_ang   = R_vals * 0.529177          # Bohr → Å
    Req_ang = R_eq   * 0.529177

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('H₂ Molecule  —  HF / STO-3G  |  From-Scratch Python',
                 fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.35)

    # ── Potential energy curve ──
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(R_ang, E_vals, 'royalblue', lw=2.5, label='E(R)  HF/STO-3G')
    ax.axhline(E_inf,   color='tomato',   ls='--', lw=1.5,
               label=f'Dissociation limit = {E_inf:.4f} Eh')
    ax.axvline(Req_ang, color='seagreen', ls=':',  lw=1.5,
               label=f'R_eq = {Req_ang:.4f} Å')
    ax.scatter([Req_ang], [E_eq], color='seagreen', s=120, zorder=6)
    ax.annotate(f'  R={Req_ang:.3f} Å\n  E={E_eq:.4f} Eh',
                xy=(Req_ang, E_eq), xytext=(Req_ang+0.2, E_eq+0.04),
                fontsize=9, color='seagreen',
                arrowprops=dict(arrowstyle='->', color='seagreen', lw=1.2))
    ax.set(xlabel='R (Å)', ylabel='E (Hartree)', title='Potential Energy Curve')
    ax.set_xlim(R_ang[0], 3.5)
    ax.set_ylim(E_eq-0.1, E_inf+0.05)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── Binding energy (relative to dissociation limit) ──
    ax = fig.add_subplot(gs[0, 2])
    mask = (R_ang > 0.4) & (R_ang < 2.8)
    ax.plot(R_ang[mask], (E_vals[mask] - E_inf) * 1000, 'royalblue', lw=2)
    ax.axhline(0, color='tomato', ls='--')
    ax.axvline(Req_ang, color='seagreen', ls=':', lw=1.5)
    ax.set(xlabel='R (Å)', ylabel='E − E_∞ (mHartree)', title='Binding Energy')
    ax.grid(alpha=0.3)

    # ── Bonding and antibonding orbital densities ──
    ax  = fig.add_subplot(gs[1, :2])
    xv  = np.linspace(-3, R_eq+3, 500)      # grid along bond axis (Bohr)
    RA  = np.zeros(3);  RB = np.array([R_eq, 0, 0])
    chi = {}
    for tag, cent in [('A', RA), ('B', RB)]:
        g   = CGO(H_PRIMS, cent)
        psi = np.zeros_like(xv)
        for al, co in zip(g.exps, g.coeffs):
            psi += co * np.exp(-al*(xv - cent[0])**2)
        chi[tag] = psi

    bond  = chi['A'] + chi['B']   # bonding MO:  constructive interference
    abond = chi['A'] - chi['B']   # antibonding: destructive interference
    for arr in [bond, abond]:
        n = np.sqrt(np.trapezoid(arr**2, xv))
        arr /= n

    xv_ang = xv * 0.529177
    ax.plot(xv_ang, bond**2,  color='royalblue', lw=2.5, label='Bonding MO σ (1sσ)')
    ax.plot(xv_ang, abond**2, color='tomato',    lw=2.0, ls='--', label='Antibonding MO σ* (1sσ*)')
    ax.fill_between(xv_ang, bond**2, alpha=0.12, color='royalblue')
    for tag, col in [('A', 'seagreen'), ('B', 'darkorange')]:
        c_ = np.zeros(3) if tag == 'A' else np.array([R_eq, 0, 0])
        ax.axvline(c_[0]*0.529177, color=col, lw=0.8, ls='-', alpha=0.6)
    ax.set(xlabel='x (Å)', ylabel='|ψ(x)|²  (y=z=0)',
           title=f'H₂ Bonding / Antibonding Orbitals  (R_eq={Req_ang:.3f} Å)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── Results summary box ──
    ax = fig.add_subplot(gs[1, 2]); ax.axis('off')
    data = [
        ("RESULTS  HF/STO-3G",       '',  12, 'bold'),
        ("",                          "",   8, 'normal'),
        ("R_eq",  f"{Req_ang:.4f} Å",    "Exp: 0.741 Å"),
        ("De",    f"{De_meV:.0f} meV",   "Exp: 4750 meV"),
        ("k",     f"{k_SI:.0f} N/m",     "Exp: 575 N/m"),
        ("ν̃",    f"{nu:.0f} cm⁻¹",     "Exp: 4401 cm⁻¹"),
        ("E_eq",  f"{E_eq:.6f} Eh",      ""),
        ("",      "",                     ""),
        ("Why errors?",               '',  10, 'bold'),
        ("• STO-3G = minimal basis",  "",   9, 'normal'),
        ("• HF misses correlation",   "",   9, 'normal'),
        ("• Fix: CCSD(T)/cc-pVTZ",   "",   9, 'normal'),
    ]
    y = 0.97
    for row in data:
        if len(row) == 4:
            ax.text(0.05, y, row[0], transform=ax.transAxes,
                    fontsize=row[2], fontweight=row[3], va='top')
            y -= 0.06
        else:
            label_, val, ref = row
            ax.text(0.05, y, f"{label_:<6} = {val}", transform=ax.transAxes,
                    fontsize=10, va='top')
            if ref:
                ax.text(0.95, y, ref, transform=ax.transAxes,
                        fontsize=8, va='top', ha='right',
                        color='gray', style='italic')
            y -= 0.08

    plt.savefig('h2_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n  📊 Plot saved: h2_results.png")


# ═══════════════════════════ MAIN ════════════════════════

if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║   H₂ MOLECULE  -  HF/STO-3G  -  Geometry Optimization              ║
╚══════════════════════════════════════════════════════════════════════╝""")

    t0 = time.time()

    # Step 1: Potential energy curve scan
    print("\n━━━  STEP 1: Energy scan  ━━━")
    R_vals = np.linspace(0.5, 6.0, 60)
    print(f"  Computing {len(R_vals)} points...", flush=True)
    E_vals = np.array([h2_energy(R)[0] for R in R_vals])
    i_min  = np.argmin(E_vals)
    print(f"  Scan minimum: R ≈ {R_vals[i_min]:.3f} Bohr  "
          f"({R_vals[i_min]*0.529177:.3f} Å),  E = {E_vals[i_min]:.6f} Eh")

    # Step 2: Geometry optimization
    print("\n━━━  STEP 2: Golden-section geometry optimization  ━━━")
    R_eq, E_eq = golden_section(lambda R: h2_energy(R)[0], 1.0, 2.5)
    Req_ang = R_eq * 0.529177
    print(f"  R_eq = {R_eq:.7f} Bohr  =  {Req_ang:.7f} Å")
    print(f"  E_eq = {E_eq:.8f} Hartree")

    # Step 3: Binding energy
    E_inf  = 2 * (-0.46658185)       # dissociation limit: 2 × E(H) HF/STO-3G
    De_meV = (E_eq - E_inf) * 27211.4
    print(f"\n━━━  STEP 3: Binding Energy  ━━━")
    print(f"  2×E(H) = {E_inf:.6f} Eh  (dissociation limit)")
    print(f"  De     = {De_meV:.1f} meV  (exp: -4750 meV)")
    print(f"  De     = {(E_eq-E_inf)*627.509:.2f} kcal/mol")

    # Step 4: Vibrational frequency
    k_au, k_SI, nu = vib_freq(R_eq, E_eq)
    print(f"\n━━━  STEP 4: Vibrational Frequency  ━━━")
    print(f"  k  = {k_SI:.1f} N/m   (exp: 575 N/m)")
    print(f"  ν̃ = {nu:.0f} cm⁻¹   (exp: 4401 cm⁻¹)")

    # Summary table
    sep = "━" * 58
    print(f"""
{sep}
  📋 SUMMARY  —  H₂  HF/STO-3G
{sep}
  Property         This code      Experiment    Error
  ──────────────────────────────────────────────────
  R_eq  (Å)       {Req_ang:.4f}         0.7414        {abs(Req_ang-0.7414)/0.7414*100:.1f}%
  De    (meV)   {De_meV:8.1f}        -4750         {abs((De_meV+4750)/4750)*100:.0f}%
  ν̃    (cm⁻¹)   {nu:8.0f}         4401         {abs((nu-4401)/4401)*100:.0f}%
  k     (N/m)   {k_SI:8.1f}          575         {abs((k_SI-575)/575)*100:.0f}%
  ──────────────────────────────────────────────────
  Geometry is excellent; energy and frequency errors
  stem from basis set incompleteness and missing
  electron correlation (HF limit).
  → Solution: CCSD(T)/cc-pVTZ
{sep}""")

    print(f"\n⏱  Total time: {time.time()-t0:.2f}s")
    plot_h2(R_vals, E_vals, R_eq, E_eq)
