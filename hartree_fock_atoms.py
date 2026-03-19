"""
╔══════════════════════════════════════════════════════════════════════╗
║   HARTREE-FOCK FROM SCRATCH: Hydrogen, Helium, Lithium Atoms        ║
║   Basis set: STO-3G  |  Language: Python  |  GPU: PyTorch (CUDA)   ║
╚══════════════════════════════════════════════════════════════════════╝

Theory:
  Unrestricted Hartree-Fock (UHF) — alpha and beta spins treated separately.
  Each spin channel has its own Fock matrix → general enough to cover H/He/Li.

  UHF Fock matrix (Szabo & Ostlund, Modern Quantum Chemistry, eq. 2.186):
    F^α_μν = H_μν + Σ_λσ (P^α + P^β)_λσ (μν|λσ) - Σ_λσ P^α_λσ (μλ|νσ)
                          ↑ Coulomb (J)                  ↑ Alpha exchange (K_α)

  UHF Energy (Szabo & Ostlund, eq. 2.187):
    E = ½ Tr[P^α (H + F^α)] + ½ Tr[P^β (H + F^β)]

  Density matrices (NO factor of 2 - UHF convention):
    P^α_μν = Σ_{i∈occ_α} C_μi C_νi
    P^β_μν = Σ_{i∈occ_β} C_μi C_νi

  Why UHF?
    H  (1e): n_α=1, n_β=0 → Pb=0 → K_α=J → F_α=H_core → E=H_11 (EXACT!)
    He (2e): n_α=1, n_β=1 → mathematically equivalent to RHF
    Li (3e): n_α=2, n_β=1 → open-shell, spin polarization handled naturally
"""

import numpy as np
from scipy.special import erf
from scipy.linalg import eigh as scipy_eigh
import time
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
#  0. GPU / CPU SETUP
# ══════════════════════════════════════════════════════════

try:
    import torch
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        print(f"✅ GPU active: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = torch.device('cpu')
        print("ℹ️  PyTorch available (CPU mode)")
    USE_TORCH = True

    def gpu_eigh(M):
        """Symmetric eigenvalue decomposition on GPU."""
        t = torch.tensor(M, dtype=torch.float64, device=DEVICE)
        vals, vecs = torch.linalg.eigh(t)
        return vals.cpu().numpy(), vecs.cpu().numpy()

    def gpu_einsum(subscripts, *arrays):
        """Einstein summation on GPU."""
        ts = [torch.tensor(a, dtype=torch.float64, device=DEVICE) for a in arrays]
        return torch.einsum(subscripts, *ts).cpu().numpy()

except ImportError:
    USE_TORCH = False
    print("ℹ️  PyTorch not found → falling back to NumPy/SciPy")

    def gpu_eigh(M):
        return scipy_eigh(M)

    def gpu_einsum(subscripts, *arrays):
        return np.einsum(subscripts, *arrays)


# ══════════════════════════════════════════════════════════
#  1. STO-3G BASIS SET
# ══════════════════════════════════════════════════════════
"""
Each STO-3G orbital is a linear combination of 3 Gaussian primitives.
Primitive (s-type): g(α, r) = N(α) exp(-α|r|²),  N(α) = (2α/π)^(3/4)

Source: Hehre, Stewart & Pople, J. Chem. Phys. 51, 2657 (1969)
"""

# (exponent α, contraction coefficient d) pairs
STO3G = {
    'H':  {'1s': [(3.4252509, 0.1543290),
                  (0.6239137, 0.5353281),
                  (0.1688554, 0.4446345)]},
    'He': {'1s': [(6.3624214, 0.1543290),
                  (1.1589229, 0.5353281),
                  (0.3136498, 0.4446345)]},
    'Li': {'1s': [(16.1195750, 0.1543290),
                  (2.9362007,  0.5353281),
                  (0.7946505,  0.4446345)],
           '2s': [(0.6362897, -0.0999672),
                  (0.1478601,  0.3995128),
                  (0.0480887,  0.7001155)]},
}

ATOMIC_NUMBER = {'H': 1,  'He': 2,  'Li': 3}

# UHF electron configuration: (n_alpha, n_beta)
#   H:  1s¹      → (1, 0)
#   He: 1s²      → (1, 1)  [UHF reduces to RHF]
#   Li: [He] 2s¹ → (2, 1)  [1s↑, 2s↑, 1s↓]
UHF_CONFIG = {'H': (1, 0), 'He': (1, 1), 'Li': (2, 1)}


# ══════════════════════════════════════════════════════════
#  2. CONTRACTED GAUSSIAN ORBITAL (CGO)
# ══════════════════════════════════════════════════════════

class CGO:
    """
    Contracted Gaussian Orbital:
        χ(r) = Σ_k c_k · N(α_k) · exp(-α_k |r - R|²)

    The normalization constant N(α) = (2α/π)^(3/4) is pre-multiplied
    into the contraction coefficients for efficiency.
    """
    def __init__(self, primitives, center):
        self.center = np.array(center, dtype=float)
        self.exps, self.coeffs = [], []
        for alpha, d in primitives:
            N = (2 * alpha / np.pi) ** 0.75
            self.exps.append(alpha)
            self.coeffs.append(d * N)   # pre-normalized coefficient
        self.exps   = np.array(self.exps)
        self.coeffs = np.array(self.coeffs)


def build_basis(atom, center=(0., 0., 0.)):
    """Build the list of STO-3G basis functions for a given atom."""
    orbs = []
    for name, prims in STO3G[atom].items():
        g = CGO(prims, center)
        g.name = name
        orbs.append(g)
    return orbs


# ══════════════════════════════════════════════════════════
#  3. INTEGRAL ENGINE  (analytic, s-type Gaussians)
# ══════════════════════════════════════════════════════════

def boys_F0(t):
    """
    Boys function F₀(t) = ∫₀¹ exp(-t u²) du

    Closed-form evaluation:
      t > 0 : F₀ = √(π/4t) · erf(√t)
      t → 0 : Taylor expansion: F₀ → 1 - t/3

    Used to handle the Coulomb singularity in nuclear attraction
    and electron repulsion integrals.
    """
    t = float(t)
    if t < 1e-10:
        return 1.0 - t / 3.0
    return 0.5 * np.sqrt(np.pi / t) * erf(np.sqrt(t))


def gauss_product(a1, R1, a2, R2):
    """
    Gaussian Product Theorem:
        exp(-a1|r-R1|²) · exp(-a2|r-R2|²) = K · exp(-γ|r-RP|²)

    γ  = a1 + a2
    RP = (a1·R1 + a2·R2) / γ   (product center)
    K  = exp(-a1·a2/γ · |R1-R2|²)

    This theorem makes all four-center integrals analytically tractable.
    """
    g  = a1 + a2
    RP = (a1 * R1 + a2 * R2) / g
    K  = np.exp(-a1 * a2 / g * np.sum((R1 - R2) ** 2))
    return g, RP, K


# ── 3a. Overlap integral ───────────────────────────────────
def prim_S(a1, R1, a2, R2):
    """
    Overlap between two s-type Gaussian primitives:
        S = ∫ g(a1,R1) g(a2,R2) d³r = (π/γ)^(3/2) · K

    Derivation: ∫ K exp(-γ|r-RP|²) d³r = K · (π/γ)^(3/2)
    """
    g, _, K = gauss_product(a1, R1, a2, R2)
    return K * (np.pi / g) ** 1.5


def cgo_S(g1, g2):
    """Overlap integral between two contracted GTOs."""
    S = 0.0
    for a1, c1 in zip(g1.exps, g1.coeffs):
        for a2, c2 in zip(g2.exps, g2.coeffs):
            S += c1 * c2 * prim_S(a1, g1.center, a2, g2.center)
    return S


# ── 3b. Kinetic energy integral ────────────────────────────
def prim_T(a1, R1, a2, R2):
    """
    Kinetic energy of an electron in the field of two Gaussians:
        T = <g1| -½∇² |g2>

    Analytic result for s-type Gaussians:
        T = (a1·a2/γ) · [3 - 2·(a1·a2/γ)·|R1-R2|²] · S(a1,R1,a2,R2)

    Derivation: apply -½∇² to exp(-a2|r-R2|²), collect terms.
    """
    g, _, K = gauss_product(a1, R1, a2, R2)
    mu  = a1 * a2 / g
    r2  = np.sum((R1 - R2) ** 2)
    return mu * (3 - 2 * mu * r2) * K * (np.pi / g) ** 1.5


def cgo_T(g1, g2):
    """Kinetic energy integral between two contracted GTOs."""
    T = 0.0
    for a1, c1 in zip(g1.exps, g1.coeffs):
        for a2, c2 in zip(g2.exps, g2.coeffs):
            T += c1 * c2 * prim_T(a1, g1.center, a2, g2.center)
    return T


# ── 3c. Nuclear attraction integral ───────────────────────
def prim_V(a1, R1, a2, R2, RC, Z):
    """
    Nuclear attraction integral:
        V = <g1| -Z/|r-RC| |g2>

    Evaluated using the Boys function to handle the Coulomb singularity:
        V = -Z · (2π/γ) · K · F₀(γ·|RP-RC|²)

    Derivation: use the Laplace representation 1/|r-RC| = (2/√π)∫₀^∞ exp(-t²|r-RC|²)dt,
    then combine with the Gaussian product to obtain the Boys function form.
    """
    g, RP, K = gauss_product(a1, R1, a2, R2)
    t = g * np.sum((RP - RC) ** 2)
    return -Z * (2 * np.pi / g) * K * boys_F0(t)


def cgo_V(g1, g2, RC, Z):
    """Nuclear attraction integral between two contracted GTOs."""
    V = 0.0
    for a1, c1 in zip(g1.exps, g1.coeffs):
        for a2, c2 in zip(g2.exps, g2.coeffs):
            V += c1 * c2 * prim_V(a1, g1.center, a2, g2.center, RC, Z)
    return V


# ── 3d. Two-electron repulsion integral (ERI) ─────────────
def prim_ERI(a1, R1, a2, R2, a3, R3, a4, R4):
    """
    Two-electron repulsion integral (ERI):
        (μν|λσ) = ∫∫ g_μ(r1) g_ν(r1) · 1/r₁₂ · g_λ(r2) g_σ(r2) dr1 dr2

    Analytic result for four s-type Gaussians:
        (μν|λσ) = 2π^(5/2) / (γ₁₂·γ₃₄·√(γ₁₂+γ₃₄)) · K₁₂·K₃₄·F₀(ρ·|RP-RQ|²)

    where ρ = γ₁₂·γ₃₄/(γ₁₂+γ₃₄) is the reduced Gaussian exponent.

    Note: O(N⁴) formal scaling — the computational bottleneck of HF.
    For H/He (N=1) and Li (N=2) this is trivially cheap.
    """
    g12, RP, K12 = gauss_product(a1, R1, a2, R2)
    g34, RQ, K34 = gauss_product(a3, R3, a4, R4)
    rho = g12 * g34 / (g12 + g34)
    t   = rho * np.sum((RP - RQ) ** 2)
    pre = 2.0 * np.pi ** 2.5 / (g12 * g34 * np.sqrt(g12 + g34))
    return pre * K12 * K34 * boys_F0(t)


def contract(func, g1, g2, *extra):
    """
    Evaluate a contracted integral: Σ_ij c_i c_j · prim_func(a_i, R1, a_j, R2, *extra)
    """
    result = 0.0
    for a1, c1 in zip(g1.exps, g1.coeffs):
        for a2, c2 in zip(g2.exps, g2.coeffs):
            result += c1 * c2 * func(a1, g1.center, a2, g2.center, *extra)
    return result


def contract4(g1, g2, g3, g4):
    """
    Contracted ERI: Σ_ijkl c_i c_j c_k c_l · (ij|kl)
    """
    eri = 0.0
    for a1, c1 in zip(g1.exps, g1.coeffs):
        for a2, c2 in zip(g2.exps, g2.coeffs):
            for a3, c3 in zip(g3.exps, g3.coeffs):
                for a4, c4 in zip(g4.exps, g4.coeffs):
                    eri += (c1 * c2 * c3 * c4
                            * prim_ERI(a1, g1.center, a2, g2.center,
                                       a3, g3.center, a4, g4.center))
    return eri


def build_integrals(basis, Z_list, R_list):
    """
    Compute all one- and two-electron integral matrices:
      S[i,j]        = overlap
      H_core[i,j]   = T[i,j] + V[i,j]   (one-electron Hamiltonian)
      ERI[i,j,k,l]  = two-electron repulsion
    """
    N   = len(basis)
    S   = np.zeros((N, N))
    Hc  = np.zeros((N, N))
    ERI = np.zeros((N, N, N, N))
    RC  = [np.array(r, float) for r in R_list]

    print(f"\n  📐 Computing integrals ({N} basis functions)...")
    t0 = time.time()

    for i in range(N):
        for j in range(N):
            S[i, j]  = contract(prim_S, basis[i], basis[j])
            Hc[i, j] = contract(prim_T, basis[i], basis[j])
            for Z, RC_ in zip(Z_list, RC):
                Hc[i, j] += contract(prim_V, basis[i], basis[j], RC_, Z)

    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    ERI[i, j, k, l] = contract4(basis[i], basis[j],
                                                 basis[k], basis[l])

    print(f"  ⏱  Elapsed: {time.time()-t0:.3f}s")
    return S, Hc, ERI


# ══════════════════════════════════════════════════════════
#  4. UNRESTRICTED HARTREE-FOCK (UHF) SCF
# ══════════════════════════════════════════════════════════
"""
UHF equations (Szabo & Ostlund, Modern Quantum Chemistry):

  Density matrices (no factor of 2 — each spin is tracked separately):
    P^α_μν = Σ_{i∈occ_α} C^α_μi C^α_νi
    P^β_μν = Σ_{i∈occ_β} C^β_μi C^β_νi

  Coulomb operator (built from total density):
    J_μν = Σ_λσ (P^α + P^β)_λσ (μν|λσ)

  Exchange operators (each spin sees only its own density):
    K^α_μν = Σ_λσ P^α_λσ (μλ|νσ)
    K^β_μν = Σ_λσ P^β_λσ (μλ|νσ)

  Fock matrices:
    F^α = H_core + J - K^α
    F^β = H_core + J - K^β

  Total electronic energy:
    E_elec = ½ Tr[P^α (H + F^α)] + ½ Tr[P^β (H + F^β)]

  Special case — H (1 electron):
    P^β = 0  →  J = K^α  →  F^α = H_core  →  E = H_11  (EXACT, no correlation)

  Special case — He (n_α=1, n_β=1):
    P^α = P^β (same spatial orbital)  →  F^α = F^β  →  identical to closed-shell RHF
"""

def löwdin_orthogonalize(S):
    """
    Löwdin (symmetric) orthogonalization: X = S^(-1/2)

    Diagonalize S = U Λ U†  →  X = U Λ^(-1/2) U†

    Transforms the generalized eigenvalue problem  F C = S C ε
    into the standard form  F' C' = C' ε  where  F' = X† F X.
    Uses GPU (torch.linalg.eigh) if available, otherwise scipy.
    """
    vals, U = gpu_eigh(S)
    return U @ np.diag(vals ** -0.5) @ U.T


def density_matrix(C, n_occ):
    """
    Build the UHF density matrix for one spin channel:
        P^σ_μν = Σ_{i<n_occ} C_μi C_νi
    Returns a zero matrix when n_occ == 0.
    """
    if n_occ == 0:
        return np.zeros((C.shape[0], C.shape[0]))
    return C[:, :n_occ] @ C[:, :n_occ].T


def build_fock(H, Pa, Pb, ERI):
    """
    Build the UHF Fock matrices for both spin channels.
    Uses GPU via torch.einsum when available.

        F^α = H + J - K^α
        F^β = H + J - K^β
        J_μν   = Σ_λσ (Pα+Pβ)_λσ (μν|λσ)   Coulomb (total density)
        K^α_μν = Σ_λσ Pα_λσ (μλ|νσ)          Alpha exchange
        K^β_μν = Σ_λσ Pβ_λσ (μλ|νσ)          Beta exchange

    Important: this differs from closed-shell RHF where F = H + J - K/2.
    In UHF the density matrices contain no factor of 2, so the exchange
    term is NOT halved.
    """
    Pt = Pa + Pb
    J  = gpu_einsum('ls,mnls->mn', Pt, ERI)    # Coulomb
    Ka = gpu_einsum('ls,mlns->mn', Pa, ERI)    # Alpha exchange
    Kb = gpu_einsum('ls,mlns->mn', Pb, ERI)    # Beta exchange
    return H + J - Ka, H + J - Kb


def uhf_scf(H, S, ERI, n_alpha, n_beta,
            max_iter=100, tol=1e-9, verbose=True):
    """
    Unrestricted Hartree-Fock Self-Consistent Field solver.

    Algorithm:
      1. X = S^(-1/2)  (Löwdin orthogonalization)
      2. Initial guess from core Hamiltonian eigenvectors
      3. Build Fock matrices F^α, F^β
      4. Solve  F'σ C'σ = C'σ εσ  in orthogonal basis  →  Cσ = X C'σ
      5. Update density matrices  Pσ = Cσ_occ (Cσ_occ)†
      6. Compute energy  E = ½ Tr[Pα(H+Fα)] + ½ Tr[Pβ(H+Fβ)]
      7. Check convergence  |ΔE| < tol

    Returns:
      E_elec   : electronic energy
      Ca, Cb   : alpha/beta MO coefficient matrices
      epsa,epsb: orbital energies
      Pa, Pb   : density matrices
      n_iter   : number of iterations to convergence
    """
    X = löwdin_orthogonalize(S)

    # Initial guess: diagonalize core Hamiltonian in orthogonal basis
    _, vecs = gpu_eigh(X.T @ H @ X)
    C0 = X @ vecs
    Pa = density_matrix(C0, n_alpha)
    Pb = density_matrix(C0, n_beta)

    if verbose:
        n_e = n_alpha + n_beta
        print(f"\n  🔄 UHF SCF | {n_e} electrons (α:{n_alpha}, β:{n_beta})")
        print(f"  {'─'*55}")
        print(f"  {'Iter':>5}  {'E (Hartree)':>18}  {'ΔE':>14}  {'Status'}")
        print(f"  {'─'*55}")

    E_prev, converged = 0.0, False

    for it in range(1, max_iter + 1):
        Fa, Fb = build_fock(H, Pa, Pb, ERI)

        epsa, Ca_new = gpu_eigh(X.T @ Fa @ X);  Ca_new = X @ Ca_new
        epsb, Cb_new = gpu_eigh(X.T @ Fb @ X);  Cb_new = X @ Cb_new

        Pa_new = density_matrix(Ca_new, n_alpha)
        Pb_new = density_matrix(Cb_new, n_beta)

        E  = (0.5 * np.sum(Pa_new * (H + Fa))
            + 0.5 * np.sum(Pb_new * (H + Fb)))

        dE   = E - E_prev
        flag = "⬇" if dE < 0 else "⬆"
        if verbose:
            print(f"  {it:>5}  {E:>18.10f}  {dE:>14.2e}  {flag}")

        if abs(dE) < tol and it > 1:
            converged = True
            break

        Pa, Pb, E_prev = Pa_new, Pb_new, E

    if verbose:
        print(f"  {'─'*55}")
        print(f"  {'✅ Converged' if converged else '⚠️  Did not converge'} ({it} iter)")

    return E, Ca_new, Cb_new, epsa, epsb, Pa_new, Pb_new, it


# ══════════════════════════════════════════════════════════
#  5. FULL ATOM CALCULATION
# ══════════════════════════════════════════════════════════

def run_atom(atom):
    """
    Run a complete HF calculation for a single atom.

    The nucleus is placed at the origin [0,0,0] (units: Bohr).
    For a single atom there is no nuclear repulsion term (V_nuc = 0).
    """
    Z        = ATOMIC_NUMBER[atom]
    n_a, n_b = UHF_CONFIG[atom]

    print(f"\n{'═'*62}")
    print(f"  ⚛️  {atom}  (Z={Z}, {n_a+n_b} electrons)  |  STO-3G / UHF")
    print(f"{'═'*62}")

    basis = build_basis(atom)
    print(f"\n  Basis functions:")
    for g in basis:
        print(f"    {atom} {g.name}  |  α = {np.round(g.exps, 5)}")

    S, Hc, ERI = build_integrals(basis, [Z], [[0., 0., 0.]])

    print(f"\n  Overlap matrix S:\n{np.round(S, 6)}")
    print(f"  Core Hamiltonian H = T+V:\n{np.round(Hc, 6)}")

    E_elec, Ca, Cb, epsa, epsb, Pa, Pb, nit = uhf_scf(
        Hc, S, ERI, n_a, n_b)

    E_total = E_elec  # single atom: no nuclear repulsion

    print(f"\n  ─── RESULTS {'─'*42}")
    print(f"  Total energy    : {E_total:>16.8f}  Hartree")
    print(f"                  : {E_total * 27.2114:>16.6f}  eV")
    print(f"                  : {E_total * 627.509:>16.4f}  kcal/mol")

    shells = list(STO3G[atom].keys())
    print(f"\n  Orbital energies:")
    for i, sh in enumerate(shells):
        ea = epsa[i]
        oa = "↑" if i < n_a else " "
        ob = "↓" if i < n_b else " "
        print(f"    {sh} {oa}{ob}  εα={ea:>10.6f} Eh  ({ea*27.2114:.3f} eV)")

    print(f"  Alpha density matrix P^α:\n{np.round(Pa, 6)}")

    # Reference values (HF/STO-3G limit) and FCI/exact values
    REF   = {'H': -0.46658185, 'He': -2.80778396, 'Li': -7.31552986}
    EXACT = {'H': -0.50000000, 'He': -2.90372440, 'Li': -7.47806030}
    err  = abs(E_total - REF[atom])   * 1000   # mHartree
    corr = abs(E_total - EXACT[atom]) * 1000   # mHartree

    print(f"\n  Reference HF/STO-3G : {REF[atom]:.8f} Eh")
    print(f"  Δ = {err:.3f} mEh  {'✅' if err < 1.5 else '⚠️'}")
    print(f"  Correlation energy  : {corr:.2f} mEh")

    return {'atom': atom, 'E_total': E_total,
            'epsa': epsa, 'epsb': epsb,
            'Pa': Pa, 'Pb': Pb, 'n_iter': nit}


# ══════════════════════════════════════════════════════════
#  6. VISUALIZATION
# ══════════════════════════════════════════════════════════

def plot_all(results):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not found — skipping plots"); return

    REF   = {'H': -0.46658185, 'He': -2.80778396, 'Li': -7.31552986}
    EXACT = {'H': -0.50000000, 'He': -2.90372440, 'Li': -7.47806030}
    CLRS  = {'H': '#4472C4', 'He': '#ED7D31', 'Li': '#70AD47'}
    atoms = [r['atom'] for r in results]

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle('Hartree-Fock / STO-3G  ·  From-Scratch Python Implementation',
                 fontsize=14, fontweight='bold')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── Panel 1: Total energy ──
    ax = fig.add_subplot(gs[0, 0])
    x        = np.arange(len(atoms))
    E_vals   = [r['E_total'] for r in results]
    ref_vals = [REF[a] for a in atoms]
    ax.bar(x - 0.2, E_vals,   0.35, label='This code', color=[CLRS[a] for a in atoms], alpha=0.9)
    ax.bar(x + 0.2, ref_vals, 0.35, label='Reference', color=[CLRS[a] for a in atoms], alpha=0.35)
    ax.set_xticks(x); ax.set_xticklabels(atoms)
    ax.set_ylabel('E (Hartree)'); ax.set_title('Total Energy')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

    # ── Panel 2: Deviation from reference ──
    ax   = fig.add_subplot(gs[0, 1])
    errs = [abs(r['E_total'] - REF[r['atom']]) * 1000 for r in results]
    bars = ax.bar(atoms, errs, color=[CLRS[a] for a in atoms], alpha=0.9)
    for bar, err in zip(bars, errs):
        ax.text(bar.get_x() + bar.get_width()/2, err + 0.001,
                f'{err:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Δ (mHartree)'); ax.set_title('Deviation from Reference')
    ax.grid(axis='y', alpha=0.3)

    # ── Panel 3: Correlation energy ──
    ax   = fig.add_subplot(gs[0, 2])
    corrs = [abs(r['E_total'] - EXACT[r['atom']]) * 1000 for r in results]
    bars  = ax.bar(atoms, corrs, color=[CLRS[a] for a in atoms], alpha=0.9)
    for bar, c in zip(bars, corrs):
        ax.text(bar.get_x() + bar.get_width()/2, c + 0.5,
                f'{c:.1f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('E_corr (mHartree)'); ax.set_title('Correlation Energy')
    ax.grid(axis='y', alpha=0.3)

    # ── Panel 4: Orbital energy level diagram ──
    ax = fig.add_subplot(gs[1, :2])
    for ri, r in enumerate(results):
        a  = r['atom']
        x  = ri * 3.0
        shells = list(STO3G[a].keys())
        na, nb = UHF_CONFIG[a]
        for si, sh in enumerate(shells):
            ea = r['epsa'][si]
            ax.hlines(ea * 27.2114, x - 0.4, x + 0.4,
                      colors=CLRS[a], linewidths=3)
            lbl = f"{sh}\n{ea*27.2114:.2f} eV"
            ax.text(x + 0.5, ea * 27.2114, lbl,
                    va='center', fontsize=8, color=CLRS[a])
            if si < na:
                ax.text(x - 0.15, ea * 27.2114 + 0.5, '↑', fontsize=10, ha='center')
            if si < nb:
                ax.text(x + 0.15, ea * 27.2114 + 0.5, '↓', fontsize=10, ha='center')
        ax.text(x, min(r['epsa']) * 27.2114 - 8, a,
                ha='center', fontsize=14, fontweight='bold', color=CLRS[a])
    ax.set_ylabel('Orbital Energy (eV)')
    ax.set_title('Alpha Orbital Energy Levels')
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.set_xticks([])
    ax.grid(axis='y', alpha=0.2)

    # ── Panel 5: Radial probability density ──
    ax = fig.add_subplot(gs[1, 2])
    rr = np.linspace(0.01, 6.0, 600)  # radial grid in Bohr

    for r in results:
        a    = r['atom']
        orbs = build_basis(a)
        # Evaluate the 1s contracted orbital along the radial axis
        psi  = np.zeros_like(rr)
        for gto in orbs[:1]:       # 1s shell only
            for alpha, c in zip(gto.exps, gto.coeffs):
                psi += c * np.exp(-alpha * rr ** 2)
        norm = np.sqrt(np.trapezoid(4 * np.pi * rr**2 * psi**2, rr))
        if norm > 1e-8:
            psi /= norm
        ax.plot(rr, rr**2 * psi**2, color=CLRS[a], linewidth=2, label=a)

    ax.set_xlabel('r (Bohr)'); ax.set_ylabel('r²|ψ(r)|²')
    ax.set_title('Radial Probability Density (1s)')
    ax.legend(); ax.grid(alpha=0.3)

    plt.savefig('hf_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n  📊 Plot saved: hf_results.png")


# ══════════════════════════════════════════════════════════
#  7. SUMMARY TABLE
# ══════════════════════════════════════════════════════════

def print_summary(results):
    REF   = {'H': -0.46658185, 'He': -2.80778396, 'Li': -7.31552986}
    EXACT = {'H': -0.50000000, 'He': -2.90372440, 'Li': -7.47806030}
    sep = '═' * 74
    print(f"\n\n{sep}")
    print(f"  📋  SUMMARY  |  UHF / STO-3G")
    print(sep)
    print(f"  {'Atom':<5} {'This code':>14} {'Ref HF/STO-3G':>15} "
          f"{'Δ(mEh)':>9} {'Exact/FCI':>13} {'E_corr(mEh)':>12}")
    print(f"  {'─'*72}")
    for r in results:
        a    = r['atom']
        E    = r['E_total']
        err  = abs(E - REF[a])   * 1000
        corr = abs(E - EXACT[a]) * 1000
        mark = '✅' if err < 1.5 else '⚠️'
        print(f"  {a:<5} {E:>14.8f} {REF[a]:>15.8f} "
              f"{err:>9.3f} {EXACT[a]:>13.8f} {corr:>12.2f}  {mark}")
    print(f"  {'─'*72}")
    print(f"""
  Notes:
    H  : 1e → Pb=0 → J=Kα → Fα=H_core → E=H_11 (exact, no correlation!)
    He : UHF ≡ RHF  (n_α=n_β=1, same spatial orbital)
    Li : open-shell UHF captures spin polarization
    E_corr = |E_UHF/STO-3G - E_exact|  ← requires post-HF methods (MP2, CCSD...)
{sep}
""")


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║   HARTREE-FOCK SCF — From Scratch  |  UHF/STO-3G  |  GPU: PyTorch  ║
║   H · He · Li atoms                                                 ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    t0      = time.time()
    results = [run_atom(a) for a in ['H', 'He', 'Li']]
    print(f"\n⏱  Total time: {time.time()-t0:.3f}s")
    print_summary(results)
    plot_all(results)
