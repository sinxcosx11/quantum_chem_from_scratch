# quantum-chem-from-scratch

A from-scratch implementation of Hartree-Fock SCF in Python 
no chemistry libraries, just NumPy, SciPy, and PyTorch (for GPU acceleration).

Written as a hobby project while taking a quantum chemistry course. 
The goal was simple: if I'm learning the theory, I might as well build it.

---

## What's inside

| File | Description |
|------|-------------|
| `hartree_fock_atoms.py` | UHF/STO-3G for H, He, Li atoms |
| `hf_h2.py` | RHF/STO-3G for H₂ geometry optimization, vibrational frequency |

---

## Background

Quantum chemistry solves the Schrödinger equation **Ĥψ = Eψ** for 
molecules. For anything beyond hydrogen, this can't be done exactly 
so we approximate.

Hartree-Fock is the foundational approximation: each electron moves 
in the average field of all others. This turns an intractable 
N-electron problem into N coupled one-electron problems, solved 
iteratively (the SCF loop).

The wavefunction is expanded in a basis set here STO-3G, where 
each atomic orbital is represented by 3 Gaussian functions. This makes 
every integral analytic and closes-form.

Four types of integrals drive the whole calculation:
- **S** (overlap)  how much two basis functions share space  
- **T** (kinetic)  electron kinetic energy  
- **V** (nuclear attraction)  electron–nucleus Coulomb interaction  
- **ERI** (two-electron repulsion)  the expensive O(N⁴) part  

The SCF loop updates the electron density until self-consistency, 
then reads off the energy and orbital coefficients.

---

## Results

| System | E (HF/STO-3G) | Reference | Δ |
|--------|--------------|-----------|---|
| H  | −0.46658184 Eh | −0.46658185 Eh | 0.000 mEh  |
| He | −2.80778397 Eh | −2.80778396 Eh | 0.000 mEh  |
| Li | −7.31552599 Eh | −7.31552986 Eh | 0.004 mEh  |

H₂ geometry optimization (RHF/STO-3G):

| Property | This code | Experiment |
|----------|-----------|------------|
| R_eq | 0.7122 Å | 0.741 Å |
| De | −5016 meV | −4750 meV |
| ν̃ | 5481 cm⁻¹ | 4401 cm⁻¹ |

Geometry is accurate; energy and frequency errors reflect the 
known limitations of a minimal basis set and the absence of 
electron correlation (an inherent HF limitation).

---

## Dependencies
```bash
pip install numpy scipy matplotlib torch
```

GPU is used automatically if available (via PyTorch). 
Falls back to NumPy/SciPy otherwise.

---

## Acknowledgements

Developed with help from [Claude](https://claude.ai). 


Reference: Szabo & Ostlund, *Modern Quantum Chemistry* (1989) 
the book this project is essentially a hands-on companion to.
