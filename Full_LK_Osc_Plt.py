import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, e, k, m_e

# =============================================
#  Parameters (Bi₂Se₃-like field-induced metal)
# =============================================
m_star = 0.15 * m_e          # effective mass
F      = 500.0               # Onsager frequency (T)
T      = 0.3                 # temperature (K)
T_D    = 10.4                # Dingle temperature (K)
g      = 20.0                # g-factor
p_max  = 3                   # harmonics to include

B = np.linspace(10, 60, 2000)  # high resolution
omega_c = e * B / m_star

# =============================================
#  LK Factors (vectorized over B)
# =============================================
def R_T(p, B):
    x = 2 * np.pi**2 * p * k * T / (hbar * omega_c)
    # Stable x/sinh(x)
    return np.where(x < 30,
                    x / (np.sinh(x) + 1e-15),
                    2 * x * np.exp(-x))

def R_D(p, B):
    x = 2 * np.pi**2 * p * k * T_D / (hbar * omega_c)
    return np.exp(-x)

def R_s(p, B):
    nu = g * m_star / (2 * m_e)
    return np.cos(np.pi * p * nu)

def curvature(B):
    return np.sqrt(B)

# =============================================
#  Full LK oscillatory signal (p-sum)
# =============================================
def lk_osc(B, p_max, F, mode='sdh'):
    osc = np.zeros_like(B)
    for p in range(1, p_max + 1):
        phase = 2 * np.pi * (p * F / B - 0.5)   # -0.5 from Berry/Maslov
        amp = curvature(B) * R_T(p, B) * R_D(p, B) * R_s(p, B)
        if mode == 'dhva':
            amp /= np.sqrt(p)                   # dHvA has 1/√p
            amp *= B                            # dHvA ∝ B^{3/2}
        osc += amp * np.cos(phase)
    return osc

# =============================================
#  Generate signals
# =============================================
sdh_osc = lk_osc(B, p_max, F, mode='sdh')     # SdH: Δρ/ρ₀
dhva_osc = lk_osc(B, p_max, F, mode='dhva')   # dHvA: M

# Background (classical)
rho_0 = 1.0
MR_factor = 1 + (e * B * 1e-12 / m_star)**2   # fake τ = 1 ps
rho_bg = rho_0 * MR_factor

# =============================================
#  Plot
# =============================================
plt.figure(figsize=(14, 9))

# --- SdH: Resistivity ---
plt.subplot(2, 1, 1)
plt.plot(B, rho_bg + 0.08 * sdh_osc, 'b-', lw=1.5, label=r'$\rho_{xx}(B)$ (SdH)')
plt.plot(B, rho_bg, 'k--', alpha=0.7, label='Classical MR')
plt.fill_between(B, rho_bg, rho_bg + 0.08 * sdh_osc, color='blue', alpha=0.1)
plt.xlabel('Magnetic Field $B$ (T)')
plt.ylabel(r'$\rho_{xx}$ (a.u.)')
plt.title('Shubnikov–de Haas (SdH) Oscillations – Full LK')
plt.legend()
plt.grid(alpha=0.3)

# --- dHvA: Magnetization ---
plt.subplot(2, 1, 2)
plt.plot(B, dhva_osc, 'r-', lw=1.5, label=r'$\tilde{M}(B)$ (dHvA)')
plt.axhline(0, color='k', linewidth=0.5)
plt.fill_between(B, 0, dhva_osc, color='red', alpha=0.1)
plt.xlabel('Magnetic Field $B$ (T)')
plt.ylabel(r'$\tilde{M}$ (a.u.)')
plt.title('de Haas–van Alphen (dHvA) Oscillations – Full LK')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================
#  Print key values
# =============================================
print(f"""
LK Oscillation Summary
{'='*50}
Fermi pocket:          F = {F} T
Effective mass:        m* = {m_star/m_e:.3f} m_e
Temperature:           T = {T} K
Dingle temperature:    T_D = {T_D} K
g-factor:              g = {g}
Harmonics included:    p = 1 to {p_max}
SdH amplitude scale:   ±8% of ρ₀
dHvA amplitude:        ~ B^{3/2} × LK factors
{'='*50}
""")
