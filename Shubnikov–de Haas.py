import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, hbar, e, k, m_e, physical_constants

# ========================
# SdH Simulation Parameters
# ========================

# Material: Bi2Se3-like (field-induced bulk metal)
m_star = 0.15 * m_e           # Effective mass
n = 1e17                      # Carrier density (m^-3) — for 3D → adjust for 2D
tau = 1e-12                   # Scattering time (s)
T = 0.3                       # Temperature (K)
g_factor = 20                 # g-factor (topological materials often large)
F = 500                       # Onsager frequency (Tesla) → A_k = 2π e F / ℏ
p_max = 3                     # Max harmonic

# Magnetic field range
B_min, B_max = 5, 60          # Tesla
B = np.linspace(B_min, B_max, 5000)
inv_B = 1 / B

# cyclotron frequency
omega_c = e * B / m_star

# Dingle temperature
T_D = hbar / (2 * np.pi * k * tau)  # in Kelvin

# Background resistivity (Drude)
rho_0 = m_star / (n * e**2 * tau)   # Ohm-m (3D)
mu = e * tau / m_star               # Mobility
wc_tau = omega_c * tau

# Classical magnetoresistance factor
MR_factor = 1 + (wc_tau)**2

# ========================
# Lifshitz-Kosevich Factors
# ========================

def R_T(p, B, T, m_star):
    x = 2 * np.pi**2 * p * k * T / (hbar * e * B / m_star)
    return x / np.sinh(x)

def R_D(p, B, m_star):
    return np.exp(-2 * np.pi**2 * p * k * T_D / (hbar * e * B / m_star))

def R_s(p, B, m_star, g):
    nu = g * m_star / (2 * m_e)
    return np.cos(np.pi * p * nu)

# SdH oscillatory term (diffusion-dominated, high mobility)
def sdH_osc(B, p, F, m_star, T, g):
    phase = 2 * np.pi * (p * F / B - 0.5)  # -1/2 from Berry phase or Maslov
    amp = np.sqrt(B) * R_T(p, B, T, m_star) * R_D(p, B, m_star) * R_s(p, B, m_star, g)
    return amp * np.cos(phase)

# Total oscillatory resistivity
delta_rho = np.zeros_like(B)
for p in range(1, p_max + 1):
    delta_rho += sdH_osc(B, p, F, m_star, T, g_factor)

# Normalize
delta_rho /= np.max(np.abs(delta_rho))  # Normalize for visibility
delta_rho *= 0.1 * rho_0                 # Scale to ~10% oscillation

# Total resistivity: classical MR + SdH
rho_xx = rho_0 * MR_factor + delta_rho

# Optional: low-field scattering-dominated (rare)
# Uncomment for comparison:
# rho_xx_scat = rho_0 * (1 + (wc_tau)**2 * (1 + 4 * delta_rho / rho_0))

# ========================
# Plotting
# ========================

plt.figure(figsize=(12, 8))

# Plot 1: rho_xx vs B
plt.subplot(2, 1, 1)
plt.plot(B, rho_xx * 1e6, 'b-', linewidth=1.5, label=r'$\rho_{xx}(B)$ (SdH + MR)')
plt.plot(B, rho_0 * MR_factor * 1e6, 'k--', alpha=0.7, label='Classical MR')
plt.xlabel('Magnetic Field $B$ (T)')
plt.ylabel(r'$\rho_{xx}$ ($\mu\Omega \cdot m$)')
plt.title('Shubnikov–de Haas Effect Simulation')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: rho_xx vs 1/B → clear oscillations
plt.subplot(2, 1, 2)
plt.plot(inv_B, rho_xx * 1e6, 'r-', linewidth=1.2)
plt.xlabel(r'$1/B$ (T$^{-1}$)')
plt.ylabel(r'$\rho_{xx}$ ($\mu\Omega \cdot m$)')
plt.title('Oscillations vs $1/B$ → Linear in $1/B$ → Constant Frequency')
plt.grid(True, alpha=0.3)

# Mark period
period = 1 / F
plt.axvline(1/B_max, color='gray', linestyle=':', alpha=0.7)
plt.axvline(1/B_min, color='gray', linestyle=':', alpha=0.7)
plt.text(0.95 * inv_B[0], plt.ylim()[1]*0.9, f'$F = {F}$ T\n$\\Delta(1/B) = {period:.4f}$ T$^{{-1}}$',
         fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat"))

plt.tight_layout()
plt.show()

# ========================
# Output Key Parameters
# ========================
print(f"""
SdH Simulation Results:
-----------------------
Effective mass:     m* = {m_star/m_e:.3f} m_e
Onsager frequency: F = {F} T
Period in 1/B:     Δ(1/B) = {1/F:.5f} T⁻¹
Cyclotron freq @ 60T: ω_c = {e*60/m_star / 1e12:.1f} THz
ω_c τ @ 60T:       {e*60*tau/m_star:.1f}
Dingle temp:       T_D = {T_D:.1f} K
Mobility:          μ = {mu:.1f} m²/Vs
""")
