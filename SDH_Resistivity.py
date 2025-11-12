import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, e, k, m_e

# --------------------------------------------------------------
#  Parameters (same as in the Bi₂Se₃ example)
# --------------------------------------------------------------
m_star = 0.15 * m_e          # effective mass
F      = 500.0               # Onsager frequency (T)
T      = 0.3                 # temperature (K)
T_D    = 10.4                # Dingle temperature (K)
g      = 20.0                # g‑factor
p      = 1                   # harmonic (fundamental)

# Magnetic‑field range
B = np.linspace(10, 60, 1000)          # 10 → 60 T
omega_c = e * B / m_star               # cyclotron frequency

# --------------------------------------------------------------
#  1. Thermal factor R_T = x / sinh(x)
# --------------------------------------------------------------
x_T = 2 * np.pi**2 * p * k * T / (hbar * omega_c)   # can be > 700 at low B
#  sinh(x) → ∞ → use the stable form  x / sinh(x) = 2x * exp(-x) for x>>1
R_T = np.where(
    x_T < 30,
    x_T / np.sinh(x_T + 1e-12),                # safe for moderate x
    2 * x_T * np.exp(-x_T)                     # asymptotic limit for x ≫ 1
)

# --------------------------------------------------------------
#  2. Dingle factor R_D = exp(-x_D)
# --------------------------------------------------------------
x_D = 2 * np.pi**2 * p * k * T_D / (hbar * omega_c)
R_D = np.exp(-x_D)

# --------------------------------------------------------------
#  3. Spin factor R_s = cos(π p g m*/2m_e)
# --------------------------------------------------------------
nu = g * m_star / (2 * m_e)                # dimensionless
R_s = np.cos(np.pi * p * nu) * np.ones_like(B)   # constant for a given harmonic

# --------------------------------------------------------------
#  4. Curvature prefactor √B
# --------------------------------------------------------------
curvature = np.sqrt(B)

# --------------------------------------------------------------
#  Plot
# --------------------------------------------------------------
plt.figure(figsize=(10, 6))

plt.plot(B, curvature,   label=r'$\sqrt{B}$ (curvature)', lw=2)
plt.plot(B, R_T,         label=r'$R_T$ (thermal)',       lw=2)
plt.plot(B, R_D,         label=r'$R_D$ (Dingle)',        lw=2)
plt.plot(B, R_s,         label=r'$R_s$ (spin)',          lw=2)
plt.plot(B, curvature * R_T * R_D * R_s,
         'k--', lw=3, label='Total amplitude factor')

plt.xlabel('Magnetic field $B$ (T)')
plt.ylabel('LK factor')
plt.title('Lifshitz‑Kosevich amplitude factors')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
