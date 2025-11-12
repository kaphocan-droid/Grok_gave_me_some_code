import numpy as np
import matplotlib.pyplot as plt

B = np.linspace(10, 60, 1000)
omega_c = 1.76e11 * B  # m* = 0.15 m_e
F = 500
p = 1

# LK factors
x_T = 2*np.pi**2 * p * 8.617e-5 * 0.3 / (1.054e-34 * omega_c)
R_T = x_T / np.sinh(x_T)

T_D = 10.4
x_D = 2*np.pi**2 * p * 8.617e-5 * T_D / (1.054e-34 * omega_c)
R_D = np.exp(-x_D)

g = 20; m_star = 0.15*9.11e-31
R_s = np.cos(np.pi * p * g * m_star / (2*9.11e-31))

plt.figure(figsize=(10,6))
plt.plot(B, np.sqrt(B), label=r'$\sqrt{B}$ (curvature)', lw=2)
plt.plot(B, R_T, label=r'$R_T$ (thermal)', lw=2)
plt.plot(B, R_D, label=r'$R_D$ (Dingle)', lw=2)
plt.plot(B, R_s, label=r'$R_s$ (spin)', lw=2)
plt.plot(B, np.sqrt(B)*R_T*R_D*R_s, 'k--', lw=3, label='Total Amplitude')
plt.xlabel('B (T)'); plt.ylabel('Factor')
plt.title('Lifshitz-Kosevich Factors')
plt.legend(); plt.grid(alpha=0.3)
plt.show()
