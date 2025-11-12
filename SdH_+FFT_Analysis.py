import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, e, k, m_e
from scipy.signal import blackmanharris, find_peaks

# =============================================
#  Parameters
# =============================================
m_star = 0.15 * m_e
F      = 500.0
T      = 0.3
T_D    = 10.4
g      = 20.0
p_max  = 3

B = np.linspace(10, 60, 5000)  # high resolution for FFT
omega_c = e * B / m_star

# =============================================
#  LK Factors
# =============================================
def R_T(p, B):
    x = 2 * np.pi**2 * p * k * T / (hbar * omega_c)
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
#  SdH Oscillatory Signal
# =============================================
osc = np.zeros_like(B)
for p in range(1, p_max + 1):
    phase = 2 * np.pi * (p * F / B - 0.5)
    amp = curvature(B) * R_T(p, B) * R_D(p, B) * R_s(p, B)
    osc += amp * np.cos(phase)

# Background + signal
rho_0 = 1.0
tau = 1e-12
MR_factor = 1 + (e * B * tau / m_star)**2
rho_bg = rho_0 * MR_factor
rho_xx = rho_bg + 0.08 * osc

# =============================================
#  FFT Analysis
# =============================================
# 1. Extract oscillatory part
rho_osc = rho_xx - rho_bg

# 2. Uniform 1/B grid
inv_B = 1.0 / B
inv_B_uniform = np.linspace(inv_B.min(), inv_B.max(), len(B))
rho_osc_uniform = np.interp(inv_B_uniform, inv_B, rho_osc)

# 3. Apply window
window = blackmanharris(len(rho_osc_uniform))
rho_windowed = rho_osc_uniform * window

# 4. FFT
N = len(rho_osc_uniform)
d_invB = inv_B_uniform[1] - inv_B_uniform[0]
fft_freq = np.fft.rfftfreq(N, d=d_invB)
fft_amp = np.abs(np.fft.rfft(rho_windowed))
fft_amp /= np.max(fft_amp)  # normalize

# 5. Peak detection
peaks, _ = find_peaks(fft_amp, height=0.05, distance=100)
peak_freqs = fft_freq[peaks]
peak_amps = fft_amp[peaks]
idx = np.argsort(peak_amps)[::-1]
peak_freqs = peak_freqs[idx]
peak_amps = peak_amps[idx]

# =============================================
#  Plot: 3 Panels
# =============================================
plt.figure(figsize=(14, 10))

# --- Panel 1: SdH vs B ---
plt.subplot(3, 1, 1)
plt.plot(B, rho_xx, 'b-', lw=1.2, label=r'$\rho_{xx}(B)$')
plt.plot(B, rho_bg, 'k--', alpha=0.7, label='Classical MR')
plt.fill_between(B, rho_bg, rho_xx, color='blue', alpha=0.15)
plt.xlabel('B (T)')
plt.ylabel(r'$\rho_{xx}$ (a.u.)')
plt.title('SdH Oscillations (Full LK)')
plt.legend()
plt.grid(alpha=0.3)

# --- Panel 2: Oscillatory Part vs 1/B ---
plt.subplot(3, 1, 2)
plt.plot(inv_B_uniform, rho_osc_uniform, 'r-', lw=1)
plt.xlabel(r'$1/B$ (T$^{-1}$)')
plt.ylabel(r'$\Delta\rho_{xx}$')
plt.title('Oscillatory Signal vs $1/B$')
plt.grid(alpha=0.3)

# --- Panel 3: FFT Spectrum ---
plt.subplot(3, 1, 3)
plt.plot(fft_freq, fft_amp, 'g-', lw=1.2)
plt.plot(peak_freqs, peak_amps, 'ro', ms=8, label='Detected Peaks')
for i, (f, a) in enumerate(zip(peak_freqs[:3], peak_amps[:3])):
    plt.text(f, a + 0.05, f'$F_{i+1} = {f:.0f}$ T', fontsize=11, ha='center')
plt.xlim(0, 1800)
plt.xlabel('Frequency $F$ (Tesla)')
plt.ylabel('Normalized Amplitude')
plt.title('FFT → Extracted Onsager Frequencies')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================
#  Results
# =============================================
print(f"""
FFT ANALYSIS RESULTS
{'='*50}
True Frequency:        F = {F} T
Detected Frequencies (Top 3):
""")
for i in range(min(3, len(peak_freqs))):
    f = peak_freqs[i]
    err = abs(f - (i+1)*F)
    print(f"  p={i+1}:  F = {f:.1f} T  (ΔF = {err:.1f} T)")

print(f"\nParameters:")
print(f"  m* = {m_star/m_e:.3f} m_e,  T = {T} K,  T_D = {T_D} K,  g = {g}")
print(f"  Harmonics: p = 1 to {p_max}")
print(f"{'='*50}")
