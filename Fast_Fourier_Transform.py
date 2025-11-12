import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, hbar, e, k, m_e
from scipy.signal import find_peaks, blackmanharris

# ========================
# Simulation Parameters
# ========================

m_star = 0.15 * m_e
n = 1e17
tau = 1e-12
T = 0.3
g_factor = 20
F_true = 500          # True Onsager frequency (T)
p_max = 3

B_min, B_max = 10, 60
B = np.linspace(B_min, B_max, 10000)  # High resolution for FFT
inv_B = 1 / B

omega_c = e * B / m_star
T_D = hbar / (2 * np.pi * k * tau)
mu = e * tau / m_star
wc_tau = omega_c * tau

rho_0 = m_star / (n * e**2 * tau)
MR_factor = 1 + (wc_tau)**2

# ========================
# LK Factors
# ========================

def R_T(p, B):
    x = 2 * np.pi**2 * p * k * T / (hbar * e * B / m_star)
    return x / np.sinh(x + 1e-12)  # Avoid sinh(0)

def R_D(p, B):
    return np.exp(-2 * np.pi**2 * p * k * T_D / (hbar * e * B / m_star))

def R_s(p, B):
    nu = g_factor * m_star / (2 * m_e)
    return np.cos(np.pi * p * nu)

def sdH_osc(B, p, F):
    phase = 2 * np.pi * (p * F / B - 0.5)
    amp = np.sqrt(B) * R_T(p, B) * R_D(p, B) * R_s(p, B)
    return amp * np.cos(phase)

# Total oscillation
delta_rho = np.zeros_like(B)
for p in range(1, p_max + 1):
    delta_rho += sdH_osc(B, p, F_true)

# Scale
delta_rho *= 0.08 * rho_0

# Total resistivity
rho_xx = rho_0 * MR_factor + delta_rho

# ========================
# FFT Analysis
# ========================

# Step 1: Extract oscillatory part
rho_osc = rho_xx - rho_0 * MR_factor  # Subtract classical MR

# Step 2: Interpolate to uniform 1/B grid
inv_B_uniform = np.linspace(inv_B.min(), inv_B.max(), len(B))
rho_osc_uniform = np.interp(inv_B_uniform, inv_B, rho_osc)

# Step 3: Apply window (Blackman-Harris)
window = blackmanharris(len(inv_B_uniform))
rho_windowed = rho_osc_uniform * window

# Step 4: FFT
N = len(inv_B_uniform)
fft_freq = np.fft.rfftfreq(N, d=(inv_B_uniform[1] - inv_B_uniform[0]))
fft_amp = np.abs(np.fft.rfft(rho_windowed))

# Normalize
fft_amp /= np.max(fft_amp)

# Step 5: Find peaks
peaks, _ = find_peaks(fft_amp, height=0.05, distance=50)
peak_freqs = fft_freq[peaks]
peak_amps = fft_amp[peaks]

# Sort by amplitude
idx = np.argsort(peak_amps)[::-1]
peak_freqs = peak_freqs[idx]
peak_amps = peak_amps[idx]

# ========================
# Plotting
# ========================

plt.figure(figsize=(14, 10))

# --- 1. rho vs B ---
plt.subplot(2, 2, 1)
plt.plot(B, rho_xx * 1e6, 'b-', lw=1.2, label=r'$\rho_{xx}(B)$')
plt.plot(B, rho_0 * MR_factor * 1e6, 'k--', alpha=0.7, label='Classical MR')
plt.xlabel('B (T)')
plt.ylabel(r'$\rho_{xx}$ ($\mu\Omega \cdot m$)')
plt.title('SdH Oscillations')
plt.legend()
plt.grid(True, alpha=0.3)

# --- 2. rho vs 1/B ---
plt.subplot(2, 2, 2)
plt.plot(inv_B_uniform, rho_osc_uniform * 1e6, 'r-', lw=1)
plt.xlabel(r'$1/B$ (T$^{-1}$)')
plt.ylabel(r'$\Delta\rho_{xx}$ ($\mu\Omega \cdot m$)')
plt.title('Oscillatory Part vs $1/B$')
plt.grid(True, alpha=0.3)

# --- 3. FFT Power Spectrum ---
plt.subplot(2, 2, (3,4))
plt.plot(fft_freq, fft_amp, 'g-', lw=1.2)
plt.plot(peak_freqs, peak_amps, 'ro', ms=8, label='Detected Peaks')
for i, (f, a) in enumerate(zip(peak_freqs[:3], peak_amps[:3])):
    plt.text(f, a + 0.05, f'$F_{i+1} = {f:.0f}$ T', fontsize=10, ha='center')
plt.xlim(0, 2000)
plt.xlabel('Frequency $F$ (Tesla)')
plt.ylabel('Normalized FFT Amplitude')
plt.title('FFT Analysis → Extracted Onsager Frequencies')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ========================
# Print Results
# ========================

print(f"""
SdH + FFT Analysis Results
{'='*50}
True Frequency:        F = {F_true} T

Detected Frequencies (Top 3):
{'-'*40}
""")
for i in range(min(3, len(peak_freqs))):
    f = peak_freqs[i]
    error = abs(f - (i+1)*F_true)
    print(f"  Harmonic p={i+1}:  F = {f:.1f} T  (ΔF = {error:.1f} T)")

print(f"\nParameters:")
print(f"  m* = {m_star/m_e:.3f} m_e")
print(f"  T = {T} K,  T_D = {T_D:.1f} K")
print(f"  μ = {mu:.1f} m²/Vs")
print(f"  ω_c τ @ 60T = {e*60*tau/m_star:.1f}")
