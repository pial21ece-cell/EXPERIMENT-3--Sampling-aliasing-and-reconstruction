import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMETER DEFINITION
# ==========================================
f1 = 10.0  # Hz (First component)
f2 = 25.0  # Hz (Second component)
f_max = max(f1, f2)
f_nyquist = 2 * f_max  # 50 Hz

# Define sampling frequencies for the 3 cases
sampling_cases = {
    "Above Nyquist Rate (fs = 100 Hz)": 100.0,
    "At Nyquist Rate (fs = 50 Hz)": 50.0,
    "Below Nyquist Rate (fs = 35 Hz)": 35.0
}

# Continuous reference time grid (high resolution for ground truth)
f_cont = 4000.0  # High plotting resolution in Hz
t_duration = 0.5 # Duration in seconds
t_cont = np.linspace(0, t_duration, int(f_cont * t_duration), endpoint=False)

# Signal function x(t) = sin(2*pi*f1*t) + 0.5*sin(2*pi*f2*t)
def signal(t):
    return np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

x_cont = signal(t_cont)

# Sinc Interpolation Reconstruction Function
def sinc_interpolation(x_sampled, t_sampled, t_fine, fs):
    """
    Reconstructs continuous signal using Sinc Interpolation:
    x_rec(t) = sum( x[n] * sinc(fs * (t - n*Ts)) )
    Note: np.sinc(u) calculates sin(pi*u)/(pi*u)
    """
    dt_matrix = (t_fine[:, np.newaxis] - t_sampled[np.newaxis, :]) * fs
    return np.dot(np.sinc(dt_matrix), x_sampled)

# Helper function to compute magnitude spectrum
def compute_spectrum(x, fs):
    N = len(x)
    X_fft = np.fft.rfft(x) / N
    freqs = np.fft.rfftfreq(N, d=1/fs)
    magnitude = 2 * np.abs(X_fft)  # Single-sided magnitude spectrum
    return freqs, magnitude

# ==========================================
# 2. SIMULATION & PLOTTING
# ==========================================
fig, axes = plt.subplots(4, 3, figsize=(18, 14), sharex='row')
fig.suptitle("Experiment 3: Sampling, Aliasing, and Sinc Reconstruction", fontsize=16, fontweight='bold')

for col_idx, (title, fs) in enumerate(sampling_cases.items()):
    # Discrete sampling
    t_sampled = np.arange(0, t_duration, 1/fs)
    x_sampled = signal(t_sampled)
    
    # Sinc Reconstruction
    x_rec = sinc_interpolation(x_sampled, t_sampled, t_cont, fs)
    
    # Reconstruction Error
    rec_error = x_cont - x_rec
    mse = np.mean(rec_error**2)
    
    # Frequency Analysis
    freqs_cont, mag_cont = compute_spectrum(x_cont, f_cont)
    freqs_rec, mag_rec = compute_spectrum(x_rec, f_cont)
    
    # -------------------------------------------------------------
    # Plot 1: Reference vs Sampled Signal
    # -------------------------------------------------------------
    axes[0, col_idx].plot(t_cont, x_cont, 'k--', label='Reference Signal', alpha=0.7)
    axes[0, col_idx].stem(t_sampled, x_sampled, linefmt='r-', markerfmt='ro', basefmt='r-', label='Sampled Points')
    axes[0, col_idx].set_title(title, fontsize=11, fontweight='bold')
    axes[0, col_idx].set_ylabel("Amplitude")
    axes[0, col_idx].grid(True)
    axes[0, col_idx].legend(loc='upper right')
    
    # -------------------------------------------------------------
    # Plot 2: Reconstructed Waveform
    # -------------------------------------------------------------
    axes[1, col_idx].plot(t_cont, x_cont, 'k--', label='Original x(t)', alpha=0.5)
    axes[1, col_idx].plot(t_cont, x_rec, 'b-', label='Sinc Reconstructed')
    axes[1, col_idx].set_ylabel("Amplitude")
    axes[1, col_idx].grid(True)
    axes[1, col_idx].legend(loc='upper right')
    
    # -------------------------------------------------------------
    # Plot 3: Reconstruction Error
    # -------------------------------------------------------------
    axes[2, col_idx].plot(t_cont, rec_error, 'm-')
    axes[2, col_idx].set_ylabel("Error")
    axes[2, col_idx].set_title(f"MSE = {mse:.5f}", fontsize=10)
    axes[2, col_idx].grid(True)
    
    # -------------------------------------------------------------
    # Plot 4: Magnitude Spectrum
    # -------------------------------------------------------------
    axes[3, col_idx].plot(freqs_cont, mag_cont, 'k--', label='Original Spectrum', alpha=0.6)
    axes[3, col_idx].plot(freqs_rec, mag_rec, 'g-', label='Reconstructed Spectrum')
    axes[3, col_idx].set_xlim(0, 60)
    axes[3, col_idx].set_xlabel("Frequency (Hz)")
    axes[3, col_idx].set_ylabel("Magnitude")
    axes[3, col_idx].grid(True)
    axes[3, col_idx].legend(loc='upper right')

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()

# ==========================================
# 3. MANDATORY VALIDATION (CALCULATION)
# ==========================================
print("="*60)
print("MANDATORY VALIDATION: Theoretical Aliased Frequency Calculation")
print("="*60)

fs_under = 35.0  # Below Nyquist rate
print(f"Given parameters:")
print(f"  Component 1: f1 = {f1} Hz (Amplitude = 1.0)")
print(f"  Component 2: f2 = {f2} Hz (Amplitude = 0.5)")
print(f"  Sampling rate (fs): {fs_under} Hz")
print(f"  Nyquist frequency limit (fs/2): {fs_under / 2} Hz\n")

# Aliasing Calculation Formula: f_alias = |f - k * fs| where k is an integer
# For f1 = 10 Hz: 10 Hz < 17.5 Hz -> No Aliasing
f1_alias = abs(f1 - round(f1 / fs_under) * fs_under)

# For f2 = 25 Hz: 25 Hz > 17.5 Hz -> Aliasing occurs
# k = round(25/35) = 1 => f2_alias = |25 - 1*35| = 10 Hz
f2_alias = abs(f2 - round(f2 / fs_under) * fs_under)

print(f"Theoretical Analysis:")
print(f"  - f1 ({f1} Hz): Within Nyquist limit ({fs_under/2} Hz) -> Reconstructed as {f1_alias} Hz.")
print(f"  - f2 ({f2} Hz): Exceeds Nyquist limit ({fs_under/2} Hz) -> Folds to |{f2} - {fs_under}| = {f2_alias} Hz.")
print(f"\nConclusion for Undersampled Case:")
print(f"  The high frequency component ({f2} Hz) aliases directly onto {f1_alias} Hz.")
print(f"  Both components sum at {f1_alias} Hz, corrupting the reconstructed spectrum.")
