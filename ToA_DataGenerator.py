"""
================================================================================
ToA_DataGenerator.py
================================================================================
PURPOSE:
    Generate synthetic single-link multipath ToA (Time of Arrival) channel data.
    This is for testing LIKES, SLIM, and Weighted SPICE algorithms.

SCENARIO:
    - Single transmitter, single receiver (one-way propagation)
    - Single snapshot (one full observation vector across all tones, not one scalar time sample)
    - Narrowband OFDM-like: M frequency subcarriers (M=64)
    - Multiple multipath echoes (propagation paths) each with different delay τ

PHYSICAL MODEL:
    Received signal at frequency f_m and delay τ_k:
        φ(f_m, τ_k) = -2π * f_m * τ_k          [radians, phase rotation]
    
    Steering vector entry:
        a_m(τ_k) = exp(j * φ(f_m, τ_k)) = exp(-j*2π*f_m*τ_k)
    
    Full observation at each frequency f_m:
        y_m = Σ_k exp(-j*2π*f_m*τ_k) * γ_k + noise_m
    where:
        - τ_k = delay of k-th path [seconds]
        - γ_k = sqrt(P_k) * exp(j*φ_k), the complex amplitude of path k
        - φ_k is drawn uniformly from [0, 2π)
        - P_k = power of k-th path [linear scale, not dB]
        - noise_m ~ CN(0, σ²_noise)

VARIABLE MEANINGS & CHOICES:

    M (int): Number of subcarriers / frequency samples
        - Default: 64
        - Why: 2^6, powers of 2 common in OFDM
        
    bw (float, Hz): Bandwidth in Hz
        - Default: 40e6 ⟹ 40 MHz
        - Why: WiFi/UWB-like bandwidth
        
    f0 (float, Hz): Baseband center frequency
        - Default: 0.0 (baseband)
        - Why: Baseband style, consistent with old SPICE code
        
    tau_max (float, seconds): Maximum delay
        - Default: 200e-9 ⟹ 200 nanoseconds
        - Why: Realistic for indoor/short-range propagation
        
    true_tau (array-like or None): True delays
        - If truth_mode == "on_grid": chosen from tau_grid
        ****- If truth_mode == "off_grid": continuous values (thinking to add it later on)
        - Default depends on truth_mode
        
    true_tau_idx (array-like or None): Indices in tau_grid for on_grid mode
        - Only used when truth_mode == "on_grid"
        
    P_true (array-like or None): Power of each multipath component
        - Example: [1, 3, 2] (linear scale, not dB)
        - Default: [1.0, 3.0, 2.0]
        
    SNR_dB (float, dB): Signal-to-noise ratio
        - Definition: SNR = signal_power / noise_power
        - Default: 15.0 dB
        
    delay_step (float, seconds): Spacing of delay grid
        - Default: 0.5e-9 ⟹ 0.5 nanoseconds
        - Why: Fine grid for estimation
        
    truth_mode (str): "on_grid" or "off_grid"
        - "on_grid": clean benchmark, true delays exactly on grid
        - "off_grid": realistic, true delays continuous
        
    random_seed (int or None): For reproducibility

OUTPUT:
    y_noisy (np.array, shape (M,1), complex): Noisy received signal
    A (np.array, shape (M,K), complex): Steering/dictionary matrix
    tau_grid (np.array, shape (K,), float): Delay grid
    true_tau (np.array, shape (C,), float): True delays
    true_gamma (np.array, shape (C,), complex): True complex path amplitudes
    P_true (np.array, shape (C,), float): True powers
    SNR_dB (float): SNR in dB
    noise_power (float): Noise power
    signal_power (float): Signal power

================================================================================
"""

import numpy as np


def steering_vector(tau, f):
    """
    Compute steering vector for delay tau at frequencies f.
    Parameters:
    - tau: scalar delay
    - f: frequency vector, shape (M,1)
    Returns:
    - steering vector, shape (M,1)
    """
    return np.exp(-1j * 2 * np.pi * f * tau)


def generate_toa_data(
    M=64,
    bw=40e6,
    f0=0.0,
    tau_max=200e-9,
    true_tau=None,
    true_tau_idx=None,
    P_true=None,
    SNR_dB=15.0,
    delay_step=0.5e-9,
    truth_mode="on_grid",
    random_seed=None,
):
    """
    Generate synthetic multipath ToA channel data in baseband.
    
    Parameters
    ----------
    M : int
        Number of subcarriers / frequency samples.
        Default: 64.
        
    bw : float
        Bandwidth in Hz.
        Default: 40e6.
        
    f0 : float
        Baseband center frequency (should be 0.0).
        Default: 0.0.
        
    tau_max : float
        Maximum delay in seconds.
        Default: 200e-9.
        
    true_tau : array-like or None
        True delays in seconds (for off_grid mode).
        If None, uses defaults.
        
    true_tau_idx : array-like or None
        Indices in tau_grid (for on_grid mode).
        If None, uses defaults.
        
    P_true : array-like or None
        Linear power of each multipath component.
        If None, defaults to [1.0, 3.0, 2.0].
        
    SNR_dB : float
        Signal-to-noise ratio in dB.
        Default: 15.0.
        
    delay_step : float
        Spacing of delay grid in seconds.
        Default: 0.5e-9.
        
    truth_mode : str
        "on_grid" or "off_grid".
        Default: "on_grid".
        
    random_seed : int or None
        Random seed for reproducibility.
        
    Returns
    -------
    y_noisy : np.ndarray, shape (M,1), complex
        Noisy received signal.
        
    A : np.ndarray, shape (M,K), complex
        Steering matrix.
        
    tau_grid : np.ndarray, shape (K,), float
        Delay grid.
        
    true_tau : np.ndarray, shape (C,), float
        True delays.
        
    true_gamma : np.ndarray, shape (C,), complex
        True complex amplitudes.
        
    P_true : np.ndarray, shape (C,), float
        True powers.
        
    SNR_dB : float
        SNR in dB.
        
    noise_power : float
        Noise power.
        
    signal_power : float
        Signal power.
    """
    
    # RNG setup
    rng = np.random.default_rng(random_seed)
    
    # Frequency grid
    delta_f = bw / M
    f = f0 + (np.arange(M).reshape(-1,1)) * delta_f
    
    # Delay grid
    K = int(round(tau_max / delay_step)) + 1
    tau_grid = np.linspace(0.0, tau_max, K)
    
    # Steering matrix
    A = np.hstack([steering_vector(tau, f) for tau in tau_grid])
    
    # Truth generation based on mode
    if truth_mode == "on_grid":
        if true_tau_idx is None:
            # Exact on-grid defaults for delay_step = 0.5 ns
            true_tau_idx = [
                int(round(20e-9 / delay_step)),
                int(round(80e-9 / delay_step)),
                int(round(149.5e-9 / delay_step)),
            ]
        true_tau_idx = np.asarray(true_tau_idx, dtype=int)

        if not (np.all(true_tau_idx >= 0) and np.all(true_tau_idx < K)):
            raise ValueError("true_tau_idx out of range")

        true_tau = tau_grid[true_tau_idx]

    elif truth_mode == "off_grid":
        if true_tau is None:
            true_tau = np.array([20.3e-9, 79.7e-9, 149.2e-9], dtype=float)
        else:
            true_tau = np.asarray(true_tau, dtype=float)

        if not (np.all(true_tau >= 0) and np.all(true_tau <= tau_max)):
            raise ValueError("true_tau out of range")

        true_tau_idx = None

    else:
        raise ValueError("truth_mode must be 'on_grid' or 'off_grid'")
    
    # Powers
    if P_true is None:
        P_true = np.array([1.0, 3.0, 2.0])
    P_true = np.asarray(P_true, dtype=float)
    C = len(true_tau)
    if len(P_true) != C:
        raise ValueError("len(P_true) must equal number of true paths")
    
    # True amplitudes
    phi_true = rng.uniform(0.0, 2.0*np.pi, size=C)
    true_gamma = np.sqrt(P_true) * np.exp(1j * phi_true)
    
    # Noiseless signal
    if truth_mode == "on_grid":
        y = A[:, true_tau_idx] @ true_gamma.reshape(-1,1)
    else:
        A_true = np.hstack([steering_vector(tau, f) for tau in true_tau])
        y = A_true @ true_gamma.reshape(-1,1)
    
    # SNR and noise
    signal_power = np.mean(np.abs(y)**2)
    noise_power = signal_power / (10**(SNR_dB/10))
    noise = np.sqrt(noise_power/2) * (rng.standard_normal((M,1)) + 1j*rng.standard_normal((M,1)))
    y_noisy = y + noise
    
    return y_noisy, A, tau_grid, true_tau, true_gamma, P_true, SNR_dB, noise_power, signal_power


def validate_data_consistency(y_noisy, A, tau_grid, true_tau, true_gamma, P_true, SNR_dB, noise_power, signal_power):
    """
    Sanity checks on generated data.
    Returns: True if consistent, False otherwise.
    """
    checks_passed = True
    
    # Check shapes
    M, _ = y_noisy.shape
    M_A, K = A.shape
    C = len(true_tau)
    
    if M_A != M:
        print(f"ERROR: Steering matrix row count {M_A} ≠ signal length {M}")
        checks_passed = False
    
    if len(P_true) != C or len(true_gamma) != C:
        print(f"ERROR: Inconsistent number of paths: P_true {len(P_true)}, true_gamma {len(true_gamma)}, true_tau {C}")
        checks_passed = False
    
    # Check powers
    measured_power = np.mean(np.abs(y_noisy)**2)
    expected_power = signal_power + noise_power
    
    print(f"Data shape OK: y_noisy∈ℂ^[{M}×1], A∈ℂ^[{M}×{K}]")
    print(f"True paths: C={C}, τ_true={true_tau*1e9} ns, P_true={P_true}")
    print(f"SNR_dB={SNR_dB}, noise_power={noise_power:.3e}, signal_power={signal_power:.3f}")
    print(f"Measured received power ~ {measured_power:.3f} (expected ~{expected_power:.3f})")
    
    return checks_passed


if __name__ == "__main__":
    # Test data generation
    print("=" * 80)
    print("Testing ToA_DataGenerator.generate_toa_data()")
    print("=" * 80)
    
    y_noisy, A, tau_grid, true_tau, true_gamma, P_true, SNR_dB, noise_power, signal_power = generate_toa_data(
        M=64,
        bw=40e6,
        f0=0,
        tau_max=200e-9,
        SNR_dB=15,
        delay_step=0.5e-9,
        truth_mode="on_grid",
        random_seed=42,
    )
    
    print(f"\nGenerated data shapes:")
    print(f"  y_noisy (received signal): {y_noisy.shape} complex")
    print(f"  A (steering matrix):       {A.shape} complex")
    print(f"  tau_grid (delay grid):     {tau_grid.shape} real")
    print(f"  true_tau (true delays):    {true_tau.shape} real")
    print(f"  true_gamma (true amps):    {true_gamma.shape} complex")
    print(f"  P_true (true powers):      {P_true.shape} real")
    
    validate_data_consistency(y_noisy, A, tau_grid, true_tau, true_gamma, P_true, SNR_dB, noise_power, signal_power)
    
    print(f"\nParameter values:")
    print(f"  Delay range:     {tau_grid[0]*1e9:.1f} to {tau_grid[-1]*1e9:.1f} ns")
    print(f"  Delay grid step: {(tau_grid[1]-tau_grid[0])*1e12:.3f} ps")
    print(f"  Frequency span:  {40e6/1e6:.1f} MHz")
    print(f"  Subcarriers:     {y_noisy.shape[0]}")
    print(f"  SNR:             {SNR_dB} dB")
    print(f"  Signal power:    {signal_power:.3f}")
    print(f"  Noise power:     {noise_power:.3e}")
