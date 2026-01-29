"""
MOND Core Analysis Module
=========================

This module implements the MOND analysis from the Channel Impedance Framework:
- Effective mass deviation at low accelerations
- Impedance-based coupling efficiency
- Frequency-dependent signatures

Theory
------
The framework derives MOND from impedance matching between mass and spacetime.
At accelerations below a₀ ≈ 1.2×10⁻¹⁰ m/s², the coupling efficiency decreases,
leading to the characteristic MOND force law: F = m·√(a·a₀)
"""

import numpy as np
from scipy import signal, optimize
from scipy.stats import chi2
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
import json
from pathlib import Path


# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

A_0 = 1.2e-10       # m/s², MOND acceleration scale
C = 3e8             # m/s, speed of light  
SIGMA_LOG = 1.5     # impedance matching width parameter
G = 6.674e-11       # m³/(kg·s²), gravitational constant
HBAR = 1.055e-34    # J·s, reduced Planck constant


# ============================================================================
# MOND PHYSICS FROM FRAMEWORK
# ============================================================================

def mond_frequency(acceleration: float) -> float:
    """
    Calculate the characteristic frequency associated with an acceleration.
    
    From the framework: f_acc = a / (2πc)
    
    At low accelerations, this frequency drops below the gravitational channel
    coupling threshold, causing impedance mismatch.
    
    Parameters
    ----------
    acceleration : float
        Acceleration in m/s²
        
    Returns
    -------
    float
        Characteristic frequency in Hz
    """
    return acceleration / (2 * np.pi * C)


def impedance_matching(f: float, f_0: float, sigma: float = SIGMA_LOG) -> float:
    """
    Calculate resonance coupling efficiency from the framework.
    
    R(f, f₀) = exp(-ln(f/f₀)² / (2σ²))
    
    Parameters
    ----------
    f : float
        Signal frequency
    f_0 : float
        Resonance frequency
    sigma : float
        Bandwidth parameter (default: SIGMA_LOG)
        
    Returns
    -------
    float
        Coupling efficiency between 0 and 1
    """
    if f <= 0 or f_0 <= 0:
        return 0.0
    log_ratio = np.log(f / f_0)
    return np.exp(-log_ratio**2 / (2 * sigma**2))


def mond_interpolation_standard(a: np.ndarray, a_0: float = A_0) -> np.ndarray:
    """
    Standard MOND interpolation function (simple form).
    
    μ(x) = x / √(1 + x²)  where x = a/a₀
    
    Behavior:
    - For a >> a₀: μ → 1 (Newtonian)
    - For a << a₀: μ → a/a₀ (deep MOND)
    
    Parameters
    ----------
    a : array-like
        Acceleration(s) in m/s²
    a_0 : float
        MOND critical acceleration (default: A_0)
        
    Returns
    -------
    ndarray
        MOND interpolation function values
    """
    x = np.asarray(a) / a_0
    return x / np.sqrt(1 + x**2)


def mond_interpolation_rar(a: np.ndarray, a_0: float = A_0) -> np.ndarray:
    """
    MOND interpolation from Radial Acceleration Relation (McGaugh 2016).
    
    μ(x) = 1 / (1 - exp(-√x))  where x = a/a₀
    
    This form better fits galaxy rotation curves.
    
    Parameters
    ----------
    a : array-like
        Acceleration(s) in m/s²
    a_0 : float
        MOND critical acceleration (default: A_0)
        
    Returns
    -------
    ndarray
        RAR interpolation function values
    """
    x = np.asarray(a) / a_0
    # Avoid division by zero
    exp_term = np.exp(-np.sqrt(np.maximum(x, 1e-20)))
    return 1.0 / (1.0 - exp_term)


def framework_effective_mass_ratio(a: np.ndarray, a_0: float = A_0) -> np.ndarray:
    """
    Calculate effective inertial mass ratio from the framework.
    
    From the derivation: m_eff/m₀ = 1/√R where R is impedance coupling efficiency
    
    In the MOND regime (a < a₀): m_eff/m₀ ≈ √(a₀/a)
    
    This gives F = m₀·√(a·a₀) in deep MOND.
    
    Parameters
    ----------
    a : array-like
        Acceleration(s) in m/s²
    a_0 : float
        MOND critical acceleration (default: A_0)
        
    Returns
    -------
    ndarray
        Effective mass ratio m_eff/m₀
    """
    a = np.asarray(a)
    ratio = np.ones_like(a, dtype=float)
    
    # Deep MOND regime
    deep_mond = a < 0.1 * a_0
    ratio[deep_mond] = np.sqrt(a_0 / a[deep_mond])
    
    # Transition regime
    transition = (a >= 0.1 * a_0) & (a <= 10 * a_0)
    if np.any(transition):
        mu = mond_interpolation_standard(a[transition], a_0)
        ratio[transition] = 1.0 / mu
    
    # Newtonian regime: ratio = 1
    return ratio


def framework_force(m: float, a: np.ndarray, a_0: float = A_0) -> np.ndarray:
    """
    Calculate force at given acceleration using the framework.
    
    In Newtonian regime: F = m·a
    In MOND regime: F = m·a/μ(a/a₀) = m·√(a·a₀)
    
    Parameters
    ----------
    m : float
        Mass in kg
    a : array-like
        Acceleration(s) in m/s²
    a_0 : float
        MOND critical acceleration (default: A_0)
        
    Returns
    -------
    ndarray
        Force in Newtons
    """
    mu = mond_interpolation_standard(a, a_0)
    return m * a / mu


# ============================================================================
# DATA ANALYSIS CLASSES AND FUNCTIONS  
# ============================================================================

@dataclass
class MONDFitResult:
    """Results from fitting MOND vs Newtonian model."""
    newtonian_chi2: float
    mond_chi2: float
    newtonian_params: Dict
    mond_params: Dict
    mond_preferred: bool
    delta_chi2: float
    p_value: float
    fitted_a0: float
    residuals_newton: np.ndarray
    residuals_mond: np.ndarray


def fit_mond_model(
    acceleration: np.ndarray,
    measured_force: np.ndarray,
    force_uncertainty: Optional[np.ndarray] = None,
    mass_estimate: float = 1.0
) -> MONDFitResult:
    """
    Fit both Newtonian and MOND models to force-acceleration data.
    
    Performs least-squares fitting of both models and computes
    statistical comparison metrics.
    
    Parameters
    ----------
    acceleration : array
        Measured accelerations (m/s²)
    measured_force : array  
        Measured forces (N)
    force_uncertainty : array, optional
        Uncertainties in force measurements
    mass_estimate : float
        Initial estimate for mass (kg)
        
    Returns
    -------
    MONDFitResult
        Contains fit statistics and model comparison
    """
    if force_uncertainty is None:
        force_uncertainty = np.ones_like(measured_force)
    
    weights = 1.0 / force_uncertainty**2
    
    # Newtonian model: F = m·a
    def newton_residuals(params):
        m = params[0]
        model = m * acceleration
        return (measured_force - model) * np.sqrt(weights)
    
    # MOND model: F = m·a/μ(a/a₀)
    def mond_residuals(params):
        m, a0 = params
        mu = mond_interpolation_standard(acceleration, a0)
        model = m * acceleration / mu
        return (measured_force - model) * np.sqrt(weights)
    
    # Fit Newtonian
    result_newton = optimize.least_squares(
        newton_residuals, 
        x0=[mass_estimate],
        bounds=([0], [np.inf])
    )
    m_newton = result_newton.x[0]
    chi2_newton = np.sum(result_newton.fun**2)
    residuals_newton = measured_force - m_newton * acceleration
    
    # Fit MOND
    result_mond = optimize.least_squares(
        mond_residuals,
        x0=[mass_estimate, A_0],
        bounds=([0, 1e-15], [np.inf, 1e-5])
    )
    m_mond, a0_fit = result_mond.x
    chi2_mond = np.sum(result_mond.fun**2)
    mu_fit = mond_interpolation_standard(acceleration, a0_fit)
    residuals_mond = measured_force - m_mond * acceleration / mu_fit
    
    # Model comparison
    n_data = len(acceleration)
    
    # F-test for nested models
    delta_chi2 = chi2_newton - chi2_mond
    
    # p-value from chi-squared distribution
    p_value = 1 - chi2.cdf(delta_chi2, df=1)
    
    # MOND preferred if significant improvement (p < 0.05)
    mond_preferred = (p_value < 0.05) and (chi2_mond < chi2_newton)
    
    return MONDFitResult(
        newtonian_chi2=chi2_newton,
        mond_chi2=chi2_mond,
        newtonian_params={'mass': m_newton},
        mond_params={'mass': m_mond, 'a0': a0_fit},
        mond_preferred=mond_preferred,
        delta_chi2=delta_chi2,
        p_value=p_value,
        fitted_a0=a0_fit,
        residuals_newton=residuals_newton,
        residuals_mond=residuals_mond
    )


def analyze_residual_spectrum(
    time: np.ndarray,
    residual: np.ndarray,
    sample_rate: float
) -> Dict:
    """
    Analyze power spectrum of residuals for framework signatures.
    
    The framework predicts:
    1. Enhanced low-frequency power (long coherence times)
    2. Possible peaks at orbital harmonics
    3. 24-hour modulation from galactic acceleration direction
    
    Parameters
    ----------
    time : array
        Time values
    residual : array
        Residual values
    sample_rate : float
        Sampling rate in Hz
        
    Returns
    -------
    dict
        Spectral analysis results
    """
    results = {}
    
    # Basic statistics
    results['mean'] = np.mean(residual)
    results['std'] = np.std(residual)
    results['n_points'] = len(residual)
    results['mean_error'] = results['std'] / np.sqrt(len(residual))
    results['mean_significance'] = abs(results['mean']) / results['mean_error']
    
    # Power spectral density
    nperseg = min(len(residual) // 4, 2048)
    if nperseg < 16:
        results['psd_computed'] = False
        return results
        
    freqs, psd = signal.welch(residual, fs=sample_rate, nperseg=nperseg)
    results['psd_computed'] = True
    results['frequencies'] = freqs
    results['psd'] = psd
    
    # Low-frequency excess (framework signature)
    freq_threshold = sample_rate / 100
    low_freq_mask = freqs < freq_threshold
    high_freq_mask = freqs >= freq_threshold
    
    if np.any(low_freq_mask) and np.any(high_freq_mask):
        low_power = np.mean(psd[low_freq_mask])
        high_power = np.mean(psd[high_freq_mask])
        results['low_freq_excess'] = low_power / high_power
        results['excess_significant'] = results['low_freq_excess'] > 3.0
    
    # Check for 24-hour periodicity
    duration = (time[-1] - time[0])
    if duration > 2 * 86400:  # At least 2 days
        freq_24h = 1.0 / 86400  # Hz
        if freq_24h < freqs[-1]:
            idx_24h = np.argmin(np.abs(freqs - freq_24h))
            window = max(1, len(freqs) // 50)
            local_bg = np.median(psd[max(0, idx_24h-window):min(len(psd), idx_24h+window)])
            results['power_24h'] = psd[idx_24h]
            results['background_24h'] = local_bg
            results['significance_24h'] = psd[idx_24h] / local_bg
    
    return results


# ============================================================================
# SIMULATION FOR TESTING
# ============================================================================

def generate_synthetic_data(
    n_points: int = 1000,
    a_min: float = 1e-12,
    a_max: float = 1e-8,
    mass: float = 0.1,
    noise_level: float = 1e-15,
    include_mond: bool = True,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic force-acceleration data for testing.
    
    Parameters
    ----------
    n_points : int
        Number of data points
    a_min, a_max : float
        Acceleration range (m/s²)
    mass : float
        Test mass (kg)
    noise_level : float
        Force measurement noise (N)
    include_mond : bool
        If True, generate data with MOND physics; if False, Newtonian only
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    acceleration, true_force, measured_force : arrays
    """
    np.random.seed(seed)
    
    # Logarithmically spaced accelerations
    acceleration = np.logspace(np.log10(a_min), np.log10(a_max), n_points)
    
    if include_mond:
        true_force = framework_force(mass, acceleration)
    else:
        true_force = mass * acceleration
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, n_points)
    measured_force = true_force + noise
    
    return acceleration, true_force, measured_force


# ============================================================================
# MAIN ANALYSIS ROUTINE
# ============================================================================

def run_mond_analysis(
    acceleration: np.ndarray,
    force: np.ndarray,
    force_error: Optional[np.ndarray] = None,
    mass_estimate: float = 1.0,
    plot: bool = True,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Complete MOND analysis pipeline.
    
    Parameters
    ----------
    acceleration : array
        Acceleration values (m/s²)
    force : array
        Force measurements (N)
    force_error : array, optional
        Force uncertainties (N)
    mass_estimate : float
        Estimated mass (kg)
    plot : bool
        Whether to generate plots
    output_dir : Path, optional
        Directory for output files
        
    Returns
    -------
    dict
        Analysis results including fit statistics
    """
    results = {}
    
    # 1. Data summary
    results['n_points'] = len(acceleration)
    results['a_min'] = float(np.min(acceleration))
    results['a_max'] = float(np.max(acceleration))
    results['a_0_regime'] = float(np.sum(acceleration < A_0)) / len(acceleration)
    results['transition_regime'] = float(np.sum(
        (acceleration >= 0.1*A_0) & (acceleration <= 10*A_0)
    )) / len(acceleration)
    
    print(f"Data Summary:")
    print(f"  Points: {results['n_points']}")
    print(f"  Acceleration range: {results['a_min']:.2e} to {results['a_max']:.2e} m/s²")
    print(f"  Fraction in MOND regime (a < a₀): {results['a_0_regime']:.1%}")
    print(f"  Fraction in transition (0.1·a₀ < a < 10·a₀): {results['transition_regime']:.1%}")
    
    # 2. Fit models
    print("\nFitting models...")
    fit_result = fit_mond_model(
        acceleration, force, force_error, mass_estimate
    )
    
    results['newtonian_chi2'] = fit_result.newtonian_chi2
    results['mond_chi2'] = fit_result.mond_chi2
    results['delta_chi2'] = fit_result.delta_chi2
    results['p_value'] = fit_result.p_value
    results['fitted_a0'] = fit_result.fitted_a0
    results['mond_preferred'] = fit_result.mond_preferred
    results['mass_newton'] = fit_result.newtonian_params['mass']
    results['mass_mond'] = fit_result.mond_params['mass']
    
    print(f"\nFit Results:")
    print(f"  Newtonian χ²: {fit_result.newtonian_chi2:.2f}")
    print(f"  MOND χ²: {fit_result.mond_chi2:.2f}")
    print(f"  Δχ²: {fit_result.delta_chi2:.2f}")
    print(f"  p-value: {fit_result.p_value:.4f}")
    print(f"  Fitted a₀: {fit_result.fitted_a0:.2e} m/s² (expected: {A_0:.2e})")
    print(f"  MOND preferred: {fit_result.mond_preferred}")
    
    # 3. Residual analysis
    print("\nAnalyzing Newtonian residuals...")
    time_proxy = np.arange(len(acceleration), dtype=float)
    sample_rate = 1.0
    
    spectrum_results = analyze_residual_spectrum(
        time_proxy, fit_result.residuals_newton, sample_rate
    )
    
    results['residual_mean'] = spectrum_results['mean']
    results['residual_std'] = spectrum_results['std']
    results['mean_significance'] = spectrum_results['mean_significance']
    
    if 'low_freq_excess' in spectrum_results:
        results['low_freq_excess'] = spectrum_results['low_freq_excess']
        print(f"  Low-frequency power excess: {spectrum_results['low_freq_excess']:.2f}×")
    
    # 4. Plots
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Force vs acceleration
        ax = axes[0, 0]
        ax.loglog(acceleration, force, 'b.', alpha=0.5, label='Data')
        a_plot = np.logspace(np.log10(acceleration.min()), np.log10(acceleration.max()), 100)
        ax.loglog(a_plot, fit_result.newtonian_params['mass'] * a_plot, 'g-', 
                  label=f'Newtonian (m={fit_result.newtonian_params["mass"]:.4f})')
        ax.loglog(a_plot, framework_force(fit_result.mond_params['mass'], a_plot, fit_result.fitted_a0),
                  'r--', label=f'MOND (a₀={fit_result.fitted_a0:.2e})')
        ax.axvline(A_0, color='orange', linestyle=':', label=f'a₀={A_0:.2e}')
        ax.set_xlabel('Acceleration (m/s²)')
        ax.set_ylabel('Force (N)')
        ax.legend()
        ax.set_title('Force vs Acceleration')
        
        # Residuals vs acceleration
        ax = axes[0, 1]
        ax.semilogx(acceleration, fit_result.residuals_newton, 'g.', alpha=0.5, label='Newton')
        ax.semilogx(acceleration, fit_result.residuals_mond, 'r.', alpha=0.5, label='MOND')
        ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
        ax.axvline(A_0, color='orange', linestyle=':', alpha=0.7)
        ax.set_xlabel('Acceleration (m/s²)')
        ax.set_ylabel('Residual Force (N)')
        ax.legend()
        ax.set_title('Fit Residuals')
        
        # Residual histogram
        ax = axes[1, 0]
        ax.hist(fit_result.residuals_newton, bins=50, alpha=0.5, label='Newton', density=True)
        ax.hist(fit_result.residuals_mond, bins=50, alpha=0.5, label='MOND', density=True)
        ax.set_xlabel('Residual (N)')
        ax.set_ylabel('Probability Density')
        ax.legend()
        ax.set_title('Residual Distribution')
        
        # Power spectrum
        ax = axes[1, 1]
        if spectrum_results.get('psd_computed', False):
            ax.loglog(spectrum_results['frequencies'], spectrum_results['psd'])
            ax.set_xlabel('Frequency')
            ax.set_ylabel('Power Spectral Density')
            ax.set_title('Newtonian Residual Spectrum')
        else:
            ax.text(0.5, 0.5, 'Insufficient data for spectrum', 
                   ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            fig.savefig(output_dir / 'mond_analysis.png', dpi=150)
            print(f"\nPlot saved to: {output_dir / 'mond_analysis.png'}")
        
        plt.show()
    
    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MOND SIGNATURE ANALYSIS")
    print("Channel Impedance Framework")
    print("=" * 70)
    
    # Test with synthetic data
    print("\n[TEST] Generating synthetic data with MOND physics...\n")
    
    a, f_true, f_measured = generate_synthetic_data(
        n_points=1000,
        a_min=1e-12,   # Deep into MOND regime
        a_max=1e-7,    # Well into Newtonian regime
        mass=0.1,      # 100 grams
        noise_level=1e-15,  # 1 femtonewton noise
        include_mond=True
    )
    
    # Run analysis
    output_dir = Path(__file__).parent.parent / "output"
    results = run_mond_analysis(
        a, f_measured,
        mass_estimate=0.1,
        plot=True,
        output_dir=output_dir
    )
    
    # Save results
    output_dir.mkdir(exist_ok=True)
    
    def convert_for_json(v):
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        elif isinstance(v, (np.bool_, bool)):
            return bool(v)
        return v
    
    json_results = {k: convert_for_json(v) 
                    for k, v in results.items() if not isinstance(v, np.ndarray)}
    
    with open(output_dir / 'analysis_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir / 'analysis_results.json'}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    if results['mond_preferred']:
        print("\n⚠️  MOND model is statistically preferred!")
        print(f"   p-value: {results['p_value']:.2e}")
        print(f"   Fitted a₀: {results['fitted_a0']:.2e} m/s²")
        print(f"   Expected a₀: {A_0:.2e} m/s²")
    else:
        print("\n✓ Newtonian model adequate for this data")
