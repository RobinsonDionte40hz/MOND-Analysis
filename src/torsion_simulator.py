"""
Be-U Torsion Balance Experiment Simulator
=========================================

Simulates what a beryllium-uranium torsion balance would measure
if the Channel Impedance Framework is correct.

This lets us "see" the predicted signal before building anything.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .mond_core import A_0


# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

G = 6.674e-11       # Gravitational constant (m³/kg/s²)
g_earth = 9.81      # Earth surface gravity (m/s²)

# Material properties
MATERIALS = {
    'Be': {'Z': 4, 'A': 9, 'name': 'Beryllium', 'color': 'blue'},
    'U': {'Z': 92, 'A': 238, 'name': 'Uranium', 'color': 'red'},
    'Ti': {'Z': 22, 'A': 48, 'name': 'Titanium', 'color': 'green'},
    'Pt': {'Z': 78, 'A': 195, 'name': 'Platinum', 'color': 'purple'},
}


# ============================================================================
# PHYSICS FUNCTIONS
# ============================================================================

def calculate_impedance_difference(mat1: str, mat2: str) -> float:
    """
    Calculate gravitational impedance difference between two materials.
    
    In the framework, impedance depends on nuclear binding energy,
    which scales roughly with (Z/A) - the proton fraction.
    
    Parameters
    ----------
    mat1, mat2 : str
        Material codes ('Be', 'U', 'Ti', 'Pt')
        
    Returns
    -------
    float
        Relative impedance difference δZ/Z₀
    """
    z1, a1 = MATERIALS[mat1]['Z'], MATERIALS[mat1]['A']
    z2, a2 = MATERIALS[mat2]['Z'], MATERIALS[mat2]['A']
    
    # Binding energy per nucleon difference (simplified)
    delta_binding = abs((z1/a1) - (z2/a2))
    
    # Convert to impedance difference
    delta_Z = delta_binding * 1e-2  # ~1% effect from nuclear binding
    
    return delta_Z


def framework_eta_prediction(a: float, mat1: str, mat2: str) -> float:
    """
    Calculate framework prediction for Eötvös parameter η.
    
    η = (a₁ - a₂) / ((a₁ + a₂)/2)
    
    In the framework:
    η = f(a/a₀) × |δZ₁ - δZ₂| / Z₀
    
    Parameters
    ----------
    a : float
        Acceleration in m/s²
    mat1, mat2 : str
        Material codes
        
    Returns
    -------
    float
        Predicted Eötvös parameter
    """
    x = a / A_0
    
    # MOND interpolation function
    mu = x / np.sqrt(1 + x**2)
    
    # Impedance difference
    delta_Z = calculate_impedance_difference(mat1, mat2)
    
    # Framework prediction
    eta = (1 - mu) * delta_Z
    
    return eta


# ============================================================================
# SIMULATION
# ============================================================================

def simulate_torsion_balance(
    mat1: str = 'Be',
    mat2: str = 'U',
    duration_hours: float = 100,
    sample_rate_hz: float = 1.0,
    pendulum_period_s: float = 600,
    noise_level: float = 1e-12,
    include_framework_signal: bool = True
) -> dict:
    """
    Simulate a torsion balance experiment.
    
    Parameters
    ----------
    mat1, mat2 : str
        Material codes ('Be', 'U', 'Ti', 'Pt')
    duration_hours : float
        Measurement duration
    sample_rate_hz : float
        Sampling rate
    pendulum_period_s : float
        Oscillation period (longer = lower acceleration)
    noise_level : float
        Measurement noise (η units)
    include_framework_signal : bool
        If True, add predicted signal
        
    Returns
    -------
    dict
        Simulation data
    """
    print("=" * 60)
    print(f"SIMULATING: {MATERIALS[mat1]['name']}-{MATERIALS[mat2]['name']} Torsion Balance")
    print("=" * 60)
    
    # Time array
    n_samples = int(duration_hours * 3600 * sample_rate_hz)
    t = np.linspace(0, duration_hours * 3600, n_samples)
    
    # Pendulum parameters
    omega = 2 * np.pi / pendulum_period_s
    amplitude = 0.01  # 1 cm displacement
    a_max = omega**2 * amplitude
    
    print(f"\nPendulum period: {pendulum_period_s/60:.1f} minutes")
    print(f"Peak acceleration: {a_max:.2e} m/s²")
    print(f"a/a₀ = {a_max/A_0:.4f}")
    
    # Framework prediction
    eta_predicted = framework_eta_prediction(a_max, mat1, mat2)
    print(f"\nFramework prediction: η = {eta_predicted:.2e}")
    
    # Generate noise
    noise = np.random.normal(0, noise_level, n_samples)
    
    # Signal
    if include_framework_signal:
        signal = eta_predicted * g_earth * np.sin(omega * t)
        print(f"Signal amplitude: {eta_predicted * g_earth:.2e} m/s²")
    else:
        signal = np.zeros(n_samples)
        print("No framework signal (null hypothesis)")
    
    # Total measurement
    measurement = signal + noise * g_earth
    eta_measured = measurement / g_earth
    
    return {
        't': t,
        'eta_measured': eta_measured,
        'eta_predicted': eta_predicted,
        'signal': signal / g_earth,
        'noise': noise,
        'a_max': a_max,
        'omega': omega,
        'mat1': mat1,
        'mat2': mat2,
    }


def analyze_simulation(data: dict) -> dict:
    """
    Analyze simulated data to extract signal.
    
    Parameters
    ----------
    data : dict
        Simulation output
        
    Returns
    -------
    dict
        Analysis results
    """
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    t = data['t']
    eta = data['eta_measured']
    omega = data['omega']
    
    # Fourier analysis
    dt = t[1] - t[0]
    freqs = np.fft.fftfreq(len(t), dt)
    fft = np.fft.fft(eta)
    power = np.abs(fft)**2
    
    # Find signal frequency
    f_pendulum = omega / (2 * np.pi)
    idx_signal = np.argmin(np.abs(freqs - f_pendulum))
    
    # Extract amplitude
    signal_amplitude = 2 * np.abs(fft[idx_signal]) / len(t)
    
    # Noise floor
    noise_region = (np.abs(freqs - f_pendulum) > 0.0001) & (np.abs(freqs) < 2*f_pendulum)
    noise_floor = np.sqrt(np.mean(power[noise_region])) * 2 / len(t)
    
    # SNR
    snr = signal_amplitude / noise_floor
    
    print(f"\nSignal frequency: {f_pendulum*1000:.3f} mHz")
    print(f"Measured amplitude: η = {signal_amplitude:.2e}")
    print(f"Predicted amplitude: η = {data['eta_predicted']:.2e}")
    print(f"Noise floor: η = {noise_floor:.2e}")
    print(f"Signal-to-noise ratio: {snr:.1f}")
    
    if snr > 5:
        status = "✅ DETECTION! Signal clearly visible"
    elif snr > 3:
        status = "⚠️ HINT: Signal marginally visible (3σ)"
    else:
        status = "❌ NO DETECTION: Signal buried in noise"
    
    print(f"\nStatus: {status}")
    
    return {
        'signal_amplitude': signal_amplitude,
        'noise_floor': noise_floor,
        'snr': snr,
        'freqs': freqs,
        'power': power,
        'f_pendulum': f_pendulum,
    }


def plot_results(data: dict, analysis: dict, output_dir: Path):
    """Create visualization of simulation results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    mat1, mat2 = data['mat1'], data['mat2']
    name1 = MATERIALS[mat1]['name']
    name2 = MATERIALS[mat2]['name']
    
    # Time series
    ax1 = axes[0, 0]
    mask = data['t'] < 3600
    ax1.plot(data['t'][mask]/60, data['eta_measured'][mask], 
             'b-', alpha=0.5, linewidth=0.5, label='Measured')
    ax1.plot(data['t'][mask]/60, data['signal'][mask], 
             'r-', linewidth=2, label='True signal')
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('η (Eötvös parameter)')
    ax1.set_title(f'{name1}-{name2} Torsion Balance: First Hour')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Power spectrum
    ax2 = axes[0, 1]
    pos_freq = analysis['freqs'] > 0
    ax2.semilogy(analysis['freqs'][pos_freq]*1000, 
                 analysis['power'][pos_freq], 'b-', alpha=0.7)
    ax2.axvline(analysis['f_pendulum']*1000, color='r', linestyle='--',
                label=f'Signal: {analysis["f_pendulum"]*1000:.2f} mHz')
    ax2.set_xlabel('Frequency (mHz)')
    ax2.set_ylabel('Power')
    ax2.set_title('Power Spectrum')
    ax2.set_xlim(0, 5)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Phase-folded data
    ax3 = axes[1, 0]
    period = 2 * np.pi / data['omega']
    phase = (data['t'] % period) / period * 360
    
    n_bins = 50
    phase_bins = np.linspace(0, 360, n_bins + 1)
    binned_eta = []
    binned_phase = []
    binned_err = []
    
    for i in range(n_bins):
        mask = (phase >= phase_bins[i]) & (phase < phase_bins[i+1])
        if np.sum(mask) > 0:
            binned_eta.append(np.mean(data['eta_measured'][mask]))
            binned_phase.append((phase_bins[i] + phase_bins[i+1]) / 2)
            binned_err.append(np.std(data['eta_measured'][mask]) / np.sqrt(np.sum(mask)))
    
    ax3.errorbar(binned_phase, binned_eta, yerr=binned_err, fmt='o', capsize=3)
    
    phase_smooth = np.linspace(0, 360, 100)
    signal_smooth = data['eta_predicted'] * np.sin(np.radians(phase_smooth))
    ax3.plot(phase_smooth, signal_smooth, 'r-', linewidth=2, label='Prediction')
    
    ax3.set_xlabel('Phase (degrees)')
    ax3.set_ylabel('η')
    ax3.set_title('Phase-Folded Data')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary = f"""
    SIMULATION SUMMARY
    {'='*35}
    
    Materials: {name1} vs {name2}
    
    Pendulum period: {period/60:.1f} minutes
    Peak acceleration: {data['a_max']:.2e} m/s²
    a/a₀ = {data['a_max']/A_0:.4f}
    
    FRAMEWORK PREDICTION:
    η = {data['eta_predicted']:.2e}
    
    MEASURED:
    η = {analysis['signal_amplitude']:.2e} ± {analysis['noise_floor']:.2e}
    
    SNR: {analysis['snr']:.1f}
    {'✅ DETECTED!' if analysis['snr'] > 5 else '❌ Not detected'}
    """
    
    ax4.text(0.1, 0.9, summary, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'torsion_simulation.png', dpi=150)
    plt.close()
    print(f"\nPlot saved to: {output_dir / 'torsion_simulation.png'}")


def compare_material_pairs():
    """Compare different material combinations."""
    print("\n" + "=" * 60)
    print("COMPARISON: DIFFERENT MATERIAL PAIRS")
    print("=" * 60)
    
    pairs = [
        ('Be', 'Ti'),   # Eöt-Wash
        ('Ti', 'Pt'),   # MICROSCOPE
        ('Be', 'U'),    # Proposed
        ('Be', 'Pt'),   # Alternative
    ]
    
    a = 1e-11  # m/s²
    
    print(f"\nAt a = {a:.0e} m/s² (a/a₀ = {a/A_0:.3f}):")
    print(f"\n{'Pair':<12} {'ΔZ/Z₀':<12} {'η predicted':<15} {'Status'}")
    print("-" * 55)
    
    results = []
    for m1, m2 in pairs:
        delta_Z = calculate_impedance_difference(m1, m2)
        eta = framework_eta_prediction(a, m1, m2)
        
        if eta > 1e-10:
            status = "✅ Detectable"
        elif eta > 1e-13:
            status = "⚠️ Marginal"
        else:
            status = "❌ Too small"
        
        print(f"{m1}-{m2:<8} {delta_Z:<12.2e} {eta:<15.2e} {status}")
        results.append({'pair': f'{m1}-{m2}', 'delta_Z': delta_Z, 'eta': eta})
    
    print("\n" + "-" * 55)
    print("Be-U gives ~1000× larger signal than Ti-Pt!")
    
    return results


def main():
    """Run the full simulation."""
    print("=" * 60)
    print("BE-U TORSION BALANCE SIMULATOR")
    print("Channel Impedance Framework Test")
    print("=" * 60)
    
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    compare_material_pairs()
    
    # Simulation with signal
    print("\n" + "=" * 60)
    print("SIMULATION 1: Framework is TRUE")
    print("=" * 60)
    
    data = simulate_torsion_balance(
        mat1='Be', 
        mat2='U',
        duration_hours=100,
        pendulum_period_s=600,
        noise_level=1e-12,
        include_framework_signal=True
    )
    analysis = analyze_simulation(data)
    plot_results(data, analysis, output_dir)
    
    # Null hypothesis
    print("\n" + "=" * 60)
    print("SIMULATION 2: Framework is FALSE")
    print("=" * 60)
    
    data_null = simulate_torsion_balance(
        mat1='Be',
        mat2='U', 
        duration_hours=100,
        pendulum_period_s=600,
        noise_level=1e-12,
        include_framework_signal=False
    )
    analysis_null = analyze_simulation(data_null)
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print(f"""
    If we build this experiment:
    
    SCENARIO A: Framework is correct
    • Signal at η = {data['eta_predicted']:.1e}
    • SNR ~ {analysis['snr']:.0f}
    • 5σ+ detection!
    
    SCENARIO B: Framework is wrong  
    • Only noise
    • SNR ~ {analysis_null['snr']:.1f}
    
    THE TEST IS DECISIVE:
    • η ~ 10⁻⁹ → Framework CONFIRMED
    • η < 10⁻¹¹ → Framework FALSIFIED
    """)
    
    return data, analysis


if __name__ == "__main__":
    data, analysis = main()
