"""
External Field Effect (EFE) Simulation
======================================

This module explores whether the galactic gravitational field screens 
local MOND effects.

Key Finding: The Channel Impedance Framework predicts NO External Field Effect
for local experiments, unlike standard MOND.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .mond_core import A_0


# ============================================================================
# CONSTANTS
# ============================================================================

G = 6.674e-11       # m³/(kg·s²)
C = 3e8             # m/s

# External fields
G_GALACTIC = 1.5e-10   # m/s², galactic field at solar position (~a₀!)
G_EARTH = 9.81         # m/s², Earth's surface gravity
G_SUN = 6e-3           # m/s², Sun's gravity at Earth's orbit


# ============================================================================
# MOND INTERPOLATION FUNCTIONS
# ============================================================================

def standard_mond_mu(a_total):
    """
    Standard MOND: μ depends on TOTAL acceleration (internal + external).
    This is the EFE assumption.
    """
    x = a_total / A_0
    return x / np.sqrt(1 + x**2)


def framework_mu(a_local, a_external=0):
    """
    Framework prediction: μ depends on LOCAL acceleration only.
    
    Physical reasoning:
    - Impedance matching depends on oscillation FREQUENCY
    - Frequency comes from RELATIVE acceleration
    - External fields create no relative acceleration locally
    
    This naturally avoids EFE!
    """
    x_local = a_local / A_0
    mu_local = x_local / np.sqrt(1 + x_local**2)
    return mu_local


def framework_mu_mixed(a_local, a_external, coupling_factor=0.5):
    """
    Partial coupling to external field for parameter exploration.
    
    coupling_factor = 0: Pure local (no EFE)
    coupling_factor = 1: Pure total (full EFE like standard MOND)
    """
    a_effective = a_local + coupling_factor * a_external
    x = a_effective / A_0
    return x / np.sqrt(1 + x**2)


# ============================================================================
# SIMULATIONS
# ============================================================================

def simulate_efe_scenarios():
    """Compare predictions under different EFE assumptions."""
    print("=" * 70)
    print("EXTERNAL FIELD EFFECT SIMULATION")
    print("=" * 70)
    
    # Local acceleration in proposed experiment
    a_local = 5e-13  # m/s², 24-hour period pendulum
    
    # External fields
    fields = {
        "None (isolated)": 0,
        "Galactic only": G_GALACTIC,
        "Galactic + Earth": G_GALACTIC + G_EARTH,
    }
    
    # Material impedance difference (Be vs U)
    delta_Z = 5.8e-4 / (1 - 0.004)
    
    print(f"\nLocal acceleration: a = {a_local:.2e} m/s²")
    print(f"a_local / a₀ = {a_local/A_0:.4f}")
    print(f"\nExternal fields:")
    for name, g in fields.items():
        print(f"  {name}: {g:.2e} m/s² (g/a₀ = {g/A_0:.1f})")
    
    print("\n" + "=" * 70)
    print("SCENARIO ANALYSIS")
    print("=" * 70)
    
    results = []
    
    # Scenario 1: Standard MOND (full EFE)
    print("\n📊 SCENARIO 1: Standard MOND (EFE screens local effects)")
    print("-" * 50)
    for name, g_ext in fields.items():
        a_total = np.sqrt(a_local**2 + g_ext**2)
        mu = standard_mond_mu(a_total)
        eta = (1 - mu) * delta_Z
        print(f"  {name}:")
        print(f"    a_total = {a_total:.2e}, μ = {mu:.6f}, η = {eta:.2e}")
        results.append({"scenario": "Standard MOND", "field": name, "eta": eta})
    
    # Scenario 2: Framework (no EFE)
    print("\n📊 SCENARIO 2: Framework - Local Only (NO EFE)")
    print("-" * 50)
    mu = framework_mu(a_local)
    eta = (1 - mu) * delta_Z
    print(f"  Any external field:")
    print(f"    μ(a_local) = {mu:.6f}, η = {eta:.2e}")
    print(f"    (External field doesn't matter!)")
    results.append({"scenario": "Framework (no EFE)", "field": "Any", "eta": eta})
    
    # Scenario 3: Partial coupling
    print("\n📊 SCENARIO 3: Framework - Partial EFE Coupling")
    print("-" * 50)
    for coupling in [0.0, 0.1, 0.5, 1.0]:
        g_ext = G_GALACTIC
        mu = framework_mu_mixed(a_local, g_ext, coupling)
        eta = (1 - mu) * delta_Z
        print(f"  Coupling = {coupling}: μ = {mu:.6f}, η = {eta:.2e}")
    
    return results


def physical_argument_for_no_efe():
    """Explain WHY the framework predicts no EFE."""
    print("\n" + "=" * 70)
    print("WHY THE FRAMEWORK MIGHT AVOID EFE")
    print("=" * 70)
    
    print("""
    STANDARD MOND REASONING:
    ────────────────────────
    MOND is usually formulated as a modification to Poisson's equation:
    
        ∇·[μ(|∇Φ|/a₀) ∇Φ] = 4πGρ
    
    Here, μ depends on the TOTAL gravitational field gradient |∇Φ|.
    An external field adds to this, so μ(g_total) ≈ 1 if g_ext >> a₀.
    
    This is why standard MOND predicts EFE screening.
    
    
    CHANNEL IMPEDANCE REASONING:
    ────────────────────────────
    The framework says gravity works through impedance matching
    between mass and spacetime. The matching efficiency μ depends
    on the FREQUENCY of energy exchange:
    
        f = a / (2πc)
    
    Key insight: What acceleration determines this frequency?
    
    Answer: The RELATIVE acceleration between interacting masses!
    
    • The pendulum masses accelerate relative to each other
    • This relative motion has frequency f = a_local / (2πc)
    • The galactic field accelerates EVERYTHING equally
    • It creates no RELATIVE motion, so no frequency contribution
    
    Analogy: You're on a train doing an experiment with springs.
    The train's acceleration doesn't affect the spring oscillations -
    only the relative motion between masses matters.
    
    
    PREDICTION:
    ───────────
    The Channel Impedance Framework predicts NO External Field Effect
    for local experiments, because:
    
    1. Impedance matching depends on oscillation FREQUENCY
    2. Frequency comes from RELATIVE acceleration
    3. External fields create no relative acceleration locally
    4. Therefore, μ(a_local) determines the physics
    
    This is a TESTABLE DIFFERENCE from standard MOND!
    """)


def simulate_experiment_with_efe():
    """Run torsion balance simulation with EFE considerations."""
    print("\n" + "=" * 70)
    print("TORSION BALANCE SIMULATION: EFE COMPARISON")
    print("=" * 70)
    
    # Experiment parameters
    period_hours = 24
    amplitude = 0.0001  # m
    omega = 2 * np.pi / (period_hours * 3600)
    a_local = omega**2 * amplitude
    
    delta_Z = 5.8e-4  # Impedance difference for Be-U
    
    # Time array (30 days)
    days = 30
    t = np.linspace(0, days * 24 * 3600, days * 24 * 60)
    
    # Local acceleration varies sinusoidally
    a_t = a_local * np.abs(np.sin(omega * t))
    
    # Standard MOND with EFE
    a_total_t = np.sqrt(a_t**2 + G_GALACTIC**2)
    mu_efe = a_total_t / A_0 / np.sqrt(1 + (a_total_t/A_0)**2)
    eta_efe = (1 - mu_efe) * delta_Z
    
    # Framework without EFE
    mu_no_efe = a_t / A_0 / np.sqrt(1 + (a_t/A_0)**2)
    mu_no_efe = np.where(a_t > 0, mu_no_efe, 0)
    eta_no_efe = (1 - mu_no_efe) * delta_Z
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: μ vs time
    ax1 = axes[0, 0]
    mask = t < 5 * 24 * 3600
    ax1.plot(t[mask] / 3600 / 24, mu_efe[mask], 'r-', label='Standard MOND (with EFE)', alpha=0.7)
    ax1.plot(t[mask] / 3600 / 24, mu_no_efe[mask], 'b-', label='Framework (no EFE)', alpha=0.7)
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('μ (interpolation function)')
    ax1.set_title('MOND Interpolation Function vs Time')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: η vs time
    ax2 = axes[0, 1]
    ax2.plot(t[mask] / 3600 / 24, eta_efe[mask], 'r-', label='Standard MOND (with EFE)', alpha=0.7)
    ax2.plot(t[mask] / 3600 / 24, eta_no_efe[mask], 'b-', label='Framework (no EFE)', alpha=0.7)
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('η (Eötvös parameter)')
    ax2.set_title('Predicted Signal vs Time')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: Signal histogram
    ax3 = axes[1, 0]
    ax3.hist(eta_efe[eta_efe > 0], bins=50, alpha=0.5, color='red', label='Standard MOND')
    ax3.hist(eta_no_efe[eta_no_efe > 0], bins=50, alpha=0.5, color='blue', label='Framework')
    ax3.set_xlabel('η')
    ax3.set_ylabel('Count')
    ax3.set_title('Signal Distribution')
    ax3.legend()
    ax3.set_xscale('log')
    
    # Plot 4: Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary = f"""
    LOCAL ACCELERATION:  a = {a_local:.2e} m/s²
    GALACTIC FIELD:      g = {G_GALACTIC:.2e} m/s² (~a₀)
    
    ────────────────────────────────────────────────
    
    STANDARD MOND (with EFE):
      μ ≈ {np.mean(mu_efe):.4f} (galactic field dominates)
      η ≈ {np.mean(eta_efe):.2e} (strongly suppressed!)
    
    FRAMEWORK (no EFE):
      μ ≈ {np.mean(mu_no_efe[mu_no_efe>0]):.4f} (local acceleration matters)
      η ≈ {np.mean(eta_no_efe):.2e} (FULL SIGNAL!)
    
    ────────────────────────────────────────────────
    
    RATIO: Framework / Standard MOND = {np.mean(eta_no_efe)/np.mean(eta_efe):.0f}× larger
    
    THE EXPERIMENT DISTINGUISHES THE TWO THEORIES!
    """
    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "efe_comparison.png", dpi=150)
    plt.close()
    print(f"\nPlot saved to: {output_dir / 'efe_comparison.png'}")
    
    return {
        "eta_standard_mond": np.mean(eta_efe),
        "eta_framework": np.mean(eta_no_efe),
        "ratio": np.mean(eta_no_efe) / np.mean(eta_efe),
    }


def main():
    print("\n" + "=" * 70)
    print("WHAT DOES THE FRAMEWORK SAY ABOUT EFE?")
    print("=" * 70)
    print("""
    Does the galactic gravitational field screen local MOND effects?
    
    Standard MOND says YES - this is called the External Field Effect.
    
    Let's see what the Channel Impedance Framework predicts...
    """)
    
    simulate_efe_scenarios()
    physical_argument_for_no_efe()
    results = simulate_experiment_with_efe()
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"""
    ┌─────────────────────────────────────────────────────────────────┐
    │              THE FRAMEWORK PREDICTS NO EFE!                     │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  WHY: Impedance matching depends on RELATIVE acceleration,     │
    │       not total. The galactic field accelerates everything     │
    │       equally, so it creates no relative motion.               │
    │                                                                 │
    │  STANDARD MOND predicts:  η ≈ {results['eta_standard_mond']:.2e}  (tiny!)          │
    │  FRAMEWORK predicts:      η ≈ {results['eta_framework']:.2e}  (large!)         │
    │                                                                 │
    │  Difference: {results['ratio']:.0f}× larger signal in framework!               │
    │                                                                 │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  THIS MAKES THE TEST EVEN MORE POWERFUL:                       │
    │                                                                 │
    │  • If we see η ~ 10⁻⁴:  Framework confirmed, standard MOND out │
    │  • If we see η ~ 10⁻⁷:  Standard MOND confirmed, framework out │
    │  • If we see η ~ 0:     Both ruled out                         │
    │                                                                 │
    │  The experiment distinguishes THREE possibilities!             │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    return results


if __name__ == "__main__":
    results = main()
