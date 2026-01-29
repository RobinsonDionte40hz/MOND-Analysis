"""
Unit tests for MOND core module.
"""

import pytest
import numpy as np
from src.mond_core import (
    A_0,
    mond_frequency,
    impedance_matching,
    mond_interpolation_standard,
    mond_interpolation_rar,
    framework_effective_mass_ratio,
    framework_force,
    fit_mond_model,
    generate_synthetic_data,
)


class TestConstants:
    """Test physical constants are reasonable."""
    
    def test_a0_value(self):
        """a₀ should be approximately 1.2e-10 m/s²."""
        assert 1e-10 < A_0 < 2e-10
    
    def test_a0_units(self):
        """Verify a₀ is in sensible range for MOND."""
        # Should be ~10^-10 m/s²
        assert np.log10(A_0) == pytest.approx(-10, abs=0.5)


class TestMONDFunctions:
    """Test MOND physics functions."""
    
    def test_mond_frequency_scaling(self):
        """Frequency should scale linearly with acceleration."""
        f1 = mond_frequency(1e-10)
        f2 = mond_frequency(2e-10)
        assert f2 == pytest.approx(2 * f1)
    
    def test_impedance_matching_peak(self):
        """Matching should be 1.0 when f = f₀."""
        assert impedance_matching(1.0, 1.0) == pytest.approx(1.0)
    
    def test_impedance_matching_decay(self):
        """Matching should decay for f ≠ f₀."""
        assert impedance_matching(10.0, 1.0) < 1.0
        assert impedance_matching(0.1, 1.0) < 1.0
    
    def test_impedance_matching_symmetry(self):
        """Log-ratio symmetry: R(f, f₀) = R(f₀/f, 1)."""
        r1 = impedance_matching(10.0, 1.0)
        r2 = impedance_matching(0.1, 1.0)
        assert r1 == pytest.approx(r2)


class TestMONDInterpolation:
    """Test MOND interpolation functions."""
    
    def test_standard_newtonian_limit(self):
        """μ → 1 for a >> a₀."""
        a_high = 1e-5  # Much greater than a₀
        mu = mond_interpolation_standard(a_high)
        assert mu == pytest.approx(1.0, rel=1e-3)
    
    def test_standard_mond_limit(self):
        """μ → a/a₀ for a << a₀."""
        a_low = 1e-14  # Much less than a₀
        mu = mond_interpolation_standard(a_low)
        expected = a_low / A_0
        assert mu == pytest.approx(expected, rel=0.1)
    
    def test_standard_transition(self):
        """μ should transition smoothly around a₀."""
        mu_at_a0 = mond_interpolation_standard(A_0)
        # At a = a₀, μ = 1/√2 ≈ 0.707
        assert mu_at_a0 == pytest.approx(1 / np.sqrt(2))
    
    def test_rar_newtonian_limit(self):
        """RAR interpolation: μ → 1 for a >> a₀."""
        a_high = 1e-5
        mu = mond_interpolation_rar(a_high)
        assert mu == pytest.approx(1.0, rel=0.1)
    
    def test_array_input(self):
        """Functions should handle array input."""
        a = np.array([1e-12, 1e-11, 1e-10, 1e-9])
        mu = mond_interpolation_standard(a)
        assert len(mu) == 4
        assert np.all(mu > 0)
        assert np.all(mu <= 1)


class TestEffectiveMass:
    """Test effective mass calculations."""
    
    def test_newtonian_regime(self):
        """Mass ratio = 1 in Newtonian regime."""
        a_high = np.array([1e-5])
        ratio = framework_effective_mass_ratio(a_high)
        assert ratio[0] == pytest.approx(1.0, rel=0.1)
    
    def test_mond_regime(self):
        """Mass ratio > 1 in MOND regime."""
        a_low = np.array([1e-12])
        ratio = framework_effective_mass_ratio(a_low)
        assert ratio[0] > 1.0


class TestFrameworkForce:
    """Test force calculations."""
    
    def test_newtonian_force(self):
        """F ≈ ma in Newtonian regime."""
        m = 1.0  # kg
        a = 1.0  # m/s² (well above a₀)
        F = framework_force(m, np.array([a]))
        assert F[0] == pytest.approx(m * a, rel=0.01)
    
    def test_mond_force(self):
        """F ≈ m√(a·a₀) in deep MOND regime."""
        m = 1.0
        a = 1e-14  # Deep MOND
        F = framework_force(m, np.array([a]))
        expected = m * np.sqrt(a * A_0)
        assert F[0] == pytest.approx(expected, rel=0.5)


class TestDataGeneration:
    """Test synthetic data generation."""
    
    def test_generate_data_shape(self):
        """Generated arrays should have correct length."""
        n = 100
        a, f_true, f_measured = generate_synthetic_data(n_points=n)
        assert len(a) == n
        assert len(f_true) == n
        assert len(f_measured) == n
    
    def test_generate_data_noise(self):
        """Measured force should differ from true force."""
        a, f_true, f_measured = generate_synthetic_data()
        assert not np.allclose(f_true, f_measured)
    
    def test_reproducibility(self):
        """Same seed should give same data."""
        a1, _, _ = generate_synthetic_data(seed=42)
        a2, _, _ = generate_synthetic_data(seed=42)
        assert np.allclose(a1, a2)


class TestModelFitting:
    """Test model fitting routines."""
    
    def test_fit_newtonian_data(self):
        """Should recover Newtonian when no MOND."""
        a, _, f = generate_synthetic_data(include_mond=False, noise_level=1e-16)
        result = fit_mond_model(a, f)
        # χ² should be similar for both models
        assert result.newtonian_chi2 < result.mond_chi2 * 1.5
    
    def test_fit_mond_data(self):
        """Should prefer MOND when data includes MOND physics."""
        a, _, f = generate_synthetic_data(include_mond=True, noise_level=1e-16)
        result = fit_mond_model(a, f)
        # MOND should fit better
        assert result.mond_chi2 < result.newtonian_chi2
    
    def test_fit_result_attributes(self):
        """Fit result should have all expected attributes."""
        a, _, f = generate_synthetic_data()
        result = fit_mond_model(a, f)
        
        assert hasattr(result, 'newtonian_chi2')
        assert hasattr(result, 'mond_chi2')
        assert hasattr(result, 'p_value')
        assert hasattr(result, 'fitted_a0')
        assert hasattr(result, 'mond_preferred')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
