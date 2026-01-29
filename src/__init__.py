"""
MOND Analysis Package
=====================

A framework for analyzing Modified Newtonian Dynamics using 
channel impedance matching principles.
"""

from .mond_core import (
    A_0, G, C, SIGMA_LOG, HBAR,
    mond_frequency,
    impedance_matching,
    mond_interpolation_standard,
    mond_interpolation_rar,
    framework_effective_mass_ratio,
    framework_force,
    fit_mond_model,
    run_mond_analysis,
    MONDFitResult,
)

__version__ = "0.1.0"
__author__ = "Your Name"
__all__ = [
    "A_0", "G", "C", "SIGMA_LOG", "HBAR",
    "mond_frequency",
    "impedance_matching", 
    "mond_interpolation_standard",
    "mond_interpolation_rar",
    "framework_effective_mass_ratio",
    "framework_force",
    "fit_mond_model",
    "run_mond_analysis",
    "MONDFitResult",
]
