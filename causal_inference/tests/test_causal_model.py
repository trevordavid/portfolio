"""Tests for the causal model module."""
import unittest
import pandas as pd
import numpy as np
from src.models.causal_model import ExoplanetCausalModel

class TestExoplanetCausalModel(unittest.TestCase):
    """Test case for the ExoplanetCausalModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple synthetic dataset for testing
        np.random.seed(42)
        n_samples = 100
        
        # Generate synthetic data
        age = np.random.normal(5, 2, n_samples)  # System Age (A)
        mass = np.random.normal(1, 0.2, n_samples)  # Stellar Mass (M)
        metallicity = age * 0.01 + np.random.normal(0, 0.1, n_samples)  # Metallicity (F)
        radius = 0.8 * mass + 0.01 * age + np.random.normal(0, 0.05, n_samples)  # Stellar Radius (R)
        temperature = 5000 + 500 * mass - 10 * age + np.random.normal(0, 100, n_samples)  # Temperature (T)
        gravity = mass / (radius**2) + np.random.normal(0, 0.01, n_samples)  # Surface Gravity (G)
        period = np.random.normal(10, 5, n_samples)  # Orbital Period (P)
        
        # Exoplanet size (Y) with causal effect from age
        planet_size = 1.0 - 0.002 * age + 0.5 * metallicity + 0.1 * radius + 0.001 * period + np.random.normal(0, 0.1, n_samples)
        
        # Create the DataFrame
        self.test_data = pd.DataFrame({
            'A': age,
            'M': mass,
            'F': metallicity,
            'R': radius,
            'T': temperature,
            'G': gravity,
            'P': period,
            'Y': planet_size
        })
        
        # Initialize the model
        self.model = ExoplanetCausalModel(self.test_data)
        
    def test_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model)
        self.assertIsNone(self.model.model)
        self.assertIsNone(self.model.identified_estimand)
        self.assertIsNone(self.model.estimate)
        self.assertEqual(self.model.data.shape, self.test_data.shape)
        
    def test_fit(self):
        """Test model fitting."""
        self.model.fit()
        self.assertIsNotNone(self.model.model)
        
    def test_identify_effect(self):
        """Test effect identification."""
        self.model.fit()
        self.model.identify_effect()
        self.assertIsNotNone(self.model.identified_estimand)
        
    def test_estimate_effect(self):
        """Test effect estimation."""
        self.model.fit()
        self.model.identify_effect()
        self.model.estimate_effect()
        self.assertIsNotNone(self.model.estimate)
        # Check that the estimated effect is close to the true effect (-0.002)
        self.assertAlmostEqual(float(self.model.estimate.value), -0.002, delta=0.05)
        
    def test_get_results(self):
        """Test results retrieval."""
        self.model.fit()
        self.model.identify_effect()
        self.model.estimate_effect()
        results = self.model.get_results()
        self.assertIn('causal_estimate', results)
        self.assertIn('confidence_intervals', results)
        self.assertIn('refutation_results', results)
        self.assertIn('model_metadata', results)
        
if __name__ == '__main__':
    unittest.main() 