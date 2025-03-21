import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
from pathlib import Path

class ExoplanetVisualizer:
    """Handles visualization of exoplanet data and causal inference results."""
    
    def __init__(self, data: pd.DataFrame, results: Dict[str, Any]):
        """
        Initialize the visualizer.
        
        Args:
            data: Preprocessed DataFrame containing exoplanet data
            results: Dictionary containing causal inference results
        """
        self.data = data
        self.results = results
        plt.style.use('seaborn')
        
    def plot_correlation_matrix(self, save_path: Optional[str] = None) -> None:
        """
        Plot correlation matrix of all variables.
        
        Args:
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            self.data.corr(),
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f'
        )
        plt.title("Correlation Matrix of Variables")
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def plot_age_vs_planet_size(self, save_path: Optional[str] = None) -> None:
        """
        Plot the relationship between stellar age and planet size.
        
        Args:
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=self.data,
            x='A',
            y='Y',
            alpha=0.5,
            s=20
        )
        
        # Add trend line - convert pandas Series to numpy arrays
        x_values = self.data['A'].values
        y_values = self.data['Y'].values
        z = np.polyfit(x_values, y_values, 1)
        p = np.poly1d(z)
        plt.plot(
            x_values,
            p(x_values),
            "r--",
            alpha=0.8,
            label=f"Trend: {z[0]:.3f}x + {z[1]:.3f}"
        )
        
        plt.xlabel("Stellar Age (Gyr)")
        plt.ylabel("Planet Radius (Earth Radii)")
        plt.title("Planet Size vs. Stellar Age")
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def plot_metallicity_vs_planet_size(self, save_path: Optional[str] = None) -> None:
        """
        Plot the relationship between stellar metallicity and planet size.
        
        Args:
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=self.data,
            x='F',
            y='Y',
            alpha=0.5,
            s=20
        )
        
        # Add trend line - convert pandas Series to numpy arrays
        x_values = self.data['F'].values
        y_values = self.data['Y'].values
        z = np.polyfit(x_values, y_values, 1)
        p = np.poly1d(z)
        plt.plot(
            x_values,
            p(x_values),
            "r--",
            alpha=0.8,
            label=f"Trend: {z[0]:.3f}x + {z[1]:.3f}"
        )
        
        plt.xlabel("Stellar Metallicity [Fe/H]")
        plt.ylabel("Planet Radius (Earth Radii)")
        plt.title("Planet Size vs. Stellar Metallicity")
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def plot_stellar_evolution(self, save_path: Optional[str] = None) -> None:
        """
        Plot stellar evolution relationships.
        
        Args:
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Stellar Evolution Relationships")
        
        # Age vs Temperature
        sns.scatterplot(
            data=self.data,
            x='A',
            y='T',
            alpha=0.5,
            s=20,
            ax=axes[0, 0]
        )
        # Add trend line - convert to numpy arrays
        x_values = self.data['A'].values
        y_values = self.data['T'].values
        z = np.polyfit(x_values, y_values, 1)
        p = np.poly1d(z)
        axes[0, 0].plot(x_values, p(x_values), "r--", alpha=0.8)
        axes[0, 0].set_xlabel("Stellar Age (Gyr)")
        axes[0, 0].set_ylabel("Effective Temperature (K)")
        axes[0, 0].set_title("Age vs. Temperature")
        
        # Age vs Radius
        sns.scatterplot(
            data=self.data,
            x='A',
            y='R',
            alpha=0.5,
            s=20,
            ax=axes[0, 1]
        )
        # Add trend line - convert to numpy arrays
        x_values = self.data['A'].values
        y_values = self.data['R'].values
        z = np.polyfit(x_values, y_values, 1)
        p = np.poly1d(z)
        axes[0, 1].plot(x_values, p(x_values), "r--", alpha=0.8)
        axes[0, 1].set_xlabel("Stellar Age (Gyr)")
        axes[0, 1].set_ylabel("Stellar Radius (Solar Radii)")
        axes[0, 1].set_title("Age vs. Radius")
        
        # Age vs Metallicity
        sns.scatterplot(
            data=self.data,
            x='A',
            y='F',
            alpha=0.5,
            s=20,
            ax=axes[1, 0]
        )
        # Add trend line - convert to numpy arrays
        x_values = self.data['A'].values
        y_values = self.data['F'].values
        z = np.polyfit(x_values, y_values, 1)
        p = np.poly1d(z)
        axes[1, 0].plot(x_values, p(x_values), "r--", alpha=0.8)
        axes[1, 0].set_xlabel("Stellar Age (Gyr)")
        axes[1, 0].set_ylabel("Metallicity [Fe/H]")
        axes[1, 0].set_title("Age vs. Metallicity")
        
        # Mass vs Radius
        sns.scatterplot(
            data=self.data,
            x='M',
            y='R',
            alpha=0.5,
            s=20,
            ax=axes[1, 1]
        )
        # Add trend line - convert to numpy arrays
        x_values = self.data['M'].values
        y_values = self.data['R'].values
        z = np.polyfit(x_values, y_values, 1)
        p = np.poly1d(z)
        axes[1, 1].plot(x_values, p(x_values), "r--", alpha=0.8)
        axes[1, 1].set_xlabel("Stellar Mass (Solar Masses)")
        axes[1, 1].set_ylabel("Stellar Radius (Solar Radii)")
        axes[1, 1].set_title("Mass vs. Radius")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def plot_all(self, output_dir: str = "figures") -> None:
        """
        Generate all plots and save them to the specified directory.
        
        Args:
            output_dir: Directory to save the plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        self.plot_correlation_matrix(output_path / "correlation_matrix.png")
        self.plot_age_vs_planet_size(output_path / "age_vs_planet_size.png")
        self.plot_metallicity_vs_planet_size(output_path / "metallicity_vs_planet_size.png")
        self.plot_stellar_evolution(output_path / "stellar_evolution.png") 