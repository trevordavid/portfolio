import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import pyvo
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ExoplanetDataFetcher:
    """Handles fetching and preprocessing of exoplanet data from NASA Exoplanet Archive."""
    
    def __init__(self, cache_dir: str = "data"):
        """
        Initialize the data fetcher.
        
        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.tap_service = pyvo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP")
        
    def fetch_data(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch exoplanet data from NASA Exoplanet Archive using TAP service.
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            pd.DataFrame: DataFrame containing exoplanet data
        """
        cache_file = self.cache_dir / "nasa_exoplanet_data.csv"
        
        if use_cache and cache_file.exists():
            logger.info("Loading cached data...")
            return pd.read_csv(cache_file)
            
        logger.info("Fetching fresh data from NASA Exoplanet Archive...")
        
        # Define the query to get all necessary columns
        query = """
        SELECT 
            pl_name, hostname, discoverymethod,
            pl_orbper, pl_rade, pl_bmasse,
            st_age, st_mass, st_teff, st_rad, 
            st_logg, st_met, sy_snum, sy_pnum
        FROM pscomppars
        """
        
        try:
            # Execute the query
            results = self.tap_service.search(query)
            
            # Convert to pandas DataFrame
            df = results.to_table().to_pandas()
            
            # Save to cache
            df.to_csv(cache_file, index=False)
            logger.info(f"Retrieved {len(df)} exoplanet records")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            if cache_file.exists():
                logger.info("Falling back to cached data...")
                return pd.read_csv(cache_file)
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the exoplanet data for causal inference analysis.
        
        Args:
            df: Raw DataFrame from NASA Exoplanet Archive
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame ready for analysis
        """
        logger.info("Preprocessing data...")
        
        # Filter for transiting planets
        mask = df['discoverymethod'] == 'Transit'
        logger.info(f"Transiting planets: {mask.sum()}")
        
        # Apply reasonable cuts for stellar mass (solar masses)
        mask &= df['st_mass'] >= 0.8
        mask &= df['st_mass'] <= 1.2
        logger.info(f"After mass cuts: {mask.sum()}")
        
        # Filter for reasonable orbital periods (days)
        mask &= df['pl_orbper'] < 100
        logger.info(f"After period cuts: {mask.sum()}")
        
        # Filter for reasonable planet radii (Earth radii)
        mask &= df['pl_rade'] <= 10
        logger.info(f"After radius cuts: {mask.sum()}")
        
        # Remove rows with missing values in key parameters
        key_params = [
            'st_age', 'st_mass', 'st_teff', 'st_rad', 
            'st_logg', 'st_met', 'pl_orbper', 'pl_rade'
        ]
        for param in key_params:
            mask &= np.isfinite(df[param])
            logger.info(f"After {param} cuts: {mask.sum()}")
        
        # Apply the mask
        df_filtered = df[mask].copy()
        
        # Rename columns for clarity
        column_mapping = {
            'st_age': 'A',    # System Age
            'st_mass': 'M',   # Stellar Mass
            'st_teff': 'T',   # Stellar Effective Temperature
            'st_rad': 'R',    # Stellar Radius
            'st_logg': 'G',   # Stellar Surface Gravity
            'st_met': 'F',    # Stellar Metallicity
            'pl_orbper': 'P', # Orbital Period
            'pl_rade': 'Y'    # Exoplanet Size
        }
        df_filtered = df_filtered.rename(columns=column_mapping)
        
        # Select only the columns we need
        result = df_filtered[list(column_mapping.values())]
        logger.info(f"Final dataset size: {len(result)}")
        
        return result
    
    def get_analysis_data(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Get preprocessed data ready for causal inference analysis.
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame ready for analysis
        """
        raw_data = self.fetch_data(use_cache)
        return self.preprocess_data(raw_data) 