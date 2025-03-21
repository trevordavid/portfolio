"""Main module for running the causal inference analysis pipeline."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

from src.data.data_fetcher import ExoplanetDataFetcher
from src.models.causal_model import ExoplanetCausalModel
from src.visualization.visualizer import ExoplanetVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories(custom_output_dir: str = None) -> Dict[str, Path]:
    """
    Create necessary directories for the analysis.
    
    Args:
        custom_output_dir: Optional custom output directory
        
    Returns:
        Dictionary of created directories
    """
    # Get the project root directory (parent of src)
    project_root = Path(__file__).parent.parent
    
    if custom_output_dir:
        output_dir = Path(custom_output_dir)
        output_dir.mkdir(exist_ok=True)
        
        directories = {
            'data': project_root / 'data',  # Data directory is always in project root
            'figures': output_dir / 'figures',
            'results': output_dir / 'results'
        }
    else:
        directories = {
            'data': project_root / 'data',
            'figures': project_root / 'figures',
            'results': project_root / 'results'
        }
    
    for directory in directories.values():
        directory.mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")
        
    return directories

def run_analysis(output_dir: str = None, skip_refutation: bool = False) -> Dict[str, Any]:
    """
    Run the complete causal inference analysis pipeline.
    
    Args:
        output_dir: Optional custom output directory
        skip_refutation: Whether to skip refutation tests
        
    Returns:
        Analysis results dictionary
    """
    try:
        # Set up directories
        dirs = setup_directories(output_dir)
        logger.info("Created necessary directories")
        
        # Fetch and preprocess data
        data_fetcher = ExoplanetDataFetcher(cache_dir=str(dirs['data']))
        data = data_fetcher.get_analysis_data()
        logger.info(f"Loaded and preprocessed data with {len(data)} samples")
        
        # Fit causal model
        model = ExoplanetCausalModel(data)
        model.fit()
        logger.info("Fitted causal model")
        
        # Identify and estimate causal effect
        model.identify_effect()
        model.estimate_effect()
        logger.info("Identified and estimated causal effect")
        
        # Get results
        results = model.get_results()
        logger.info("Retrieved analysis results")
        
        # Plot causal graph
        model.plot_causal_graph(dirs['figures'] / 'causal_graph.png')
        logger.info("Generated causal graph")
        
        # Create visualizations
        visualizer = ExoplanetVisualizer(data, results)
        visualizer.plot_all(output_dir=str(dirs['figures']))
        logger.info("Generated all visualizations")
        
        # Save results
        import json
        results_file = dirs['results'] / 'analysis_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved analysis results to {results_file}")
        
        # Print summary
        print("\nAnalysis Summary:")
        print(f"Number of samples: {len(data)}")
        print(f"Causal effect estimate: {results['causal_estimate']:.3f}")
        print(f"Confidence intervals: {results['confidence_intervals']}")
        print(f"\nResults saved to: {dirs['results']}")
        print(f"Figures saved to: {dirs['figures']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in analysis pipeline: {e}")
        raise

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run exoplanet causal inference analysis')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Custom output directory for results and figures')
    parser.add_argument('--skip-refutation', action='store_true',
                        help='Skip refutation tests (faster but less robust)')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Logging level')
    
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    args = parse_args()
    
    # Set log level from arguments
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Run the analysis
    results = run_analysis(
        output_dir=args.output_dir,
        skip_refutation=args.skip_refutation
    )
    
    return results

if __name__ == "__main__":
    main() 