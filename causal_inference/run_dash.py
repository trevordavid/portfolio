#!/usr/bin/env python3
"""Script to launch the exoplanet causal inference Dash dashboard."""

import subprocess
import os
import sys
from pathlib import Path

def main():
    """Launch the Dash dashboard."""
    # Get the dashboard path
    dashboard_path = Path(__file__).parent / "src" / "dashboard_dash" / "app.py"
    
    if not dashboard_path.exists():
        print(f"Error: Dashboard file not found at {dashboard_path}")
        sys.exit(1)
    
    print(f"Launching Dash dashboard from {dashboard_path}")
    
    # Run the dashboard module
    try:
        # Import and run the dashboard directly
        sys.path.append(str(Path(__file__).parent))
        from src.dashboard_dash.app import main as run_dashboard
        run_dashboard()
    except ImportError as e:
        print(f"Error importing dashboard module: {e}")
        print("Make sure you have installed all required packages:")
        print("pip install dash dash-bootstrap-components plotly")
        sys.exit(1)
    except Exception as e:
        print(f"Error launching dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 