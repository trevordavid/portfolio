#!/bin/bash
# Script to initialize Git repository and make first commit

# Initialize Git repository if it doesn't exist
if [ ! -d .git ]; then
    echo "Initializing Git repository..."
    git init
else
    echo "Git repository already exists"
fi

# Clean up __pycache__ directories
echo "Cleaning up Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type f -name "*.pyd" -delete

# Create necessary directories if they don't exist
echo "Ensuring required directories exist..."
mkdir -p data
mkdir -p figures
mkdir -p results
mkdir -p src/dashboard_dash/assets

# Create initial .gitkeep files to track empty directories
touch data/.gitkeep
touch figures/.gitkeep
touch results/.gitkeep
touch src/dashboard_dash/assets/.gitkeep

# Git add files with proper exclusions
echo "Adding files to Git..."
git add .

# Initial commit
echo "Making initial commit..."
git commit -m "Initial commit: Exoplanet causal inference project

This project applies causal inference methodologies to analyze the effect
of stellar age on exoplanet sizes, using data from the NASA Exoplanet Archive.

The project includes:
- Data fetching from NASA Exoplanet Archive
- Causal modeling and analysis
- Interactive Dash dashboard
- Visualization utilities

See README.md for more details."

echo "Git repository initialized and first commit completed!"
echo ""
echo "Next steps:"
echo "1. Connect to a remote repository with: git remote add origin <repository-url>"
echo "2. Push your code with: git push -u origin main"
echo ""
echo "Done!" 