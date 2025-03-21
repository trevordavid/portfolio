# Exoplanet Causal Inference Analysis

This project applies causal inference methodologies to analyze the effect of stellar age on exoplanet sizes, using data from the NASA Exoplanet Archive.

## Project Overview

The main goal of this analysis is to determine whether a causal relationship exists between stellar age and exoplanet radius, while controlling for other stellar properties that might affect both variables.

The causal model incorporates stellar evolutionary theory, considering:
- Stellar properties (mass, radius, temperature, metallicity, surface gravity)
- System age 
- Exoplanet size
- Orbital parameters

## Features

- **Automated Data Fetching**: Downloads data from NASA Exoplanet Archive with local caching
- **Causal Modeling**: Implements the causal inference framework using DoWhy
- **Statistical Analysis**: Performs causal effect estimation and refutation tests
- **Visualization**: Generates plots for data exploration and causal relationships
- **Interactive Dashboard**: Dash-based dashboard for exploring data and results

## Project Structure

```
.
├── data/                # Data directory with cached NASA exoplanet data
├── figures/             # Generated visualizations
├── results/             # Analysis results in JSON format
├── src/                 # Source code
│   ├── dashboard_dash/  # Interactive Dash dashboard
│   ├── data/            # Data fetching and preprocessing
│   ├── models/          # Causal modeling
│   ├── visualization/   # Plotting and visualization
│   └── main.py          # Main script to execute the pipeline
├── tests/               # Unit tests
└── requirements.txt     # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.8+
- GraphViz (for causal graph visualization)

### Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Install GraphViz:
   - macOS: `brew install graphviz`
   - Ubuntu/Debian: `apt-get install graphviz`
   - Windows: Download from the [GraphViz website](https://graphviz.org/download/)

### Running the Analysis

Execute the main script from the project root directory:

```
python -m src.main
```

### Using the Interactive Dashboard

The project includes an interactive Dash dashboard for data exploration and visualization of results:

```
python run_dash.py
```

The dashboard provides:
- Data exploration tools
- Interactive visualization of relationships between variables
- Display of causal analysis results
- Interface for running new analyses with different parameters

## Results

The analysis produces:

1. A causal graph visualization showing the relationships between variables
2. Effect estimates of stellar age on exoplanet size
3. Refutation test results to validate causal claims
4. Visualizations of key relationships in the data

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- NASA Exoplanet Archive for providing the dataset
- DoWhy library for causal inference tooling 