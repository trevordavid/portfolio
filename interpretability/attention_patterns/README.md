# Attention Pattern Analysis Tool

This tool visualizes and analyzes attention patterns in transformer models, helping identify important interactions between tokens. It provides insights into how different parts of a transformer model attend to each other during processing.

## Features

- Extract attention patterns from transformer models
- Visualize attention weights between tokens
- Analyze attention patterns across different layers and heads
- Interactive visualization using Dash
- Support for various transformer architectures

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```python
from attention_analyzer import AttentionAnalyzer

# Initialize the analyzer with a model
analyzer = AttentionAnalyzer("bert-base-uncased")

# Analyze attention patterns for a given text
text = "The quick brown fox jumps over the lazy dog"
attention_data = analyzer.analyze(text)

# Visualize the attention patterns
analyzer.visualize(attention_data)
```

## Project Structure

- `attention_analyzer.py`: Core functionality for extracting and analyzing attention patterns
- `visualization.py`: Visualization tools for attention patterns
- `app.py`: Dash web application for interactive visualization
- `utils.py`: Utility functions and helper methods

## Contributing

Feel free to submit issues and enhancement requests! 