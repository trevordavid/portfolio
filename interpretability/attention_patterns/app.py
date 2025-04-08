import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import json
import os
from attention_analyzer import AttentionAnalyzer

# Initialize the Dash app with a minimalist theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Attention Pattern Analysis"

# Define the layout with a clean, minimalist design
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Attention Pattern Analysis", className="text-center my-4", 
                   style={"fontFamily": "Arial, sans-serif", "fontWeight": "300"}),
            html.Hr()
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H4("Select Text Example", className="mb-3", 
                   style={"fontFamily": "Arial, sans-serif", "fontWeight": "400"}),
            dcc.Dropdown(
                id="text-dropdown",
                options=[
                    {"label": "Attention is All You Need (Abstract)", "value": "attention_is_all_you_need"},
                    {"label": "The Unbearable Lightness of Being", "value": "unbearable_lightness"},
                    {"label": "Into Thin Air", "value": "into_thin_air"}
                ],
                value="attention_is_all_you_need",
                clearable=False,
                style={"fontFamily": "Arial, sans-serif"}
            ),
            html.Div(id="text-display", className="mt-3 p-3 bg-light rounded", 
                    style={"fontFamily": "Arial, sans-serif", "minHeight": "100px"}),
            html.Hr()
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H4("Model Configuration", className="mb-3", 
                   style={"fontFamily": "Arial, sans-serif", "fontWeight": "400"}),
            dbc.Row([
                dbc.Col([
                    html.Label("Model", style={"fontFamily": "Arial, sans-serif"}),
                    dcc.Dropdown(
                        id="model-dropdown",
                        options=[
                            {"label": "DistilBERT", "value": "distilbert-base-uncased"},
                            {"label": "BERT", "value": "bert-base-uncased"},
                            {"label": "RoBERTa", "value": "roberta-base"}
                        ],
                        value="distilbert-base-uncased",
                        clearable=False,
                        style={"fontFamily": "Arial, sans-serif"}
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Layer", style={"fontFamily": "Arial, sans-serif"}),
                    dcc.Slider(
                        id="layer-slider",
                        min=0,
                        max=11,  # Will be updated based on model
                        step=1,
                        value=0,
                        marks={i: str(i) for i in range(12)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Head", style={"fontFamily": "Arial, sans-serif"}),
                    dcc.Slider(
                        id="head-slider",
                        min=0,
                        max=11,  # Will be updated based on model
                        step=1,
                        value=0,
                        marks={i: str(i) for i in range(12)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], width=12)
            ], className="mt-3"),
            dbc.Button("Analyze", id="analyze-button", color="primary", className="mt-3",
                      style={"fontFamily": "Arial, sans-serif"}),
            html.Hr()
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H4("Attention Visualization", className="mb-3", 
                   style={"fontFamily": "Arial, sans-serif", "fontWeight": "400"}),
            dcc.Loading(
                id="loading-attention-plot",
                type="circle",
                children=[
                    dcc.Graph(id="attention-plot", style={"height": "600px"})
                ]
            )
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H4("Attention Statistics", className="mb-3", 
                   style={"fontFamily": "Arial, sans-serif", "fontWeight": "400"}),
            html.Div(id="attention-stats", className="p-3 bg-light rounded", 
                    style={"fontFamily": "Arial, sans-serif"})
        ], width=12)
    ]),
    
    # Store for attention data
    dcc.Store(id="attention-data-store")
], fluid=True, className="py-4")

# Load text examples
def load_text_examples():
    examples_dir = os.path.join(os.path.dirname(__file__), "examples")
    examples = {}
    
    print(f"Loading examples from: {examples_dir}")
    
    # Create examples directory if it doesn't exist
    if not os.path.exists(examples_dir):
        print("Examples directory doesn't exist, creating it and example files...")
        os.makedirs(examples_dir)
        
        # Create example files
        with open(os.path.join(examples_dir, "attention_is_all_you_need.txt"), "w") as f:
            f.write("The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.")
        
        with open(os.path.join(examples_dir, "unbearable_lightness.txt"), "w") as f:
            f.write("The idea of eternal return is a mysterious one, and Nietzsche has often perplexed other philosophers with it: to think that everything recurs as we once experienced it, and that the recurrence itself recurs ad infinitum! What does this mad myth signify?")
        
        with open(os.path.join(examples_dir, "into_thin_air.txt"), "w") as f:
            f.write("Straddling the top of the world, one foot in Tibet and the other in Nepal, I cleared the ice from my oxygen mask, hunched a shoulder against the wind, and stared absently down at the vastness of Tibet. I understood on some dim, detached level that the sweep of earth beneath my feet was a spectacular sight. I'd been fantasizing about this moment, and the release of emotion that would accompany it, for many months. But now that I was finally here, standing on the roof of the world, I just couldn't summon the energy to care.")
    else:
        print(f"Examples directory exists, contains: {os.listdir(examples_dir)}")
    
    # Load examples
    for filename in os.listdir(examples_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(examples_dir, filename), "r") as f:
                examples[filename.replace(".txt", "")] = f.read()
    
    print(f"Loaded {len(examples)} examples: {list(examples.keys())}")
    return examples

# Callback to update text display
@app.callback(
    Output("text-display", "children"),
    Input("text-dropdown", "value")
)
def update_text_display(selected_text):
    examples = load_text_examples()
    if selected_text in examples:
        return html.P(examples[selected_text], style={"whiteSpace": "pre-wrap"})
    return "Text not found."

# Callback to analyze text and update attention data and sliders
@app.callback(
    [Output("attention-data-store", "data"),
     Output("layer-slider", "max"),
     Output("head-slider", "max")],
    [Input("analyze-button", "n_clicks")],
    [State("text-dropdown", "value"),
     State("model-dropdown", "value")]
)
def analyze_text_and_update_sliders(n_clicks, selected_text, model_name):
    print(f"analyze_text_and_update_sliders called: n_clicks={n_clicks}, selected_text={selected_text}, model_name={model_name}")
    
    if n_clicks is None:
        print("No clicks, returning None")
        return None, 11, 11
    
    examples = load_text_examples()
    if selected_text not in examples:
        print(f"Text '{selected_text}' not found in examples")
        return None, 11, 11
    
    text = examples[selected_text]
    print(f"Analyzing text: {text[:50]}...")
    
    # Initialize analyzer
    try:
        print(f"Initializing analyzer with model: {model_name}")
        analyzer = AttentionAnalyzer(model_name)
        
        # Analyze text
        print("Calling analyzer.analyze()")
        attention_data = analyzer.analyze(text)
        print(f"analyzer.analyze() returned data with keys: {attention_data.keys() if attention_data else 'None'}")
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {
            'patterns': [],
            'text': text,
            'num_layers': attention_data['num_layers'],
            'num_heads': attention_data['num_heads']
        }
        
        for layer_data in attention_data['patterns']:
            serializable_data['patterns'].append({
                'layer': layer_data['layer'],
                'attention': layer_data['attention'].tolist(),
                'tokens': layer_data['tokens']
            })
        
        return serializable_data, attention_data['num_layers'] - 1, attention_data['num_heads'] - 1
    except Exception as e:
        print(f"Error analyzing text: {e}")
        return None, 11, 11

# Callback to update attention plot
@app.callback(
    Output("attention-plot", "figure"),
    [Input("attention-data-store", "data"),
     Input("layer-slider", "value"),
     Input("head-slider", "value")]
)
def update_attention_plot(data, layer_idx, head_idx):
    print(f"update_attention_plot called: layer_idx={layer_idx}, head_idx={head_idx}")
    print(f"attention_data_store contains data: {data is not None}")
    
    if data is None:
        print("No data, returning empty figure")
        return go.Figure()
    
    print(f"Number of patterns in data: {len(data['patterns']) if 'patterns' in data else 'None'}")
    
    # Extract data for the selected layer and head
    layer_data = data['patterns'][layer_idx]
    attention_matrix = np.array(layer_data['attention'][head_idx])
    tokens = layer_data['tokens']
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=attention_matrix,
        x=tokens,
        y=tokens,
        colorscale='Viridis',
        hoverongaps=False,
        hovertemplate='Source: %{y}<br>Target: %{x}<br>Attention: %{z:.4f}<extra></extra>'
    ))
    
    # Update layout for minimalist design
    fig.update_layout(
        title=f"Layer {layer_idx}, Head {head_idx}",
        xaxis_title="Target Tokens",
        yaxis_title="Source Tokens",
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            gridcolor='lightgray',
            zerolinecolor='lightgray',
            tickangle=45
        ),
        yaxis=dict(
            gridcolor='lightgray',
            zerolinecolor='lightgray'
        )
    )
    
    return fig

# Callback to update attention statistics
@app.callback(
    Output("attention-stats", "children"),
    [Input("attention-data-store", "data"),
     Input("layer-slider", "value"),
     Input("head-slider", "value")]
)
def update_attention_stats(attention_data, layer, head):
    if attention_data is None:
        return "No data available. Please analyze a text."
    
    # Extract data for the selected layer and head
    layer_data = attention_data['patterns'][layer]
    attention_matrix = np.array(layer_data['attention'][head])
    
    # Calculate statistics
    mean_attention = np.mean(attention_matrix)
    max_attention = np.max(attention_matrix)
    entropy = -np.sum(attention_matrix * np.log2(attention_matrix + 1e-10))
    
    return html.Div([
        html.P(f"Mean Attention: {mean_attention:.4f}"),
        html.P(f"Max Attention: {max_attention:.4f}"),
        html.P(f"Entropy: {entropy:.4f}"),
        html.P(f"Most Attended Token: {layer_data['tokens'][np.argmax(np.mean(attention_matrix, axis=1))]}")
    ])

if __name__ == "__main__":
    app.run(debug=True) 