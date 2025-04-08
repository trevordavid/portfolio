import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from attention_analyzer import AttentionAnalyzer

class AttentionVisualizer:
    def __init__(self, model_name="bert-base-uncased"):
        self.analyzer = AttentionAnalyzer(model_name)
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.setup_layout()
        
    def setup_layout(self):
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col(html.H1("Transformer Attention Pattern Analysis", className="text-center my-4"))
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Input Text", className="card-title"),
                            dbc.Input(id="input-text", type="text", placeholder="Enter text to analyze..."),
                            html.Br(),
                            dbc.Button("Analyze", id="analyze-button", color="primary", className="mt-2")
                        ])
                    ], className="mb-4")
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Model Configuration", className="card-title"),
                            dbc.Select(
                                id="layer-select",
                                options=[{"label": f"Layer {i}", "value": i} for i in range(12)],
                                value=0
                            ),
                            html.Br(),
                            dbc.Select(
                                id="head-select",
                                options=[{"label": f"Head {i}", "value": i} for i in range(12)],
                                value=0
                            )
                        ])
                    ], className="mb-4")
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="attention-heatmap")
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div(id="analysis-output")
                ])
            ])
        ], fluid=True)
        
        @self.app.callback(
            [dash.Output("attention-heatmap", "figure"),
             dash.Output("analysis-output", "children")],
            [dash.Input("analyze-button", "n_clicks")],
            [dash.State("input-text", "value"),
             dash.State("layer-select", "value"),
             dash.State("head-select", "value")]
        )
        def update_visualization(n_clicks, text, layer, head):
            if not text:
                return go.Figure(), "Please enter text to analyze"
                
            # Get attention patterns
            attention_patterns = self.analyzer.get_attention_patterns(text)
            layer_pattern = attention_patterns[layer][head]
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=layer_pattern,
                colorscale="Viridis",
                showscale=True
            ))
            
            fig.update_layout(
                title=f"Attention Pattern (Layer {layer}, Head {head})",
                xaxis_title="Key Tokens",
                yaxis_title="Query Tokens"
            )
            
            # Generate analysis text
            analysis = self.analyzer.analyze_pattern(layer_pattern)
            
            return fig, html.Div([
                html.H4("Pattern Analysis"),
                html.P(analysis)
            ])
    
    def run_server(self, debug=True, port=8050):
        self.app.run_server(debug=debug, port=port)

if __name__ == "__main__":
    visualizer = AttentionVisualizer()
    visualizer.run_server() 