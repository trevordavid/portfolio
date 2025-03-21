"""Dash dashboard for the exoplanet causal inference project."""

import json
import os
import sys
from pathlib import Path
import base64
import shutil
from typing import Dict, List, Optional

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State, callback
from dash.exceptions import PreventUpdate

# Add the parent directory to the path to allow importing from src
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.data_fetcher import ExoplanetDataFetcher
from src.models.causal_model import ExoplanetCausalModel

# Initialize the Dash app with a professional theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    title="Exoplanet Causal Inference",
    suppress_callback_exceptions=True,
)

# Function to refresh images from the figures directory
def refresh_images():
    """Copy the latest images from the figures directory to the assets directory."""
    project_root = Path(__file__).parent.parent.parent
    figures_dir = project_root / "figures"
    assets_dir = Path(__file__).parent / "assets"
    assets_dir.mkdir(exist_ok=True)
    
    if figures_dir.exists():
        for fig_path in figures_dir.glob("*.png"):
            dest_path = assets_dir / fig_path.name
            try:
                # Force a fresh copy by updating the modification time
                if dest_path.exists():
                    dest_path.unlink()  # Remove existing file
                shutil.copy(fig_path, dest_path)
                # Touch the file to update its timestamp
                dest_path.touch()
                print(f"Refreshed {fig_path.name} in assets directory")
            except Exception as e:
                print(f"Error copying {fig_path.name}: {e}")

# Define functions to load data and results
def load_data():
    """Load and preprocess the exoplanet data."""
    # Project root is the parent directory of the src directory
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Use the existing data fetcher
    data_fetcher = ExoplanetDataFetcher(cache_dir=str(data_dir))
    return data_fetcher.get_analysis_data()

def load_results():
    """Load the analysis results from the JSON file."""
    project_root = Path(__file__).parent.parent.parent
    results_file = project_root / "results" / "analysis_results.json"
    
    if results_file.exists():
        with open(results_file, "r") as f:
            return json.load(f)
    else:
        return None

def load_figures():
    """Load figure paths from the figures directory."""
    project_root = Path(__file__).parent.parent.parent
    figures_dir = project_root / "figures"
    
    figures = {}
    if figures_dir.exists():
        for file in figures_dir.glob("*.png"):
            figures[file.stem] = str(file)
    
    return figures

# Load data and resources at startup
data = load_data()
results = load_results()
figure_paths = load_figures()

# Define the navigation bar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="#", id="home-link", active=True)),
        dbc.NavItem(dbc.NavLink("Data Explorer", href="#", id="data-link")),
        dbc.NavItem(dbc.NavLink("Bivariate Analysis", href="#", id="bivariate-link")),
        dbc.NavItem(dbc.NavLink("Causal Analysis", href="#", id="causal-link")),
        dbc.NavItem(dbc.NavLink("Run New Analysis", href="#", id="run-link")),
    ],
    brand="Exoplanet Causal Inference",
    brand_href="#",
    color="primary",
    dark=True,
)

# Define the home page content
home_page = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Exoplanet Causal Inference Analysis", className="mt-4 mb-4"),
            html.P([
                "This dashboard presents the results of a causal inference analysis examining the effect of ",
                "stellar age on exoplanet sizes, using data from the NASA Exoplanet Archive."
            ], className="lead"),
            html.P([
                "The main research question is: ",
                html.Strong("Does stellar age have a causal effect on exoplanet radius?")
            ]),
            html.P([
                "We've applied causal inference methodologies to control for confounding variables and isolate ",
                "the direct causal effect of age on planet size."
            ]),
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H2("Causal Model Framework", className="mt-4"),
            html.P([
                "Our causal model incorporates stellar evolutionary theory, considering relationships between:"
            ]),
            html.Ul([
                html.Li("Stellar properties (mass, radius, temperature, metallicity, surface gravity)"),
                html.Li("System age"),
                html.Li("Exoplanet size"),
                html.Li("Orbital parameters")
            ]),
        ], width=5),
        
        dbc.Col([
            html.Div([
                html.Img(
                    src=app.get_asset_url("causal_graph.png") + f"?t={pd.Timestamp.now().timestamp()}" if "causal_graph" in figure_paths else "",
                    style={"width": "100%"},
                    className="img-fluid"
                ) if "causal_graph" in figure_paths else html.P("Causal graph will appear here after running the analysis.")
            ], className="p-2 border rounded bg-light")
        ], width=7)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H2("Key Findings", className="mt-4"),
            html.Div([
                html.P([
                    "Causal Effect Estimate: ",
                    html.Strong(f"{results['causal_estimate']:.4f}" if results and 'causal_estimate' in results else "Not available")
                ]),
                html.P([
                    "Based on the causal inference analysis, we found that stellar age has a ", 
                    html.Strong("minimal" if results and abs(float(results.get('causal_estimate', 0))) < 0.05 else "significant"),
                    " direct effect on exoplanet radius when controlling for other stellar properties."
                ]),
                html.P("This suggests that exoplanet size is primarily determined by formation conditions rather than long-term evolutionary processes.")
            ], className="p-3 border rounded")
        ], width=12)
    ])
], fluid=True, className="mt-4")

# Define the data explorer page
data_explorer_page = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Data Explorer", className="mt-4 mb-4"),
            html.P([
                "Explore the exoplanet dataset used in the causal analysis. The data has been preprocessed ",
                "and includes the following key variables:"
            ]),
            html.Ul([
                html.Li([html.Strong("A: "), "System Age (treatment variable)"]),
                html.Li([html.Strong("Y: "), "Exoplanet Size (outcome variable)"]),
                html.Li([html.Strong("M: "), "Stellar Mass"]),
                html.Li([html.Strong("R: "), "Stellar Radius"]),
                html.Li([html.Strong("F: "), "Stellar Metallicity"]),
                html.Li([html.Strong("G: "), "Stellar Surface Gravity"]),
                html.Li([html.Strong("T: "), "Stellar Effective Temperature"]),
                html.Li([html.Strong("P: "), "Orbital Period"])
            ])
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H2("Dataset Sample", className="mt-3"),
            html.Div([
                dash.dash_table.DataTable(
                    data=data.head(10).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in data.columns],
                    style_cell={
                        'textAlign': 'left',
                        'padding': '8px',
                        'font-family': 'Source Sans Pro, sans-serif',
                    },
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold',
                        'border': '1px solid black'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }
                    ],
                    page_size=10,
                )
            ], className="mt-2")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H2("Dataset Statistics", className="mt-4"),
            html.Div([
                dash.dash_table.DataTable(
                    data=data.describe().reset_index().rename(columns={"index": "statistic"}).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in ["statistic"] + list(data.columns)],
                    style_cell={
                        'textAlign': 'left',
                        'padding': '8px',
                        'font-family': 'Source Sans Pro, sans-serif',
                    },
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold',
                        'border': '1px solid black'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }
                    ],
                    page_size=10,
                )
            ], className="mt-2")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H2("Variable Distribution", className="mt-4"),
            html.Label("Select a variable:"),
            dcc.Dropdown(
                id="var-distribution-dropdown",
                options=[{"label": col, "value": col} for col in data.columns],
                value="A" if "A" in data.columns else data.columns[0]
            ),
            dcc.Graph(id="var-distribution-graph")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H2("Correlation Matrix", className="mt-4"),
            html.Div([
                html.Img(
                    src=app.get_asset_url("correlation_matrix.png") if "correlation_matrix" in figure_paths else "",
                    style={"width": "100%"},
                    className="img-fluid"
                ) if "correlation_matrix" in figure_paths else 
                dcc.Graph(
                    figure=px.imshow(
                        data.corr(),
                        text_auto=True,
                        color_continuous_scale="Blues",
                        title="Variable Correlations"
                    )
                )
            ], className="p-2 border rounded bg-light")
        ], width=12)
    ])
], fluid=True, className="mt-4")

# Define the bivariate explorer page
bivariate_page = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Bivariate Relationships", className="mt-4 mb-4"),
            html.P([
                "Explore relationships between pairs of variables. This can help identify potential ",
                "causal relationships and confounders."
            ])
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label("Select X variable:"),
            dcc.Dropdown(
                id="x-var-dropdown",
                options=[{"label": col, "value": col} for col in data.columns],
                value="A" if "A" in data.columns else data.columns[0]
            )
        ], width=6),
        
        dbc.Col([
            html.Label("Select Y variable:"),
            dcc.Dropdown(
                id="y-var-dropdown",
                options=[{"label": col, "value": col} for col in data.columns],
                value="Y" if "Y" in data.columns else data.columns[1]
            )
        ], width=6)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label("Color points by (optional):"),
            dcc.Dropdown(
                id="color-var-dropdown",
                options=[{"label": "None", "value": "None"}] + [{"label": col, "value": col} for col in data.columns],
                value="None"
            )
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="bivariate-scatter-plot")
        ], width=8),
        
        dbc.Col([
            html.Div([
                html.H4("Relationship Statistics", className="mt-3"),
                html.Div(id="relationship-stats", className="p-3")
            ], className="border rounded p-3 mt-4")
        ], width=4)
    ])
], fluid=True, className="mt-4")

# Define the causal analysis page
causal_page = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Causal Analysis Results", className="mt-4 mb-4"),
            html.P([
                "This page presents the results of the causal inference analysis examining the effect of ",
                "stellar age on exoplanet sizes."
            ])
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H2("Causal Graph", className="mt-3"),
            html.Div([
                html.Img(
                    src=app.get_asset_url("causal_graph.png") + f"?t={pd.Timestamp.now().timestamp()}" if "causal_graph" in figure_paths else "",
                    style={"width": "100%"},
                    className="img-fluid"
                ) if "causal_graph" in figure_paths else html.P("Causal graph will appear here after running the analysis.")
            ], className="p-2 border rounded bg-light")
        ], width=6),
        
        dbc.Col([
            html.H2("Key Visualizations", className="mt-3"),
            dbc.Tabs([
                dbc.Tab(
                    dbc.Card(dbc.CardImg(
                        src=app.get_asset_url("age_vs_planet_size.png") if "age_vs_planet_size" in figure_paths else "",
                        top=True
                    )),
                    label="Age vs. Planet Size"
                ),
                dbc.Tab(
                    dbc.Card(dbc.CardImg(
                        src=app.get_asset_url("metallicity_vs_planet_size.png") if "metallicity_vs_planet_size" in figure_paths else "",
                        top=True
                    )),
                    label="Metallicity vs. Planet Size"
                ),
                dbc.Tab(
                    dbc.Card(dbc.CardImg(
                        src=app.get_asset_url("stellar_evolution.png") if "stellar_evolution" in figure_paths else "",
                        top=True
                    )),
                    label="Stellar Evolution"
                )
            ])
        ], width=6)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H2("Causal Effect Estimate", className="mt-4"),
            html.Div([
                html.P([
                    "Estimated causal effect: ",
                    html.Strong(f"{results['causal_estimate']:.4f}" if results and 'causal_estimate' in results else "Not available")
                ]),
                html.P([
                    "Confidence interval: ",
                    html.Span(
                        f"[{results['confidence_intervals'][0][0]:.4f}, {results['confidence_intervals'][0][1]:.4f}]" 
                        if results and 'confidence_intervals' in results and results['confidence_intervals'] 
                        else "Not available"
                    )
                ]),
                html.P([
                    "Interpretation: Based on the causal inference analysis, we found that stellar age has a ",
                    html.Strong(
                        "minimal" if results and abs(float(results.get('causal_estimate', 0))) < 0.05 else "significant"
                    ),
                    " direct effect on exoplanet radius when controlling for other stellar properties."
                ])
            ], className="p-3 border rounded")
        ], width=6),
        
        dbc.Col([
            html.H2("Refutation Tests", className="mt-4"),
            html.Div([
                html.P("Placebo Treatment Test Results:"),
                html.Ul([
                    html.Li([
                        "Placebo Effect: ",
                        html.Span(
                            f"{float(results['refutation_results']['placebo_treatment_refuter']['new_effect']):.4e}" 
                            if results and 'refutation_results' in results and 'placebo_treatment_refuter' in results['refutation_results'] 
                            else "Not available"
                        )
                    ]),
                    html.Li([
                        "P-value: ",
                        html.Span(
                            f"{float(results['refutation_results']['placebo_treatment_refuter']['p_value']):.4f}" 
                            if results and 'refutation_results' in results and 'placebo_treatment_refuter' in results['refutation_results'] 
                            else "Not available"
                        )
                    ]),
                    html.Li([
                        "Result: ",
                        html.Strong(
                            "Passed" 
                            if results and 'refutation_results' in results and 'placebo_treatment_refuter' in results['refutation_results'] and 
                            float(results['refutation_results']['placebo_treatment_refuter']['p_value']) < 0.05 
                            else "Failed"
                        ),
                        " - The causal estimate is robust to placebo treatments."
                    ])
                ])
            ], className="p-3 border rounded")
        ], width=6)
    ])
], fluid=True, className="mt-4")

# Define the run new analysis page
run_analysis_page = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Run New Analysis", className="mt-4 mb-4"),
            html.P([
                "Configure and run a new causal inference analysis on the exoplanet data."
            ])
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Analysis Configuration"),
                dbc.CardBody([
                    dbc.Form([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Treatment Variable"),
                                dcc.Dropdown(
                                    id="treatment-dropdown",
                                    options=[{"label": col, "value": col} for col in data.columns],
                                    value="A" if "A" in data.columns else data.columns[0]
                                )
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Outcome Variable"),
                                dcc.Dropdown(
                                    id="outcome-dropdown",
                                    options=[{"label": col, "value": col} for col in data.columns],
                                    value="Y" if "Y" in data.columns else data.columns[1]
                                )
                            ], width=6)
                        ]),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Estimator Method"),
                                dcc.Dropdown(
                                    id="estimator-dropdown",
                                    options=[
                                        {"label": "Linear Regression", "value": "linear_regression"},
                                        {"label": "Propensity Score Matching", "value": "propensity_score_matching"},
                                        {"label": "Propensity Score Stratification", "value": "propensity_score_stratification"}
                                    ],
                                    value="linear_regression"
                                )
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Run Refutation Tests"),
                                dbc.Checklist(
                                    id="refutation-checklist",
                                    options=[
                                        {"label": "Placebo Treatment", "value": "placebo"},
                                        {"label": "Random Common Cause", "value": "random_cause"}
                                    ],
                                    value=["placebo"],
                                    inline=True
                                )
                            ], width=6)
                        ]),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("Run Analysis", id="run-analysis-btn", color="primary", className="mt-3")
                            ], width=12)
                        ])
                    ])
                ])
            ])
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div(id="analysis-results-container", className="mt-4")
        ], width=12)
    ])
], fluid=True, className="mt-4")

# Define the main app layout
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    navbar,
    html.Div(id="page-content", children=home_page),
    # Hidden div to trigger refresh_images on page load
    html.Div(id="refresh-trigger", style={"display": "none"})
])

# Define the callback for page navigation
@app.callback(
    Output("page-content", "children"),
    [
        Input("home-link", "n_clicks"),
        Input("data-link", "n_clicks"),
        Input("bivariate-link", "n_clicks"),
        Input("causal-link", "n_clicks"),
        Input("run-link", "n_clicks")
    ],
    prevent_initial_call=True
)
def navigate(home_clicks, data_clicks, bivariate_clicks, causal_clicks, run_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return home_page
    
    # Refresh images from figures directory
    refresh_images()
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == "home-link":
        return home_page
    elif button_id == "data-link":
        return data_explorer_page
    elif button_id == "bivariate-link":
        return bivariate_page
    elif button_id == "causal-link":
        return causal_page
    elif button_id == "run-link":
        return run_analysis_page
    else:
        return home_page

# Callbacks for interactive components
@app.callback(
    Output("var-distribution-graph", "figure"),
    Input("var-distribution-dropdown", "value")
)
def update_distribution(selected_var):
    if not selected_var:
        raise PreventUpdate
    
    fig = px.histogram(
        data, 
        x=selected_var, 
        title=f"Distribution of {selected_var}",
        color_discrete_sequence=["#2C3E50"]
    )
    fig.update_layout(
        xaxis_title=selected_var,
        yaxis_title="Count",
        plot_bgcolor="#f9f9f9",
        paper_bgcolor="#f9f9f9",
        font=dict(family="Source Sans Pro, sans-serif")
    )
    return fig

@app.callback(
    [Output("bivariate-scatter-plot", "figure"),
     Output("relationship-stats", "children")],
    [Input("x-var-dropdown", "value"),
     Input("y-var-dropdown", "value"),
     Input("color-var-dropdown", "value")]
)
def update_bivariate_plot(x_var, y_var, color_var):
    if not x_var or not y_var:
        raise PreventUpdate
    
    if color_var == "None":
        fig = px.scatter(
            data, 
            x=x_var, 
            y=y_var,
            trendline="ols",
            title=f"{x_var} vs {y_var}",
            color_discrete_sequence=["#2C3E50"],
            opacity=0.7
        )
    else:
        # Create a binned version of the color variable for better visualization
        data_temp = data.copy()
        try:
            data_temp[f"{color_var}_binned"] = pd.qcut(
                data_temp[color_var], 
                q=5, 
                duplicates="drop"
            ).astype(str)
            
            fig = px.scatter(
                data_temp, 
                x=x_var, 
                y=y_var,
                color=f"{color_var}_binned",
                title=f"{x_var} vs {y_var} (colored by {color_var})",
                opacity=0.7
            )
        except Exception:
            # Fall back to using the raw color variable
            fig = px.scatter(
                data, 
                x=x_var, 
                y=y_var,
                color=color_var,
                title=f"{x_var} vs {y_var} (colored by {color_var})",
                opacity=0.7
            )
    
    fig.update_layout(
        xaxis_title=x_var,
        yaxis_title=y_var,
        plot_bgcolor="#f9f9f9",
        paper_bgcolor="#f9f9f9",
        font=dict(family="Source Sans Pro, sans-serif")
    )
    
    # Calculate relationship statistics
    corr = data[x_var].corr(data[y_var])
    mean_x = data[x_var].mean()
    mean_y = data[y_var].mean()
    std_x = data[x_var].std()
    std_y = data[y_var].std()
    
    correlation_strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.3 else "weak"
    correlation_direction = "positive" if corr > 0 else "negative"
    
    stats_div = [
        html.P(f"Correlation: {corr:.4f}"),
        html.P(f"{x_var} mean: {mean_x:.4f}"),
        html.P(f"{y_var} mean: {mean_y:.4f}"),
        html.P(f"{x_var} std dev: {std_x:.4f}"),
        html.P(f"{y_var} std dev: {std_y:.4f}"),
        html.Hr(),
        html.P([
            f"Interpretation: There is a {abs(corr):.2f} ",
            html.Strong(correlation_strength),
            f" {correlation_direction} correlation between {x_var} and {y_var}."
        ])
    ]
    
    return fig, stats_div

@app.callback(
    Output("analysis-results-container", "children"),
    Input("run-analysis-btn", "n_clicks"),
    [State("treatment-dropdown", "value"),
     State("outcome-dropdown", "value"),
     State("estimator-dropdown", "value"),
     State("refutation-checklist", "value")]
)
def run_new_analysis(n_clicks, treatment, outcome, estimator, refutation_tests):
    if not n_clicks:
        raise PreventUpdate

    if treatment == outcome:
        return dbc.Alert(
            "Error: Treatment and outcome variables must be different",
            color="danger"
        )
    
    # Note: In a production app, this would actually run the analysis
    # Here we just provide a sample result
    result_card = dbc.Card([
        dbc.CardHeader(html.H3("Analysis Results")),
        dbc.CardBody([
            html.P(f"Treatment: {treatment}"),
            html.P(f"Outcome: {outcome}"),
            html.P(f"Estimator: {estimator}"),
            html.P(f"Refutation tests: {', '.join(refutation_tests)}"),
            html.Hr(),
            html.P([
                "Sample result: The causal effect of ",
                html.Strong(f"{treatment}"),
                " on ",
                html.Strong(f"{outcome}"),
                " is estimated to be ",
                html.Strong("-0.0019"),
                " using the ",
                html.Strong(f"{estimator}"),
                " estimator."
            ]),
            dbc.Alert(
                "Note: This is just a demonstration. To run a real analysis, use the Python API or command line interface.",
                color="info"
            )
        ])
    ])
    
    return result_card

# Callback to refresh images on page load
@app.callback(
    Output("refresh-trigger", "children"),
    Input("refresh-trigger", "id")
)
def refresh_on_load(trigger_id):
    refresh_images()
    return "Images refreshed"

# Main function to run the server
def main():
    # Refresh images when starting the app
    refresh_images()
    
    # Start the server with a specific port
    print("Dashboard will be available at: http://127.0.0.1:8060/")
    app.run(debug=True, port=8060)

if __name__ == "__main__":
    main() 