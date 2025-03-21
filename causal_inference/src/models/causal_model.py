from typing import Optional, Dict, Any
import pandas as pd
from dowhy import CausalModel
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)

class ExoplanetCausalModel:
    """Handles causal inference analysis for exoplanet data."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the causal model.
        
        Args:
            data: Preprocessed DataFrame containing exoplanet data
        """
        self.data = data
        self.model = None
        self.identified_estimand = None
        self.estimate = None
        
        # Define the causal graph based on stellar evolutionary theory
        # Note: Using proper DOT syntax with semicolons after each statement
        self.graph = """
        digraph {
            A [label="System Age"];
            Y [label="Exoplanet Size"];
            M [label="Stellar Mass"];
            R [label="Stellar Radius"];
            F [label="Stellar Metallicity"];
            G [label="Stellar Surface Gravity"];
            T [label="Stellar Effective Temperature"];
            P [label="Orbital Period"];

            # Stellar evolution relationships
            A -> F;  # Age affects metallicity (galactic chemical evolution)
            A -> R;  # Age affects stellar radius (stellar evolution)
            A -> T;  # Age affects temperature (stellar evolution)
            M -> R;  # Mass affects radius (stellar structure)
            M -> G;  # Mass affects surface gravity
            M -> T;  # Mass affects temperature
            F -> R;  # Metallicity affects radius
            R -> G;  # Radius affects surface gravity

            # Direct effects on planet size
            A -> Y;  # Age affects planet size (formation/evolution)
            F -> Y;  # Metallicity affects planet size (formation)
            R -> Y;  # Stellar radius affects planet size (measurement)
            P -> Y;  # Orbital period affects planet size (evolution)
        }
        """
        
    def fit(self) -> None:
        """Fit the causal model to the data."""
        try:
            # Try to create the causal model with the DOT graph
            self.model = CausalModel(
                data=self.data,
                treatment='A',  # System Age
                outcome='Y',    # Exoplanet Size
                graph=self.graph
            )
            logger.info("Successfully fitted causal model")
        except Exception as e:
            logger.error(f"Error fitting causal model: {e}")
            
            # Fallback to defining common causes directly if the graph fails
            logger.info("Falling back to manual common causes definition")
            try:
                self.model = CausalModel(
                    data=self.data,
                    treatment='A',  # System Age
                    outcome='Y',    # Exoplanet Size
                    common_causes=['M', 'R', 'F', 'G', 'T', 'P']
                )
                logger.info("Successfully fitted causal model with manual common causes")
            except Exception as e2:
                logger.error(f"Fallback also failed: {e2}")
                raise
        
    def identify_effect(self) -> None:
        """Identify the causal effect using the backdoor criterion."""
        if self.model is None:
            raise ValueError("Model must be fit before identifying effect")
        try:
            self.identified_estimand = self.model.identify_effect()
            logger.info("Successfully identified causal effect")
        except Exception as e:
            logger.error(f"Error identifying effect: {e}")
            raise
        
    def estimate_effect(self, method: str = "backdoor.linear_regression") -> None:
        """
        Estimate the causal effect using specified method.
        
        Args:
            method: Estimation method to use
        """
        if self.identified_estimand is None:
            raise ValueError("Effect must be identified before estimation")
        try:
            self.estimate = self.model.estimate_effect(
                self.identified_estimand,
                method_name=method,
                confidence_intervals=True
            )
            logger.info(f"Successfully estimated effect using {method}")
        except Exception as e:
            logger.error(f"Error estimating effect: {e}")
            raise
        
    def refute_estimate(self, method: str = "placebo_treatment_refuter") -> Dict[str, Any]:
        """
        Refute the causal estimate using specified method.
        
        Args:
            method: Refutation method to use
            
        Returns:
            Dict containing refutation results
        """
        if self.estimate is None:
            raise ValueError("Effect must be estimated before refutation")
        try:
            result = self.model.refute_estimate(
                self.identified_estimand,
                self.estimate,
                method_name=method
            )
            logger.info(f"Successfully refuted estimate using {method}")
            return result
        except Exception as e:
            logger.error(f"Error refuting estimate: {e}")
            # Return a placeholder if refutation fails
            return {"error": str(e), "refutation_failed": True}
        
    def plot_causal_graph(self, save_path: Optional[str] = None) -> None:
        """
        Plot the causal graph with clear directional arrows.
        
        Args:
            save_path: Optional path to save the plot
        """
        if self.model is None:
            raise ValueError("Model must be fit before plotting")
            
        try:
            # Create a simplified and clear visualization with proper directionality
            # This approach ensures we have full control over the graph layout
            plt.figure(figsize=(12, 8))
            
            # Create a directed graph with our specific causal structure
            G = nx.DiGraph()
            
            # Add nodes
            nodes = ['A', 'Y', 'M', 'R', 'F', 'G', 'T', 'P']
            for node in nodes:
                G.add_node(node)
            
            # Add edges with the correct causal direction
            edges = [
                # Age causes changes in stellar properties
                ('A', 'F'),  # Age → Metallicity
                ('A', 'R'),  # Age → Radius
                ('A', 'T'),  # Age → Temperature
                ('A', 'Y'),  # Age → Planet Size
                
                # Mass affects stellar properties
                ('M', 'R'),  # Mass → Radius
                ('M', 'G'),  # Mass → Surface Gravity
                ('M', 'T'),  # Mass → Temperature
                
                # Other causal relationships
                ('F', 'R'),  # Metallicity → Radius
                ('F', 'Y'),  # Metallicity → Planet Size
                ('R', 'G'),  # Radius → Surface Gravity
                ('R', 'Y'),  # Radius → Planet Size
                ('P', 'Y')   # Orbital Period → Planet Size
            ]
            G.add_edges_from(edges)
            
            # Use a hierarchical layout to better show causality
            # This positions causes above their effects
            pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
            
            # Node properties
            node_labels = {
                'A': 'System Age',
                'Y': 'Planet Size',
                'M': 'Stellar Mass',
                'R': 'Stellar Radius',
                'F': 'Metallicity',
                'G': 'Surface Gravity',
                'T': 'Temperature',
                'P': 'Orbital Period'
            }
            
            # Draw nodes with labels
            nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', alpha=0.8)
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, font_weight='bold')
            
            # Draw edges with arrows that are clearly visible
            nx.draw_networkx_edges(
                G, pos, 
                arrows=True,
                arrowsize=20,
                width=1.5,
                edge_color='navy',
                arrowstyle='-|>',  # Stronger arrow style
                connectionstyle='arc3,rad=0.1'  # Slightly curved edges
            )
            
            plt.title('Causal Graph: Effect of System Age on Exoplanet Size', fontsize=16)
            
            plt.axis('off')  # Hide axes
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved causal graph to {save_path}")
            
            plt.close()
            return True
            
        except Exception as e:
            logger.warning(f"Could not plot causal graph with hierarchical layout: {e}")
            logger.info("Falling back to basic graph visualization")
            
            try:
                # Fallback to a simpler visualization
                plt.figure(figsize=(12, 8))
                G = nx.DiGraph()
                
                # Add nodes with descriptive labels
                G.add_node('A', label='System Age')
                G.add_node('Y', label='Planet Size')
                G.add_node('M', label='Stellar Mass')
                G.add_node('R', label='Stellar Radius')
                G.add_node('F', label='Metallicity')
                G.add_node('G', label='Surface Gravity')
                G.add_node('T', label='Temperature')
                G.add_node('P', label='Orbital Period')
                
                # Add edges (same as above)
                edges = [
                    ('A', 'F'), ('A', 'R'), ('A', 'T'), ('A', 'Y'),
                    ('M', 'R'), ('M', 'G'), ('M', 'T'),
                    ('F', 'R'), ('F', 'Y'),
                    ('R', 'G'), ('R', 'Y'),
                    ('P', 'Y')
                ]
                G.add_edges_from(edges)
                
                # Use spring layout but with fixed positions to ensure clarity
                pos = nx.spring_layout(G, seed=42)
                
                # Manually adjust positions for clarity
                # Position Age at the top to show it as a primary cause
                pos['A'] = np.array([0.5, 0.9])
                pos['Y'] = np.array([0.5, 0.1])  # Planet size at bottom
                
                # Draw the graph with clear directional arrows
                nx.draw(G, pos, with_labels=True, 
                        node_color='lightblue',
                        node_size=2000,
                        font_weight='bold', 
                        font_size=12, 
                        edge_color='navy',
                        arrows=True,
                        arrowsize=15,
                        width=1.5)
                
                # Add a title and note about arrow direction
                plt.title('Causal Graph: Effect of System Age on Exoplanet Size', fontsize=16)
                
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Saved causal graph to {save_path}")
                plt.close()
                return True
            except Exception as e2:
                logger.error(f"All graph visualization methods failed: {e2}")
                return False
        
    def get_results(self) -> Dict[str, Any]:
        """
        Get the complete results of the causal analysis.
        
        Returns:
            Dict containing all analysis results
        """
        if self.estimate is None:
            raise ValueError("Analysis must be completed before getting results")
            
        try:
            # Helper function to convert numpy types to Python native types
            def convert_numpy_types(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(convert_numpy_types(item) for item in obj)
                return obj
            
            # Get confidence intervals safely
            try:
                confidence_intervals = self.estimate.get_confidence_intervals()
                confidence_intervals = convert_numpy_types(confidence_intervals)
            except Exception as e:
                logger.warning(f"Failed to get confidence intervals: {e}")
                confidence_intervals = None
            
            # Get refutation results safely
            try:
                refutation_results = self.refute_estimate()
                
                # Convert complex refutation results to serializable form
                if hasattr(refutation_results, 'to_dict'):
                    refutation_dict = refutation_results.to_dict()
                else:
                    refutation_dict = {"method": "placebo_treatment_refuter"}
                    
                    if hasattr(refutation_results, "new_effect"):
                        refutation_dict["new_effect"] = float(refutation_results.new_effect)
                    
                    if hasattr(refutation_results, "refutation_result"):
                        refutation_dict["refutation_result"] = str(refutation_results.refutation_result)
                    
                    if hasattr(refutation_results, "p_value"):
                        refutation_dict["p_value"] = float(refutation_results.p_value)
                
                refutation_results = convert_numpy_types(refutation_dict)
            except Exception as e:
                logger.warning(f"Failed to refute estimate: {e}")
                refutation_results = {"error": str(e)}
            
            # Make sure all values are JSON serializable
            causal_estimate = float(self.estimate.value)
            
            return {
                "identified_estimand": str(self.identified_estimand),
                "causal_estimate": causal_estimate,
                "confidence_intervals": confidence_intervals,
                "refutation_results": refutation_results,
                "model_metadata": {
                    "treatment": "System Age (A)",
                    "outcome": "Exoplanet Size (Y)",
                    "estimation_method": self.estimate.params.get("method_name", "backdoor.linear_regression"),
                    "sample_size": len(self.data)
                }
            }
        except Exception as e:
            logger.error(f"Error getting results: {e}")
            # Return partial results
            return {
                "error": str(e),
                "causal_estimate": float(self.estimate.value) if self.estimate else None
            } 