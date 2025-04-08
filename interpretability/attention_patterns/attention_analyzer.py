import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class AttentionAnalyzer:
    def __init__(self, model_name: str = "bert-base-uncased"):
        print(f"Initializing AttentionAnalyzer with model_name: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        try:
            print(f"Loading model from {model_name}")
            self.model = AutoModel.from_pretrained(model_name, output_attentions=True).to(self.device)
            print(f"Model loaded successfully, type: {type(self.model)}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"Tokenizer loaded successfully, type: {type(self.tokenizer)}")
        except Exception as e:
            print(f"Error loading model or tokenizer: {e}")
            raise
        
    def get_attention_patterns(self, text: str) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Get attention patterns for all layers and heads for the given text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict[int, Dict[int, np.ndarray]]: Attention patterns for each layer and head
        """
        # Tokenize and prepare input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Extract attention patterns
        attention_patterns = {}
        for layer_idx, layer_attention in enumerate(outputs.attentions):
            attention_patterns[layer_idx] = {}
            # Shape: [batch_size, num_heads, seq_length, seq_length]
            layer_attention = layer_attention[0].cpu().numpy()
            
            for head_idx in range(layer_attention.shape[0]):
                attention_patterns[layer_idx][head_idx] = layer_attention[head_idx]
                
        return attention_patterns
    
    def analyze_pattern(self, attention_matrix: np.ndarray) -> str:
        """
        Analyze an attention pattern matrix and provide insights.
        
        Args:
            attention_matrix (np.ndarray): Attention pattern matrix
            
        Returns:
            str: Analysis of the attention pattern
        """
        # Calculate statistics
        mean_attention = np.mean(attention_matrix)
        max_attention = np.max(attention_matrix)
        min_attention = np.min(attention_matrix)
        
        # Check for strong diagonal patterns (self-attention)
        diagonal_strength = np.mean(np.diag(attention_matrix))
        
        # Check for uniform distribution
        std_dev = np.std(attention_matrix)
        
        # Generate analysis text
        analysis = []
        analysis.append(f"Mean attention weight: {mean_attention:.3f}")
        analysis.append(f"Maximum attention weight: {max_attention:.3f}")
        analysis.append(f"Minimum attention weight: {min_attention:.3f}")
        
        if diagonal_strength > 0.5:
            analysis.append("Strong self-attention pattern detected (high diagonal values)")
        elif diagonal_strength < 0.1:
            analysis.append("Weak self-attention pattern (low diagonal values)")
            
        if std_dev < 0.1:
            analysis.append("Uniform attention distribution")
        elif std_dev > 0.3:
            analysis.append("Highly focused attention pattern")
            
        return " | ".join(analysis)
    
    def get_token_importance(self, text: str, layer: int, head: int) -> List[Tuple[str, float]]:
        """
        Calculate token importance based on attention weights.
        
        Args:
            text (str): Input text
            layer (int): Layer index
            head (int): Head index
            
        Returns:
            List[Tuple[str, float]]: List of (token, importance) pairs
        """
        tokens = self.tokenizer.tokenize(text)
        attention_patterns = self.get_attention_patterns(text)
        attention_matrix = attention_patterns[layer][head]
        
        # Calculate importance as mean attention received by each token
        importance = np.mean(attention_matrix, axis=0)
        
        return list(zip(tokens, importance))
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze attention patterns for a given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing attention patterns and metadata
        """
        print(f"analyze() called with text: {text[:50]}...")
        
        # Tokenize input text
        try:
            print("Tokenizing input...")
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            print(f"Tokenized input: {inputs.keys()}")
            
            # Get model outputs with attention
            print("Running model...")
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            print(f"Model output keys: {outputs.keys() if hasattr(outputs, 'keys') else dir(outputs)}")
            
            # Extract attention patterns
            attention_patterns = outputs.attentions  # Tuple of attention tensors for each layer
            print(f"Got {len(attention_patterns)} attention patterns")
            
            # Process attention patterns
            processed_patterns = []
            for layer_idx, layer_attention in enumerate(attention_patterns):
                # Shape: (batch_size, num_heads, seq_length, seq_length)
                print(f"Processing layer {layer_idx}, shape: {layer_attention.shape}")
                layer_data = {
                    'layer': layer_idx,
                    'attention': layer_attention[0].cpu().numpy(),  # Remove batch dimension
                    'tokens': self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                }
                processed_patterns.append(layer_data)
                
            result = {
                'patterns': processed_patterns,
                'text': text,
                'num_layers': len(attention_patterns),
                'num_heads': attention_patterns[0].shape[1]
            }
            print(f"Returning result with {len(processed_patterns)} patterns")
            return result
        except Exception as e:
            print(f"Error in analyze(): {e}")
            raise
    
    def get_attention_stats(self, attention_data: Dict) -> Dict:
        """
        Calculate statistics about attention patterns.
        
        Args:
            attention_data: Output from analyze() method
            
        Returns:
            Dictionary containing attention statistics
        """
        stats = {}
        
        for layer_data in attention_data['patterns']:
            layer_idx = layer_data['layer']
            attention = layer_data['attention']
            
            # Calculate statistics for each head
            head_stats = {
                'mean_attention': np.mean(attention, axis=(1, 2)),
                'max_attention': np.max(attention, axis=(1, 2)),
                'attention_entropy': -np.sum(attention * np.log2(attention + 1e-10), axis=(1, 2))
            }
            
            stats[f'layer_{layer_idx}'] = head_stats
            
        return stats
    
    def visualize(self, attention_data: Dict, layer: Optional[int] = None, 
                 head: Optional[int] = None, save_path: Optional[str] = None):
        """
        Visualize attention patterns.
        
        Args:
            attention_data: Output from analyze() method
            layer: Specific layer to visualize (if None, visualize all layers)
            head: Specific attention head to visualize (if None, visualize all heads)
            save_path: Path to save the visualization (if None, display the plot)
        """
        if layer is not None:
            layers_to_plot = [layer]
        else:
            layers_to_plot = range(attention_data['num_layers'])
            
        for layer_idx in layers_to_plot:
            layer_data = attention_data['patterns'][layer_idx]
            attention = layer_data['attention']
            tokens = layer_data['tokens']
            
            if head is not None:
                heads_to_plot = [head]
            else:
                heads_to_plot = range(attention_data['num_heads'])
                
            for head_idx in heads_to_plot:
                plt.figure(figsize=(10, 8))
                sns.heatmap(attention[head_idx], 
                           xticklabels=tokens,
                           yticklabels=tokens,
                           cmap='viridis')
                plt.title(f'Layer {layer_idx}, Head {head_idx}')
                plt.xlabel('Target Tokens')
                plt.ylabel('Source Tokens')
                
                if save_path:
                    plt.savefig(f'{save_path}_layer{layer_idx}_head{head_idx}.png')
                else:
                    plt.show()
                plt.close() 