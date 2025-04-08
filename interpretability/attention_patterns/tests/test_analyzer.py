import torch
from attention_analyzer import AttentionAnalyzer
import os
import json

def main():
    print("Starting test...")
    
    # Create a test text
    text = "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks."
    
    print(f"Test text: {text}")
    
    # Initialize analyzer with a small model
    model_name = "distilbert-base-uncased"
    print(f"Using model: {model_name}")
    
    try:
        # Initialize analyzer
        analyzer = AttentionAnalyzer(model_name)
        
        # Get attention data
        print("Calling analyze()...")
        attention_data = analyzer.analyze(text)
        
        # Print structure and basic info
        print("\nAttention Data Structure:")
        print(f"- Keys: {attention_data.keys() if attention_data else None}")
        
        if attention_data and 'patterns' in attention_data:
            print(f"- Number of layers: {len(attention_data['patterns'])}")
            print(f"- First layer keys: {attention_data['patterns'][0].keys() if attention_data['patterns'] else None}")
            
            # Check attention shape for first layer, first head
            first_layer = attention_data['patterns'][0]
            if 'attention' in first_layer:
                attention = first_layer['attention']
                print(f"- Attention shape: {attention.shape}")
                print(f"- Number of heads: {attention.shape[0] if len(attention.shape) > 1 else 1}")
                
                # Check for NaN or zeros
                print(f"- Contains NaN: {torch.tensor(attention).isnan().any().item()}")
                print(f"- Contains only zeros: {(torch.tensor(attention) == 0).all().item()}")
                print(f"- Min value: {torch.tensor(attention).min().item()}")
                print(f"- Max value: {torch.tensor(attention).max().item()}")
            
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 