"""
Text generation utilities for trained transformer models.
"""
import os
import sys
import argparse
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shakespeare_data import ShakespeareDataset
from transformer_model import TransformerLanguageModel
from config import Config


def load_model(checkpoint_dir, config):
    """Load model from checkpoint."""
    model = TransformerLanguageModel(
        config=config.model,
        multiplier_config=config.multiplier
    )
    
    # Build model by running a forward pass
    dummy_input = tf.zeros((1, config.model.sequence_length), dtype=tf.int32)
    _ = model(dummy_input, training=False)
    
    # Load checkpoint
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    
    return model


def generate_text(
    model,
    dataset,
    prompt="ROMEO:",
    max_length=500,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    num_samples=1
):
    """
    Generate text from prompt.
    
    Args:
        model: Trained transformer model
        dataset: Dataset (for encoding/decoding)
        prompt: Starting text
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        num_samples: Number of samples to generate
        
    Returns:
        List of generated texts
    """
    # Encode prompt
    encoded_prompt = dataset.encode(prompt)
    
    samples = []
    for _ in range(num_samples):
        start_tokens = tf.constant([encoded_prompt], dtype=tf.int32)
        
        # Generate
        generated = model.generate(
            start_tokens,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        # Decode
        generated_text = dataset.decode(generated[0].numpy())
        samples.append(generated_text)
    
    return samples


def main():
    parser = argparse.ArgumentParser(description='Generate text with trained model')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Checkpoint directory')
    parser.add_argument('--prompt', type=str, default='ROMEO:',
                       help='Starting prompt')
    parser.add_argument('--max_length', type=int, default=500,
                       help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=40,
                       help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Nucleus sampling')
    parser.add_argument('--num_samples', type=int, default=3,
                       help='Number of samples to generate')
    
    args = parser.parse_args()
    
    # Load config
    import json
    config_path = os.path.join(args.checkpoint_dir.replace('checkpoints', 'logs'), 'config.json')
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Reconstruct config
    from config import ModelConfig, TrainingConfig, DataConfig, MultiplierConfig
    config = Config(
        model=ModelConfig(**config_dict['model']),
        training=TrainingConfig(**config_dict['training']),
        data=DataConfig(**config_dict['data']),
        multiplier=MultiplierConfig(**config_dict['multiplier'])
    )
    
    # Load dataset
    dataset = ShakespeareDataset()
    dataset.download_and_load()
    dataset.build_vocabulary(dataset.text)
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint_dir, config)
    print("Model loaded!")
    
    # Generate samples
    print(f"\n{'='*80}")
    print(f"Generating {args.num_samples} samples with prompt: '{args.prompt}'")
    print(f"{'='*80}\n")
    
    samples = generate_text(
        model,
        dataset,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_samples=args.num_samples
    )
    
    for i, sample in enumerate(samples, 1):
        print(f"Sample {i}:")
        print(f"{'-'*80}")
        print(sample)
        print(f"{'-'*80}\n")


if __name__ == "__main__":
    main()