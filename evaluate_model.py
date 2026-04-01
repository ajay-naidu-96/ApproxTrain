"""
Evaluation script for trained transformer models.
"""
import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shakespeare_data import ShakespeareDataset
from transformer_model import TransformerLanguageModel
from config import Config, ModelConfig, TrainingConfig, DataConfig, MultiplierConfig
from generate_text import load_model, generate_text


def evaluate_perplexity(model, dataset, batch_size=64):
    """
    Calculate perplexity on validation set.
    
    Args:
        model: Trained model
        dataset: TensorFlow dataset
        batch_size: Batch size for evaluation
        
    Returns:
        Average loss and perplexity
    """
    loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    total_loss = 0.0
    num_batches = 0
    
    for inputs, targets in dataset:
        logits = model(inputs, training=False)
        loss = loss_fn(targets, logits)
        total_loss += tf.reduce_mean(loss).numpy()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    perplexity = np.exp(avg_loss)
    
    return avg_loss, perplexity


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Checkpoint directory')
    parser.add_argument('--generate_samples', action='store_true',
                       help='Generate text samples')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to generate')
    
    args = parser.parse_args()
    
    # Load config
    config_path = os.path.join(args.checkpoint_dir.replace('checkpoints', 'logs'), 'config.json')
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = Config(
        model=ModelConfig(**config_dict['model']),
        training=TrainingConfig(**config_dict['training']),
        data=DataConfig(**config_dict['data']),
        multiplier=MultiplierConfig(**config_dict['multiplier'])
    )
    
    # Load dataset
    print("Loading dataset...")
    dataset_loader = ShakespeareDataset(
        data_dir=config.data.data_dir,
        sequence_length=config.data.sequence_length
    )
    train_data, val_data = dataset_loader.prepare_data(config.data.train_split)
    
    val_dataset = dataset_loader.create_tf_dataset(
        val_data,
        batch_size=config.training.batch_size,
        shuffle=False
    )
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint_dir, config)
    print(f"Model loaded! Parameters: {model.count_params():,}")
    
    # Evaluate perplexity
    print("\nEvaluating perplexity...")
    val_loss, val_perplexity = evaluate_perplexity(model, val_dataset)
    
    print(f"\n{'='*80}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Perplexity: {val_perplexity:.2f}")
    print(f"{'='*80}\n")
    
    # Generate samples
    if args.generate_samples:
        prompts = [
            "ROMEO:",
            "JULIET:",
            "First Citizen:",
            "KING HENRY:",
            "The "
        ]
        
        print("Generating samples...\n")
        
        for prompt in prompts[:args.num_samples]:
            samples = generate_text(
                model,
                dataset_loader,
                prompt=prompt,
                max_length=200,
                temperature=0.8,
                top_k=40,
                num_samples=1
            )
            
            print(f"Prompt: '{prompt}'")
            print(f"{'-'*80}")
            print(samples[0])
            print(f"{'-'*80}\n")
    
    # Save evaluation results
    results = {
        'val_loss': float(val_loss),
        'val_perplexity': float(val_perplexity),
        'model_params': int(model.count_params()),
        'multiplier': config.multiplier.multiplier_name
    }
    
    results_path = os.path.join(args.checkpoint_dir.replace('checkpoints', 'logs'), 
                                'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
