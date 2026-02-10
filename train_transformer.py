"""
Main training script for transformer language model.
Supports both native FP32 and approximate multipliers.
"""
import os
import sys
import argparse
import time
import json
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Mean

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, ModelConfig, TrainingConfig, DataConfig, MultiplierConfig
from shakespeare_data import ShakespeareDataset
from transformer_model import TransformerLanguageModel


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Custom learning rate schedule with warmup."""
    
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class Trainer:
    """Trainer for transformer language model."""
    
    def __init__(self, config: Config, experiment_name: str):
        """
        Initialize trainer.
        
        Args:
            config: Complete configuration
            experiment_name: Name for this experiment (for logging/checkpointing)
        """
        self.config = config
        self.experiment_name = experiment_name
        
        # Create directories
        self.checkpoint_dir = os.path.join(config.training.checkpoint_dir, experiment_name)
        self.log_dir = os.path.join(config.training.log_dir, experiment_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Save config
        self._save_config()
        
        # Initialize data
        print("Loading data...")
        self.dataset = ShakespeareDataset(
            data_dir=config.data.data_dir,
            sequence_length=config.data.sequence_length
        )
        train_data, val_data = self.dataset.prepare_data(config.data.train_split)
        
        self.train_dataset = self.dataset.create_tf_dataset(
            train_data,
            batch_size=config.training.batch_size,
            shuffle=True,
            buffer_size=config.data.buffer_size,
            prefetch_size=config.data.prefetch_size
        )
        
        self.val_dataset = self.dataset.create_tf_dataset(
            val_data,
            batch_size=config.training.batch_size,
            shuffle=False,
            prefetch_size=config.data.prefetch_size
        )
        
        # Update vocab size in config
        self.config.model.vocab_size = self.dataset.vocab_size
        
        # Initialize model
        print("Creating model...")
        self.model = TransformerLanguageModel(
            config=config.model,
            multiplier_config=config.multiplier
        )
        
        # Build model to initialize weights (required for count_params)
        dummy_input = tf.zeros((1, config.data.sequence_length), dtype=tf.int32)
        _ = self.model(dummy_input, training=False)
        
        # Initialize optimizer
        learning_rate = CustomSchedule(config.model.d_model, config.training.warmup_steps)
        self.optimizer = Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        
        # Initialize loss and metrics
        self.loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.train_loss = Mean(name='train_loss')
        self.val_loss = Mean(name='val_loss')
        
        # Checkpoint manager
        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint,
            self.checkpoint_dir,
            max_to_keep=5
        )
        
        # TensorBoard writer
        self.train_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, 'train'))
        self.val_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, 'val'))
        
        # Metrics tracking
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_perplexity': [],
            'val_perplexity': [],
            'epoch_times': [],
            'tokens_per_sec': []
        }
        
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def _save_config(self):
        """Save configuration to JSON."""
        config_dict = {
            'model': self.config.model.__dict__,
            'training': self.config.training.__dict__,
            'data': self.config.data.__dict__,
            'multiplier': self.config.multiplier.__dict__
        }
        
        config_path = os.path.join(self.log_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @tf.function
    def train_step(self, inputs, targets):
        """Single training step."""
        with tf.GradientTape() as tape:
            logits = self.model(inputs, training=True)
            loss = self.loss_fn(targets, logits)
            loss = tf.reduce_mean(loss)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Clip gradients
        gradients, _ = tf.clip_by_global_norm(
            gradients,
            self.config.training.gradient_clip_norm
        )
        
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss
    
    @tf.function
    def val_step(self, inputs, targets):
        """Single validation step."""
        logits = self.model(inputs, training=False)
        loss = self.loss_fn(targets, logits)
        loss = tf.reduce_mean(loss)
        return loss
    
    def generate_sample(self, prompt="ROMEO:", max_length=200, temperature=0.8):
        """Generate text sample."""
        # Encode prompt
        encoded_prompt = self.dataset.encode(prompt)
        start_tokens = tf.constant([encoded_prompt], dtype=tf.int32)
        
        # Generate
        generated = self.model.generate(
            start_tokens,
            max_length=max_length,
            temperature=temperature,
            top_k=40
        )
        
        # Decode
        generated_text = self.dataset.decode(generated[0].numpy())
        return generated_text
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.train_loss.reset_state()
        
        epoch_start = time.time()
        num_batches = 0
        total_tokens = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_dataset):
            if self.config.training.max_steps_per_epoch is not None and batch_idx >= self.config.training.max_steps_per_epoch:
                break
                
            loss = self.train_step(inputs, targets)
            self.train_loss.update_state(loss)
            
            num_batches += 1
            total_tokens += inputs.shape[0] * inputs.shape[1]
            self.global_step += 1
            
            # Log to TensorBoard
            if self.global_step % self.config.training.log_freq == 0:
                with self.train_writer.as_default():
                    tf.summary.scalar('loss', loss, step=self.global_step)
                    tf.summary.scalar('learning_rate', 
                                    self.optimizer.learning_rate,
                                    step=self.global_step)
            
            # Generate samples
            if self.global_step % self.config.training.sample_freq == 0:
                sample = self.generate_sample()
                print(f"\n{'='*80}")
                print(f"Sample at step {self.global_step}:")
                print(f"{'='*80}")
                print(sample)
                print(f"{'='*80}\n")
        
        epoch_time = time.time() - epoch_start
        tokens_per_sec = total_tokens / epoch_time
        
        return epoch_time, tokens_per_sec
    
    def validate(self):
        """Run validation."""
        self.val_loss.reset_state()
        
        for inputs, targets in self.val_dataset:
            loss = self.val_step(inputs, targets)
            self.val_loss.update_state(loss)
        
        return self.val_loss.result().numpy()
    
    def train(self):
        """Main training loop."""
        print(f"\nStarting training: {self.experiment_name}")
        print(f"Multiplier: {self.config.multiplier.multiplier_name}")
        print(f"Model parameters: {self.model.count_params():,}")
        print(f"{'='*80}\n")
        
        for epoch in range(self.config.training.epochs):
            print(f"Epoch {epoch + 1}/{self.config.training.epochs}")
            
            # Train
            epoch_time, tokens_per_sec = self.train_epoch(epoch)
            train_loss = self.train_loss.result().numpy()
            train_perplexity = np.exp(train_loss)
            
            # Validate
            if (epoch + 1) % self.config.training.val_freq == 0:
                val_loss = self.validate()
                val_perplexity = np.exp(val_loss)
                
                # Log to TensorBoard
                with self.val_writer.as_default():
                    tf.summary.scalar('loss', val_loss, step=epoch)
                    tf.summary.scalar('perplexity', val_perplexity, step=epoch)
                
                # Save metrics
                self.metrics['val_loss'].append(float(val_loss))
                self.metrics['val_perplexity'].append(float(val_perplexity))
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.checkpoint_manager.save()
                    print(f"  ✓ Saved checkpoint (best val_loss: {val_loss:.4f})")
            
            # Save metrics
            self.metrics['train_loss'].append(float(train_loss))
            self.metrics['train_perplexity'].append(float(train_perplexity))
            self.metrics['epoch_times'].append(float(epoch_time))
            self.metrics['tokens_per_sec'].append(float(tokens_per_sec))
            
            # Print epoch summary
            print(f"  Train Loss: {train_loss:.4f} | Perplexity: {train_perplexity:.2f}")
            if (epoch + 1) % self.config.training.val_freq == 0:
                print(f"  Val Loss: {val_loss:.4f} | Perplexity: {val_perplexity:.2f}")
            print(f"  Time: {epoch_time:.2f}s | Tokens/sec: {tokens_per_sec:.0f}")
            print()
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.training.save_freq == 0:
                self.checkpoint_manager.save()
        
        # Save final metrics
        self._save_metrics()
        
        print(f"\n{'='*80}")
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*80}\n")
    
    def _save_metrics(self):
        """Save metrics to JSON."""
        metrics_path = os.path.join(self.log_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Saved metrics to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description='Train transformer language model')
    
    # Multiplier configuration
    parser.add_argument('--multiplier', type=str, default='fp32',
                       choices=['fp32', 'mbm_7', 'mbm_5', 'mbm_3', 'mbm_1',
                               'mit_7', 'mit_5', 'mit_3', 'mit_1'],
                       help='Multiplier type to use')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (default: auto-generated)')
    
    # Model configuration
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    
    # Test mode
    parser.add_argument('--test_mode', action='store_true',
                       help='Run in test mode (1 epoch, small batch)')
    
    args = parser.parse_args()
    
    # Create configuration
    if args.multiplier == 'fp32':
        config = Config.create_fp32_config()
    else:
        # Parse multiplier name
        mul_type, bits = args.multiplier.split('_')
        lut_file = f"lut/{mul_type.upper()}_{bits}.bin"
        config = Config.create_approximate_config(lut_file, args.multiplier.upper())
    
    # Override with command line arguments
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.model.d_model = args.d_model
    config.model.num_layers = args.num_layers
    config.model.num_heads = args.num_heads
    
    # Test mode adjustments
    if args.test_mode:
        config.training.epochs = 1
        config.training.batch_size = 4
        config.training.log_freq = 10
        config.training.sample_freq = 50
        config.training.max_steps_per_epoch = 1000
    
    # Generate experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"{args.multiplier}_{timestamp}"
    
    # Create trainer and train
    trainer = Trainer(config, args.experiment_name)
    trainer.train()


if __name__ == "__main__":
    main()
