"""
Transformer model for character-level language modeling.
Supports both native FP32 and approximate multipliers.
"""
import os
import sys
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, Dropout

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ammha.decoder import Decoder
from config import ModelConfig, MultiplierConfig


class TransformerLanguageModel(Model):
    """
    Decoder-only transformer for autoregressive language modeling.
    """
    
    def __init__(self, config: ModelConfig, multiplier_config: MultiplierConfig, **kwargs):
        """
        Initialize transformer language model.
        
        Args:
            config: Model configuration
            multiplier_config: Multiplier configuration (FP32 or approximate)
        """
        super().__init__(**kwargs)
        
        self.config = config
        self.multiplier_config = multiplier_config
        
        # Set environment variable for multiplier type
        if multiplier_config.use_approximate:
            os.environ['mul'] = 'APPROXIMATE'
        else:
            os.environ['mul'] = 'FP32'
        
        # Token embedding
        self.embedding = Embedding(
            input_dim=config.vocab_size,
            output_dim=config.d_model,
            name="token_embedding"
        )
        
        # Decoder stack
        lut_file = multiplier_config.lut_file if multiplier_config.use_approximate else None
        self.decoder = Decoder(
            lut_file=lut_file,
            vocab_size=config.vocab_size,
            sequence_length=config.sequence_length,
            h=config.num_heads,
            d_k=config.d_k,
            d_v=config.d_v,
            d_model=config.d_model,
            d_ff=config.d_ff,
            n=config.num_layers,
            rate=config.dropout_rate,
            name="decoder"
        )
        
        # Output projection to vocabulary
        if multiplier_config.use_approximate:
            from python.keras.layers.amdenselayer import denseam
            self.output_layer = denseam(
                config.vocab_size,
                mant_mul_lut=lut_file,
                name="output_projection"
            )
        else:
            self.output_layer = Dense(
                config.vocab_size,
                name="output_projection"
            )
        
        self.dropout = Dropout(config.dropout_rate)
    
    def create_lookahead_mask(self, size):
        """
        Create causal mask for autoregressive generation.
        
        Args:
            size: Sequence length
            
        Returns:
            Mask of shape (size, size)
        """
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (size, size)
    
    def call(self, inputs, training=False):
        """
        Forward pass.
        
        Args:
            inputs: Input token indices of shape (batch_size, sequence_length)
            training: Whether in training mode
            
        Returns:
            Logits of shape (batch_size, sequence_length, vocab_size)
        """
        seq_len = tf.shape(inputs)[1]
        
        # Create causal mask
        lookahead_mask = self.create_lookahead_mask(seq_len)
        
        # Embed tokens (decoder will add positional encoding)
        # Note: The Decoder class has its own positional encoding
        # so we pass the token indices directly
        
        # Pass through decoder
        decoder_output = self.decoder(
            output_target=inputs,
            encoder_output=None,  # No encoder for language modeling
            lookahead_mask=lookahead_mask,
            padding_mask=None,  # No padding mask for now
            training=training
        )
        
        # Project to vocabulary
        logits = self.output_layer(decoder_output)
        
        return logits
    
    def generate(
        self,
        start_tokens,
        max_length=100,
        temperature=1.0,
        top_k=None,
        top_p=None
    ):
        """
        Generate text autoregressively.
        
        Args:
            start_tokens: Starting token indices of shape (batch_size, start_len)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated token indices of shape (batch_size, max_length)
        """
        batch_size = tf.shape(start_tokens)[0]
        generated = start_tokens
        
        for _ in range(max_length - tf.shape(start_tokens)[1]):
            # Get predictions for current sequence
            logits = self(generated, training=False)
            
            # Get logits for last position
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                top_k_logits, top_k_indices = tf.nn.top_k(next_token_logits, k=top_k)
                next_token_logits = tf.where(
                    tf.reduce_any(tf.equal(tf.expand_dims(tf.range(self.config.vocab_size), 0), 
                                          tf.expand_dims(top_k_indices, -1)), axis=1),
                    next_token_logits,
                    -1e10 * tf.ones_like(next_token_logits)
                )
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits = tf.sort(next_token_logits, direction='DESCENDING')
                sorted_probs = tf.nn.softmax(sorted_logits)
                cumulative_probs = tf.cumsum(sorted_probs, axis=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift to keep first token above threshold
                sorted_indices_to_remove = tf.concat([
                    tf.zeros_like(sorted_indices_to_remove[:, :1]),
                    sorted_indices_to_remove[:, :-1]
                ], axis=-1)
                
                indices_to_remove = tf.argsort(tf.argsort(next_token_logits, direction='DESCENDING'))
                indices_to_remove = tf.gather(sorted_indices_to_remove, indices_to_remove, batch_dims=1)
                next_token_logits = tf.where(indices_to_remove, -1e10, next_token_logits)
            
            # Sample next token
            next_token = tf.random.categorical(next_token_logits, num_samples=1, dtype=tf.int32)
            
            # Append to generated sequence
            generated = tf.concat([generated, next_token], axis=-1)
        
        return generated


def test_model():
    """Test model creation and forward pass."""
    print("Testing Transformer Language Model...")
    
    from config import Config
    
    # Create FP32 config
    config = Config.create_fp32_config()
    
    # Create model
    model = TransformerLanguageModel(
        config=config.model,
        multiplier_config=config.multiplier
    )
    
    # Test forward pass
    batch_size = 4
    seq_len = 32
    dummy_input = tf.random.uniform(
        (batch_size, seq_len),
        minval=0,
        maxval=config.model.vocab_size,
        dtype=tf.int32
    )
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Forward pass
    logits = model(dummy_input, training=True)
    print(f"Output shape: {logits.shape}")
    
    # Test generation
    start_tokens = tf.constant([[1, 2, 3]], dtype=tf.int32)
    generated = model.generate(start_tokens, max_length=20, temperature=1.0)
    print(f"Generated shape: {generated.shape}")
    print(f"Generated tokens: {generated.numpy()}")
    
    print("\n✓ Model test passed!")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run test")
    args = parser.parse_args()
    
    if args.test:
        test_model()
