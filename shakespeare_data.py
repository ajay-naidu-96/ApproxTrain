"""
Shakespeare dataset loading and preprocessing for character-level language modeling.
"""
import os
import numpy as np
import tensorflow as tf
from typing import Tuple, Dict


class ShakespeareDataset:
    """Shakespeare text dataset for character-level language modeling."""
    
    def __init__(self, data_dir: str = "data", sequence_length: int = 128):
        """
        Initialize Shakespeare dataset.
        
        Args:
            data_dir: Directory to store/load data
            sequence_length: Length of input sequences
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.text = None
        self.vocab = None
        self.char_to_idx = None
        self.idx_to_char = None
        self.vocab_size = None
        
        os.makedirs(data_dir, exist_ok=True)
        
    def download_and_load(self) -> str:
        """Download and load Shakespeare text."""
        filepath = os.path.join(self.data_dir, "shakespeare.txt")
        
        if not os.path.exists(filepath):
            print("Downloading Shakespeare dataset...")
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            
            try:
                import urllib.request
                urllib.request.urlretrieve(url, filepath)
                print(f"Downloaded to {filepath}")
            except Exception as e:
                print(f"Error downloading: {e}")
                print("Please manually download from:")
                print(url)
                raise
        
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"Loaded {len(text)} characters")
        return text
    
    def build_vocabulary(self, text: str):
        """Build character vocabulary."""
        self.vocab = sorted(set(text))
        self.vocab_size = len(self.vocab)
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.vocab)}
        self.idx_to_char = {idx: ch for idx, ch in enumerate(self.vocab)}
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Vocabulary: {''.join(self.vocab[:20])}...")
        
    def encode(self, text: str) -> np.ndarray:
        """Encode text to integer indices."""
        return np.array([self.char_to_idx[ch] for ch in text])
    
    def decode(self, indices: np.ndarray) -> str:
        """Decode integer indices to text."""
        return ''.join([self.idx_to_char[idx] for idx in indices])
    
    def create_sequences(self, encoded_text: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input-target sequence pairs.
        
        Args:
            encoded_text: Encoded text as integer array
            
        Returns:
            inputs: Input sequences of shape (num_sequences, sequence_length)
            targets: Target sequences of shape (num_sequences, sequence_length)
        """
        # Create sequences with sliding window
        sequences = []
        for i in range(0, len(encoded_text) - self.sequence_length):
            seq = encoded_text[i:i + self.sequence_length + 1]
            sequences.append(seq)
        
        sequences = np.array(sequences)
        
        # Split into inputs and targets
        inputs = sequences[:, :-1]
        targets = sequences[:, 1:]
        
        return inputs, targets
    
    def prepare_data(self, train_split: float = 0.9) -> Tuple[Dict, Dict]:
        """
        Prepare complete dataset.
        
        Args:
            train_split: Fraction of data for training
            
        Returns:
            train_data: Dictionary with 'inputs' and 'targets'
            val_data: Dictionary with 'inputs' and 'targets'
        """
        # Load and encode text
        self.text = self.download_and_load()
        self.build_vocabulary(self.text)
        encoded = self.encode(self.text)
        
        # Create sequences
        inputs, targets = self.create_sequences(encoded)
        
        # Split into train/val
        split_idx = int(len(inputs) * train_split)
        
        train_data = {
            'inputs': inputs[:split_idx],
            'targets': targets[:split_idx]
        }
        
        val_data = {
            'inputs': inputs[split_idx:],
            'targets': targets[split_idx:]
        }
        
        print(f"Training sequences: {len(train_data['inputs'])}")
        print(f"Validation sequences: {len(val_data['inputs'])}")
        
        return train_data, val_data
    
    def create_tf_dataset(
        self,
        data: Dict,
        batch_size: int,
        shuffle: bool = True,
        buffer_size: int = 10000,
        prefetch_size: int = 2
    ) -> tf.data.Dataset:
        """
        Create TensorFlow dataset.
        
        Args:
            data: Dictionary with 'inputs' and 'targets'
            batch_size: Batch size
            shuffle: Whether to shuffle data
            buffer_size: Buffer size for shuffling
            prefetch_size: Number of batches to prefetch
            
        Returns:
            TensorFlow dataset
        """
        dataset = tf.data.Dataset.from_tensor_slices((data['inputs'], data['targets']))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size)
        
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(prefetch_size)
        
        return dataset


def test_dataset():
    """Test dataset loading and preprocessing."""
    print("Testing Shakespeare dataset...")
    
    dataset = ShakespeareDataset(sequence_length=128)
    train_data, val_data = dataset.prepare_data(train_split=0.9)
    
    # Create TF datasets
    train_dataset = dataset.create_tf_dataset(train_data, batch_size=32)
    val_dataset = dataset.create_tf_dataset(val_data, batch_size=32, shuffle=False)
    
    # Test one batch
    for inputs, targets in train_dataset.take(1):
        print(f"\nBatch shapes:")
        print(f"  Inputs: {inputs.shape}")
        print(f"  Targets: {targets.shape}")
        
        # Decode first sequence
        print(f"\nFirst sequence (input):")
        print(dataset.decode(inputs[0].numpy()))
        print(f"\nFirst sequence (target):")
        print(dataset.decode(targets[0].numpy()))
    
    print("\n✓ Dataset test passed!")
    return dataset


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run test")
    args = parser.parse_args()
    
    if args.test:
        test_dataset()
