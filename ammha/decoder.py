from tensorflow.keras.layers import Layer, Dropout
from ammha.multihead_attention import MultiHeadAttention
from ammha.positional_encoding import PositionEmbeddingFixedWeights
from ammha.encoder import AddNormalization, FeedForward
import os
# No longer using global 'mul' as it is captured at import time.
# Implementing the Decoder Layer
class DecoderLayer(Layer):
    def __init__(self, lut_file, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
        super().__init__(**kwargs)
        self.multihead_attention1 = MultiHeadAttention(lut_file, h, d_k, d_v, d_model)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalization()
        self.multihead_attention2 = MultiHeadAttention(lut_file, h, d_k, d_v, d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalization()
        self.feed_forward = FeedForward(lut_file, d_ff, d_model)
        self.dropout3 = Dropout(rate)
        self.add_norm3 = AddNormalization()

    def call(self, x, encoder_output, lookahead_mask, padding_mask, training):
        # Multi-head self-attention layer
        multihead_output1 = self.multihead_attention1(x, keys=x, values=x, mask=lookahead_mask)
        multihead_output1 = self.dropout1(multihead_output1, training=training)
        
        # Followed by an Add & Norm layer
        x = self.add_norm1(x, sublayer_x=multihead_output1)

        # Cross-attention layer (only if encoder output is provided)
        if encoder_output is not None:
            multihead_output2 = self.multihead_attention2(x, keys=encoder_output,
                                                          values=encoder_output, mask=padding_mask)
            multihead_output2 = self.dropout2(multihead_output2, training=training)
            
            # Followed by another Add & Norm layer
            x = self.add_norm2(x, sublayer_x=multihead_output2)

        # Followed by a fully connected layer
        feedforward_output = self.feed_forward(x)
        feedforward_output = self.dropout3(feedforward_output, training=training)

        # Followed by another Add & Norm layer
        return self.add_norm3(x, sublayer_x=feedforward_output)

# Implementing the Decoder
class Decoder(Layer):
    def __init__(self, lut_file, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate,
                       **kwargs):
        super().__init__(**kwargs)
        self.pos_encoding = PositionEmbeddingFixedWeights(vocab_size,
                                                          d_model)
        self.dropout = Dropout(rate)
        self.decoder_layer = [DecoderLayer(lut_file, h, d_k, d_v, d_model, d_ff, rate)
                              for _ in range(n)]

    def call(self, output_target, encoder_output, lookahead_mask, padding_mask, training):
        # Generate the positional encoding
        pos_encoding_output = self.pos_encoding(output_target)
        # Expected output shape = (number of sentences, sequence_length, d_model)

        # Add in a dropout layer
        x = self.dropout(pos_encoding_output, training=training)

        # Pass on the positional encoded values to each encoder layer
        for i, layer in enumerate(self.decoder_layer):
            x = layer(x, encoder_output=encoder_output, lookahead_mask=lookahead_mask,
                      padding_mask=padding_mask, training=training)

        return x
