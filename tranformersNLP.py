import tensorflow as tf
import keras

from keras._tf_keras.keras.models import Model as M, Sequential as sq
from keras._tf_keras.keras import ops
import keras._tf_keras.keras.layers as L

class TransformerBlock(L.Layer):
    
    def __init__(self, embedding_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        
        self.attention = L.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim
            )
        
        self.ffn = sq(
            [
                L.Dense(ff_dim, activation=tf.nn.relu),
                L.Dense(embedding_dim),
            ]
        )
        
        self.layer_normalization1 = L.LayerNormalization(epsilon=1e-6)
        self.layer_normalization2 = L.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = L.Dropout(rate=dropout_rate)
        self.dropout2 = L.Dropout(rate=dropout_rate)
        
    def call(self, inputs):
        attention_output = self.attention(inputs, inputs)
        attention_output = self.dropout1(attention_output)
        out1 = self.layer_normalization1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layer_normalization2(out1 + ffn_output)
    
    
class TokenAndPositioningEmbedding(L.Layer):
    
    def __init__(self, maxlen, vocab_size, embedding_dim):
        super().__init__()
            
        self.token_embedding = L.Embedding(
            input_dim=vocab_size, 
            output_dim=embedding_dim
            )
        self.positional_embedding = L.Embedding(
            input_dim=maxlen, 
            output_dim=embedding_dim
            )

    def call(self, x):
        maxlen = ops.shape(x)[-1]
        positions = ops.arange(start=0, stop=maxlen, step=1)
        positions = self.positional_embedding(positions)
        x = self.token_embedding(x)
        return x + positions