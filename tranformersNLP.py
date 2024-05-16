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
        
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

    def call(self, x):
        maxlen = ops.shape(x)[-1]
        positions = ops.arange(start=0, stop=maxlen, step=1)
        positions = self.positional_embedding(positions)
        x = self.token_embedding(x)
        return x + positions
    

class TransformerEncoder(L.Layer):
    
    def __init__(self, embedding_dim, dense_dim, num_heads, name='Transformer_Encoder', **kwargs):
        super().__init__(name=name, **kwargs)
        self.embedding_dim = embedding_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = L.MultiHeadAttention(
            num_heads=num_heads, key_dim=embedding_dim
        )
        self.dense_proj = sq(
            [
                L.Dense(dense_dim, activation=tf.nn.relu),
                L.Dense(embedding_dim)
                ]
            )
        
        self.layer_normalization1 = L.LayerNormalization()
        self.layer_normalization2 = L.LayerNormalization()
        self.supports_masking = True
        
    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = ops.cast(mask[:, None, :], dtype="int32")
        else:
            padding_mask = None
        
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        proj_input = self.layer_normalization1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layer_normalization2(proj_input + proj_output)
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "dense_dim": self.dense_dim,
                "num_heads": self.num_heads,
            }
        )
        return config
    

class TransformerDecoder(L.Layer):
    
    def __init__(self, embedding_dim, latent_dim, num_heads, name='Transformer_Decoder', **kwargs):
        super().__init__(name=name, **kwargs)
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        
        self.attention_1 = L.MultiHeadAttention(
            num_heads=num_heads, key_dim=embedding_dim
        )
        
        self.attention_2 = L.MultiHeadAttention(
            num_heads=num_heads, key_dim=embedding_dim
        )
        
        self.dense_proj = sq(
            [
                L.Dense(latent_dim, activation=tf.nn.relu),
                L.Dense(embedding_dim)
            ]
        )
        
        self.layer_normalization1 = L.LayerNormalization()
        self.layer_normalization2 = L.LayerNormalization()
        self.layer_normalization3 = L.LayerNormalization()
        self.support_masking = True
        
    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = ops.cast(mask[:, None, :], dtype="int32")
            padding_mask = ops.minimum(padding_mask, causal_mask)
        else:
            padding_mask=None
                
        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        
        out_1 = self.layer_normalization1(inputs + attention_output_1)
        
        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask
        )
        
        out_2 = self.layer_normalization2(out_1 + attention_output_2)
        
        proj_output = self.dense_proj(out_2)
        return self.layer_normalization3(out_2 + proj_output)
    
    def get_causal_attention_mask(self, inputs):
        input_shape = ops.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = ops.arange(sequence_length)[:, None]
        j = ops.arange(sequence_length)
        mask = ops.cast(i >= j, dtype="int32")
        mask = ops.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = ops.concatenate(
            [ops.expand_dims(batch_size, -1), ops.convert_to_tensor([1, 1])],
            axis=0,
        )
        return ops.tile(mask, mult)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "latent_dim": self.latent_dim,
                "num_heads": self.num_heads,
            }
        )
        
        return config


