import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

# ----------------------------------------------------------------------
# 1. Helper: Causal Dilated Convolution Layer
# ----------------------------------------------------------------------
def causal_dilated_conv1d(x, filters, kernel_size, dilation_rate, activation='gelu'):
    """
    Applies a causal dilated 1D convolution.
    - Causal padding ensures no information from the future is used.
    - Dilation increases the receptive field exponentially.
    """
    # 'causal' padding in Conv1D automatically pads the left side of the sequence
    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding='causal',
        activation=activation,
        use_bias=True
    )(x)
    return x

# ----------------------------------------------------------------------
# 2. TCN Block
# ----------------------------------------------------------------------
class TCNBlock(layers.Layer):
    """
    A stack of dilated causal convolutions that act as a local feature extractor.
    The receptive field grows exponentially with depth.

    Args:
        filters: number of filters in each convolutional layer
        kernel_size: size of the convolution kernel
        dilations: list of dilation rates for successive layers (e.g. [1,2,4,8])
        activation: activation function (default 'gelu')
    """
    def __init__(self, filters, kernel_size, dilations, activation='gelu', **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.activation = activation
        self.conv_layers = []

        for dilation in dilations:
            self.conv_layers.append(
                layers.Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    dilation_rate=dilation,
                    padding='causal',
                    activation=activation,
                    use_bias=True
                )
            )

    def call(self, inputs):
        x = inputs
        for conv in self.conv_layers:
            x = conv(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilations': self.dilations,
            'activation': self.activation,
        })
        return config

# ----------------------------------------------------------------------
# 3. Linformer Attention (Factorized Self-Attention)
# ----------------------------------------------------------------------
class LinformerAttention(layers.Layer):
    """
    Implements the factorized self-attention mechanism from Linformer [3].
    Projects the keys and values to a lower dimension (k) to achieve O(L·k)
    complexity instead of O(L²).

    Args:
        dim: model dimension (D) – input feature depth
        proj_dim: projection dimension (k), with k << L
        num_heads: number of attention heads (default 1 for simplicity)
    """
    def __init__(self, dim, proj_dim, num_heads=1, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        # Ensure dim is divisible by num_heads if multi-head is used
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        # Linear projections for Q, K, V (without bias to save parameters)
        self.query_dense = layers.Dense(dim, use_bias=False)
        self.key_dense = layers.Dense(dim, use_bias=False)
        self.value_dense = layers.Dense(dim, use_bias=False)

        # Projection matrices E and F for keys and values (Linformer)
        self.E = layers.Dense(proj_dim, use_bias=False)   # projects K: (L, dim) -> (k, dim)
        self.F = layers.Dense(proj_dim, use_bias=False)   # projects V: (L, dim) -> (k, dim)

        # Output projection
        self.out_dense = layers.Dense(dim, use_bias=False)

    def call(self, inputs):
        # inputs shape: (batch, L, dim)
        batch_size, seq_len, _ = tf.shape(inputs)[0], tf.shape(inputs)[1], self.dim

        # 1. Generate Q, K, V
        Q = self.query_dense(inputs)      # (batch, L, dim)
        K = self.key_dense(inputs)        # (batch, L, dim)
        V = self.value_dense(inputs)      # (batch, L, dim)

        # 2. Project K and V to lower dimension (k) using E and F
        #    Note: We need to apply the same projection across the batch,
        #    so we treat the projection as a Dense layer applied to the last dimension.
        #    However, Linformer projects the *sequence* dimension, not the feature dim.
        #    We achieve this by swapping axes: (batch, L, dim) -> (batch, dim, L)
        #    then apply a Dense layer with proj_dim units along the L dimension,
        #    then swap back.
        #    Alternative: use a Conv1D with kernel_size=1 and proj_dim filters.
        #    Here we use a time-distributed Dense that projects the feature dim,
        #    which is not correct. We need to project the sequence length.
        #    Correct implementation: Use a Dense layer with input shape (L,) but that's
        #    not typical. Instead, we can use a Conv1D with filters=proj_dim and
        #    kernel_size=1 along the time dimension? No, that would change feature dim.
        #    The standard Linformer uses a linear projection matrix of shape (k, L)
        #    applied to the transposed tensor. We'll implement it as a Dense layer
        #    on the transposed tensor.

        # Transpose to (batch, dim, L)
        K_T = tf.transpose(K, perm=[0, 2, 1])   # (batch, dim, L)
        V_T = tf.transpose(V, perm=[0, 2, 1])   # (batch, dim, L)

        # Apply projection E (units = proj_dim) along the L dimension
        # Dense layer applied to the last dimension (L) will output (batch, dim, proj_dim)
        K_proj = self.E(K_T)                    # (batch, dim, proj_dim)
        V_proj = self.F(V_T)                    # (batch, dim, proj_dim)

        # Transpose back to (batch, proj_dim, dim) for attention computation
        K_proj = tf.transpose(K_proj, perm=[0, 2, 1])   # (batch, proj_dim, dim)
        V_proj = tf.transpose(V_proj, perm=[0, 2, 1])   # (batch, proj_dim, dim)

        # At this point:
        # Q: (batch, L, dim)
        # K_proj: (batch, proj_dim, dim)
        # V_proj: (batch, proj_dim, dim)

        # 3. Compute scaled dot-product attention with projected K/V
        #    Scores = Q * K_proj^T  -> (batch, L, proj_dim)
        scores = tf.matmul(Q, K_proj, transpose_b=True) / tf.sqrt(float(self.dim))
        attn_weights = tf.nn.softmax(scores, axis=-1)   # (batch, L, proj_dim)

        # 4. Apply attention to projected V
        #    Output = attn_weights * V_proj  -> (batch, L, dim)
        out = tf.matmul(attn_weights, V_proj)            # (batch, L, dim)

        # 5. Final linear projection
        out = self.out_dense(out)                         # (batch, L, dim)

        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'proj_dim': self.proj_dim,
            'num_heads': self.num_heads,
        })
        return config

# ----------------------------------------------------------------------
# 4. Full Hybrid Model
# ----------------------------------------------------------------------
def create_tcn_transformer_model(
    input_length=625,
    input_features=1,
    tcn_filters=32,
    tcn_kernel_size=3,
    tcn_dilations=(1, 2, 4, 8),
    tcn_activation='gelu',
    transformer_dim=64,
    transformer_proj_dim=16,
    transformer_heads=1,
    num_classes=1
):
    """
    Builds the lightweight hybrid TCN-Transformer model.

    Args:
        input_length: length of the input sequence (L)
        input_features: number of features per time step (F_in)
        tcn_filters: number of filters in each TCN layer
        tcn_kernel_size: kernel size for TCN convolutions
        tcn_dilations: tuple of dilation rates for the TCN block
        tcn_activation: activation after each convolution
        transformer_dim: model dimension (D) after TCN block
        transformer_proj_dim: projection dimension (k) for Linformer
        transformer_heads: number of attention heads (currently only 1 works with our Linformer implementation)
        num_classes: number of output classes (1 for binary anomaly detection)

    Returns:
        A Keras Model.
    """
    inputs = layers.Input(shape=(input_length, input_features))

    # ---- TCN Block ----
    # First, project input features to transformer_dim (if needed)
    x = layers.Conv1D(filters=transformer_dim, kernel_size=1, padding='same')(inputs)
    # Apply TCN block (which already uses causal padding)
    x = TCNBlock(
        filters=tcn_filters,
        kernel_size=tcn_kernel_size,
        dilations=tcn_dilations,
        activation=tcn_activation
    )(x)
    # Adjust dimension back to transformer_dim if tcn_filters != transformer_dim
    if tcn_filters != transformer_dim:
        x = layers.Conv1D(filters=transformer_dim, kernel_size=1, padding='same')(x)

    # ---- Linformer Attention Block ----
    x = LinformerAttention(
        dim=transformer_dim,
        proj_dim=transformer_proj_dim,
        num_heads=transformer_heads
    )(x)

    # ---- Classification Head ----
    x = layers.GlobalAveragePooling1D()(x)          # aggregate over time
    x = layers.Dense(32, activation='gelu')(x)      # optional small hidden layer
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs, name='TCN_Transformer')
    return model

# ----------------------------------------------------------------------
# 5. Example Usage and Model Summary
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Hyperparameters from the paper (typical values)
    MODEL_CONFIG = {
        'input_length': 625,           # 5 seconds at 125 Hz
        'input_features': 1,            # single-lead ECG
        'tcn_filters': 32,              # number of TCN filters
        'tcn_kernel_size': 3,           # kernel size
        'tcn_dilations': (1, 2, 4, 8),  # 4 layers, receptive field = 2^4 * (3-1) + 1 = 31? Actually dilated conv receptive field formula: 1 + sum_{i} (kernel_size-1)*dilation_i. Here: 1 + 2*1 + 2*2 + 2*4 + 2*8 = 1+2+4+8+16=31. That's local. Global context is handled by attention.
        'tcn_activation': 'gelu',
        'transformer_dim': 64,           # D
        'transformer_proj_dim': 16,      # k (projection dimension)
        'transformer_heads': 1,
        'num_classes': 1
    }

    model = create_tcn_transformer_model(**MODEL_CONFIG)
    model.summary()

    # Estimate model size (parameters * bytes)
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params}")
    # With INT8 quantization, size ≈ total_params bytes
    print(f"Estimated INT8 model size: {total_params} bytes (~{total_params/1024:.2f} KB)")

    # Example forward pass
    dummy_input = tf.random.normal((1, MODEL_CONFIG['input_length'], MODEL_CONFIG['input_features']))
    output = model(dummy_input)
    print(f"Output shape: {output.shape} (anomaly probability)")

    # Note: The model is ready for training with binary crossentropy.
    # After training, it can be converted to TensorFlow Lite for Microcontrollers
    # using tf.lite.TFLiteConverter with optimizations for INT8 quantization.
