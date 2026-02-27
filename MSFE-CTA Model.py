import tensorflow as tf
from tensorflow.keras import layers, models, Input, backend as K

class ResidualTCNBlock(layers.Layer):
    """
    A single Residual TCN Block as shown in Fig 3(b).
    Contains: Dilated Conv1D -> BN -> ReLU -> Dropout -> 1x1 Conv -> BN -> ReLU -> Dropout -> Add
    """
    def __init__(self, filters, kernel_size, dilation_rate, dropout_rate=0.3):
        super(ResidualTCNBlock, self).__init__()
        
        self.dilated_conv = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal', 
            use_bias=True
        )
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.dropout1 = layers.Dropout(dropout_rate)
        
        self.conv_1x1 = layers.Conv1D(filters=filters, kernel_size=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(dropout_rate)
        
        self.res_conv = layers.Conv1D(filters=filters, kernel_size=1, padding='same')

    def call(self, inputs, training=False):
        x = self.dilated_conv(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.dropout1(x, training=training)
        
        x = self.conv_1x1(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)
        x = self.dropout2(x, training=training)
        
        res = self.res_conv(inputs)
        
        return layers.Add()([x, res])

class InceptionTCNModule(layers.Layer):
    def __init__(self, filters):
        super(InceptionTCNModule, self).__init__()
        
        self.bottleneck = layers.Conv1D(filters, kernel_size=1, padding='same', activation='relu')
        
        self.branch_d1 = ResidualTCNBlock(filters, kernel_size=3, dilation_rate=1)
        self.branch_d2 = ResidualTCNBlock(filters, kernel_size=3, dilation_rate=2)
        self.branch_d4 = ResidualTCNBlock(filters, kernel_size=3, dilation_rate=4)
        self.branch_d8 = ResidualTCNBlock(filters, kernel_size=3, dilation_rate=8)
        
        self.concat = layers.Concatenate(axis=-1)
        self.bn = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

    def call(self, inputs, training=False):
        x = self.bottleneck(inputs)
        
        b1 = self.branch_d1(x, training=training)
        b2 = self.branch_d2(x, training=training)
        b3 = self.branch_d4(x, training=training)
        b4 = self.branch_d8(x, training=training)
        
        out = self.concat([b1, b2, b3, b4])
        out = self.bn(out, training=training)
        out = self.relu(out)
        return out

class ChannelAttention(layers.Layer):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.channels = channels
        
        self.gap = layers.GlobalAveragePooling1D()
        self.gmp = layers.GlobalMaxPooling1D()
        
        self.shared_conv = tf.keras.Sequential([
            layers.Conv1D(filters=1, kernel_size=3, dilation_rate=1, padding='same', use_bias=False),
            layers.Activation('relu'),
            layers.Conv1D(filters=1, kernel_size=3, dilation_rate=2, padding='same', use_bias=False)
        ])
        
        self.sigmoid = layers.Activation('sigmoid')

    def call(self, inputs):
        avg_pool = self.gap(inputs)
        max_pool = self.gmp(inputs)
        
        avg_pool_reshaped = tf.expand_dims(avg_pool, axis=-1)
        max_pool_reshaped = tf.expand_dims(max_pool, axis=-1)
        
        avg_out = self.shared_conv(avg_pool_reshaped)
        max_out = self.shared_conv(max_pool_reshaped)
        
        added = layers.Add()([avg_out, max_out])
        att_map = self.sigmoid(added)
        
        att_map = layers.Permute((2, 1))(att_map)
        
        return layers.Multiply()([inputs, att_map])

class TemporalAttention(layers.Layer):
    def __init__(self, kernel_sizes=[1, 3, 5]):
        super(TemporalAttention, self).__init__()
        
        self.convs = []
        for k in kernel_sizes:
            self.convs.append(
                tf.keras.Sequential([
                    layers.DepthwiseConv1D(kernel_size=k, padding='same', use_bias=False),
                    layers.Conv1D(filters=1, kernel_size=1, padding='same', use_bias=False) 
                ])
            )
        
        self.concat = layers.Concatenate(axis=-1)
        self.sigmoid = layers.Activation('sigmoid')
        self.final_conv = layers.Conv1D(filters=1, kernel_size=1, padding='same', activation='sigmoid')

    def call(self, inputs):
        outputs = [conv(inputs) for conv in self.convs]
        
        concat_out = self.concat(outputs)
        
        t_map = self.final_conv(concat_out)
        
        return layers.Multiply()([inputs, t_map])

def build_msfe_cta_model(input_shape=(5400, 1566)):
    inputs = Input(shape=input_shape, name='video_input')
    
    scale1 = InceptionTCNModule(filters=32)(inputs)
    pool1 = layers.MaxPooling1D(pool_size=2)(scale1)
    
    scale2 = InceptionTCNModule(filters=64)(pool1)
    pool2 = layers.MaxPooling1D(pool_size=2)(scale2)
    
    scale3 = InceptionTCNModule(filters=128)(pool2)
    
    scale1_aligned = layers.MaxPooling1D(pool_size=4)(scale1)
    scale2_aligned = layers.MaxPooling1D(pool_size=2)(scale2)
    
    msfe_features = layers.Concatenate(axis=-1)([scale1_aligned, scale2_aligned, scale3])
    
    ch_attended = ChannelAttention(channels=msfe_features.shape[-1])(msfe_features)
    
    final_features = TemporalAttention()(ch_attended)
    
    x = layers.GlobalAveragePooling1D()(final_features)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(1, activation='linear', name='depression_score')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="MSFE_CTA_Net")
    return model

if __name__ == "__main__":
    model = build_msfe_cta_model(input_shape=(5400, 1566))
    
    model.summary()
    
    total_params = model.count_params()
    print(f"\nTotal Parameters: {total_params:,}")
