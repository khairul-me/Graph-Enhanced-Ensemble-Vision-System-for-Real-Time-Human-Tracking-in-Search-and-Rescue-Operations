import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

class PerceptualModule:
    """
    Complete implementation of the Perceptual Module from the paper
    "Learning Physical Dynamics for Object-centric Visual Prediction"
    """
    
    def __init__(self, num_keypoints=4, rotation_dim=4, sigma=2.0):
        self.num_keypoints = num_keypoints
        self.rotation_dim = rotation_dim
        self.sigma = sigma
        
        # Build encoder and decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        
        # Initialize rotation kernels (Gabor-like filters)
        self.rotation_kernels = self.create_rotation_kernels()
    
    def build_encoder(self):
        """
        Three-branch encoder: ω_feat(I), ω_pose^pos(I), ω_pose^coef(I)
        """
        inputs = layers.Input(shape=(None, None, 3), name='input_image')
        
        # Shared convolutional backbone
        x = layers.Conv2D(32, 5, strides=2, padding='same', activation='relu', name='conv1')(inputs)
        x = layers.Conv2D(64, 5, strides=2, padding='same', activation='relu', name='conv2')(x)
        x = layers.Conv2D(128, 5, strides=2, padding='same', activation='relu', name='conv3')(x)
        
        # Branch 1: Spatial features F (static environment/appearance)
        spatial_features = layers.Conv2D(128, 3, padding='same', activation='relu', name='spatial_features')(x)
        
        # Branch 2: Position heatmaps (N keypoints)
        heatmaps = layers.Conv2D(self.num_keypoints, 1, name='heatmaps')(x)
        
        # Branch 3: Scale and rotation coefficients 
        coeffs = layers.Conv2D(self.num_keypoints * (1 + self.rotation_dim), 1, name='pose_coeffs')(x)
        
        return models.Model(
            inputs=inputs, 
            outputs=[heatmaps, coeffs, spatial_features], 
            name='PerceptualEncoder'
        )
    
    def build_decoder(self):
        """
        Decoder ε that combines spatial features and Gaussian maps
        Implements: Î_tgt = ε([F_ref, G_tgt])
        """
        # Input 1: Spatial features from reference frame
        feature_input = layers.Input(shape=(None, None, 128), name='spatial_features')
        
        # Input 2: Gaussian maps from target frame  
        gaussian_input = layers.Input(shape=(None, None, self.num_keypoints * self.rotation_dim), name='gaussian_maps')
        
        # Concatenate features (Equation 6)
        x = layers.Concatenate(axis=-1, name='feature_fusion')([feature_input, gaussian_input])
        
        # Decoder network (upsampling path)
        x = layers.Conv2D(128, 3, padding='same', activation='relu', name='dec_conv1')(x)
        x = layers.UpSampling2D(2, name='upsample1')(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu', name='dec_conv2')(x)
        x = layers.UpSampling2D(2, name='upsample2')(x)
        x = layers.Conv2D(32, 3, padding='same', activation='relu', name='dec_conv3')(x)
        x = layers.UpSampling2D(2, name='upsample3')(x)
        
        # Output reconstructed image
        output = layers.Conv2D(3, 3, padding='same', activation='sigmoid', name='reconstructed_image')(x)
        
        return models.Model(
            inputs=[feature_input, gaussian_input], 
            outputs=output, 
            name='PerceptualDecoder'
        )
    
    def create_rotation_kernels(self):
        """
        Create rotation kernels K_i for different orientations
        These act as learnable filters for Equation 5
        """
        kernels = []
        for i in range(self.rotation_dim):
            # Initialize with random values (could be Gabor filters)
            kernel = tf.Variable(
                tf.random.normal([5, 5, 1, 1], stddev=0.1),
                trainable=True,
                name=f'rotation_kernel_{i}'
            )
            kernels.append(kernel)
        return kernels
    
    def spatial_softargmax(self, heatmaps):
        """
        Implements Equations 2 & 3: Convert heatmaps to 2D coordinates
        
        Equation 2: H_n(i,j) = exp(H_n(i,j)) / Σ exp(H_n(i,j))
        Equation 3: p_n = Σ u(i,j) · H_n(i,j)
        """
        B, H, W, N = tf.shape(heatmaps)[0], tf.shape(heatmaps)[1], tf.shape(heatmaps)[2], tf.shape(heatmaps)[3]
        
        # Equation 2: Spatial softmax normalization
        flat = tf.reshape(heatmaps, [B, -1, N])
        softmax = tf.nn.softmax(flat, axis=1)
        softmax = tf.reshape(softmax, [B, H, W, N])
        
        # Create coordinate grids
        x_range = tf.cast(tf.range(W), tf.float32)
        y_range = tf.cast(tf.range(H), tf.float32)
        xx, yy = tf.meshgrid(x_range, y_range)
        
        xx = tf.reshape(xx, [1, H, W, 1])
        yy = tf.reshape(yy, [1, H, W, 1])
        
        # Equation 3: Weighted summation to get coordinates
        x = tf.reduce_sum(softmax * xx, axis=[1, 2])
        y = tf.reduce_sum(softmax * yy, axis=[1, 2])
        
        return tf.stack([x, y], axis=-1), softmax
    
    def extract_pose_coefficients(self, coeffs):
        """
        Extract scale and rotation coefficients per keypoint
        """
        B = tf.shape(coeffs)[0]
        H, W = tf.shape(coeffs)[1], tf.shape(coeffs)[2]
        
        # Reshape to separate keypoints and their coefficients
        coeffs_reshaped = tf.reshape(coeffs, [B, H, W, self.num_keypoints, 1 + self.rotation_dim])
        
        # Global average pooling per keypoint
        scale_factors = tf.reduce_mean(coeffs_reshaped[..., 0], axis=[1, 2])  # [B, num_keypoints]
        rotation_coeffs = tf.reduce_mean(coeffs_reshaped[..., 1:], axis=[1, 2])  # [B, num_keypoints, rotation_dim]
        
        return scale_factors, rotation_coeffs
    
    def generate_gaussian_maps(self, keypoints, scale_factors, height, width):
        """
        Implements Equation 4: Generate isotropic Gaussian maps
        G_n = s_n · exp(-1/(2σ²) ||u - p_n||²)
        """
        B = tf.shape(keypoints)[0]
        N = tf.shape(keypoints)[1]
        
        # Create coordinate grids
        y = tf.range(height, dtype=tf.float32)
        x = tf.range(width, dtype=tf.float32)
        yy, xx = tf.meshgrid(y, x, indexing='ij')
        yy = tf.reshape(yy, [1, 1, height, width])
        xx = tf.reshape(xx, [1, 1, height, width])
        
        # Keypoint positions
        mu_x = tf.expand_dims(tf.expand_dims(keypoints[..., 0], -1), -1)
        mu_y = tf.expand_dims(tf.expand_dims(keypoints[..., 1], -1), -1)
        scale_factors = tf.expand_dims(tf.expand_dims(scale_factors, -1), -1)
        
        # Equation 4: Gaussian distribution
        gaussian_maps = scale_factors * tf.exp(-((xx - mu_x)**2 + (yy - mu_y)**2) / (2.0 * self.sigma**2))
        
        return gaussian_maps
    
    def apply_rotation_kernels(self, gaussian_maps, rotation_coeffs):
        """
        Implements Equation 5: Apply rotation kernels
        G_n^i = r_n^i · (K_i * G_n)
        """
        B, N, H, W = tf.shape(gaussian_maps)[0], tf.shape(gaussian_maps)[1], tf.shape(gaussian_maps)[2], tf.shape(gaussian_maps)[3]
        C = self.rotation_dim
        
        # Reshape for convolution [B*N, H, W, 1]
        g_maps_reshaped = tf.reshape(gaussian_maps, [B * N, H, W, 1])
        
        rotated_maps = []
        for i in range(C):
            # Equation 5: Convolution with rotation kernel
            conv = tf.nn.conv2d(g_maps_reshaped, self.rotation_kernels[i], strides=1, padding='SAME')
            conv = tf.reshape(conv, [B, N, H, W])
            
            # Weight by rotation coefficient
            r_i = tf.expand_dims(tf.expand_dims(rotation_coeffs[:, :, i], -1), -1)
            rotated_maps.append(conv * r_i)
        
        # Concatenate along keypoint dimension: [B, N*C, H, W]
        return tf.concat(rotated_maps, axis=1)
    
    def forward_encode(self, image):
        """
        Forward pass through encoder: extract keypoints and features
        """
        # Encoder forward pass
        heatmaps, coeffs, spatial_features = self.encoder(image)
        
        # Extract keypoint coordinates (Equations 2 & 3)
        keypoints, softmax_maps = self.spatial_softargmax(heatmaps)
        
        # Extract pose coefficients
        scale_factors, rotation_coeffs = self.extract_pose_coefficients(coeffs)
        
        return {
            'keypoints': keypoints,
            'scale_factors': scale_factors,
            'rotation_coeffs': rotation_coeffs,
            'spatial_features': spatial_features,
            'heatmaps': heatmaps,
            'softmax_maps': softmax_maps
        }
    
    def generate_gaussian_representation(self, keypoints, scale_factors, rotation_coeffs, height, width):
        """
        Complete Gaussian map generation process (Equations 4 & 5)
        """
        # Step 1: Generate isotropic Gaussian maps (Equation 4)
        gaussian_maps = self.generate_gaussian_maps(keypoints, scale_factors, height, width)
        
        # Step 2: Apply rotation kernels (Equation 5)
        rotated_maps = self.apply_rotation_kernels(gaussian_maps, rotation_coeffs)
        
        return rotated_maps, gaussian_maps
    
    def forward_decode(self, spatial_features, gaussian_maps):
        """
        Forward pass through decoder: reconstruct image
        Implements Equation 6: Î_tgt = ε([F_ref, G_tgt])
        """
        return self.decoder([spatial_features, gaussian_maps])
    
    def perceptual_loss(self, I_target, I_reconstructed, omega=0.1):
        """
        Implements Equation 7: Perceptual loss with gradient regularization
        L_per = ||I_tgt - Î_tgt||²₂ + ω||∇I_tgt - ∇Î_tgt||²₂
        """
        # Pixel-wise L2 loss
        pixel_loss = tf.reduce_mean(tf.square(I_target - I_reconstructed))
        
        # Sobel operators for gradient computation
        sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
        sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)
        sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
        sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])
        
        # Convert to grayscale for gradient computation
        I_target_gray = tf.reduce_mean(I_target, axis=-1, keepdims=True)
        I_recon_gray = tf.reduce_mean(I_reconstructed, axis=-1, keepdims=True)
        
        # Compute gradients
        grad_target_x = tf.nn.conv2d(I_target_gray, sobel_x, strides=1, padding='SAME')
        grad_recon_x = tf.nn.conv2d(I_recon_gray, sobel_x, strides=1, padding='SAME')
        grad_target_y = tf.nn.conv2d(I_target_gray, sobel_y, strides=1, padding='SAME')
        grad_recon_y = tf.nn.conv2d(I_recon_gray, sobel_y, strides=1, padding='SAME')
        
        # Gradient difference loss
        grad_loss = tf.reduce_mean(tf.square(grad_target_x - grad_recon_x)) + \
                   tf.reduce_mean(tf.square(grad_target_y - grad_recon_y))
        
        return pixel_loss + omega * grad_loss
    
    def full_forward_pass(self, I_reference, I_target):
        """
        Complete forward pass implementing the full perceptual module pipeline
        """
        # Extract features from reference frame
        ref_encoding = self.forward_encode(I_reference)
        F_ref = ref_encoding['spatial_features']
        
        # Extract keypoints from target frame  
        tgt_encoding = self.forward_encode(I_target)
        keypoints_tgt = tgt_encoding['keypoints']
        scale_factors_tgt = tgt_encoding['scale_factors']
        rotation_coeffs_tgt = tgt_encoding['rotation_coeffs']
        
        # Generate Gaussian maps for target frame
        H, W = tf.shape(I_target)[1], tf.shape(I_target)[2]
        G_tgt, _ = self.generate_gaussian_representation(
            keypoints_tgt, scale_factors_tgt, rotation_coeffs_tgt, H//8, W//8  # Accounting for downsampling
        )
        
        # Reconstruct target image (Equation 6)
        I_reconstructed = self.forward_decode(F_ref, G_tgt)
        
        return I_reconstructed, {
            'ref_encoding': ref_encoding,
            'tgt_encoding': tgt_encoding,
            'gaussian_maps': G_tgt
        }

# Demo usage
def demo_perceptual_module():
    """
    Demonstration of the complete perceptual module
    """
    # Initialize module
    perceptual_module = PerceptualModule(num_keypoints=4, rotation_dim=4)
    
    # Create dummy data
    batch_size = 2
    I_ref = tf.random.uniform([batch_size, 128, 128, 3])
    I_tgt = tf.random.uniform([batch_size, 128, 128, 3])
    
    # Forward pass
    I_reconstructed, intermediate_results = perceptual_module.full_forward_pass(I_ref, I_tgt)
    
    # Compute loss
    loss = perceptual_module.perceptual_loss(I_tgt, I_reconstructed)
    
    print("✅ Perceptual Module Demo Results:")
    print(f"Input shape: {I_ref.shape}")
    print(f"Reconstructed shape: {I_reconstructed.shape}")
    print(f"Reconstruction loss: {loss.numpy():.4f}")
    print(f"Keypoints shape: {intermediate_results['tgt_encoding']['keypoints'].shape}")
    print(f"Gaussian maps shape: {intermediate_results['gaussian_maps'].shape}")
    print(f"Spatial features shape: {intermediate_results['ref_encoding']['spatial_features'].shape}")

if __name__ == "__main__":
    demo_perceptual_module()
