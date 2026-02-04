import numpy as np
import h5py

class SimpleNumpyModel:
    def __init__(self, h5_path):
        self.weights = {}
        self.load_weights(h5_path)

    def load_weights(self, h5_path):
        try:
            with h5py.File(h5_path, 'r') as f:
                # This assumes the standard Keras H5 structure
                # We need to manually match the layer names from make_model.py
                # Layer names are usually: conv2d, conv2d_1, dense, dense_1
                
                # Helper to print structure if needed
                # def print_structure(name, obj):
                #     print(name)
                # f.visititems(print_structure)

                # Robust Loading Logic
                g_weights = f['model_weights']
                layer_names = sorted(list(g_weights.keys()))
                print(f"[INFO] Found layers in H5: {layer_names}")

                conv_layers = []
                dense_layers = []

                # Recursive helper to find weights in a group
                def find_weights_in_group(group):
                    found_w = None
                    found_b = None
                    
                    # Helper to visit all items
                    def visitor(name, obj):
                        nonlocal found_w, found_b
                        if isinstance(obj, h5py.Dataset):
                            base_name = name.split('/')[-1]
                            if 'kernel' in base_name or 'W' in base_name:
                                found_w = obj[()]
                            elif 'bias' in base_name or 'b' in base_name:
                                found_b = obj[()]
                                
                    group.visititems(visitor)
                    return found_w, found_b

                for name in layer_names:
                    # Ignore non-layer keys like top_level_model_weights
                    if name in ['top_level_model_weights']:
                        continue
                        
                    group = g_weights[name]
                    W, b = find_weights_in_group(group)
                    
                    if W is not None and b is not None:
                        if len(W.shape) == 4:
                            conv_layers.append((name, W, b))
                        elif len(W.shape) == 2:
                            dense_layers.append((name, W, b))
                
                # Sort to ensure order
                conv_layers.sort(key=lambda x: x[0])
                dense_layers.sort(key=lambda x: x[0])
                
                if len(conv_layers) < 2 or len(dense_layers) < 2:
                    raise ValueError(f"Expected at least 2 conv and 2 dense layers. Found {len(conv_layers)} conv, {len(dense_layers)} dense.")
                    
                print(f"[OK] Loaded {len(conv_layers)} Conv layers and {len(dense_layers)} Dense layers.")

                self.W_conv1, self.b_conv1 = conv_layers[0][1], conv_layers[0][2]
                self.W_conv2, self.b_conv2 = conv_layers[1][1], conv_layers[1][2]
                self.W_dense1, self.b_dense1 = dense_layers[0][1], dense_layers[0][2]
                self.W_dense2, self.b_dense2 = dense_layers[1][1], dense_layers[1][2]
                
        except Exception as e:
            print(f"[ERROR] Failed to load weights: {e}")
            raise e


    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def max_pool_2x2(self, x):
        # x shape: (H, W, C)
        # Output shape: (H/2, W/2, C)
        h, w, c = x.shape
        new_h = h // 2
        new_w = w // 2
        x_reshaped = x[:new_h*2, :new_w*2, :].reshape(new_h, 2, new_w, 2, c)
        return x_reshaped.max(axis=(1, 3))

    def conv2d(self, x, W, b):
        # x: (H_in, W_in, C_in)
        # W: (3, 3, C_in, C_out) - TF/Keras format
        # b: (C_out,)
        
        # Naive implementation is slow, let's try a slightly optimized valid convolution
        # Since make_model.py used default padding='valid', output size reduces by 2
        
        h_in, w_in, c_in = x.shape
        f_h, f_w, _, c_out = W.shape
        
        h_out = h_in - f_h + 1
        w_out = w_in - f_w + 1
        
        # Output buffer
        out = np.zeros((h_out, w_out, c_out))
        
        # This is the slow part. For 48x48 input it might be acceptable (~50ms)
        # Optimized loop:
        for i in range(h_out):
            for j in range(w_out):
                # Extract patch
                patch = x[i:i+f_h, j:j+f_w, :]  # (3, 3, C_in)
                
                # Vectorized dot product over kernel dims
                # W is (3, 3, C_in, C_out)
                # We want sum(patch * W[..., k])
                
                # Reshape patch to broadcast: (3, 3, C_in, 1)
                # Multiply and sum over spatial+channel dims
                res = np.tensordot(patch, W, axes=([0, 1, 2], [0, 1, 2])) 
                out[i, j, :] = res
                
        return out + b

    def predict(self, face_img):
        # face_img: (48, 48) grayscale, normalized 0-1
        
        # 1. Expand input dim to (48, 48, 1)
        x = np.expand_dims(face_img, axis=-1)
        
        # 2. Conv1
        x = self.conv2d(x, self.W_conv1, self.b_conv1)
        x = self.relu(x)
        
        # 3. Pool1
        x = self.max_pool_2x2(x)
        
        # 4. Conv2
        x = self.conv2d(x, self.W_conv2, self.b_conv2)
        x = self.relu(x)
        
        # 5. Pool2
        x = self.max_pool_2x2(x)
        
        # 6. Flatten
        x = x.flatten()
        
        # 7. Dense1
        x = np.dot(x, self.W_dense1) + self.b_dense1
        x = self.relu(x)
        
        # 8. Dense2 (Output)
        x = np.dot(x, self.W_dense2) + self.b_dense2
        x = self.softmax(x)
        
        return x

