import torch
import tensorflow as tf

class TriangularCausalMask__Pytorch():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class TriangularCausalMask__Tensorflow():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with tf.device(device):
            self._mask = tf.linalg.band_part(tf.ones(mask_shape, dtype=tf.bool), num_lower=-1, num_upper=0)

    @property
    def mask(self):
        return self._mask

class ProbMask__Pytorch():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask
    
class ProbMask__TensorFlow():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = tf.ones((L, scores.shape[-1]), dtype=tf.bool).numpy().triu(1)
        _mask_ex = tf.tile(tf.expand_dims(tf.expand_dims(_mask, 0), 0), [B, H, 1, 1])
        indices = tf.stack([tf.range(B)[:, None, None], tf.range(H)[None, :, None], index], axis=3)
        indicator = tf.gather_nd(_mask_ex, indices)
        self._mask = tf.reshape(indicator, scores.shape)

    @property
    def mask(self):
        return self._mask
