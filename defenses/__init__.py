"""Defense mechanisms against adversarial attacks"""
from .freqpress import (
    apply_freqpress_defense,
    butterworth_low_pass_filter,
    webp_compression,
    preprocess_batch_butterworth_webp,
    save_unnormalized_tensor_image,
    normalize_tensor
)
