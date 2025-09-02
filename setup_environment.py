#!/usr/bin/env python
"""
Environment setup script to handle CUDA/CPU configuration for experiments
"""

import os
import logging
import tensorflow as tf
import warnings

logger = logging.getLogger(__name__)

def setup_tf_environment(force_cpu=False, verbose=True):
    """
    Setup TensorFlow environment with proper GPU/CPU configuration
    
    Args:
        force_cpu: Force CPU usage even if GPU is available
        verbose: Print setup information
    """
    
    if force_cpu:
        # Force CPU usage
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Reduce TF warnings
        if verbose:
            logger.info("🖥️  Forced CPU mode - GPU disabled")
    else:
        # Setup XLA flags for CUDA compilation (fixes libdevice and ptxas issues)
        cuda_dir = os.path.expanduser('~/.local/cuda')
        os.environ['XLA_FLAGS'] = f'--xla_gpu_cuda_data_dir={cuda_dir} --xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found'
        
        # Try to use GPU, fallback to CPU if issues
        try:
            # Check if CUDA is available
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                if verbose:
                    logger.info(f"🚀 GPU detected: {len(gpus)} device(s)")
                # Configure GPU memory growth to avoid allocation issues
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            else:
                if verbose:
                    logger.info("🖥️  No GPU detected, using CPU")
        except Exception as e:
            # If GPU setup fails, force CPU
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            if verbose:
                logger.warning(f"⚠️  GPU setup failed ({e}), falling back to CPU")
    
    # Suppress protobuf warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')
    
    # Set optimized CPU flags
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
    
    if verbose:
        logger.info("✅ TensorFlow environment configured")

def main():
    """Test the environment setup"""
    import sys
    
    force_cpu = '--cpu' in sys.argv or '--force-cpu' in sys.argv
    setup_tf_environment(force_cpu=force_cpu, verbose=True)
    
    # Test basic TF operation
    try:
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        y = tf.matmul(x, x)
        logger.info(f"✅ TensorFlow test successful: {tf.reduce_sum(y).numpy()}")
        logger.info(f"   Device: {y.device}")
    except Exception as e:
        logger.error(f"❌ TensorFlow test failed: {e}")

if __name__ == "__main__":
    main()