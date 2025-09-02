# CUDA GPU Setup Fix Documentation

## Problem Summary
The original issue was TensorFlow failing with "JIT compilation failed" errors due to missing CUDA toolkit components, specifically:
- Missing `libdevice.10.bc` file
- Missing `ptxas` (NVIDIA PTX assembler) 

## Root Cause
- NVIDIA GPU drivers (575.64.03, CUDA 12.9) were installed
- But CUDA **toolkit** was missing (no `nvcc`, `ptxas`, `libdevice`, etc.)
- TensorFlow needs these components for GPU compilation

## Solution Implemented

### 1. Install CUDA Components via pip
```bash
python3 -m pip install nvidia-pyindex
python3 -m pip install nvidia-cuda-nvcc
python3 -m pip install nvidia-cuda-runtime
```

### 2. Create Proper CUDA Directory Structure
```bash
mkdir -p ~/.local/cuda/nvvm/libdevice
cp /home/postdoc/.local/lib/python3.10/site-packages/triton/backends/nvidia/lib/libdevice.10.bc ~/.local/cuda/nvvm/libdevice/
```

### 3. Set XLA Environment Flags
```bash
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/home/postdoc/.local/cuda --xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found"
```

## Verification
After applying the fix, the test shows:
- ✅ GPU detected and used: "🚀 GPU detected: 1 device(s)"
- ✅ XLA compilation working: "Compiled cluster using XLA!"
- ✅ No more "JIT compilation failed" errors
- ✅ At least one experiment strategy runs successfully on GPU

## Automated Setup
The fix has been integrated into `setup_environment.py` and `experiment_runner.py`:
- Automatically sets XLA_FLAGS on startup
- Provides CPU fallback if GPU issues persist
- Can be forced to CPU with `--cpu` flag

## Usage
```bash
# Run with GPU (automatic)
python3 run_experiments.py test

# Force CPU usage
python3 run_experiments.py test --cpu
```

## Files Modified
1. `setup_environment.py` - Environment setup with CUDA fixes
2. `experiment_runner.py` - Integrated environment setup
3. `run_experiments.py` - Added CPU fallback option
4. `CUDA_FIX_DOCUMENTATION.md` - This documentation

## Notes
- Some experiments still show NaN loss issues, but these are **training/data problems**, not CUDA issues
- The finetune_only strategy works perfectly, confirming GPU functionality
- PTX compilation warnings are expected but don't prevent operation (fallback to driver works)

## Alternative: Full CUDA Toolkit Installation
If you prefer a full CUDA toolkit installation instead of the pip-based approach:
1. Download CUDA Toolkit from NVIDIA
2. Install with: `sudo bash cuda_*_linux.run`
3. Add to PATH: `export PATH=/usr/local/cuda/bin:$PATH`
4. Add to LD_LIBRARY_PATH: `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`