# deep-learning-performance-analysis
Performance optimization study comparing CPU vs GPU, mixed precision, and quantization for deep learning workloads
# PyTorch Performance Optimization Study

> Comprehensive analysis of GPU acceleration, mixed precision training, and quantization techniques for deep learning optimization

## Overview

This project benchmarks and implements various PyTorch optimization techniques on computer vision tasks. Through systematic experimentation, I demonstrate significant performance improvements achievable through proper GPU utilization, automatic mixed precision, and model quantization.

## Key Results

- **GPU Acceleration**: Achieved significant speedup using NVIDIA T4 GPU vs CPU baseline
- **Mixed Precision Training**: Reduced memory usage by ~40-50% with FP16 operations while maintaining accuracy
- **Dynamic Quantization**: 75% model size reduction with INT8 quantization for faster inference
- **Operator Fusion**: Leveraged `torch.compile()` for automatic kernel optimization

## Optimization Techniques Implemented

### 1. GPU vs CPU Performance Comparison
- Benchmarked identical CNN training on CPU vs NVIDIA T4 GPU
- Analyzed compute utilization and memory bandwidth
- Demonstrated the necessity of GPU acceleration for modern deep learning

### 2. Automatic Mixed Precision (AMP)
```python
scaler = torch.cuda.amp.GradScaler()

for images, labels in dataloader:
    with torch.cuda.amp.autocast():
        outputs = model(images)
        loss = criterion(outputs, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```
- Implemented FP16 training with gradient scaling
- Reduced memory footprint enabling larger batch sizes
- Maintained numerical stability with automatic loss scaling

### 3. Dynamic Quantization
```python
quantized_model = torch.quantization.quantize_dynamic(
    model, 
    {torch.nn.Linear, torch.nn.Conv2d}, 
    dtype=torch.qint8
)
```
- Post-training INT8 quantization for inference optimization
- 4x model compression with minimal accuracy loss
- Optimized for CPU deployment scenarios

### 4. Torch Compile & Operator Fusion
```python
compiled_model = torch.compile(model, mode='default')
```
- Utilized PyTorch 2.0+ automatic graph optimization
- Reduced kernel launch overhead through fusion
- Improved throughput without code changes

## Performance Comparison

| Optimization | Hardware | Memory Impact | Speed | Accuracy |
|-------------|----------|---------------|-------|----------|
| GPU Acceleration | NVIDIA T4 | Higher VRAM | ⭐⭐⭐⭐⭐ | No change |
| Mixed Precision | GPU | -40-50% | ⭐⭐⭐⭐ | Negligible |
| Quantization | CPU | -75% | ⭐⭐⭐ | <1% loss |
| Torch Compile | Both | Minimal | ⭐⭐ | No change |

## Technical Stack

- **Framework**: PyTorch 2.0+
- **Hardware**: NVIDIA T4 GPU (16GB), CPU baseline
- **Datasets**: MNIST, CIFAR-10, CIFAR-100
- **Key APIs**: `torch.cuda.amp`, `torch.quantization`, `torch.compile()`

## Implementation Details

### Dataset Pipeline
- Efficient data loading with optimized DataLoader configuration
- Custom transforms and preprocessing
- Support for built-in and custom datasets

### Model Architecture
- Custom CNN implementation (SimpleCNN)
- Modular design for easy experimentation
- Device-agnostic training code

### Optimization Pipeline
```
1. Baseline (CPU) → 2. GPU Acceleration → 3. Mixed Precision → 4. Quantization
```

## Quick Start

### Prerequisites
```bash
pip install torch torchvision torchaudio matplotlib
```

### Running the Experiments
```python
# Clone and navigate to the repository
git clone https://github.com/yourusername/pytorch-performance-optimization
cd pytorch-performance-optimization

# Open the notebook
jupyter notebook pytorch_performance_optimization.ipynb

# For GPU experiments: Runtime → Change runtime type → T4 GPU
```

## Key Insights

1. **GPU acceleration is essential** - Training time reduction is dramatic for CNNs
2. **AMP is nearly free performance** - Modern GPUs with Tensor Cores benefit significantly
3. **Quantization enables edge deployment** - Critical for resource-constrained environments
4. **DataLoader configuration matters** - `num_workers` and `pin_memory` impact training speed
5. **Different optimizations target different bottlenecks** - Combine techniques strategically

##  Skills Demonstrated

**Deep Learning Systems**
- GPU programming and CUDA memory management
- Mixed precision training implementation
- Model quantization workflows
- Performance profiling and benchmarking

**PyTorch Internals**
- Automatic Mixed Precision API
- Dynamic quantization techniques
- Computational graph optimization
- DataLoader pipeline optimization

**Software Engineering**
- Reproducible experiments
- Systematic benchmarking methodology
- Clean, modular code architecture
- Technical documentation

## Datasets Used

- **MNIST**: 60K training samples, 28×28 grayscale images, 10 classes
- **CIFAR-10**: 50K training samples, 32×32 RGB images, 10 classes
- **CIFAR-100**: 50K training samples, 32×32 RGB images, 100 fine-grained classes

##  Optimization Guidelines

### Use GPU Acceleration When:
✅ Training models with >1M parameters  
✅ Batch size > 32  
✅ Working with CNNs or Transformers

### Use Mixed Precision When:
✅ GPU has Tensor Cores (V100, A100, T4)  
✅ Memory is constrained  
✅ Training time is the bottleneck

### Use Quantization When:
✅ Deploying to CPU or edge devices  
✅ Model size is critical  
✅ Inference latency must be minimized

### Use Torch Compile When:
✅ Production deployment  
✅ Using PyTorch 2.0+  
✅ Model has repetitive operations

## Connect With Me

I'm actively seeking opportunities in **ML Engineering**, **MLOps**, and **AI Infrastructure**.

- 📧 Email: rheanair071@gmail.com
- 💼 LinkedIn: [linkedin.com/in/yourprofile](www.linkedin.com/in/rheanair07)


</div>
