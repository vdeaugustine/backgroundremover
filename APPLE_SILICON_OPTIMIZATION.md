# Apple Silicon (M-Series) Optimization Guide

This document provides tips and best practices for running BackgroundRemover efficiently on Apple Silicon Macs (M1, M1 Pro/Max/Ultra, M2, M2 Pro/Max/Ultra, M3, M3 Pro/Max, M4, M4 Pro/Max).

## Hardware Acceleration (MPS)

BackgroundRemover has been optimized to use **Metal Performance Shaders (MPS)** on Apple Silicon. This means the AI inference runs on the GPU/Neural Engine rather than the CPU, resulting in **significant speed improvements** (often 3-10x faster).

### Verifying MPS is Active

To verify MPS is being used, run:

```bash
python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

You should see:
```
MPS available: True
```

## Optimal Settings for M-Series Macs

### For M4 Max (64GB RAM) - Recommended Settings

| Setting | Recommended Value | Notes |
|---------|------------------|-------|
| GPU Batch Size (`-gb`) | 4-8 | Higher values process more frames in parallel |
| Worker Nodes (`-wn`) | 1-2 | With MPS, fewer workers are needed |
| Model | Depends on use case | See model selection below |

### For M3/M2/M1 Pro/Max (16-96GB RAM)

| Setting | Recommended Value | Notes |
|---------|------------------|-------|
| GPU Batch Size (`-gb`) | 2-4 | Scale based on available unified memory |
| Worker Nodes (`-wn`) | 1-2 | Multiple workers may cause memory pressure |
| Model | Depends on use case | See model selection below |

### For M3/M2/M1 Base (8-16GB RAM)

| Setting | Recommended Value | Notes |
|---------|------------------|-------|
| GPU Batch Size (`-gb`) | 1-2 | Conservative to avoid memory pressure |
| Worker Nodes (`-wn`) | 1 | Single worker recommended |
| Model | `u2netp` for speed | Smaller model uses less memory |

## Model Selection Guide

Choose the right model based on your use case:

| Model | Speed | Quality | Best For |
|-------|-------|---------|----------|
| `u2net` | Medium | High | General objects, products |
| `u2netp` | Fast | Good | Quick previews, batch processing |
| `u2net_human_seg` | Medium | High (for humans) | People, portraits, selfies |

### Quick Reference Commands

```bash
# Human subject - best quality
./backgroundremover_m4 -i "photo.jpg" -m "u2net_human_seg" -a -o "output.png"

# Product photo - balanced
./backgroundremover_m4 -i "product.jpg" -m "u2net" -o "output.png"

# Quick preview - fastest
./backgroundremover_m4 -i "image.jpg" -m "u2netp" -o "preview.png"

# Video processing - optimized for M4 Max
./backgroundremover_m4 -i "video.mp4" -tv -gb 8 -wn 2 -o "output.mov"
```

## Memory Management

### Unified Memory Architecture

Apple Silicon uses **Unified Memory**, shared between CPU and GPU. This is advantageous for ML workloads as there's no data transfer overhead between CPU and GPU memory.

**Tips:**
- Close memory-intensive apps before processing large videos
- Monitor memory usage with Activity Monitor
- If you encounter "out of memory" errors, reduce `-gb` batch size

### Thermal Management

Apple Silicon is designed to run efficiently without active cooling (on some models). For extended batch processing:

1. **Keep MacBook well-ventilated** - Don't obstruct air vents
2. **Monitor temperature** - Use Activity Monitor or third-party tools
3. **Consider breaks** - For very long batch jobs, allow cooling between batches

## Video Processing Optimization

Video processing benefits most from optimization. Here are tuned settings:

### Transparent Video Output (Best Quality)

```bash
# M4 Max optimized
./backgroundremover_m4 -i "video.mp4" -tv -gb 8 -wn 2 -m "u2net" -o "output.mov"

# M1/M2 optimized
./backgroundremover_m4 -i "video.mp4" -tv -gb 4 -wn 1 -m "u2net" -o "output.mov"
```

### Transparent GIF Output (Smaller Files)

```bash
./backgroundremover_m4 -i "video.mp4" -tg -gb 4 -wn 2 -o "output.gif"
```

### Matte Key for Video Editing

```bash
./backgroundremover_m4 -i "video.mp4" -mk -gb 8 -wn 2 -o "matte.mp4"
```

## Batch Processing

For processing multiple files efficiently:

```bash
# Process entire folder of images
./backgroundremover_m4 -if "/path/to/images" -of "/path/to/output"

# Process folder of videos to transparent MOV
./backgroundremover_m4 -if "/path/to/videos" -of "/path/to/output" -tv -gb 8 -wn 2
```

## Performance Benchmarks

*Approximate times on different Apple Silicon chips (results may vary):*

### Image Processing (1920x1080 image)

| Chip | u2netp | u2net | u2net_human_seg |
|------|--------|-------|-----------------|
| M4 Max | ~0.3s | ~0.8s | ~0.8s |
| M3 Pro | ~0.5s | ~1.2s | ~1.2s |
| M2 | ~0.8s | ~2.0s | ~2.0s |
| M1 | ~1.0s | ~2.5s | ~2.5s |

### Video Processing (30s @ 30fps, 1080p)

| Chip | GB=4, WN=2 | GB=8, WN=2 |
|------|-----------|-----------|
| M4 Max | ~45s | ~35s |
| M3 Pro | ~75s | ~60s |
| M2 | ~120s | N/A (memory) |

## Troubleshooting

### MPS Not Available

If `torch.backends.mps.is_available()` returns `False`:

1. **Update macOS** - MPS requires macOS 12.3+
2. **Update PyTorch** - Use PyTorch 2.0+ for best MPS support
3. **Reinstall PyTorch**:
   ```bash
   pip install --upgrade torch torchvision
   ```

### Slow Processing Despite MPS

1. **Check Activity Monitor** - Verify GPU is being utilized
2. **Reduce other GPU load** - Close graphics-intensive apps
3. **Restart Python** - Sometimes MPS context needs refresh

### Memory Errors

1. **Reduce batch size** - Use `-gb 1` or `-gb 2`
2. **Use single worker** - Use `-wn 1`
3. **Close other apps** - Free up unified memory
4. **Use lighter model** - Switch to `u2netp`

### Model Download Issues

Models are downloaded on first use to `~/.u2net/`. If downloads fail:

1. **Check internet connection**
2. **Clear corrupted downloads**:
   ```bash
   rm -rf ~/.u2net/
   ```
3. **Re-run the command** - Models will re-download

## Energy Efficiency

Apple Silicon is exceptionally power-efficient. For battery operation:

- Use `u2netp` model for faster processing (less energy per image)
- Reduce batch size to lower peak power draw
- Process in batches rather than one-at-a-time for better efficiency

## Recommended Workflow for M4 Max

For the best experience on M4 Max (64GB):

1. **Use the GUI App** - Located in Applications folder
2. **For batch via CLI**:
   ```bash
   ./backgroundremover_m4 -if "input_folder" -of "output_folder" -m "u2net" -gb 8
   ```
3. **For video**:
   ```bash
   ./backgroundremover_m4 -i "video.mp4" -tv -m "u2net" -gb 8 -wn 2 -o "output.mov"
   ```

---

*This guide was created for BackgroundRemover optimized for Apple Silicon. For general usage, see README.md.*
