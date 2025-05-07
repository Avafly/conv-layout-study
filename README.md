# conv-layout-study

NCHW and NHWC are data layouts that specify how tensors are stored in memory. PyTorch defaults to the NCHW layout, while TensorFlow defaults to NHWC. This repo studies the convolutional layer's performance under NCHW and NHWC layouts. The convolutional layer is composed of [im2col](https://caffe.berkeleyvision.org/tutorial/convolution.html) and [GEMM](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3). Since optimized GEMM can approach the CPU's peak performance in most cases, the main focus of this study is im2col. The test devices are a RPi 4B and a VPS with the Intel Xeon Platinum processor. This repo was inspired by [this discussion](https://github.com/Avafly/tiny-cnn/issues/1).

TL;DR: For CPU inference, the convolutional layer under NHWC layout may be faster in most cases, and the benefit is bigger with more channels.

## How to run

Parameters can be specified via args.

```bash
Usage: argv[0] in_c in_h in_w kn_size out_c stride padding
```

## Results

### Im2col layers

Kernel size = 3, stride = 1, padding = 0.

| (Channel, Height, Width) | RPi 4B<br/>NCHW \| NHWC | NHWCVPS<br/>NCHW \| NHWC |
| :----------------------: | :---------------------: | :----------------------: |
|       (8, 16, 16)        |   0.07 ms \| 0.06 ms    |    0.03 ms \| 0.02 ms    |
|       (64, 16, 16)       |   0.60 ms \| 0.43 ms    |    0.24 ms \| 0.17 ms    |
|      (128, 16, 16)       |   1.24 ms \| 0.84 ms    |    0.49 ms \| 0.33 ms    |
|                          |                         |                          |
|       (8, 80, 80)        |   2.35 ms \| 2.36 ms    |    0.97 ms \| 0.75 ms    |
|       (64, 80, 80)       |  19.61 ms \| 16.83 ms   |    5.20 ms \| 2.44 ms    |
|      (128, 80, 80)       |  38.86 ms \| 31.92 ms   |   10.68 ms \| 6.39 ms    |
|                          |                         |                          |
|      (8, 320, 320)       |  39.28 ms \| 42.24 ms   |    10.1 ms \| 7.99 ms    |
|      (64, 320, 320)      | 316.88 ms \| 287.20 ms  |   93.07 ms \| 64.85 ms   |
|     (128, 320, 320)      | 628.09 ms \| 541.26 ms  |  186.13 ms \| 134.85 ms  |

### Convolutional layers

In `conv_layers.c`, the convolutional layers for both NCHW and NHWC layouts are implemented using im2col functions above. I simulated several common convolutional layers used in YOLO models and measured their execution time.

| Layers                                            | RPi 4B<br/>NCHW \| NHWC |  VPS<br/>NCHW \| NHWC  |
| ------------------------------------------------- | :---------------------: | :--------------------: |
| Input shape: (64, 320, 320), k=3, s=1, p=1, c=128 | 849.08 ms \| 850.50 ms  | 272.96 ms \| 230.79 ms |
| Input shape: (256, 80, 80), k=3, s=2, p=1, c=256  |  90.46 ms \| 84.68 ms   |  24.49 ms \| 20.98 ms  |
| Input shape: (512, 40, 40), k=3, s=2, p=1, c=512  |  75.44 ms \| 75.42 ms   |  20.21 ms \| 17.86 ms  |
| Input shape: (256, 80, 80), k=1, s=1, p=0, c=256  |  42.34 ms \| 41.53 ms   |  13.05 ms \| 12.63 ms  |
| Input shape: (256, 80, 80), k=3, s=1, p=1, c=256  | 341.81 ms \| 319.62 ms  |  92.25 ms \| 77.02 ms  |
| Input shape: (3, 640, 640), k=3, s=2, p=1, c=64   |  58.19 ms \| 69.31 ms   |  20.41 ms \| 20.64 ms  |
| Input shape: (3, 640, 640), k=6, s=2, p=2, c=64   | 147.65 ms \| 164.38 ms  |  48.11 ms \| 53.24 ms  |

`conv_layers.c` also saves the results to binary files. Running `torch_test.py` can verify the computation results.
