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

#### k=1/s=1/p=0

| (Channel, Height, Width) | RPi 4B NCHW \| NHWC  |   VPS NCHW \| NHWC   |
| :----------------------: | :------------------: | :------------------: |
|       (8, 16, 16)        |  0.02 ms \| 0.02 ms  |  0.01 ms \| 0.01 ms  |
|      (128, 16, 16)       |  0.53 ms \| 0.23 ms  |  0.10 ms \| 0.04 ms  |
|                          |                      |                      |
|       (8, 80, 80)        |  0.74 ms \| 0.58 ms  |  0.15 ms \| 0.11 ms  |
|      (128, 80, 80)       | 12.32 ms \| 8.40 ms  |  1.84 ms \| 0.84 ms  |
|                          |                      |                      |
|      (8, 320, 320)       | 10.42 ms \| 10.26 ms |  1.30 ms \| 1.55 ms  |
|     (128, 320, 320)      | 73.13 ms \| 76.73 ms | 23.74 ms \| 17.22 ms |

#### k=3/s=1/p=1

| (Channel, Height, Width) |  RPi 4B NCHW \| NHWC   |    VPS NCHW \| NHWC    |
| :----------------------: | :--------------------: | :--------------------: |
|       (8, 16, 16)        |   0.20 ms \| 0.16 ms   |   0.05 ms \| 0.04 ms   |
|      (128, 16, 16)       |   3.47 ms \| 2.04 ms   |   0.67 ms \| 0.45 ms   |
|                          |                        |                        |
|       (8, 80, 80)        |   6.43 ms \| 5.40 ms   |   1.27 ms \| 1.01 ms   |
|      (128, 80, 80)       |  43.25 ms \| 33.94 ms  |  12.91 ms \| 7.97 ms   |
|                          |                        |                        |
|      (8, 320, 320)       |  43.15 ms \| 46.50 ms  |  13.10 ms \| 9.61 ms   |
|     (128, 320, 320)      | 683.17 ms \| 581.01 ms | 202.45 ms \| 134.55 ms |

#### k=5/s=1/p=2

| (Channel, Height, Width) |   RPi 4B NCHW \| NHWC    |    VPS NCHW \| NHWC    |
| :----------------------: | :----------------------: | :--------------------: |
|       (8, 16, 16)        |    0.44 ms \| 0.34 ms    |   0.10 ms \| 0.08 ms   |
|      (128, 16, 16)       |    7.16 ms \| 4.42 ms    |   1.37 ms \| 0.94 ms   |
|                          |                          |                        |
|       (8, 80, 80)        |   17.28 ms \| 14.20 ms   |   2.30 ms \| 1.57 ms   |
|      (128, 80, 80)       |  111.07 ms \| 84.48 ms   |  33.56 ms \| 21.19 ms  |
|                          |                          |                        |
|      (8, 320, 320)       |  118.94 ms \| 125.90 ms  |  35.09 ms \| 25.67 ms  |
|     (128, 320, 320)      | 1867.34 ms \| 1519.18 ms | 550.35 ms \| 366.64 ms |

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
