#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "im2col.h"

#define ALIGN_SIZE      16

int main(int argc, char *argv[])
{
    // Usage: argv[0] in_c in_h in_w kn_size out_c stride padding
    /* define tensor shape */
    int in_c = 2, in_h = 4, in_w = 4, kn_size = 3, out_c = 2, stride = 1, padding = 0;

    if (argc >= 2 && atoi(argv[1]) > 0)
        in_c = atoi(argv[1]);
    if (argc >= 3 && atoi(argv[2]) > 0)
        in_h = atoi(argv[2]);
    if (argc >= 4 && atoi(argv[3]) > 0)
        in_w = atoi(argv[3]);
    if (argc >= 5 && atoi(argv[4]) > 0)
        kn_size = atoi(argv[4]);
    if (argc >= 6 && atoi(argv[5]) > 0)
        out_c = atoi(argv[5]);
    if (argc >= 7 && atoi(argv[6]) > 0)
        stride = atoi(argv[6]);
    if (argc >= 8 && atoi(argv[7]) > 0)
        padding = atoi(argv[7]);

    int out_h = in_h - kn_size + 1, out_w = in_w - kn_size + 1;
    const int K = kn_size * kn_size * in_c, N = out_h * out_w;
    const int in_buf_size = in_c * in_h * in_w;
    printf("in_c: %d, in_h: %d, in_w: %d, kn: %d, stride: %d, padding: %d\n", in_c, in_h, in_w, kn_size, stride, padding);

    /* create buffers */
    float *in_buf = aligned_alloc(ALIGN_SIZE, sizeof(float) * in_buf_size);
    float *col_buf = aligned_alloc(ALIGN_SIZE, sizeof(float) * K * N);
    if (!in_buf || !col_buf)
    {
        fprintf(stderr, "Failed to create buffers\n");
        return -1;
    }

    /* init */
    for (int i = 0; i < in_buf_size; ++i)
        in_buf[i] = (float)i + 1.0f;

    double start_time = omp_get_wtime();

    /* im2col nchw */
    im2col_nchw(
        in_buf, col_buf,
        in_c, in_h, in_w,
        1, out_h, out_w,
        kn_size, kn_size, padding, stride
    );

    printf("Elapsed time: %.2f ms\n", (omp_get_wtime() - start_time) * 1000.0);

    /* show results */
    if (in_c * in_h * in_w > 1024)
        return 0;

    // in_buf
    printf("In NCHW (%d, %d, %d)\n", in_c, in_h, in_w);
    for (int c = 0; c < in_c; ++c)
    {
        for (int h = 0; h < in_h; ++h)
        {
            for (int w = 0; w < in_w; ++w)
            {
                printf("%-4.0f", in_buf[c * in_h * in_w + h * in_w + w]);
            }
            printf("\n");
        }
        printf("--- %d\n", c + 1);
    }

    // col_buf
    printf("\ndata_col %d x %d\n", K, N);
    for (int row = 0; row < K; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            printf("%-4.0f", col_buf[row * N + col]);
        }
        printf("\n");
    }

    /* release buffers */
    free(in_buf);   in_buf = NULL;
    free(col_buf);  col_buf = NULL;

    return 0;
}
