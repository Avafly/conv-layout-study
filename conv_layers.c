#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <cblas.h>

#include "im2col.h"

#define ALIGN_SIZE      64

// tensor
void chw2hwc(
    const float *chw, float *hwc,
    const int channel, const int rows, const int cols
)
{
    for (int h = 0; h < rows; ++h)
    {
        for (int w = 0; w < cols; ++w)
        {
            for (int c = 0; c < channel; ++c)
            {
                int hwc_i = h * cols * channel + w * channel + c;
                int chw_i = c * rows * cols + h * cols + w;
                hwc[hwc_i] = chw[chw_i];
            }
        }
    }
}

// kernel
void oihw2hwio(
    const float *oihw, float *hwio,
    const int out_c, const int in_c, const int rows, const int cols
)
{
    for (int h = 0; h < rows; ++h)
    {
        for (int w = 0; w < cols; ++w)
        {
            for (int ic = 0; ic < in_c; ++ic)
            {
                for (int oc = 0; oc < out_c; ++oc)
                {
                    int hwio_i = h * cols * in_c * out_c + w * in_c * out_c + ic * out_c + oc;
                    int oihw_i = oc * in_c * rows * cols + ic * rows * cols + h * cols + w;
                    hwio[hwio_i] = oihw[oihw_i];
                }
            }
        }
    }
}

int save_to_bin(
    const char *filename, const float *data, size_t num_elements
)
{
    FILE *fp = fopen(filename, "wb");
    if (!fp)
    {
        fprintf(stderr, "Failed opening file for writing\n");
        return -1;
    }
    size_t num_written = fwrite(data, sizeof(float), num_elements, fp);
    fclose(fp);

    if (num_written != num_elements)
    {
        fprintf(stderr, "Error: wrote %u/%u elements\n", num_written, num_elements);
        return -1;
    }
    return 0;
}

int main(int argc, char *argv[])
{
    // Usage: argv[0] in_c in_h in_w kn_size out_c stride padding
    /* define shapes */
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

    printf("in_c: %d, in_h: %d, in_w: %d, kn_size: %d, out_c: %d, stride: %d, padding: %d\n",
        in_c, in_h, in_w, kn_size, out_c, stride, padding);

    const int out_h = (in_h - kn_size + 2 * padding) / stride + 1;
    const int out_w = (in_w - kn_size + 2 * padding) / stride + 1;

    // nchw
    const int nchw_m = out_c;
    const int nchw_n = out_h * out_w;
    const int nchw_k = kn_size * kn_size * in_c;
    // nhwc
    const int nhwc_m = out_h * out_w;
    const int nhwc_n = out_c;
    const int nhwc_k = kn_size * kn_size * in_c;

    // sizes
    const int in_buf_size = in_c * in_h * in_w;
    const int kn_buf_size = kn_size * kn_size * in_c * out_c;
    const int col_buf_size = nchw_k * nchw_n;
    const int out_buf_size = out_c * out_h * out_w;

    // random seed
    srand(15);

    // create buffers for nchw layout
    float *nchw_in_buf = aligned_alloc(ALIGN_SIZE, sizeof(float) * in_buf_size);
    float *nchw_kn_buf = aligned_alloc(ALIGN_SIZE, sizeof(float) * kn_buf_size);
    float *nchw_col_buf = aligned_alloc(ALIGN_SIZE, sizeof(float) * col_buf_size);
    float *nchw_out_buf = aligned_alloc(ALIGN_SIZE, sizeof(float) * out_buf_size);
    if (!nchw_in_buf || !nchw_kn_buf || !nchw_col_buf || !nchw_out_buf)
    {
        fprintf(stderr, "Failed to create buffers\n");
        return -1;
    }
    // create buffers for nhwc layout
    float *nhwc_in_buf = aligned_alloc(ALIGN_SIZE, sizeof(float) * in_buf_size);
    float *nhwc_kn_buf = aligned_alloc(ALIGN_SIZE, sizeof(float) * kn_buf_size);
    float *nhwc_col_buf = aligned_alloc(ALIGN_SIZE, sizeof(float) * col_buf_size);
    float *nhwc_out_buf = aligned_alloc(ALIGN_SIZE, sizeof(float) * out_buf_size);
    if (!nhwc_in_buf || !nhwc_kn_buf || !nhwc_col_buf || !nhwc_out_buf)
    {
        fprintf(stderr, "Failed to create buffers\n");
        return -1;
    }

    /*
     * 1. NCHW LAYOUT
     */

    // init
    for (int i = 0; i < in_buf_size; ++i)
        nchw_in_buf[i] = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;  // float numbers in -1 to 1
    for (int i = 0; i < kn_buf_size; ++i)
        nchw_kn_buf[i] = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;

    // timer
    double start_time = omp_get_wtime();

    // im2col
    im2col_nchw(
        nchw_in_buf, nchw_col_buf,
        in_c, in_h, in_w,
        out_c, out_h, out_w,
        kn_size, kn_size, padding, stride
    );

    // gemm
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        nchw_m, nchw_n, nchw_k, 1.0f, nchw_kn_buf, nchw_k,
        nchw_col_buf, nchw_n, 0.0f, nchw_out_buf, nchw_n
    );

    printf("Elapsed time for NCHW layout: %.2f ms\n", (omp_get_wtime() - start_time) * 1000.0);

    /*
     * 2. NHWC LAYOUT
     */

    // init
    chw2hwc(nchw_in_buf, nhwc_in_buf, in_c, in_h, in_w);
    oihw2hwio(nchw_kn_buf, nhwc_kn_buf, out_c, in_c, kn_size, kn_size);

    // timer
    start_time = omp_get_wtime();
    
    // im2col
    im2col_nhwc(
        nhwc_in_buf, nhwc_col_buf,
        in_c, in_h, in_w,
        out_c, out_h, out_w,
        kn_size, kn_size, padding, stride
    );

    // gemm
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        nhwc_m, nhwc_n, nhwc_k, 1.0f, nhwc_col_buf, nhwc_k,
        nhwc_kn_buf, nhwc_n, 0.0f, nhwc_out_buf, nhwc_n
    );

    printf("Elapsed time for NHWC layout: %.2f ms\n", (omp_get_wtime() - start_time) * 1000.0);

    /* show results */
    if (out_c * out_h * out_w < 1024)
    {
        // nchw
        printf("nchw_out_buf (%d, %d, %d)\n", out_c, out_h, out_w);
        for (int c = 0; c < out_c; ++c)
        {
            for (int h = 0; h < out_h; ++h)
            {
                for (int w = 0; w < out_w; ++w)
                {
                    printf("%-8.3f", nchw_out_buf[c * out_h * out_w + h * out_w + w]);
                }
                printf("\t");
            }
            printf("\n");
        }
        // nhwc
        printf("nhwc_out_buf (%d, %d, %d)\n", out_h, out_w, out_c);
        for (int h = 0; h < out_h; ++h)
        {
            for (int w = 0; w < out_w; ++w)
            {
                for (int c = 0; c < out_c; ++c)
                {
                    printf("%-8.3f", nhwc_out_buf[h * out_w * out_c + w * out_c + c]);
                }
                printf("\t");
            }
            printf("\n");
        }
    }

    // save data for pytorch check
    FILE *fp = fopen("meta.txt", "w");
    if (!fp)
    {
        fprintf(stderr, "Failed saving results\n");
    }
    else
    {
        // save meta data
        fprintf(fp, "in_c: %d, in_h: %d, in_w: %d, kn_size: %d, out_c: %d, out_h: %d, out_w: %d, stride: %d, padding: %d\n",
            in_c, in_h, in_w, kn_size, out_c, out_h, out_w, stride, padding);
        fclose(fp);

        // save results for unit test
        save_to_bin("in_buf.bin", nchw_in_buf, in_buf_size);
        save_to_bin("kn_buf.bin", nchw_kn_buf, kn_buf_size);
        save_to_bin("nchw_out.bin", nchw_out_buf, out_buf_size);
        save_to_bin("nhwc_out.bin", nhwc_out_buf, out_buf_size);
    }

    // release resources
    free(nchw_in_buf);  nchw_in_buf = NULL;
    free(nchw_kn_buf);  nchw_kn_buf = NULL;
    free(nchw_col_buf); nchw_col_buf = NULL;
    free(nchw_out_buf); nchw_out_buf = NULL;
    free(nhwc_in_buf);  nhwc_in_buf = NULL;
    free(nhwc_kn_buf);  nhwc_kn_buf = NULL;
    free(nhwc_col_buf); nhwc_col_buf = NULL;
    free(nhwc_out_buf); nhwc_out_buf = NULL;

    return 0;
}
