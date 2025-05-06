#ifndef IM2COL_H_
#define IM2COL_H_

static inline void im2col_nchw(
    const float *data_im, float *data_col,
    const int in_c, const int in_h, const int in_w,
    const int out_c, const int out_h, const int out_w,
    const int kn_h, const int kn_w, const int padding, const int stride
)
{
    for (int ch = 0; ch < in_c; ++ch)
    {
        for (int kh = 0; kh < kn_h; ++kh)
        {
            for (int kw = 0; kw < kn_w; ++kw)
            {
                int row_i = ch * kn_h * kn_w + kh * kn_w + kw;
                int col_i = 0;
                for (int oh = 0; oh < out_h; ++oh)
                {
                    int ih = oh * stride + kh - padding;
                    for (int ow = 0; ow < out_w; ++ow)
                    {
                        int iw = ow * stride + kw - padding;
                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w)
                            data_col[row_i * (out_h * out_w) + col_i++] = data_im[ch * (in_h * in_w) + ih * in_w + iw];
                        else
                            data_col[row_i * (out_h * out_w) + col_i++] = 0.0f;
                    }
                }
            }
        }
    }
}

static inline void im2col_nhwc(
    const float *data_im, float *data_col,
    const int in_c, const int in_h, const int in_w,
    const int out_c, const int out_h, const int out_w,
    const int kn_h, const int kn_w, const int padding, const int stride
)
{
    int col_i = 0;
    for (int oh = 0; oh < out_h; ++oh)
    {
        for (int ow = 0; ow < out_w; ++ow)
        {
            // top left corner of data_im
            int im_tl_h = oh * stride - padding;
            int im_tl_w = ow * stride - padding;

            for (int kh = 0; kh < kn_h; ++kh)
            {
                for (int kw = 0; kw < kn_w; ++kw)
                {
                    int im_h = im_tl_h + kh;
                    int im_w = im_tl_w + kw;
                    if (im_h >= 0 && im_h < in_h && im_w >= 0 && im_w < in_w)
                        memcpy(&data_col[col_i], &data_im[(im_h * in_w + im_w) * in_c], sizeof(float) * in_c);
                    else
                        memset(&data_col[col_i], 0, sizeof(float) * in_c);
                    col_i += in_c;
                }
            }
        }
    }
}

#endif  // IM2COL_H_
