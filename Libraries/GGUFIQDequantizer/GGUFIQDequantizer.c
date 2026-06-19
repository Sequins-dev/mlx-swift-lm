#include "GGUFIQDequantizer.h"

#include <string.h>

#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

static float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = ((uint32_t) h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t) h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t) h & 0x03ffu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp -= 1;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }

    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

int gguf_iq_block_size(GGUFIQType type) {
    switch (type) {
    case GGUFIQTypeIQ2XS:
        return sizeof(block_iq2_xs);
    case GGUFIQTypeIQ3XXS:
        return sizeof(block_iq3_xxs);
    case GGUFIQTypeIQ3S:
        return sizeof(block_iq3_s);
    case GGUFIQTypeIQ2S:
        return sizeof(block_iq2_s);
    case GGUFIQTypeIQ4XS:
        return sizeof(block_iq4_xs);
    }
    return 0;
}

static void dequantize_iq2_xs(const block_iq2_xs *x, float *y, int64_t value_count) {
    const int64_t block_count = value_count / QK_K;
    float db[2];

    for (int64_t i = 0; i < block_count; i++) {
        const float d = fp16_to_fp32(x[i].d);

        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            db[0] = d * (0.5f + (x[i].scales[ib32] & 0x0f)) * 0.25f;
            db[1] = d * (0.5f + (x[i].scales[ib32] >> 4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t *grid = (const uint8_t *) (iq2xs_grid + (x[i].qs[4 * ib32 + l] & 511));
                const uint8_t signs = ksigns_iq2xs[x[i].qs[4 * ib32 + l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db[l / 2] * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                }
                y += 8;
            }
        }
    }
}

static void dequantize_iq2_s(const block_iq2_s *x, float *y, int64_t value_count) {
    const int64_t block_count = value_count / QK_K;
    float db[2];

    for (int64_t i = 0; i < block_count; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const uint8_t *qs = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const uint8_t *signs = qs + QK_K / 8;

        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            db[0] = d * (0.5f + (x[i].scales[ib32] & 0x0f)) * 0.25f;
            db[1] = d * (0.5f + (x[i].scales[ib32] >> 4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const float dl = db[l / 2];
                const uint8_t *grid = (const uint8_t *) (iq2s_grid + (qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl * grid[j] * ((signs[l] & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                }
                y += 8;
            }
            qs += 4;
            signs += 4;
        }
    }
}

static void dequantize_iq3_xxs(const block_iq3_xxs *x, float *y, int64_t value_count) {
    const int64_t block_count = value_count / QK_K;

    for (int64_t i = 0; i < block_count; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const uint8_t *qs = x[i].qs;
        const uint8_t *scales_and_signs = qs + QK_K / 4;

        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            uint32_t aux32;
            memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
            const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
                const uint8_t *grid1 = (const uint8_t *) (iq3xxs_grid + qs[2 * l + 0]);
                const uint8_t *grid2 = (const uint8_t *) (iq3xxs_grid + qs[2 * l + 1]);
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db * grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.0f : 1.0f);
                    y[j + 4] = db * grid2[j] * ((signs & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
                }
                y += 8;
            }
            qs += 8;
        }
    }
}

static void dequantize_iq3_s(const block_iq3_s *x, float *y, int64_t value_count) {
    const int64_t block_count = value_count / QK_K;

    for (int64_t i = 0; i < block_count; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const uint8_t *qs = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const uint8_t *signs = x[i].signs;

        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const float db1 = d * (1 + 2 * (x[i].scales[ib32 / 2] & 0x0f));
            const float db2 = d * (1 + 2 * (x[i].scales[ib32 / 2] >> 4));

            for (int l = 0; l < 4; ++l) {
                const uint8_t *grid1 = (const uint8_t *) (iq3s_grid + (qs[2 * l + 0] | ((qh[0] << (8 - 2 * l)) & 256)));
                const uint8_t *grid2 = (const uint8_t *) (iq3s_grid + (qs[2 * l + 1] | ((qh[0] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db1 * grid1[j] * ((signs[l] & kmask_iq2xs[j + 0]) ? -1.0f : 1.0f);
                    y[j + 4] = db1 * grid2[j] * ((signs[l] & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
                }
                y += 8;
            }

            qs += 8;
            signs += 4;
            for (int l = 0; l < 4; ++l) {
                const uint8_t *grid1 = (const uint8_t *) (iq3s_grid + (qs[2 * l + 0] | ((qh[1] << (8 - 2 * l)) & 256)));
                const uint8_t *grid2 = (const uint8_t *) (iq3s_grid + (qs[2 * l + 1] | ((qh[1] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db2 * grid1[j] * ((signs[l] & kmask_iq2xs[j + 0]) ? -1.0f : 1.0f);
                    y[j + 4] = db2 * grid2[j] * ((signs[l] & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
                }
                y += 8;
            }

            qh += 2;
            qs += 8;
            signs += 4;
        }
    }
}

static void dequantize_iq4_xs(const block_iq4_xs *x, float *y, int64_t value_count) {
    const int64_t block_count = value_count / QK_K;

    for (int64_t i = 0; i < block_count; i++) {
        const uint8_t *qs = x[i].qs;
        const float d = fp16_to_fp32(x[i].d);

        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const int ls = ((x[i].scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0f) | (((x[i].scales_h >> (2 * ib)) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j + 0] = dl * kvalues_iq4nl[qs[j] & 0x0f];
                y[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
            }
            y += 32;
            qs += 16;
        }
    }
}

int gguf_iq_dequantize(GGUFIQType type, const uint8_t *input, float *output, int64_t value_count) {
    if (value_count % QK_K != 0) {
        return 0;
    }

    switch (type) {
    case GGUFIQTypeIQ2XS:
        dequantize_iq2_xs((const block_iq2_xs *) input, output, value_count);
        return 1;
    case GGUFIQTypeIQ3XXS:
        dequantize_iq3_xxs((const block_iq3_xxs *) input, output, value_count);
        return 1;
    case GGUFIQTypeIQ3S:
        dequantize_iq3_s((const block_iq3_s *) input, output, value_count);
        return 1;
    case GGUFIQTypeIQ2S:
        dequantize_iq2_s((const block_iq2_s *) input, output, value_count);
        return 1;
    case GGUFIQTypeIQ4XS:
        dequantize_iq4_xs((const block_iq4_xs *) input, output, value_count);
        return 1;
    }
    return 0;
}
