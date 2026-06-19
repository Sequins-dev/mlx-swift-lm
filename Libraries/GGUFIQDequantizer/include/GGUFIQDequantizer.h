#pragma once

#include <stdint.h>

typedef enum GGUFIQType {
    GGUFIQTypeIQ2XS = 17,
    GGUFIQTypeIQ3XXS = 18,
    GGUFIQTypeIQ3S = 21,
    GGUFIQTypeIQ2S = 22,
    GGUFIQTypeIQ4XS = 23,
} GGUFIQType;

int gguf_iq_block_size(GGUFIQType type);
int gguf_iq_dequantize(GGUFIQType type, const uint8_t *input, float *output, int64_t value_count);
