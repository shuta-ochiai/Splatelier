// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// 5×5 Census Transform (PocketGS Sec. III-A 3, Fife & Archibald 2013). Matching cost for MVS.
// census_transform: compares the 24 non-center neighbors against the center, 24 bits (LSB-first,
// row-major).

StructuredBuffer<float> luma;       // [H * W] luma
StructuredBuffer<int>   dims;       // [W, H]

RWStructuredBuffer<int> census;     // [H * W]

[shader("compute")]
[numthreads(16, 16, 1)]
void census_transform(uint3 tid : SV_DispatchThreadID)
{
    int imageWidth  = dims[0];
    int imageHeight = dims[1];
    int x = int(tid.x);
    int y = int(tid.y);
    if (x >= imageWidth || y >= imageHeight) { return; }

    int pix = y * imageWidth + x;
    float c = luma[pix];
    int bits = 0;
    int bit = 0;

    // Clamp coordinates at the image border (replicate)
    [unroll]
    for (int dy = -2; dy <= 2; dy++) {
        [unroll]
        for (int dx = -2; dx <= 2; dx++) {
            if (dx == 0 && dy == 0) { continue; }
            int nx = clamp(x + dx, 0, imageWidth - 1);
            int ny = clamp(y + dy, 0, imageHeight - 1);
            float n = luma[ny * imageWidth + nx];
            if (n >= c) {
                bits = bits | (1 << bit);
            }
            bit = bit + 1;
        }
    }
    census[pix] = bits;
}

// ---------------------------------------------------------------------------
// Center-Mean Census Transform (Fife & Archibald 2013 Sec. III). Instead of the center pixel,
// compares all 25 pixels against the 5×5 mean, so noise in the center pixel does not corrupt every
// bit. 25 bits.
// ---------------------------------------------------------------------------

[shader("compute")]
[numthreads(16, 16, 1)]
void improved_census_transform(uint3 tid : SV_DispatchThreadID)
{
    int imageWidth  = dims[0];
    int imageHeight = dims[1];
    int x = int(tid.x);
    int y = int(tid.y);
    if (x >= imageWidth || y >= imageHeight) { return; }

    // Read the 25 pixels into registers once; use them for both the mean and the bit assembly
    float win[25];
    float sum = 0.0;
    [unroll]
    for (int dy = -2; dy <= 2; dy++) {
        [unroll]
        for (int dx = -2; dx <= 2; dx++) {
            int nx = clamp(x + dx, 0, imageWidth - 1);
            int ny = clamp(y + dy, 0, imageHeight - 1);
            float v = luma[ny * imageWidth + nx];
            int slot = (dy + 2) * 5 + (dx + 2);
            win[slot] = v;
            sum += v;
        }
    }
    float m = sum * (1.0 / 25.0);

    int bits = 0;
    [unroll]
    for (int slot = 0; slot < 25; slot++) {
        if (win[slot] >= m) {
            bits = bits | (1 << slot);
        }
    }
    census[int(tid.y) * imageWidth + int(tid.x)] = bits;
}
