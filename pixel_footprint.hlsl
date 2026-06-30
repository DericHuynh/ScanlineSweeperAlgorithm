// Scanline Sweeper — Total Pixel Footprint Compute Shader (HLSL)
// ===============================================================
//
// HLSL compute shader that processes a glyph's bezier curves and computes
// per-pixel coverage using the scanline sweep algorithm.
//
// Requires: bezier_intersect.hlsl, scanline_sweep.hlsl (included or linked)
//
// Data flow:
//   1. CPU uploads pre-subdivided (y-monotonic) quadratic bezier curves.
//   2. Each thread sweeps every curve against its pixel, accumulates coverage.
//   3. Output texture receives the per-pixel alpha.

#define ACCURATE_COVERAGE 1

// ---------------------------------------------------------------------------
// Constant buffer
// ---------------------------------------------------------------------------
cbuffer Params : register(b0)
{
    uint  curve_count;       // number of curves in the buffer
    uint  image_width;       // output image width in pixels
    uint  image_height;      // output image height in pixels
    float funits_per_px;     // font-units per pixel scale
    float2 glyph_origin;     // lower-left origin of the glyph (font units)
};

// ---------------------------------------------------------------------------
// Curve buffer (structured buffer of packed curves)
// ---------------------------------------------------------------------------
struct Curve
{
    float2 p0;
    float2 p1;
    float2 p2;
};

StructuredBuffer<Curve> curves : register(t0);

// ---------------------------------------------------------------------------
// Output texture
// ---------------------------------------------------------------------------
RWTexture2D<float4> output_image : register(u0);

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------
float  intersect_monotonic_bezier(float qa, float c0, float c1, float c2, float target, float delta_sign);
float2 evaluate_bezier(float2 p0, float2 p1, float2 p2, float t);
float  scanline_sweep(float2 pixel_size, float2 pixel_offset, float2 p0, float2 p1, float2 p2);

// ---------------------------------------------------------------------------
// Main — one thread per pixel
// ---------------------------------------------------------------------------
[numthreads(8, 8, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID)
{
    uint2 pixel_coord = dispatch_id.xy;

    if (pixel_coord.x >= image_width || pixel_coord.y >= image_height)
    {
        return;
    }

    // Compute the lower-left corner of this pixel in font-unit space.
    float2 pixel_offset = glyph_origin
        + float2(float(pixel_coord.x), float(pixel_coord.y)) * funits_per_px;

    float2 pixel_size = float2(funits_per_px, funits_per_px);

    // Accumulate signed coverage from every curve.
    float total_coverage = 0.f;

    for (uint i = 0; i < curve_count; ++i)
    {
        Curve c = curves[i];
        total_coverage += scanline_sweep(pixel_size, pixel_offset, c.p0, c.p1, c.p2);
    }

    // Clamp to [0, 1] — the winding rule produces continuous coverage.
    float alpha = saturate(total_coverage);

    output_image[pixel_coord] = float4(alpha, alpha, alpha, 1.f);
}
