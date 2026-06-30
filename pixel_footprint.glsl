// Scanline Sweeper — Total Pixel Footprint Compute Shader
// ========================================================
//
// This compute shader processes a glyph's bezier curves and computes per-pixel
// coverage using the scanline sweep algorithm.
//
// Each compute thread handles one pixel.  A local workgroup of 8x8 is used so
// that shared memory can cache a batch of curves for the tile.  The shader
// sums the signed coverage contribution of every curve that overlaps the pixel,
// then writes the final alpha value.
//
// Data flow:
//   1. CPU uploads pre-subdivided (y-monotonic) quadratic bezier curves.
//   2. This shader reads the curve buffer, sweeps each curve against the pixel,
//      and accumulates coverage.
//   3. The output image receives the per-pixel alpha.
//
// Requires: common.glsl, bezier_intersect.glsl, scanline_sweep.glsl

#version 460 core

// ---------------------------------------------------------------------------
// Shared helpers (normally these would be #included)
// ---------------------------------------------------------------------------
#define mad(a, b, c) ((a) * (b) + (c))
#define saturate(x) clamp((x), 0.0, 1.0)

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

// Each quadratic bezier segment is stored as 6 floats: p0.xy, p1.xy, p2.xy.
// The buffer is expected to contain only y-monotonic segments (pre-split).
// Curves are sorted by their first scanline intersection.
layout(std430, binding = 0) readonly buffer CurveBuffer {
    float curve_data[];  // 6 floats per curve
};

// Uniform parameters for the dispatch.
layout(std140, binding = 1) uniform Params {
    uint  curve_count;       // number of curves in the buffer
    uint  image_width;       // output image width in pixels
    uint  image_height;      // output image height in pixels
    float funits_per_px;     // font-units per pixel (uniform scale)
    vec2  glyph_origin;      // lower-left origin of the glyph in font units
};

// Output image — one channel (red) holds the final coverage / alpha.
layout(rgba8, binding = 2) writeonly uniform image2D output_image;

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// ---------------------------------------------------------------------------
// Helper: unpack curve i from the flat float array.
// ---------------------------------------------------------------------------
void load_curve(uint index, out vec2 p0, out vec2 p1, out vec2 p2)
{
    uint base = index * 6u;
    p0 = vec2(curve_data[base + 0u], curve_data[base + 1u]);
    p1 = vec2(curve_data[base + 2u], curve_data[base + 3u]);
    p2 = vec2(curve_data[base + 4u], curve_data[base + 5u]);
}

// ---------------------------------------------------------------------------
// Forward declarations of algorithm functions (bezier_intersect.glsl + scanline_sweep.glsl).
// ---------------------------------------------------------------------------
float intersect_monotonic_bezier(float qa, float c0, float c1, float c2,
                                 float target, float delta_sign);

vec2 evaluate_bezier(vec2 p0, vec2 p1, vec2 p2, float t);

// ACCURATE_COVERAGE is set to 1 to use the continuous-integration path,
// or 0 for the trapezoidal approximation.
#define ACCURATE_COVERAGE 1

float scanline_sweep(vec2 pixel_size, vec2 pixel_offset,
                     vec2 p0, vec2 p1, vec2 p2);

// ---------------------------------------------------------------------------
// intersect_monotonic_bezier
// ---------------------------------------------------------------------------
float intersect_monotonic_bezier(
    float qa, float c0, float c1, float c2,
    float target, float delta_sign)
{
    if (abs(qa) < 1e-2)
    {
        return (target - c0) / (c2 - c0);
    }

    float qb = mad(2.0, c1, -2.0 * c0);
    float qc = c0 - target;
    float d  = mad(qb, qb, -4.0 * qa * qc);
    float sqrt_d = d < 0.0 ? 0.0 : sqrt(d);
    float inv_2a = 0.5 / qa;

    return mad(-qb, inv_2a, delta_sign * sqrt_d * inv_2a);
}

// ---------------------------------------------------------------------------
// evaluate_bezier
// ---------------------------------------------------------------------------
vec2 evaluate_bezier(vec2 p0, vec2 p1, vec2 p2, float t)
{
    float mt = 1.0 - t;
    float a  = mt * mt;
    float b  = 2.0 * mt * t;
    float c  = t * t;
    return a * p0 + b * p1 + c * p2;
}

// ---------------------------------------------------------------------------
// scanline_sweep  (full implementation)
// ---------------------------------------------------------------------------
float scanline_sweep(
    vec2 pixel_size,
    vec2 pixel_offset,
    vec2 p0,
    vec2 p1,
    vec2 p2)
{
    // Discard curves entirely above or below the scanline.
    if (max(p0.y, p2.y) <= pixel_offset.y ||
        min(p0.y, p2.y) >= pixel_offset.y + pixel_size.y)
    {
        return 0.0;
    }

    vec2 delta = p2 - p0;

    p0 -= pixel_offset;
    p1 -= pixel_offset;
    p2 -= pixel_offset;

    // Fast path: strictly vertical segments.
    if (p0.x == p1.x && p0.x == p2.x)
    {
        if (p0.x >= pixel_size.x) { return 0.0; }

        float top    = min(max(p0.y, p2.y), pixel_size.y);
        float bottom = max(min(p0.y, p2.y), 0.0);
        float height = top - bottom;

        if (p0.x <= 0.0)
        {
            return sign(delta.y) * pixel_size.x * height;
        }

        float base = pixel_size.x - p0.x;
        return sign(delta.y) * base * height;
    }

    // Y-boundary intersections.
    float qa_y = mad(-2.0, p1.y, p0.y + p2.y);
    float bt = intersect_monotonic_bezier(qa_y, p0.y, p1.y, p2.y, 0.0, sign(delta.y));
    float tt = intersect_monotonic_bezier(qa_y, p0.y, p1.y, p2.y, pixel_size.y, sign(delta.y));

    float v_min = delta.y > 0.0 ? bt : tt;
    float v_max = delta.y > 0.0 ? tt : bt;
    vec2  v_min_crossing = evaluate_bezier(p0, p1, p2, saturate(v_min));
    vec2  v_max_crossing = evaluate_bezier(p0, p1, p2, saturate(v_max));

    if (max(v_min_crossing.x, v_max_crossing.x) <= 0.0)
    {
        return (v_max_crossing.y - v_min_crossing.y) * pixel_size.x;
    }

    if (min(v_min_crossing.x, v_max_crossing.x) >= pixel_size.x)
    {
        return 0.0;
    }

    // X-boundary intersections.
    float qa_x    = mad(-2.0, p1.x, p0.x + p2.x);
    float dx_sign = sign(delta.x);

    float h_min, h_max;

    vec4 h_check = delta.x > 0.0
        ? vec4(p0.x, p2.x, 0.0, 0.0)
        : vec4(p2.x, p0.x, pixel_size.x, 1.0);

    if (h_check.x >= h_check.z)
    {
        h_min = h_check.w;
    }
    else if (h_check.y <= h_check.z)
    {
        h_min = 1.0 - h_check.w;
    }
    else
    {
        h_min = intersect_monotonic_bezier(qa_x, p0.x, p1.x, p2.x, h_check.z, dx_sign);
    }

    float h_target_max = delta.x > 0.0 ? pixel_size.x : 0.0;

    if (h_check.x >= h_target_max)
    {
        h_max = h_check.w;
    }
    else if (h_check.y <= h_target_max)
    {
        h_max = 1.0 - h_check.w;
    }
    else
    {
        h_max = intersect_monotonic_bezier(qa_x, p0.x, p1.x, p2.x, h_target_max, dx_sign);
    }

    float t_min = saturate(max(v_min, h_min));
    float t_max = saturate(min(v_max, h_max));

    vec2 q0 = v_min >= h_max ? v_min_crossing : evaluate_bezier(p0, p1, p2, t_min);
    vec2 q1 = v_max <= h_min ? v_max_crossing : evaluate_bezier(p0, p1, p2, t_max);

    float coverage = 0.0;

    // External contributions.
    if (t_min > 0.0 && delta.x > 0.0)
    {
        float h = delta.y > 0.0
            ? q0.y - max(0.0, p0.y)
            : min(pixel_size.y, p0.y) - q0.y;
        coverage = sign(delta.y) * h * pixel_size.x;
    }

    if (t_max < 1.0 && delta.x > 0.0)
    {
        float h = delta.y > 0.0
            ? min(pixel_size.y, p2.y) - q1.y
            : q1.y - max(0.0, p2.y);
        coverage += sign(delta.y) * h * pixel_size.x;
    }

#if ACCURATE_COVERAGE
    // Continuous integration of bezier area.
    float px = pixel_size.x;

    float ax = qa_x;
    float bx = mad(2.0, p1.x, -2.0 * p0.x);
    float cx = p0.x;

    float ay = qa_y;
    float by = mad(2.0, p1.y, -2.0 * p0.y);
    float cy = p0.y;

    float c0 = -0.5 * ax * ay;
    float c1 = -(2.0 * bx * ay + ax * by) / 3.0;
    float c2 = ay * (px - cx) - 0.5 * bx * by;
    float c3 = (px - cx) * by;

    float ft_max = t_max * (c3 + t_max * (c2 + t_max * (c1 + t_max * c0)));
    float ft_min = t_min * (c3 + t_min * (c2 + t_min * (c1 + t_min * c0)));

    coverage += ft_max - ft_min;
#else
    // Trapezoidal approximation.
    float h = q1.y - q0.y;
    float b = mad(-0.5, q0.x + q1.x, pixel_size.x);
    coverage += b * h;
#endif

    return coverage;
}

// ===========================================================================
// Main compute entry point — one thread per pixel
// ===========================================================================
void main()
{
    // Global pixel coordinate in the output image.
    uvec2 pixel_coord = gl_GlobalInvocationID.xy;

    if (pixel_coord.x >= image_width || pixel_coord.y >= image_height)
    {
        return;
    }

    // Compute the lower-left corner of this pixel in font-unit space.
    vec2 pixel_offset = glyph_origin
        + vec2(float(pixel_coord.x), float(pixel_coord.y)) * funits_per_px;

    vec2 pixel_size = vec2(funits_per_px);

    // Accumulate signed coverage from every curve.
    float total_coverage = 0.0;

    for (uint i = 0u; i < curve_count; ++i)
    {
        vec2 p0, p1, p2;
        load_curve(i, p0, p1, p2);

        total_coverage += scanline_sweep(pixel_size, pixel_offset, p0, p1, p2);
    }

    // Clamp to [0, 1] — the winding rule produces a continuous coverage value.
    // Anything > 1 saturates (multiple overlapping paths).
    float alpha = saturate(total_coverage);

    // Write: pack alpha into R8 channel, G/B/A set to 1/0.
    vec4 color = vec4(alpha, alpha, alpha, 1.0);
    imageStore(output_image, ivec2(pixel_coord), color);
}
