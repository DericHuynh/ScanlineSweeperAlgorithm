// Scanline Sweeper — Tiled Pixel Footprint (GLSL Compute)
// =========================================================
//
// Optimised variant that uses shared memory to cache a batch of curves
// per workgroup, dramatically reducing global-memory bandwidth.
//
// Architecture:
//   - Each workgroup (8x8 threads) processes a tile of pixels.
//   - Curves are streamed through shared memory in batches.
//   - Every thread in the group processes the same batch against its pixel,
//     then the group advances to the next batch.
//   - Coverage is accumulated in registers and written once at the end.
//
// This is the recommended shipping variant.
//
// Requires: common.glsl, bezier_intersect.glsl, scanline_sweep.glsl

#version 460 core

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------
#define mad(a, b, c) ((a) * (b) + (c))
#define saturate(x) clamp((x), 0.0, 1.0)

// ---------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------
#define ACCURATE_COVERAGE  1    // 1 = continuous integration, 0 = trapezoidal approx
#define CURVE_BATCH_SIZE  64    // curves cached per workgroup iteration
#define TILE_SIZE_X        8
#define TILE_SIZE_Y        8

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

// Curve buffer: 6 floats per curve (p0.xy, p1.xy, p2.xy).
// Pre-condition: curves are pre-subdivided into y-monotonic segments and
//                sorted by their minimum y-extent for early culling.
layout(std430, binding = 0) readonly buffer CurveBuffer {
    float curve_data[];
};

layout(std140, binding = 1) uniform Params {
    uint  curve_count;
    uint  image_width;
    uint  image_height;
    float funits_per_px;
    vec2  glyph_origin;
};

layout(rgba8, binding = 2) writeonly uniform image2D output_image;

// ---------------------------------------------------------------------------
// Shared memory — caches one batch of curves for the workgroup.
// ---------------------------------------------------------------------------
shared vec2 s_curves[CURVE_BATCH_SIZE * 3];  // p0, p1, p2 per curve

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------
layout(local_size_x = TILE_SIZE_X, local_size_y = TILE_SIZE_Y, local_size_z = 1) in;

// ---------------------------------------------------------------------------
// Helper: cooperative load of a curve batch into shared memory.
// ---------------------------------------------------------------------------
void load_curve_batch(uint batch_start, uint batch_count)
{
    // Each thread loads a subset of the float data.
    uint total_floats = batch_count * 6u;
    uint tid = gl_LocalInvocationIndex;  // 0 .. 63

    for (uint i = tid; i < total_floats; i += TILE_SIZE_X * TILE_SIZE_Y)
    {
        uint curve_idx   = batch_start + (i / 6u);
        uint comp        = i % 6u;
        uint global_base = curve_idx * 6u + comp;
        uint local_base  = (curve_idx - batch_start) * 6u + comp;

        // Reinterpret as vec2 array index.
        s_curves[local_base / 2u][local_base % 2u] =
            (curve_idx < curve_count) ? curve_data[global_base] : 0.0;
    }

    barrier();
    memoryBarrierShared();
}

// ---------------------------------------------------------------------------
// Helper: read one curve from shared memory.
// ---------------------------------------------------------------------------
void read_curve_shared(uint local_index, out vec2 p0, out vec2 p1, out vec2 p2)
{
    uint base = local_index * 3u;
    p0 = s_curves[base + 0u];
    p1 = s_curves[base + 1u];
    p2 = s_curves[base + 2u];
}

// ===========================================================================
// Algorithm functions (inlined for the self-contained shader)
// ===========================================================================

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

vec2 evaluate_bezier(vec2 p0, vec2 p1, vec2 p2, float t)
{
    float mt = 1.0 - t;
    float a  = mt * mt;
    float b  = 2.0 * mt * t;
    float c  = t * t;
    return a * p0 + b * p1 + c * p2;
}

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
// Main — tiled compute entry point
// ===========================================================================
void main()
{
    uvec2 pixel_coord = gl_GlobalInvocationID.xy;

    // Pre-compute pixel parameters (constant across all curve batches).
    vec2 pixel_offset = glyph_origin
        + vec2(float(pixel_coord.x), float(pixel_coord.y)) * funits_per_px;
    vec2 pixel_size = vec2(funits_per_px);

    float total_coverage = 0.0;

    // Stream curves through shared memory in batches.
    for (uint batch_start = 0u; batch_start < curve_count; batch_start += CURVE_BATCH_SIZE)
    {
        uint batch_count = min(CURVE_BATCH_SIZE, curve_count - batch_start);

        // Cooperative load of this batch into shared memory.
        load_curve_batch(batch_start, batch_count);

        // Each thread processes this batch against its pixel.
        // Note: p0/p1/p2 must be copied (not aliased) since scanline_sweep
        // modifies them internally.
        for (uint i = 0u; i < batch_count; ++i)
        {
            vec2 p0, p1, p2;
            read_curve_shared(i, p0, p1, p2);

            total_coverage += scanline_sweep(pixel_size, pixel_offset, p0, p1, p2);
        }

        barrier();  // ensure all threads finish before loading next batch
    }

    // Write final alpha.
    if (pixel_coord.x < image_width && pixel_coord.y < image_height)
    {
        float alpha = saturate(total_coverage);
        vec4 color  = vec4(alpha, alpha, alpha, 1.0);
        imageStore(output_image, ivec2(pixel_coord), color);
    }
}
