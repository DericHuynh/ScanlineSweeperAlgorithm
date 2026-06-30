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

float intersect_monotonic(
    float qa, float c0, float c1, float c2,
    float target)
{
    if (abs(qa) < 1e-3)
    {
        return (target - c0) / (c2 - c0);
    }

    float qb = mad(2.0, c1, -2.0 * c0);
    float qc = c0 - target;
    float d  = mad(qb, qb, -4.0 * qa * qc);
    float sqrt_d = d < 0.0 ? 0.0 : sqrt(d);
    float inv_2a = 0.5 / qa;

    return mad(-qb, inv_2a, sign(c2 - c0) * sqrt_d * inv_2a);
}

vec2 evaluate_bezier(vec2 p0, vec2 p1, vec2 p2, float t)
{
    vec2 a = mix(p0, p1, t);
    vec2 b = mix(p1, p2, t);
    return mix(a, b, t);
}

float scanline_sweep(
    vec2 size,
    vec2 offset,
    vec2 p0,
    vec2 p1,
    vec2 p2)
{
    // Discard curves above or below the scanline.
    if (max(p0.y, p2.y) <= offset.y || min(p0.y, p2.y) >= offset.y + size.y)
    {
        return 0.0;
    }

    vec2 delta = p2 - p0;

    // Shift all control points to a coordinate system with the
    // window at the origin.
    p0 -= offset;
    p1 -= offset;
    p2 -= offset;

    // Fast path for strictly vertical segments, common in many fonts.
    if (p0.x == p1.x && p0.x == p2.x)
    {
        if (p0.x >= size.x)
        {
            // Segment is to the right of the window. Nothing to do.
            return 0.0;
        }

        float t = min(max(p0.y, p2.y), size.y);
        float b = max(min(p0.y, p2.y), 0.0);
        float h = t - b;
        float w = min(size.x, size.x - p0.x);

        // Signed area of the swept rectangle.
        return sign(delta.y) * w * h;
    }

    // qa is the second-degree coefficient for the y-coordinate
    // quadratic.
    float qa = mad(-2.0, p1.y, p0.y + p2.y);

    float bt = intersect_monotonic(qa, p0.y, p1.y, p2.y, 0.0);
    float tt = intersect_monotonic(qa, p0.y, p1.y, p2.y, size.y);

    // v_min_t and v_max_t are the crossings where the curve enters
    // and exits the scanline.
    float v_min_t = delta.y > 0.0 ? bt : tt;
    float v_max_t = delta.y > 0.0 ? tt : bt;

    vec2 v_min = evaluate_bezier(p0, p1, p2, saturate(v_min_t));
    vec2 v_max = evaluate_bezier(p0, p1, p2, saturate(v_max_t));

    if (max(v_min.x, v_max.x) <= 0.0)
    {
        // Fast path for curves entirely to the left of the window
        // within the scanline. Note that the area sign is
        // incorporated in the result.
        return (v_max.y - v_min.y) * size.x;
    }

    if (min(v_min.x, v_max.x) >= size.x)
    {
        // The curve is entirely to the right of the window within
        // the scanline, so it can be ignored.
        return 0.0;
    }

    // Solve for roots along x.
    qa = mad(-2.0, p1.x, p0.x + p2.x);

    // As with v_min_t and v_max_t, we now need the values of t where
    // the curve enters and exits the window moving horizontally.
    float h_min_t;
    float h_max_t;

    // This check vector stores the following quantities in each component:
    // - lower x bound
    // - upper x bound
    // - target value
    // - parameter associated with the lower x bound (0 or 1)
    //
    // Packing the values in this way simplifies bounds checks and intersection
    // testing, and the values depend on the direction the curve moves.
    vec4 h_check = delta.x > 0.0
        ? vec4(p0.x, p2.x, 0.0, 0.0)
        : vec4(p2.x, p0.x, size.x, 1.0);

    if (h_check.x >= h_check.z)
    {
        h_min_t = h_check.w;
    }
    else if (h_check.y <= h_check.z)
    {
        h_min_t = 1.0 - h_check.w;
    }
    else
    {
        h_min_t = intersect_monotonic(qa, p0.x, p1.x, p2.x, h_check.z);
    }

    h_check.z = size.x - h_check.z;

    if (h_check.x >= h_check.z)
    {
        h_max_t = h_check.w;
    }
    else if (h_check.y <= h_check.z)
    {
        h_max_t = 1.0 - h_check.w;
    }
    else
    {
        h_max_t = intersect_monotonic(qa, p0.x, p1.x, p2.x, h_check.z);
    }

    // Now, we can compute the values of t for which the curve enters
    // and leaves the window in any direction. Note that these values
    // are constrained to the unit interval, so it's ok if the curve
    // stops or ends within the window.
    float min_t = saturate(max(v_min_t, h_min_t));
    float max_t = saturate(min(v_max_t, h_max_t));

    // Evaluate the curve at new intersection points if needed based
    // on the newly constrained interval.
    vec2 q0 = v_min_t >= h_min_t ? v_min : evaluate_bezier(p0, p1, p2, min_t);
    vec2 q1 = v_max_t <= h_max_t ? v_max : evaluate_bezier(p0, p1, p2, max_t);

    float coverage = 0.0;

    if (min_t > 0.0 && delta.x > 0.0)
    {
        // We enter the pixel from the left, so we need to integrate the
        // swept rectangle below the entry point.
        float h = delta.y > 0.0
            ? q0.y - max(0.0, p0.y)
            : min(size.y, p0.y) - q0.y;
        coverage = sign(delta.y) * h * size.x;
    }

    if (max_t < 1.0 && delta.x < 0.0)
    {
        // We exit the pixel on the left side, so we need to integrate the
        // swept rectangle after the exit point.
        float h = delta.y > 0.0
            ? min(size.y, p2.y) - q1.y
            : q1.y - max(0.0, p2.y);
        coverage += sign(delta.y) * h * size.x;
    }

#if ACCURATE_COVERAGE
    // Continuous integration of bezier area.
    float px = size.x;

    float ax = qa;
    float bx = mad(2.0, p1.x, -2.0 * p0.x);
    float cx = p0.x;

    float ay = mad(-2.0, p1.y, p0.y + p2.y);
    float by = mad(2.0, p1.y, -2.0 * p0.y);
    float cy = p0.y;

    float c0 = -0.5 * ax * ay;
    float c1 = -(2.0 * bx * ay + ax * by) / 3.0;
    float c2 = ay * (px - cx) - 0.5 * bx * by;
    float c3 = (px - cx) * by;

    float ft_max = max_t * (c3 + max_t * (c2 + max_t * (c1 + max_t * c0)));
    float ft_min = min_t * (c3 + min_t * (c2 + min_t * (c1 + min_t * c0)));

    coverage += ft_max - ft_min;
#else
    // This implements the simple trapezoidal approximation for the
    // portion of the curve within the window.
    float h = q1.y - q0.y;

    // Sum of trapezoid bases divided by two. If q0.x or q1.x happen
    // to equal size.x, the trapezoidal area is effectively a triangle.
    float b = mad(-0.5, q0.x + q1.x, size.x);

    coverage += b * h;
#endif

    // The caller is expected to accumulate this coverage for each curve,
    // and divide the final result by the window area.
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
