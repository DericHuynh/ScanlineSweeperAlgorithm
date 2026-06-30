
// Preconditions:
// - size is the size of the window to be considered
// - offset is the position of the lower-left window corner
// - p0, p1, and p2 are control points of a monotonic Bézier curve
//
// The curve must be monotonic, such that all(p0 <= p1) and
// all(p1 <= p2) are true.
//
// Returns the estimated covered area of the window if the curve
// is swept to the right. This is a signed quantity that is
// positive for upwards moving curves, and negative for downwards
// moving curves.
float scanline_sweep(
    float2 size,
    float2 offset,
    float2 p0,
    float2 p1,
    float2 p2)
{
    // Discard curves above or below the scanline.
    if (max(p0.y, p2.y) <= offset.y || min(p0.y, p2.y) >= offset.y + size.y)
    {
        return 0.f;
    }

    float2 delta = p2 - p0;

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
            return 0.f;
        }

        float t = min(max(p0.y, p2.y), size.y);
        float b = max(min(p0.y, p2.y), 0.f);
        float h = t - b;
        float w = min(size.x, size.x - p0.x);

        // Signed area of the swept rectangle.
        return sign(delta.y) * w * h;
    }

    // qa is the second-degree coefficient for the y-coordinate
    // quadratic.
    float qa = mad(-2.f, p1.y, p0.y + p2.y);

    float bt = intersect_monotonic(qa, p0.y, p1.y, p2.y, 0.f);
    float tt = intersect_monotonic(qa, p0.y, p1.y, p2.y, size.y);

    // v_min_t and v_max_t are the crossings where the curve enters
    // and exits the scanline.
    float v_min_t = delta.y > 0.f ? bt : tt;
    float v_max_t = delta.y > 0.f ? tt : bt;

    float2 v_min = evaluate_bezier(p0, p1, p2, saturate(v_min_t));
    float2 v_max = evaluate_bezier(p0, p1, p2, saturate(v_max_t));

    if (max(v_min.x, v_max.x) <= 0.f)
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
        return 0.f;
    }

    // Solve for roots along x.
    qa = mad(-2.f, p1.x, p0.x + p2.x);

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
    float4 h_check = delta.x > 0.f
        ? float4(p0.x, p2.x, 0.f, 0.f)
        : float4(p2.x, p0.x, size.x, 1.f);

    if (h_check.x >= h_check.z)
    {
        h_min_t = h_check.w;
    }
    else if (h_check.y <= h_check.z)
    {
        h_min_t = 1.f - h_check.w;
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
        h_max_t = 1.f - h_check.w;
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
    float2 q0 = v_min_t >= h_min_t ? v_min : evaluate_bezier(p0, p1, p2, min_t);
    float2 q1 = v_max_t <= h_max_t ? v_max : evaluate_bezier(p0, p1, p2, max_t);

    float coverage = 0.f;

    if (min_t > 0.f && delta.x > 0.f)
    {
        // We enter the pixel from the left, so we need to integrate the
        // swept rectangle below the entry point.
        float h = delta.y > 0.f
            ? q0.y - max(0.f, p0.y)
            : min(size.y, p0.y) - q0.y;
        coverage = sign(delta.y) * h * size.x;
    }

    if (max_t < 1.f && delta.x < 0.f)
    {
        // We exit the pixel on the left side, so we need to integrate the
        // swept rectangle after the exit point.
        float h = delta.y > 0.f
            ? min(size.y, p2.y) - q1.y
            : q1.y - max(0.f, p2.y);
        coverage += sign(delta.y) * h * size.x;
    }

    // This implements the simple trapezoidal approximation for the
    // portion of the curve within the window.
    float h = q1.y - q0.y;

    // Sum of trapezoid bases divided by two. If q0.x or q1.x happen
    // to equal size.x, the trapezoidal area is effectively a triangle.
    float b = mad(-0.5f, q0.x + q1.x, size.x);

    coverage += b * h;

    // The caller is expected to accumulate this coverage for each curve,
    // and divide the final result by the window area.
    return coverage;
}
