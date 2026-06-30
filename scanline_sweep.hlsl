
// The goal of this function is to take a curve, sweep it to the right and compute the area that the curve overlaps with the pixel window.
// The offset provided is the location of the lower-left pixel coordinate.
// All parameter units are font units.
// For a linear segment, p1 is directly between p0 and p2.
// The bezier curve passed to this function is expected to never be horizontal.
float scanline_sweep(
    float2 pixel_size,
    float2 pixel_offset,
    float2 p0,
    float2 p1,
    float2 p2)
{
    // Discard curves entirely above or below the scanline.
    if (max(p0.y, p2.y) <= pixel_offset.y ||
        min(p0.y, p2.y) >= pixel_offset.y + pixel_size.y)
    {
        return 0.f;
    }

    float2 delta = p2 - p0;

    // After the coordinate shift below, we are in a coordinate system that looks like the following;
    // looks like the following:
    //
    // (0, pixel_size.y)
    // +---------------+ (pixel_size.x, pixel_size.y)
    // |               |
    // |               |
    // |               |
    // |               |
    // +---------------+ (pixel_size.x, 0)
    // (0, 0)
    //
    // The pixel size in funits is given by funits_per_px constant.
    //
    // Note that the font coordinate space is orientated with +x moving to the right and +y moving up.
    p0 -= pixel_offset;
    p1 -= pixel_offset;
    p2 -= pixel_offset;

    // Fast path for strictly vertical segments.
    if (p0.x == p1.x && p0.x == p2.x)
    {
        if(p0.x >= pixel_size.x)
        {
            return 0.f;
        }

        // Compute the area within the pixel to the right of the segment.
        float top = min(max(p0.y, p2.y), pixel_size.y);
        float bottom = max(min(p0.y, p2.y), 0.f);
        float height = top - bottom;

        if (p0.x <= 0.f)
        {
            return sign(delta.y) * pixel_size.x * height;
        }

        float base = pixel_size.x - p0.x;

        return sign(delta.y) * base * height;
    }

    // Constrain the segment to the top and bottom of the pixel scanline.
    float qa_y = mad(-2.f, p1.y, p0.y + p2.y);
    float bt = intersect_monotonic_bezier(qa_y, p0.y, p1.y, p2.y, 0.f, sign(delta.y));
    float tt = intersect_monotonic_bezier(qa_y, p0.y, p1.y, p2.y, pixel_size.y, sign(delta.y));

    float v_min = delta.y > 0.f ? bt : tt;
    float v_max = delta.y > 0.f ? tt : bt;
    float2 v_min_crossing = evaluate_bezier(p0, p1, p2, saturate(v_min));
    float2 v_max_crossing = evaluate_bezier(p0, p1, p2, saturate(v_max));

    if(max(v_min_crossing.x, v_max_crossing.x) <= 0.f)
    {
        // The segment is entirely to the left of the pixel.
        return (v_max_crossing.y - v_min_crossing.y) * pixel_size.x;
    }

    // This check perhaps should be further up, since it seems to be a pretty important fast-path.
    if(min(v_min_crossing.x, v_max_crossing.x) >= pixel_size.x)
    {
        // Segments entirely to the right of the pixel can be ignored.
        return 0.f;
    }

    // At this point, the curve is at least partially contained within the
    // pixel (possibly terminating within the pixel itself).
    //
    // Compute the crossing parameters for each pixel boundary, constrained to
    // the segment itself.

    // Quadratic coefficient for the x-coordinate bezier.
    float qa_x = mad(-2.f, p1.x, p0.x + p2.x);
    float dx_sign = sign(delta.x);

    // The segment direction determines which boundaries are expected initial and final crossing.
    float h_min;
    float h_max;

    // This check vector stores the following quantities in each component:
    // - lower x bound (first endpoint in traversal order)
    // - upper x bound (last endpoint in traversal order)
    // - target value (the vertical pixel boundary hit first)
    // - parameter associated with the lower x bound (0 or 1)
    //
    // Packing the values in this way simplifies bounds checks and intersection
    // testing, and the values depend on the direction the curve moves.
    float4 h_check = delta.x > 0.f
        ? float4(p0.x, p2.x, 0.f, 0.f)
        : float4(p2.x, p0.x, pixel_size.x, 1.f);

    if(h_check.x >= h_check.z)
    {
        // The curve starts at or past the target boundary.
        h_min = h_check.w;
    }
    else if(h_check.y <= h_check.z)
    {
        // The curve never reaches the target boundary.
        h_min = 1.f - h_check.w;
    }
    else
    {
        // The curve crosses the target boundary. Solve for the intersection t.
        h_min = intersect_monotonic_bezier(qa_x, p0.x, p1.x, p2.x, h_check.z, dx_sign);
    }

    // Now compute h_max: the intersection with the opposite vertical pixel boundary.
    //
    // For rightward-moving curves (delta.x > 0):
    //   h_min = intersection with x=0    (entering pixel)
    //   h_max = intersection with x=px_w (exiting pixel)
    //
    // For leftward-moving curves (delta.x < 0):
    //   h_min = intersection with x=px_w (entering pixel)
    //   h_max = intersection with x=0    (exiting pixel)
    float h_target_max = delta.x > 0.f ? pixel_size.x : 0.f;

    // Use the same bounds-check pattern as h_min.
    // The endpoint ordering is the same (first/last in traversal order).
    if (h_check.x >= h_target_max)
    {
        // The curve starts past the opposite boundary.
        h_max = h_check.w;
    }
    else if (h_check.y <= h_target_max)
    {
        // The curve never reaches the opposite boundary.
        h_max = 1.f - h_check.w;
    }
    else
    {
        // The curve crosses the opposite boundary.
        h_max = intersect_monotonic_bezier(qa_x, p0.x, p1.x, p2.x, h_target_max, dx_sign);
    }

    // Account for segments that start or terminate within the pixel itself.
    float t_min = saturate(max(v_min, h_min));
    float t_max = saturate(min(v_max, h_max));

    float2 q0 = v_min >= h_max ? v_min_crossing : evaluate_bezier(p0, p1, p2, t_min);
    float2 q1 = v_max <= h_min ? v_max_crossing : evaluate_bezier(p0, p1, p2, t_max);

    float coverage = 0.f;

    // The intervals [0, t_min] and [t_max, 1] are still of interest since we
    // need to account for coverage contributions external to the pixel as well.
    if (t_min > 0.f && delta.x > 0.f)
    {
        float h = delta.y > 0.f
            ? q0.y - max(0.f, p0.y)
            : min(pixel_size.y, p0.y) - q0.y;
        coverage = sign(delta.y) * h * pixel_size.x;
    }

    if (t_max < 1.f && delta.x > 0.f)
    {
        float h = delta.y > 0.f
            ? min(pixel_size.y, p2.y) - q1.y
            : q1.y - max(0.f, p2.y);

        coverage += sign(delta.y) * h * pixel_size.x;
    }

#if ACCURATE_COVERAGE
    // Continuous integration of the bezier curve area within the pixel.
    //
    // The area to the right of the path between t_min and t_max is:
    //   ∫ (pixel_size.x - x(t)) * y'(t) dt
    //
    // Where the bezier in power basis is:
    //   x(t) = a_x*t^2 + b_x*t + c_x
    //   y(t) = a_y*t^2 + b_y*t + c_y
    //   y'(t) = 2*a_y*t + b_y
    //
    // The antiderivative F(t) is a 4th-degree polynomial:
    //   F(t) = (px - c_x)*b_y*t
    //        + (a_y*(px - c_x) - 0.5*b_x*b_y)*t^2
    //        - (2*b_x*a_y + a_x*b_y)*t^3 / 3
    //        - 0.5*a_x*a_y*t^4

    float px = pixel_size.x;

    // Power-basis coefficients for x(t) and y(t).
    float ax = qa_x;
    float bx = mad(2.f, p1.x, -2.f * p0.x);
    float cx = p0.x;

    float ay = qa_y;
    float by = mad(2.f, p1.y, -2.f * p0.y);
    float cy = p0.y;

    // Evaluate the antiderivative F(t) at parameter t using Horner's method
    // to minimise operations:
    //   F(t) = t * (c3 + t * (c2 + t * (c1 + t * c0)))
    //
    // Polynomial: c0*t^4 + c1*t^3 + c2*t^2 + c3*t  (no constant term)
    float c0 = -0.5f * ax * ay;
    float c1 = -(2.f * bx * ay + ax * by) / 3.f;
    float c2 = ay * (px - cx) - 0.5f * bx * by;
    float c3 = (px - cx) * by;

    float ft_max = t_max * (c3 + t_max * (c2 + t_max * (c1 + t_max * c0)));
    float ft_min = t_min * (c3 + t_min * (c2 + t_min * (c1 + t_min * c0)));

    coverage += ft_max - ft_min;
#else
    // For a downwards moving segment, the trapezoidal area to the right of the path is subtracted from the coverage.
    float h = q1.y - q0.y;

    // Note that depending on how the segment intersects the boundary, the
    // "trapezoid" may actually be a triangle in the limiting case.
    float b = mad(-0.5f, q0.x + q1.x, pixel_size.x);

    coverage += b * h;
#endif

    return coverage;
}
