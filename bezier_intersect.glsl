// Compared to intersect_bezier, this intersection routine assumes that the
// bezier curve is monotonic (i.e. the control points are in increasing or decreasing order).
// The delta_sign parameter indicates whether the curve is increasing or decreasing
// over its interval.
float intersect_monotonic_bezier(
    float qa,
    float c0,
    float c1,
    float c2,
    float target,
    float delta_sign)
{
    // If we're mostly linear, this routine is good enough.
    if (abs(qa) < 1e-2)
    {
        // Fairly linear line segment.
        return (target - c0) / (c2 - c0);
    }

    // Otherwise compute the other coefficients of the quadratic expression.
    float qb = mad(2.0, c1, -2.0 * c0);
    float qc = c0 - target;

    // Compute the discriminant of the quadratic expression.
    float d = mad(qb, qb, -4.0 * qa * qc);

    // The discriminant being negative here is actually unexpected since
    // earlier code is responsible for skipping curves that lie completely above
    // or below the target. However, a zero underflow is possible due to floating point imprecision.
    float sqrt_d = d < 0.0 ? 0.0 : sqrt(d);
    float inv_2a = 0.5 / qa;

    // Because the curve is monotonic, we can know exactly which side of the
    // quadratic center line the root should be.
    //
    // There are 4 cases to consider (k := curvature, g := gradient):
    // 1. +k, +g -> root on the right (+)
    // 2. +k, -g -> root on the left  (-)
    // 3. -k, +g -> root on the left  (+)
    // 4. -k, -g -> root on the right (-)
    //
    // Note that the sign needed to produce the right and left root depends on
    // curvature sign.
    //
    // Catastrophic cancellation can occur when b^2 is much greater than 4ac.
    // Because all points are within the vicinity of the unit em-square,
    // we don't expect this condition to occur.
    //
    // If b^2 is approximately equal to 4ac, we may lose precision in the discriminant.
    // However, such a root will be very close to the inflection point of the curve itself,
    // which will be close to one of the curve endpoints.
    return mad(-qb, inv_2a, delta_sign * sqrt_d * inv_2a);
}

// Evaluate a quadratic bezier curve at parameter t.
// B(t) = (1-t)^2 * p0 + 2t(1-t) * p1 + t^2 * p2
vec2 evaluate_bezier(vec2 p0, vec2 p1, vec2 p2, float t)
{
    float mt = 1.0 - t;
    float a = mt * mt;
    float b = 2.0 * mt * t;
    float c = t * t;
    return a * p0 + b * p1 + c * p2;
}
