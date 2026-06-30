// Preconditions:
// - qa is the second-degree coefficient.
// - c0, c1, and c2 are the coordinate values of the first,
//   second, and third control points respectively.
// - target is the desired line of intersection.
//
// This routine assumes that c0 <= target <= c2. That is, the
// caller has done bounds checks that guarantee that this
// routine is needed already.
//
// This routine also assumes monotonicity:
// c0 <= c1 <= c2.
//
// Returns the value of t where the intersection occurs.
float intersect_monotonic(
    float qa,
    float c0,
    float c1,
    float c2,
    float target)
{
    if (abs(qa) < 1e-3f)
    {
        // Approximately linear case. Threshold can be adjusted
        // as needed.
        return (target - c0) / (c2 - c0);
    }

    // First-degree coefficient.
    float qb = mad(2.f, c1, -2.f * c0);
    float qc = c0 - target;

    float d = mad(qb, qb, -4.f * qa * qc);

    // Clamping d above 0 is OK because of our established
    // precondition that c0 <= target <= c2.
    float sqrt_d = d < 0.f ? 0.f : sqrt(d);
    float inv_2a = 0.5f / qa;

    // The sign dictates whether we need the positive or
    // negative root. Because of the monotonic assumption,
    // only one root ever needs consideration for each
    // curve.
    return mad(-qb, inv_2a, sign(c2 - c0) * sqrt_d * inv_2a);
}

// p0 and p2 always refer to the curve endpoints.
// p0, p1, and p2 are the curve control points in order of increasing t.
float2 evaluate_bezier(float2 p0, float2 p1, float2 p2, float t)
{
    float2 a = lerp(p0, p1, t);
    float2 b = lerp(p1, p2, t);
    return lerp(a, b, t);
}
