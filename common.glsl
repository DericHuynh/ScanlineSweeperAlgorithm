// Shared GLSL compatibility helpers for the Scanline Sweeper algorithm.
// Include this before the other shader files.

#ifndef SCANLINE_SWEEPER_COMMON_GLSL
#define SCANLINE_SWEEPER_COMMON_GLSL

// HLSL compatibility macros.
// fma() is preferred on GLSL 4.0+ / ES 3.0+, fall back to explicit mul-add.
#ifndef mad
#define mad(a, b, c) ((a) * (b) + (c))
#endif

#ifndef saturate
#define saturate(x) clamp((x), 0.0, 1.0)
#endif

#endif // SCANLINE_SWEEPER_COMMON_GLSL
