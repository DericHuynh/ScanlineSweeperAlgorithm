#!/bin/sh
set -e

# --- paths -----------------------------------------------------------
DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$DIR/main.c"
SHADER_DIR="$DIR/.."
BUILD_DIR="$DIR/build"

BEZIER_SRC="$SHADER_DIR/bezier_intersect.glsl"
SWEEP_SRC="$SHADER_DIR/scanline_sweep.glsl"

# --- embed GLSL shaders as C string headers --------------------------
mkdir -p "$BUILD_DIR"

embed() {
    var="$1"
    path="$2"
    out="$BUILD_DIR/${var}.h"
    # escape backslashes, quotes, newlines → C string literal
    awk '{ gsub(/\\/,"\\\\"); gsub(/"/,"\\\""); printf "%s\\n\"\n\"", $0 }
         END  { printf "\\n" }' "$path" > "$out.tmp"
    printf '/* Auto-generated from %s */\nstatic const char *%s =\n"' "$path" "$var" > "$out"
    cat "$out.tmp" >> "$out"
    printf '";\n' >> "$out"
    rm "$out.tmp"
}

embed BEZIER_SHADER_SRC "$BEZIER_SRC"
embed SWEEP_SHADER_SRC  "$SWEEP_SRC"

# --- compile & link --------------------------------------------------
CFLAGS="-std=c11 -O2 -Wall -I/usr/include/freetype2 -I$BUILD_DIR"
LIBS="-lglfw -lfreetype -lGL -lm -ldl -lpthread"

cc $CFLAGS -o "$BUILD_DIR/benchmark" "$SRC" $LIBS

# --- run -------------------------------------------------------------
cd "$BUILD_DIR"
exec ./benchmark
