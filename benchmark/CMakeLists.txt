cmake_minimum_required(VERSION 3.16)
project(scanline_sweeper_benchmark C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Dependencies
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Freetype REQUIRED)

# --- Embed GLSL shaders as C string arrays ---
# Each call produces a header <VAR_NAME>.h in the build directory
# with a non-static const definition that main.c #includes.
set(SHADER_DIR "${CMAKE_SOURCE_DIR}/..")

function(embed_shader VAR_NAME SHADER_PATH)
    file(READ "${SHADER_PATH}" CONTENT)
    string(REPLACE "\\" "\\\\" CONTENT "${CONTENT}")
    string(REPLACE "\"" "\\\"" CONTENT "${CONTENT}")
    string(REPLACE "\n" "\\n\"\n\"" CONTENT "${CONTENT}")
    set(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${VAR_NAME}.h")
    file(WRITE "${OUTPUT}"
        "/* Auto-generated from ${SHADER_PATH} */\n"
        "static const char *${VAR_NAME} =\n"
        "\"${CONTENT}\";\n"
    )
endfunction()

add_executable(benchmark main.c)

embed_shader(BEZIER_SHADER_SRC "${SHADER_DIR}/bezier_intersect.glsl")
embed_shader(SWEEP_SHADER_SRC  "${SHADER_DIR}/scanline_sweep.glsl")

target_include_directories(benchmark PRIVATE
    ${FREETYPE_INCLUDE_DIRS}
    ${GLFW3_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(benchmark
    OpenGL::GL
    glfw
    ${FREETYPE_LIBRARIES}
)

if(UNIX AND NOT APPLE)
    target_link_libraries(benchmark m dl pthread)
endif()
