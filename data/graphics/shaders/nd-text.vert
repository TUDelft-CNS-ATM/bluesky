#version 330
#define VERTEX_IS_LATLON 0
#define VERTEX_IS_METERS 1
#define VERTEX_IS_SCREEN 2
#define VERTEX_IS_GLXY   3
#define VERTEX_ROTATE_OWN 10
#define DEG2RAD 0.01745329252
#define SYMBOLSCALE 0.004

// Uniform block of global data
layout (std140) uniform global_data {
float ownhdg;           // Ownship heading/track [deg]
float ownlat;           // Ownship coordinates [deg]
float ownlon;           // Ownship coordinates [deg]
float zoom;             // Screen zoom factor [-]
int vertex_modifiers;   // Vertex scale type
};

// Local uniform data
uniform vec2 char_size;         // Size (wxh) of a character
uniform ivec2 block_size;         // Textblock width and height in #characters


layout (location = 0) in vec2 vertex_in;
layout (location = 1) in vec3 texcoords_in;
layout (location = 2) in float lat_in;
layout (location = 3) in float lon_in;
layout (location = 4) in float orientation_in;
layout (location = 5) in vec3 color_in;
layout (location = 6) in float texdepth_in;

out vec3 texcoords_fs;
out vec3 color_fs;

void main() {
    int vertex_scale_type     = vertex_modifiers % 10;
    bool vertex_rotate_ownhdg = (vertex_modifiers >= VERTEX_ROTATE_OWN);
    // Pass color and texture coordinates to the fragment shader
    color_fs = color_in;
    texcoords_fs = texcoords_in;
    texcoords_fs.p -= 32.0;

    vec2 flat_earth = vec2(cos(DEG2RAD*ownlat), 1.0);
    mat2 ownrot     = mat2(cos(DEG2RAD*ownhdg), sin(DEG2RAD*ownhdg),
                          -sin(DEG2RAD*ownhdg), cos(DEG2RAD*ownhdg));
    mat2 symbolrot  = mat2(cos(DEG2RAD*orientation_in),-sin(DEG2RAD*orientation_in),
                           sin(DEG2RAD*orientation_in), cos(DEG2RAD*orientation_in));

    vec2 position = vec2(lon_in - ownlon, lat_in - ownlat);
    vec2 vertex   = vertex_in;

    // When text_dims is non-zero we are drawing instanced
    if (block_size[0] > 0) {
        texcoords_fs.p = texdepth_in - 32.0;
        vertex.x += float(gl_InstanceID%block_size[0]) * char_size.x;
        vertex.y -= floor(float((gl_InstanceID%(block_size[0]*block_size[1])))/block_size[0]) * char_size.y;
    }

    switch (vertex_scale_type) {
        case VERTEX_IS_SCREEN:
            // Vertex coordinates are screen pixels, so correct for screen size
            if (vertex_rotate_ownhdg) {
                vertex    = ownrot * (zoom * flat_earth * position + SYMBOLSCALE * symbolrot * vertex);
            } else {
                vertex    = ownrot * zoom * flat_earth * position + SYMBOLSCALE * symbolrot * vertex;
            }
            break;

        case VERTEX_IS_GLXY:
            if (vertex_rotate_ownhdg) {
                vertex    = ownrot * vertex;
            } else {
                vertex = vertex;
            }
            break;
    }

    gl_Position = vec4(vertex.x, vertex.y - 0.7, 0.0, 1.0);

}
