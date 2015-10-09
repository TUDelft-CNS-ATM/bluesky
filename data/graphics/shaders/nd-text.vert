#version 330
#define DEG2RAD 0.01745329252

// Uniform block of global data
layout (std140) uniform global_data {
float ownhdg;           // Ownship heading/track [deg]
float ownlat;           // Ownship coordinates [deg]
float ownlon;           // Ownship coordinates [deg]
float zoom;             // Screen zoom factor [-]
int screen_width;       // Screen width in pixels
int screen_height;      // Screen height in pixels
int vertex_scale_type;  // Vertex scale type
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
    // Pass color and texture coordinates to the fragment shader
    color_fs = color_in;
    texcoords_fs = texcoords_in;
    texcoords_fs.p -= 32.0;

    float AR = float(screen_height) / float(screen_width);
    vec2 flat_earth = vec2(cos(DEG2RAD*ownlat), 1.0);
    mat2 mrot = mat2(cos(DEG2RAD*(orientation_in - ownhdg)), -sin(DEG2RAD*(orientation_in - ownhdg)), sin(DEG2RAD*(orientation_in - ownhdg)), cos(DEG2RAD*(orientation_in - ownhdg)));

    vec2 position = vec2(lon_in, lat_in);
    position -= vec2(ownlon, ownlat);
    position *= (zoom * flat_earth);

    vec2 vertex = vec2(0.0, -0.7) + mrot * vertex_in;

    // When text_dims is non-zero we are drawing instanced
    if (block_size[0] > 0) {
        texcoords_fs.p = texdepth_in - 32.0;
        vertex.x += float(gl_InstanceID%block_size[0]) * char_size.x;
        vertex.y -= floor(float((gl_InstanceID%(block_size[0]*block_size[1])))/block_size[0]) * char_size.y;
    }

    gl_Position = vec4(vertex.x, vertex.y, 0.0, 1.0);

}
