#version 330

// Uniform block of global data
layout (std140) uniform global_data {
int wrap_dir;           // Wrap-around direction
float wrap_lon;         // Wrap-around longitude
float panlat;           // Map panning coordinates [deg]
float panlon;           // Map panning coordinates [deg]
float zoom;             // Screen zoom factor [-]
int screen_width;       // Screen width in pixels
int screen_height;      // Screen height in pixels
int vertex_scale_type;  // Vertex scale type
float screen_pixel_ratio; // > 1 for high-DPI screens
};

// Local uniform data
uniform vec2 char_size;         // Size (wxh) of a character
uniform ivec2 block_size;         // Textblock width and height in #characters


layout (location = 0) in vec2 vertex;
layout (location = 1) in vec3 texcoords;
layout (location = 2) in float lat;
layout (location = 3) in float lon;
layout (location = 4) in float orientation;
layout (location = 5) in vec4 color;
layout (location = 6) in int texdepth;

out vec3 texcoords_fs;
out vec4 color_fs;

void main() {
    // Pass color and texture coordinates to the fragment shader
    color_fs = color;
    texcoords_fs = texcoords;
    texcoords_fs.p -= 30.0;

    vec2 vAR = vec2(1.0, float(screen_width) / float(screen_height));
    vec2 flat_earth = vec2(cos(radians(panlat)), 1.0);
    float rot = radians(orientation);
    mat2 mrot = mat2(cos(rot), -sin(rot), sin(rot), cos(rot));

    vec2 position = vec2(lon, lat);
    if (wrap_dir < -0.1 && position.x > wrap_lon) {
        position.x -= 360.0;
    } else if (wrap_dir > 0.1 && position.x < wrap_lon) {
        position.x += 360.0;
    }
    position -= vec2(panlon, panlat);
    position *= (zoom * flat_earth);

    vec2 vertex_out = mrot * vertex + ceil(0.5 * position * screen_width);

    // When text_dims is non-zero we are drawing instanced
    if (block_size[0] > 0) {
        texcoords_fs.p = texdepth - 30.0;
        vertex_out.x += float(gl_InstanceID%block_size[0]) * char_size.x;
        vertex_out.y -= floor(float((gl_InstanceID%(block_size[0]*block_size[1])))/block_size[0]) * char_size.y;
    }

    vertex_out = vec2(2.0 * vertex_out.x / float(screen_width), 2.0 * vertex_out.y / float(screen_height));
    gl_Position = vec4(vertex_out, 0.0, 1.0);
    // gl_Position = vec4(vAR * position + vertex, 0.0, 1.0);
}
