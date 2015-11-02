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
};

layout (location=0) in float lat1;
layout (location=1) in float lon1;
layout (location=2) in float alt1;
layout (location=3) in float tas1;
layout (location=4) in float trk1;

layout (location=5) in float lat0;
layout (location=6) in float lon0;
layout (location=7) in float alt0;
layout (location=8) in float tas0;
layout (location=9) in float trk0;

out GSData {
    vec2 vAR;
    vec4 ownship;
    vec4 intruder;
    float dH;
    int ac_id;
} to_gs;

void main() {
    to_gs.vAR       = vec2(1.0 / float(screen_width), 1.0 / float(screen_height));
    vec2 flat_earth = vec2(cos(radians(panlat)), 1.0);
    vec2 position   = vec2(lon0, lat0);
    if (wrap_dir < 0 && position.x > wrap_lon) {
        position.x -= 360.0;
    } else if (wrap_dir > 0 && position.x < wrap_lon) {
        position.x += 360.0;
    }
    position -= vec2(panlon, panlat);

    to_gs.ownship  = vec4(lat0, lon0, tas0, trk0);
    to_gs.intruder = vec4(lat1, lon1, tas1, trk1);
    to_gs.dH       = alt1 - alt0;
    to_gs.ac_id    = gl_InstanceID;
    gl_Position    = vec4(vec2(1.0, float(screen_width) / float(screen_height)) * zoom * flat_earth * position, 0.0, 1.0);
}
