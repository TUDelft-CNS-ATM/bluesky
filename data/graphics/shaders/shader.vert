#version 330
#define VERTEX_IS_LATLON 0
#define VERTEX_IS_METERS 1
#define VERTEX_IS_SCREEN 2
#define DEG2RAD 0.01745329252
#define RAD2DEG 57.295779513
#define REARTH_INV 1.56961231e-7

// Uniform block of global data
layout (std140) uniform global_data {
int wrap_dir;           // Wrap-around direction
float wrap_lon;         // Wrap-around longitude
vec2 pan;               // Map panning coordinates [lat,lon]
float zoom;             // Screen zoom factor [-]
int screen_width;       // Screen width in pixels
int screen_height;      // Screen height in pixels
int vertex_scale_type;  // Vertex scale type
};

layout (location = 0) in vec2 vertex_in;
layout (location = 1) in vec2 texcoords_in;
layout (location = 2) in float lat_in;
layout (location = 3) in float lon_in;
layout (location = 4) in float orientation_in;
layout (location = 5) in vec3 color_in;

out vec3 color_fs;
out vec2 texcoords_fs;
void main()
{
	// Pass color and texture coordinates to the fragment shader
	color_fs = color_in;
	texcoords_fs = texcoords_in;

	vec2 vAR = vec2(1.0, float(screen_width) / float(screen_height));
	vec2 flat_earth = vec2(cos(DEG2RAD*pan.y), 1.0);
    mat2 mrot = mat2(cos(DEG2RAD*orientation_in), -sin(DEG2RAD*orientation_in), sin(DEG2RAD*orientation_in), cos(DEG2RAD*orientation_in));

	// Vertex position and rotation calculations
	vec2 position = vec2(lon_in, lat_in);
	if (wrap_dir < 0 && position.x > wrap_lon) {
		position.x -= 360.0;
	} else if (wrap_dir > 0 && position.x < wrap_lon) {
		position.x += 360.0;
	}
	position -= pan;

	switch (vertex_scale_type) {
		case VERTEX_IS_SCREEN:
			// Vertex coordinates are screen pixels, so correct for screen size
			vec2 vertex = mrot * vertex_in;
			vertex = vec2(2.0 * vertex.x / float(screen_width), 2.0 * vertex.y / float(screen_height));
			gl_Position = vec4(vAR * zoom * flat_earth * position + vertex, 0.0, 1.0);
			break;
		case VERTEX_IS_METERS:
			gl_Position = vec4(vAR * zoom * (flat_earth * position + mrot * (vertex_in * REARTH_INV * RAD2DEG)), 0.0, 1.0);
			break;
		case VERTEX_IS_LATLON:
		default:
			gl_Position = vec4(vAR * flat_earth * zoom * (position + mrot * vertex_in), 0.0, 1.0);
			break;	
	}
}