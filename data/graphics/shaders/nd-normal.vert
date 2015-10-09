#version 330
#define VERTEX_IS_LATLON 0
#define VERTEX_IS_METERS 1
#define VERTEX_IS_SCREEN 2
#define VERTEX_IS_GLXY   3
#define DEG2RAD 0.01745329252
#define RAD2DEG 57.295779513
#define REARTH_INV 1.56961231e-7

// Uniform block of global data
layout (std140) uniform global_data {
float ownhdg;			// Ownship heading/track [deg]
float ownlat;           // Ownship coordinates [deg]
float ownlon;           // Ownship coordinates [deg]
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

	float AR = float(screen_height) / float(screen_width);
	vec2 flat_earth = vec2(cos(DEG2RAD*ownlat), 1.0);
    mat2 mrot = mat2(cos(DEG2RAD*(orientation_in - ownhdg)), -sin(DEG2RAD*(orientation_in - ownhdg)), sin(DEG2RAD*(orientation_in - ownhdg)), cos(DEG2RAD*(orientation_in - ownhdg)));

	// Vertex position and rotation calculations
	vec2 position = vec2(lon_in, lat_in);
	position -= vec2(ownlon, ownlat);
	vec2 vertex = vec2(0.0, -0.7) + mrot * vertex_in;
	gl_Position = vec4(vertex.x, vertex.y, 0.0, 1.0);
	texcoords_fs = texcoords_in;
}