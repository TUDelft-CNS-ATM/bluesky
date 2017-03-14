#version 330
#define VERTEX_IS_LATLON 0
#define VERTEX_IS_METERS 1
#define VERTEX_IS_SCREEN 2
#define REARTH_INV 1.56961231e-7

layout (location = 0) in vec2 vertex_in;
layout (location = 1) in vec2 texcoords_in;
layout (location = 2) in float lat_in;
layout (location = 3) in float lon_in;
layout (location = 4) in float orientation_in;
layout (location = 5) in vec4 color_in;

out vec4 color_fs;
out vec2 texcoords_fs;
void main()
{
	// Pass color and texture coordinates to the fragment shader
	color_fs = color_in;

	gl_Position = vec4(vertex_in.x, vertex_in.y - 0.7, 0.0, 1.0);
	texcoords_fs = texcoords_in;
}