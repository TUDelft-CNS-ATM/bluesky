#version 330
#define VERTEX_IS_LATLON 0
#define VERTEX_IS_METERS 1
#define VERTEX_IS_SCREEN 2
#define REARTH_INV 1.56961231e-7
const vec2 texcoords[6] = vec2[](vec2(0.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 0.0),
                                 vec2(1.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 1.0));

layout (location = 0) in vec2 vertex_in;
layout (location = 1) in vec3 texcoords_in;
layout (location = 2) in vec2 pos_in;
layout (location = 3) in float orientation_in;
layout (location = 5) in vec4 color_in;

out vec4 color_fs;
out vec3 texcoords_fs;
void main()
{
	// Pass color and texture coordinates to the fragment shader
	color_fs = color_in;

	gl_Position = vec4(vertex_in + pos_in, 0.0, 1.0);
	texcoords_fs = vec3(texcoords[gl_VertexID % 6], 0.0);
}