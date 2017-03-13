#version 330
#define VERTEX_IS_LATLON  0
#define VERTEX_IS_METERS  1
#define VERTEX_IS_SCREEN  2
#define VERTEX_IS_GLXY    3
#define VERTEX_ROTATE_OWN 10
#define REARTH_INV 1.56961231e-7
#define SYMBOLSCALE 0.004

// Uniform block of global data
layout (std140) uniform global_data {
float ownhdg;			// Ownship heading/track [deg]
float ownlat;           // Ownship coordinates [deg]
float ownlon;           // Ownship coordinates [deg]
float zoom;             // Screen zoom factor [-]
int ownid;
int vertex_modifiers;   // Vertex modifiers
};

layout (location = 0) in vec2 vertex_in;
layout (location = 1) in vec2 texcoords_in;
layout (location = 2) in float lat_in;
layout (location = 3) in float lon_in;
layout (location = 4) in float orientation_in;
layout (location = 5) in vec4 color_in;

out vec4 color_fs;
out vec2 texcoords_fs;
flat out int discard_instance;
void main()
{
	discard_instance = (ownid == gl_InstanceID ? 1 : 0);

	int vertex_scale_type     = vertex_modifiers % 10;
	bool vertex_rotate_ownhdg = (vertex_modifiers >= VERTEX_ROTATE_OWN);
	// Pass color and texture coordinates to the fragment shader
	color_fs = color_in;

    vec2 flat_earth = vec2(cos(radians(ownlat)), 1.0);
    mat2 ownrot     = mat2(cos(radians(ownhdg)), sin(radians(ownhdg)),
                          -sin(radians(ownhdg)), cos(radians(ownhdg)));
    mat2 symbolrot  = mat2(cos(radians(orientation_in)),-sin(radians(orientation_in)),
                           sin(radians(orientation_in)), cos(radians(orientation_in)));

    vec2 position   = vec2(lon_in - ownlon, lat_in - ownlat);
	// Vertex depends on scale type
	vec2 vertex;
	switch (vertex_scale_type) {
		case VERTEX_IS_SCREEN:
			// Vertex coordinates are screen pixels, so correct for screen size
			if (vertex_rotate_ownhdg) {
				vertex    = ownrot * (zoom * flat_earth * position + SYMBOLSCALE * symbolrot * vertex_in);
			} else {
				vertex    = ownrot * (zoom * flat_earth * position) + SYMBOLSCALE * symbolrot * vertex_in;
			}
			texcoords_fs  = texcoords_in;
			break;

		case VERTEX_IS_METERS:
			// Vertex coordinates in meters use a right-handed coordinate system, where the positive x-axis points to the north
			// The elements in each vertex therefore need to be flipped
			vertex        = ownrot * zoom * (flat_earth * position + (degrees(vertex_in.yx * REARTH_INV)));
			texcoords_fs  = texcoords_in.ts;
			break;

		case VERTEX_IS_LATLON:
			// Lat/lon vertex coordinates are flipped: lat is index 0, but screen y-axis, and lon is index 1, but screen x-axis
			vertex        = ownrot * zoom * flat_earth * (position + vertex_in.yx);
			texcoords_fs  = texcoords_in.ts;
			break;	

		case VERTEX_IS_GLXY:
			if (vertex_rotate_ownhdg) {
				vertex    = ownrot * vertex_in;
			} else {
				vertex = vertex_in;
			}
			texcoords_fs  = texcoords_in;
			break;
	}
	gl_Position   = vec4(vertex.x, vertex.y - 0.7, 0.0, 1.0);
}