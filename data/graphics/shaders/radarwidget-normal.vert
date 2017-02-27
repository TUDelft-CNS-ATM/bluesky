#version 330
#define VERTEX_IS_LATLON 0
#define VERTEX_IS_METERS 1
#define VERTEX_IS_SCREEN 2
#define REARTH_INV 1.56961231e-7

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

	vec2 vAR = vec2(1.0, float(screen_width) / float(screen_height));
	vec2 flat_earth = vec2(cos(radians(panlat)), 1.0);
	float rot = radians(orientation_in);
    mat2 mrot = mat2(cos(rot), -sin(rot), sin(rot), cos(rot));

	// Vertex position and rotation calculations
	vec2 position = vec2(lon_in, lat_in);
	if (wrap_dir < 0 && position.x > wrap_lon) {
		position.x -= 360.0;
	} else if (wrap_dir > 0 && position.x < wrap_lon) {
		position.x += 360.0;
	}
	position -= vec2(panlon, panlat);

	switch (vertex_scale_type) {
		case VERTEX_IS_SCREEN:
			// Vertex coordinates are screen pixels, so correct for screen size
			vec2 vertex = mrot * vertex_in;
			vertex = vec2(2.0 * vertex.x / float(screen_width), 2.0 * vertex.y / float(screen_height));
			gl_Position = vec4(vAR * zoom * flat_earth * position + vertex, 0.0, 1.0);
			texcoords_fs = texcoords_in;
			break;
		case VERTEX_IS_METERS:
			// Vertex coordinates in meters use a right-handed coordinate system, where the positive x-axis points to the north
			// The elements in each vertex therefore need to be flipped
			gl_Position = vec4(vAR * zoom * (flat_earth * position + mrot * (degrees(vertex_in.yx * REARTH_INV))), 0.0, 1.0);
			texcoords_fs = texcoords_in.ts;
			break;
		case VERTEX_IS_LATLON:
		default:
			// Lat/lon vertex coordinates are flipped: lat is index 0, but screen y-axis, and lon is index 1, but screen x-axis
			gl_Position = vec4(vAR * flat_earth * zoom * (position + mrot * vertex_in.yx), 0.0, 1.0);
			texcoords_fs = texcoords_in.ts;
			break;	
	}
}