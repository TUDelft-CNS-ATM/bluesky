#version 330

in vec3 color_fs;
out vec4 color_out;

void main()
{
	color_out = vec4(color_fs, 1.0);
}