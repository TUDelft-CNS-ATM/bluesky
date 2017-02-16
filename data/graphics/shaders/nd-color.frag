#version 330

flat in int discard_instance;
in vec4 color_fs;
out vec4 color_out;

void main()
{
    if (discard_instance > 0)
        discard;
	color_out = color_fs;
}