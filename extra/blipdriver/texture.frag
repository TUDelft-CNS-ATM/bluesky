#version 330
 
// Interpolated values from the vertex shaders
in vec4 color_fs;
in vec3 texcoords_fs;

// Ouput data
out vec4 color;
 
// Values that stay constant for the whole mesh.
uniform sampler2D tex_sampler;
 
void main()
{ 
    // Output color = color of the texture at the specified UV
    color = texture(tex_sampler, texcoords_fs.st);
    color.a = color_fs.a;
}