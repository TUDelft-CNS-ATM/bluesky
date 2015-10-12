#version 330

flat in int discard_instance;
// Interpolated values from the vertex shaders
in vec3 texcoords_fs;
in vec3 color_fs;

// Ouput data
out vec4 color;
 
// Values that stay constant for the whole mesh.
uniform sampler2DArray tex_sampler;
 
void main()
{ 
    if (discard_instance > 0)
        discard;
    // Output color = color of the texture at the specified UV
    //vec4 texcolor = texture(tex_sampler, texcoords_fs);
    //color = vec4(color_fs, 1.0);
    color = texture(tex_sampler, texcoords_fs) * vec4(color_fs, 1.0);
}