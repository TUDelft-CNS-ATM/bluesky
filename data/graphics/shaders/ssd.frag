#version 330
in vec2 ssd_coord;
in vec4 color_fs;
out vec4 color_out;

// Vlimits is [Vmin^2, Vmax^2, Vmax]
uniform vec3 Vlimits;

void main()
{
    float Vsquared    = dot(ssd_coord, ssd_coord);
    if (Vsquared < Vlimits[0] || Vsquared > Vlimits[1]) {
        discard;
    } else {
        //color_out = vec4(color_fs, smoothstep(Vlimits[0], Vlimits[1], Vsquared));
        color_out = color_fs;
    }
}
