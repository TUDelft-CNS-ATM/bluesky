#version 330

flat in int discard_instance;
// Interpolated values from the vertex shaders
in vec3 texcoords_fs;
in vec4 color_fs;

// Ouput data
out vec4 color;

// Values that stay constant for the whole mesh.
const float smoothing = 1.0/5.0;
const vec4 outlineColor = vec4(0.0, 0.0, 0.0, 1.0);
const float outlineDistance = 0.2;

const vec3 shadowOffset = vec3(0.05, 0.1, 0.0); // Between 0 and spread / textureSize
const float shadowSmoothing = 0.5; // Between 0 and 0.5
const vec4 shadowColor = vec4(0.0, 0.0, 0.0, 0.8);

float calcdist(vec4 tex) {
    // Calculate the median of the three distance fields r, g, b
    return max(min(tex.r, tex.g), min(max(tex.r, tex.g), tex.b));
}

uniform sampler2DArray tex_sampler;
 
void main()
{ 
    if (discard_instance > 0)
        discard;

    // With an outline
    // float dist = calcdist(texture(tex_sampler, texcoords_fs));
    // float outlineFactor = smoothstep(0.5 - smoothing, 0.5 + smoothing, dist);
    // color = mix(outlineColor, color_fs, outlineFactor);
    // float alpha = smoothstep(outlineDistance - smoothing, outlineDistance + smoothing, dist);
    // color.a *= alpha;


    // With a dropshadow
    float dist = calcdist(texture(tex_sampler, texcoords_fs));
    float alpha = smoothstep(0.5 - smoothing, 0.5 + smoothing, dist);
    vec4 text = vec4(color_fs.rgb, color_fs.a * alpha);

    float shadowDistance = calcdist(texture(tex_sampler, texcoords_fs - shadowOffset));
    float shadowAlpha = smoothstep(0.5 - shadowSmoothing, 0.5 + shadowSmoothing, shadowDistance);
    vec4 shadow = vec4(shadowColor.rgb, shadowColor.a * shadowAlpha);

    color = mix(shadow, text, text.a);
}