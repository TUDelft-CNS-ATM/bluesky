#version 410

// Input data
layout(origin_upper_left) in vec4 gl_FragCoord;

// Ouput data
out vec4 color;

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
    float screen_pixel_ratio; // > 1 for high-DPI screens
};

// Top-left tile offset and scale
uniform vec4 offset_scale;
// Tile index texture. The bound texture should be an integer texture
uniform isampler2D tile_index;
// Tile texture array
uniform sampler2DArray tile_texture;


void main()
{
    // Fragment coordinates are integer pixel coordinates with (0,0) in top-left corner
    // Dividing by viewport gives glcoords from (0,0) top left to (1,1) bottom right
    vec2 glcoords = gl_FragCoord.xy / vec2(screen_width * screen_pixel_ratio, screen_height * screen_pixel_ratio);

    // Step 1: convert GL screen coordinates to texture coordinates for the tile index texture
    vec2 idxcoords = glcoords * offset_scale.zw + offset_scale.xy;
    // We need the dimensions of the index texture
    // ivec2 idxsize = textureSize(tile_index, 0);
    // if (any(greaterThan(idxcoords, idxsize))) {
    //     // Discard pixels that point beyond the tile texture array bounds
    //     discard;
    // }
    // The index texture lookup returns a tile index, that tile's zoom factor relative
    // to the main tile zoom, and an offset into the tile for tiles that cover a bigger area
    // than the main tile.
    ivec4 offset_zoom_idx = texelFetch(tile_index, ivec2(floor(idxcoords)), 0);

    // Step 2: determine the texture coordinates for the tile texture array
    // if (offset_zoom_idx.w < 0) {
    //     // A negative array index means that there is no tile available
    //     discard;
    // }    
    // The fractional part of the floating point pixel coordinates in the index texture forms the coordinates in
    // the current cell.
    vec2 maintilecoords = fract(idxcoords);
    // The actual texture coordinate of the current tile depends on the current tile zoom, which is given relative to the main tile/cell size
    vec3 tilecoords = vec3((maintilecoords + offset_zoom_idx.xy) / offset_zoom_idx.z, offset_zoom_idx.w);
    // Output color = color of the texture at the specified UV
    color = texture(tile_texture, tilecoords);
    // color = texture(tile_texture, vec3(glcoords, 0));
}