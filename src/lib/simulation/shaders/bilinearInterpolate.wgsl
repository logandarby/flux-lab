// Manual bilinear interpolation function
fn bilinear_interpolate(texture: texture_2d<f32>, coord: vec2<f32>) -> vec2<f32> {
    let texture_size = vec2<f32>(textureDimensions(texture));
    
    // Get the four surrounding pixel coordinates
    let coord_floor = floor(coord);
    let coord_fract = coord - coord_floor;
    
    let coord_i = vec2<i32>(coord_floor);
    let coord_i_plus1 = coord_i + vec2<i32>(1, 1);
    
    // Clamp coordinates to texture bounds
    let max_coord = vec2<i32>(texture_size) - vec2<i32>(1, 1);
    let tl = clamp(coord_i, vec2<i32>(0, 0), max_coord);
    let tr = clamp(vec2<i32>(coord_i.x + 1, coord_i.y), vec2<i32>(0, 0), max_coord);
    let bl = clamp(vec2<i32>(coord_i.x, coord_i.y + 1), vec2<i32>(0, 0), max_coord);
    let br = clamp(coord_i_plus1, vec2<i32>(0, 0), max_coord);
    
    // Sample the four surrounding texels
    let val_tl = textureLoad(texture, tl, 0).xy;
    let val_tr = textureLoad(texture, tr, 0).xy;
    let val_bl = textureLoad(texture, bl, 0).xy;
    let val_br = textureLoad(texture, br, 0).xy;
    
    // Perform bilinear interpolation
    let top = mix(val_tl, val_tr, coord_fract.x);
    let bottom = mix(val_bl, val_br, coord_fract.x);
    let result = mix(top, bottom, coord_fract.y);
    
    return result;
}