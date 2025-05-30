// Advect the velocities in the texture

struct AdvectionInput {
    timestep: f32,   // delta time
    rdx: f32,        // 1 / the grid scale
};

@group(0) @binding(0) var velocity: texture_2d<f32>;
@group(0) @binding(1) var advection_in: texture_2d<f32>;
@group(0) @binding(2) var advection_out: texture_storage_2d<${OUT_FORMAT}, write>;
@group(0) @binding(3) var<uniform> input: AdvectionInput;


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

@compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
fn compute_main(
    @builtin(global_invocation_id) texture_coord: vec3<u32>,
) {
    let coord = vec2<i32>(texture_coord.xy);
    let texture_size = vec2<i32>(textureDimensions(velocity));
    // if (coord.x >= texture_size.x || coord.y >= texture_size.y) {
    //     // TODO: Indicator for glitches
    //     textureStore(advection_out, coord, vec4<f32>(1, 1, 0, 0));
    //     return;
    // }
    let current_velocity = textureLoad(velocity, coord, 0).xy;
    let prev_coord = vec2f(coord) - input.timestep * input.rdx * current_velocity;

    // Clamp to valid texture bounds
    let clamped_prev_coord = clamp(prev_coord, vec2f(0.0), vec2f(f32(texture_size.x) - 1.0, f32(texture_size.y) - 1.0));
    
    // Use manual bilinear interpolation instead of sampler
    let advected_quantity = bilinear_interpolate(advection_in, clamped_prev_coord);
    
    textureStore(advection_out, coord, vec4<f32>(advected_quantity.x, advected_quantity.y, 0.0, 0.0));
}