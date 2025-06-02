// Advect the velocities in the texture

@include "bilinearInterpolate.wgsl"

struct AdvectionInput {
    timestep: f32,   // delta time
    rdx: f32,        // 1 / the grid scale
};

@group(0) @binding(0) var velocity: texture_2d<f32>;
@group(0) @binding(1) var advection_in: texture_2d<f32>;
@group(0) @binding(2) var advection_out: texture_storage_2d<${OUT_FORMAT}, write>;
@group(0) @binding(3) var<uniform> input: AdvectionInput;


@compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
fn compute_main(
    @builtin(global_invocation_id) texture_coord: vec3<u32>,
) {
    let coord = vec2<i32>(texture_coord.xy);
    let texture_size = vec2<i32>(textureDimensions(velocity));
    let current_velocity = textureLoad(velocity, coord, 0).xy;
    let prev_coord = vec2f(coord) - input.timestep * input.rdx * current_velocity;

    // Clamp to valid texture bounds
    let clamped_prev_coord = clamp(prev_coord, vec2f(0.0), vec2f(f32(texture_size.x) - 1.0, f32(texture_size.y) - 1.0));
    
    // Use manual bilinear interpolation instead of sampler
    let advected_quantity = bilinear_interpolate(advection_in, clamped_prev_coord);
    
    textureStore(advection_out, coord, vec4<f32>(advected_quantity.x, advected_quantity.y, 0.0, 0.0));
}