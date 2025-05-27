// Advect the velocities in the texture

@group(0) @binding(0) var velocity: texture_2d<f32>;
@group(0) @binding(1) var advection_in: texture_2d<f32>;
@group(0) @binding(2) var advection_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var advection_sampler: sampler;

struct AdvectionInput {
    @location(0) timestep: f32,   // delta time
    @location(1) rdx: f32,        // 1 / the grid scale
};

@compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
fn compute_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    // // TODO: Temporary
    var input: AdvectionInput;
    input.timestep = 1.0/30.0;
    input.rdx = 0.1;
    //

    let coord = vec2<i32>(global_id.xy);
    let texture_size = vec2<i32>(textureDimensions(velocity));
    if (coord.x >= texture_size.x || coord.y >= texture_size.y) {
        return;
    }
    let current_velocity = textureLoad(velocity, coord, 0).xy;
    let prev_coord = vec2f(coord) - input.timestep * input.rdx * current_velocity;

    // Clamp to valid texture bounds instead of writing zeros
    let clamped_prev_coord = clamp(prev_coord, vec2f(0.0), vec2f(f32(texture_size.x) - 1.0, f32(texture_size.y) - 1.0));
    
    // Convert pixel coordinates to normalized coordinates for sampling
    let prev_coord_normalized = clamped_prev_coord / vec2f(texture_size);
    // let advected_quantity = vec2f(0.5, 0.5);
    let advected_quantity = textureSampleLevel(advection_in, advection_sampler, prev_coord_normalized, 0.0).xy;
    
    textureStore(advection_out, coord, vec4<f32>(advected_quantity.x, advected_quantity.y, 0.0, 0.0));
}