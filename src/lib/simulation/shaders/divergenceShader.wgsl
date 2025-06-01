// Calculate divergence of velocity field

struct DivergenceInput {
    half_rdx: f32, // Half the grid scale
}

@group(0) @binding(0) var velocity_in: texture_2d<f32>;
@group(0) @binding(1) var divergence_out: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> input: DivergenceInput;

@compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
fn compute_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let coord = vec2<i32>(global_id.xy);
    let texture_size = vec2<i32>(textureDimensions(velocity_in));

    // Edge case: Simply write 0
    if (coord.x == 0 || coord.x == texture_size.x - 1 || coord.y == 0 || coord.y == texture_size.y - 1) {
        textureStore(divergence_out, coord, vec4f(0.0));
        return;
    }

    let center = textureLoad(velocity_in, coord, 0);
    let left = textureLoad(velocity_in, coord + vec2<i32>(-1, 0), 0);
    let right = textureLoad(velocity_in, coord + vec2<i32>(1, 0), 0);
    let down = textureLoad(velocity_in, coord + vec2<i32>(0, 1), 0);
    let up = textureLoad(velocity_in, coord + vec2<i32>(0, -1), 0);

    let divergence = (right.x - left.x + down.y - up.y) * input.half_rdx;
    textureStore(divergence_out, coord, vec4f(divergence, 0.0, 0.0, 0.0));
}