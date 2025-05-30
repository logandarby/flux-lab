// Calculates the gradient of the pressure, and subtracts it from the velocity field

struct GradientSubtractionInput {
    half_rdx: f32, // Half the grid scale
};

@group(0) @binding(0) var pressure_in: texture_2d<f32>;
@group(0) @binding(1) var velocity_in: texture_2d<f32>;
@group(0) @binding(2) var subtraction_out: texture_storage_2d<rg32float, write>;
@group(0) @binding(3) var<uniform> input: GradientSubtractionInput;

@compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
fn compute_main(
    @builtin(global_invocation_id) u_coords: vec3<u32>
) {
    let coord = vec2<i32>(u_coords.xy);
    let texture_size = vec2<i32>(textureDimensions(pressure_in));

    // Edge case, just write 0
    if (coord.x == 0 || coord.x == texture_size.x - 1 || coord.y == 0 || coord.y == texture_size.y - 1) {
        textureStore(subtraction_out, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let left    = textureLoad(pressure_in, coord - vec2<i32>(1, 0), 0);
    let right   = textureLoad(pressure_in, coord + vec2<i32>(1, 0), 0);
    let up      = textureLoad(pressure_in, coord - vec2<i32>(0, 1), 0);
    let down    = textureLoad(pressure_in, coord + vec2<i32>(0, 1), 0);

    let pressure_gradient = vec2<f32>(right.x - left.x, up.x - down.x) * input.half_rdx;
    let velocity = textureLoad(velocity_in, coord, 0);
    let result = velocity.xy - pressure_gradient;
    textureStore(subtraction_out, coord, vec4<f32>(result.x, result.y, 0.0, 0.0));
}