// Uses Jacobi iterations to solve pressure Poisson equation

struct JacobiInput {
    alpha: f32,
    rBeta: f32,
};

@group(0) @binding(0) var x_in: texture_2d<f32>;
@group(0) @binding(1) var b_in: texture_2d<f32>;
@group(0) @binding(2) var texture_out: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var<uniform> input: JacobiInput;

@compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
fn compute_main(
    @builtin(global_invocation_id) u_coords: vec3<u32>
) {

    let coords = vec2u(u_coords.xy);
    let texture_size = textureDimensions(x_in);
    let origin = vec2u(0, 0);
    let max_coord = vec2u(texture_size) - vec2u(1, 1);

    let xl = textureLoad(x_in, clamp(coords - vec2u(1, 0), origin, max_coord), 0);
    let xr = textureLoad(x_in, clamp(coords + vec2u(1, 0), origin, max_coord), 0);
    let xt = textureLoad(x_in, clamp(coords - vec2u(0, 1), origin, max_coord), 0);
    let xb = textureLoad(x_in, clamp(coords + vec2u(0, 1), origin, max_coord), 0);

    let bc = textureLoad(b_in, coords, 0);

    // For pressure, we only work with the x component (red channel)
    let result = (xl.x + xr.x + xt.x + xb.x + input.alpha * bc.x) * input.rBeta;
    textureStore(texture_out, coords, vec4<f32>(result, 0.0, 0.0, 0.0));
} 