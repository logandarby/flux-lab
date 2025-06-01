// Jacobi iteration solver for Poisson equation Ax = b (A = Laplacian)
// Diffusion: x=velocity, b=velocity
// Pressure: x=pressure, b=velocity divergence

struct JacobiInput {
    alpha: f32,
    rBeta: f32,
};

@group(0) @binding(0) var x_in: texture_2d<f32>;
@group(0) @binding(1) var b_in: texture_2d<f32>;
@group(0) @binding(2) var texture_out: texture_storage_2d<${FORMAT}, write>;
@group(0) @binding(3) var<uniform> input: JacobiInput;

@compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
fn compute_main(
    @builtin(global_invocation_id) u_coords: vec3<u32>
) {
    let coords = u_coords.xy;
    let texture_size = textureDimensions(x_in);
    let i_coords = vec2i(coords);
    
    // Imposes Neumann boundaries (reflecting)
    
    let left_x = max(i_coords.x - 1, 1 - i_coords.x);
    let xl = textureLoad(x_in, vec2u(u32(left_x), coords.y), 0);
    
    let right_x = min(i_coords.x + 1, 2 * i32(texture_size.x) - 3 - i_coords.x);
    let xr = textureLoad(x_in, vec2u(u32(right_x), coords.y), 0);
    
    let top_y = max(i_coords.y - 1, 1 - i_coords.y);
    let xt = textureLoad(x_in, vec2u(coords.x, u32(top_y)), 0);
    
    let bottom_y = min(i_coords.y + 1, 2 * i32(texture_size.y) - 3 - i_coords.y);
    let xb = textureLoad(x_in, vec2u(coords.x, u32(bottom_y)), 0);

    let bc = textureLoad(b_in, coords, 0);

    let result = (xl + xr + xt + xb + input.alpha * bc) * input.rBeta;
    textureStore(texture_out, coords, result);
}