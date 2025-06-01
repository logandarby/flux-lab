// Boundary conditions shader
// Enforces different boundary conditions on vector and scalar fields

struct BoundaryInput {
    boundary_type: u32, // BoundaryType: NO_SLIP_VELOCITY = 0, FREE_SLIP_VELOCITY = 1, SCALAR_NEUMANN = 2
    scale: f32,         // For scaling boundary values
};

@group(0) @binding(0) var field_in: texture_2d<f32>;
@group(0) @binding(1) var field_out: texture_storage_2d<${FORMAT}, write>;
@group(0) @binding(2) var<uniform> input: BoundaryInput;

fn enforce_no_slip_velocity(coord: vec2<i32>, texture_size: vec2<i32>) -> vec4<f32> {
    // No-slip: velocity = 0 at boundaries
    // For velocity field (2 components)
    
    // Corners: zero velocity
    if ((coord.x == 0 || coord.x == texture_size.x - 1) && 
        (coord.y == 0 || coord.y == texture_size.y - 1)) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    // Left and right boundaries
    if (coord.x == 0) {
        let interior = textureLoad(field_in, vec2<i32>(1, coord.y), 0);
        return vec4<f32>(-interior.x, 0.0, 0.0, 0.0); // Reflect x-component, zero y-component
    }
    if (coord.x == texture_size.x - 1) {
        let interior = textureLoad(field_in, vec2<i32>(texture_size.x - 2, coord.y), 0);
        return vec4<f32>(-interior.x, 0.0, 0.0, 0.0); // Reflect x-component, zero y-component
    }
    
    // Top and bottom boundaries
    if (coord.y == 0) {
        let interior = textureLoad(field_in, vec2<i32>(coord.x, 1), 0);
        return vec4<f32>(0.0, -interior.y, 0.0, 0.0); // Zero x-component, reflect y-component
    }
    if (coord.y == texture_size.y - 1) {
        let interior = textureLoad(field_in, vec2<i32>(coord.x, texture_size.y - 2), 0);
        return vec4<f32>(0.0, -interior.y, 0.0, 0.0); // Zero x-component, reflect y-component
    }
    
    // Interior points: copy unchanged
    return textureLoad(field_in, coord, 0);
}

fn enforce_free_slip_velocity(coord: vec2<i32>, texture_size: vec2<i32>) -> vec4<f32> {
    // Free-slip: normal component = 0, tangential component unchanged
    
    // Corners: zero velocity
    if ((coord.x == 0 || coord.x == texture_size.x - 1) && 
        (coord.y == 0 || coord.y == texture_size.y - 1)) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    // Left and right boundaries (normal is x-direction)
    if (coord.x == 0) {
        let interior = textureLoad(field_in, vec2<i32>(1, coord.y), 0);
        return vec4<f32>(0.0, interior.y, 0.0, 0.0); // Zero normal (x), preserve tangential (y)
    }
    if (coord.x == texture_size.x - 1) {
        let interior = textureLoad(field_in, vec2<i32>(texture_size.x - 2, coord.y), 0);
        return vec4<f32>(0.0, interior.y, 0.0, 0.0); // Zero normal (x), preserve tangential (y)
    }
    
    // Top and bottom boundaries (normal is y-direction)
    if (coord.y == 0) {
        let interior = textureLoad(field_in, vec2<i32>(coord.x, 1), 0);
        return vec4<f32>(interior.x, 0.0, 0.0, 0.0); // Preserve tangential (x), zero normal (y)
    }
    if (coord.y == texture_size.y - 1) {
        let interior = textureLoad(field_in, vec2<i32>(coord.x, texture_size.y - 2), 0);
        return vec4<f32>(interior.x, 0.0, 0.0, 0.0); // Preserve tangential (x), zero normal (y)
    }
    
    // Interior points: copy unchanged
    return textureLoad(field_in, coord, 0);
}

fn enforce_scalar_boundary(coord: vec2<i32>, texture_size: vec2<i32>) -> vec4<f32> {
    // Neumann boundary conditions for scalar fields (pressure, density, etc.)
    // Zero-gradient: boundary value = adjacent interior value
    
    // Corners: copy from diagonal interior
    if (coord.x == 0 && coord.y == 0) {
        let interior = textureLoad(field_in, vec2<i32>(1, 1), 0);
        return interior;
    }
    if (coord.x == texture_size.x - 1 && coord.y == 0) {
        let interior = textureLoad(field_in, vec2<i32>(texture_size.x - 2, 1), 0);
        return interior;
    }
    if (coord.x == 0 && coord.y == texture_size.y - 1) {
        let interior = textureLoad(field_in, vec2<i32>(1, texture_size.y - 2), 0);
        return interior;
    }
    if (coord.x == texture_size.x - 1 && coord.y == texture_size.y - 1) {
        let interior = textureLoad(field_in, vec2<i32>(texture_size.x - 2, texture_size.y - 2), 0);
        return interior;
    }
    
    // Edge boundaries: copy from adjacent interior
    if (coord.x == 0) {
        let interior = textureLoad(field_in, vec2<i32>(1, coord.y), 0);
        return interior;
    }
    if (coord.x == texture_size.x - 1) {
        let interior = textureLoad(field_in, vec2<i32>(texture_size.x - 2, coord.y), 0);
        return interior;
    }
    if (coord.y == 0) {
        let interior = textureLoad(field_in, vec2<i32>(coord.x, 1), 0);
        return interior;
    }
    if (coord.y == texture_size.y - 1) {
        let interior = textureLoad(field_in, vec2<i32>(coord.x, texture_size.y - 2), 0);
        return interior;
    }
    
    // Interior points: copy unchanged
    return textureLoad(field_in, coord, 0);
}

@compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
fn compute_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let coord = vec2<i32>(global_id.xy);
    let texture_size = vec2<i32>(textureDimensions(field_in));
    
    // Check if we're at a boundary
    let is_boundary = coord.x == 0 || coord.x == texture_size.x - 1 || 
                     coord.y == 0 || coord.y == texture_size.y - 1;
    
    var result: vec4<f32>;
    
    if (!is_boundary) {
        // Interior points: copy unchanged
        result = textureLoad(field_in, coord, 0);
    } else {
        // Apply boundary conditions based on type
        switch (input.boundary_type) {
            case 0u: { // No-slip velocity
                result = enforce_no_slip_velocity(coord, texture_size);
            }
            case 1u: { // Free-slip velocity
                result = enforce_free_slip_velocity(coord, texture_size);
            }
            case 2u: { // Scalar (Neumann)
                result = enforce_scalar_boundary(coord, texture_size);
            }
            default: {
                result = textureLoad(field_in, coord, 0);
            }
        }
    }
    
    textureStore(field_out, coord, result);
} 