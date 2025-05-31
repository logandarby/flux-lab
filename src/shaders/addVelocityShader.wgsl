// Add velocity at a specific position based on mouse movement

struct AddVelocityInput {
    position_x: f32,
    position_y: f32,
    velocity_x: f32,
    velocity_y: f32,
    radius: f32,
    intensity: f32,
};

@group(0) @binding(0) var velocity_in: texture_2d<f32>;
@group(0) @binding(1) var velocity_out: texture_storage_2d<rg32float, write>;
@group(0) @binding(2) var<uniform> input: AddVelocityInput;

@compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
fn compute_main(
    @builtin(global_invocation_id) texture_coord: vec3<u32>,
) {
    let coord = vec2<i32>(texture_coord.xy);
    let texture_size = vec2<i32>(textureDimensions(velocity_in));
    
    if (coord.x >= texture_size.x || coord.y >= texture_size.y) {
        return;
    }
    
    // Get current velocity
    let current_velocity = textureLoad(velocity_in, coord, 0).xy;
    
    // Calculate distance from mouse position
    let pixel_pos = vec2<f32>(f32(coord.x), f32(coord.y));
    let mouse_pos = vec2<f32>(input.position_x, input.position_y);
    let distance = length(pixel_pos - mouse_pos);
    
    // Add velocity with smooth falloff
    var added_velocity = vec2<f32>(0.0, 0.0);
    if (distance <= input.radius) {
        let falloff = 1.0 - (distance / input.radius);
        let smooth_falloff = falloff * falloff * (3.0 - 2.0 * falloff); // Smoothstep
        let velocity_delta = vec2<f32>(input.velocity_x, input.velocity_y);
        added_velocity = velocity_delta * input.intensity * smooth_falloff;
    }
    
    let new_velocity = current_velocity + added_velocity;
    textureStore(velocity_out, coord, vec4<f32>(new_velocity.x, new_velocity.y, 0.0, 0.0));
} 