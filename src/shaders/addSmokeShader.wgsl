// Add smoke density at a specific position

struct AddSmokeInput {
    position_x: f32,
    position_y: f32,
    radius: f32,
    intensity: f32,
};

@group(0) @binding(0) var smoke_in: texture_2d<f32>;
@group(0) @binding(1) var smoke_out: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> input: AddSmokeInput;

@compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
fn compute_main(
    @builtin(global_invocation_id) texture_coord: vec3<u32>,
) {
    let coord = vec2<i32>(texture_coord.xy);
    let texture_size = vec2<i32>(textureDimensions(smoke_in));
    
    if (coord.x >= texture_size.x || coord.y >= texture_size.y) {
        return;
    }
    
    // Get current smoke density
    let current_smoke = textureLoad(smoke_in, coord, 0).r;
    
    // Calculate distance from click position
    let pixel_pos = vec2<f32>(f32(coord.x), f32(coord.y));
    let click_pos = vec2<f32>(input.position_x, input.position_y);
    let distance = length(pixel_pos - click_pos);
    
    // Add smoke with smooth falloff
    var added_smoke = 0.0;
    if (distance <= input.radius) {
        let falloff = 1.0 - (distance / input.radius);
        let smooth_falloff = falloff * falloff * (3.0 - 2.0 * falloff); // Smoothstep
        added_smoke = input.intensity * smooth_falloff;
    }
    
    let new_smoke = min(current_smoke + added_smoke, 1.0); // Clamp to [0,1]
    textureStore(smoke_out, coord, vec4<f32>(new_smoke, 0.0, 0.0, 0.0));
} 