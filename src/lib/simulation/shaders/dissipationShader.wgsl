struct DissipationUniforms {
  dissipation_factor: f32,
}

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<${FORMAT}, write>;
@group(0) @binding(2) var<uniform> uniforms: DissipationUniforms;

@compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
fn compute_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let coord = vec2<i32>(global_id.xy);
  let dimensions = textureDimensions(input_texture);
  
  // Check bounds
  if (coord.x >= i32(dimensions.x) || coord.y >= i32(dimensions.y)) {
    return;
  }
  
  let current_value = textureLoad(input_texture, coord, 0);
  let dissipated_value = current_value * uniforms.dissipation_factor;
  
  textureStore(output_texture, coord, vec4<f32>(dissipated_value));
} 