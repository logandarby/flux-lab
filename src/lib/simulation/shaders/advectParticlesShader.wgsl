@include "bilinearInterpolate.wgsl"

struct AdvectParticlesUniforms {
    timestep: f32,
}

@group(0) @binding(0) var velocity: texture_2d<f32>;
@group(0) @binding(1) var particle_position: texture_2d<f32>;
@group(0) @binding(2) var particle_position_out: texture_storage_2d<rg32float, write>;
@group(0) @binding(3) var<uniform> uniforms: AdvectParticlesUniforms;

@compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
fn compute_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
   let grid_size = vec2f(textureDimensions(velocity));
   let current_position = textureLoad(particle_position, global_id.xy, 0).xy;
   let velocity_at_position = bilinear_interpolate(velocity, current_position);
   let new_position = current_position + velocity_at_position * uniforms.timestep;
   textureStore(particle_position_out, global_id.xy, vec4f(new_position, 0.0, 0.0));
}