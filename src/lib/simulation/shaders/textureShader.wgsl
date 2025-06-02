// --- VERTEX --- //

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) texCoord: vec2f,
}

@vertex
fn vertex_main(
    @builtin(vertex_index) vertexIndex: u32,
) -> VertexOutput {
    let squareVertices: array<vec2f, 6> = array<vec2f, 6>(
        vec2f(-1.0, -1.0),
        vec2f(1.0, -1.0),
        vec2f(-1.0, 1.0),
        vec2f(-1.0, 1.0),
        vec2f(1.0, -1.0),
        vec2f(1.0, 1.0),
    );
    var output: VertexOutput;
    let xy = squareVertices[vertexIndex];
    output.position = vec4f(xy, 0.0, 1.0);
    // Normalize to 0-1, and flip y
    let normalized = xy * 0.5 + 0.5;
    output.texCoord = vec2f(normalized.x, 1.0 - normalized.y);
    return output;
}

// --- FRAGMENT --- //

@define PI 3.141592653589793238462
@define VELOCITY_MODE 0
@define PRESSURE_MODE 1
@define DENSITY_MODE 2
@define PARTICLES_MODE 3

struct FragmentInput {
    shaderMode: u32,
    // mean and std of gaussian noise sampling
    stddev: f32,
    mean: f32,
    // 3 pseudo random sampled values from the GPU
    offsets: vec3f,
}

@group(0) @binding(0) var texture: texture_2d<f32>;
@group(0) @binding(1) var textureSampler: sampler;
@group(0) @binding(2) var<uniform> uniformInput: FragmentInput;

// All components are in the range [0…1], including hue.
fn hsv2rgb(c: vec3f) -> vec3f
{
    let K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, vec3f(0.0), vec3f(1.0)), c.y);
}

fn shadeVelocity(value: vec2f, input: VertexOutput) -> vec4f {
    let DAMP = 30.0;
    let angle = acos((value.y * value.y) / (dot(value.xy, value.xy)));
    let magnitude = sqrt(dot(value.xy, value.xy)) / DAMP;
    let color = hsv2rgb(vec3f(angle / 8 + PI / 2, 1, 1));
    return vec4f(color * magnitude, 0);
}

fn shadePressure(value: f32, input: VertexOutput) -> vec4f {
    // Pressure
    let DAMP = 30.0;
    if (value < 0) {
        return vec4f(-value, 0, 0, 1) / DAMP;
    } else {
        return vec4f(0, 0, value, 1) / DAMP;
    }
}

fn shadeDensity(value: f32, input: VertexOutput) -> vec4f {
    let density = clamp(value, 0.0, 1.0);
    
    let color1 = vec3f(0.05, 0.05, 0.15);
    let color2 = vec3f(0.15, 0.05, 0.35);
    let color3 = vec3f(0.45, 0.15, 0.65);
    let color4 = vec3f(0.65, 0.35, 0.85);
    let color5 = vec3f(0.75, 0.65, 0.95);
    let color6 = vec3f(0.95, 0.95, 1.0);
    
    var finalColor: vec3f;
    
    // Multi-stop gradient interpolation
    if (density < 0.2) {
        let t = density / 0.2;
        finalColor = mix(color1, color2, t);
    } else if (density < 0.4) {
        let t = (density - 0.2) / 0.2;
        finalColor = mix(color2, color3, t);
    } else if (density < 0.6) {
        let t = (density - 0.4) / 0.2;
        finalColor = mix(color3, color4, t);
    } else if (density < 0.8) {
        let t = (density - 0.6) / 0.2;
        finalColor = mix(color4, color5, t);
    } else {
        let t = (density - 0.8) / 0.2;
        finalColor = mix(color5, color6, t);
    }
    
    // Add some subtle brightness boost for glow effect
    let brightness = 1.0 + density * 0.3;
    // finalColor = finalColor * brightness;
    
    // Smooth alpha transition - more transparent at low densities
    let alpha = density;
    // let alpha = density * density * (3.0 - 2.0 * density); // Smoothstep
    return vec4f(finalColor, alpha);
}

fn rand(co: vec2f) -> f32 {
    let r = fract(sin(dot(co.xy, vec2f(12.9898,78.233))) * 43758.5453);
    if (r == 0.0) {
        return 0.000000000001;
    }
    else {
        return r;
    }
}

fn gaussrand(co: vec2f) -> vec4f {
    let offsets = uniformInput.offsets;
    let U = rand(co + vec2(offsets.x, offsets.x));
    let V = rand(co + vec2(offsets.y, offsets.y));
    let R = rand(co + vec2(offsets.z, offsets.z));
    var Z = sqrt(-2.0 * log(U)) * sin(2.0 * PI * V);
    Z = Z * uniformInput.stddev + uniformInput.mean;
    return vec4f(Z, Z, Z, 0.0);
}

fn addNoise(color: vec4f, texCoord: vec2f) -> vec4f {
    let magnitude = sqrt(dot(color.xy, color.xy));
    let noise = gaussrand(texCoord) ;
    let baseNoise = gaussrand(texCoord) * 0.2;
    return color + noise * magnitude + baseNoise;
}

// TODO: This is currently inefficient. We should be using instanced particle rendering/compute particle rendering probably.
fn shadeParticles(particlePosition: vec2f, input: VertexOutput) -> vec4f {
    // Since texcoord is in [0, 1] we need to multiply by grid size to get the actual position512.0, 512.0);
    let gridSize = vec2f(512.0, 512.0);
    let currentPos = input.texCoord * gridSize;
    
    // Calculate distance from current pixel to particle position
    let distance = length(currentPos - particlePosition);
    let particleRadius = 2.0; // Particle size in pixels
    
    // Render particle as a circle with smooth falloff
    if (distance <= particleRadius) {
        let falloff = 1.0 - (distance / particleRadius);
        let intensity = falloff * falloff; // Smooth falloff
        
        // Particle color (could be based on velocity, density, etc.)
        let particleColor = vec3f(1.0, 0.8, 0.4); // Warm orange
        return vec4f(particleColor * intensity, intensity);
    }

    return vec4f(0.0, 0.0, 0.0, 0.0);
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0)vec4f {
    let value = textureSample(texture, textureSampler, input.texCoord);

    var output: vec4f;

    switch uniformInput.shaderMode {
        case VELOCITY_MODE: {
            output = shadeVelocity(value.xy, input);
        }
        case PRESSURE_MODE: {
            output = shadePressure(value.x, input);
        }
        case DENSITY_MODE: {
            output = shadeDensity(value.x, input);
        }
        case PARTICLES_MODE: {
            output = shadeParticles(value.xy, input);
        }
        default: {
            output = vec4(1, 1, 1, 1);
        }
    }

    return addNoise(output, input.texCoord);
}