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
    // RGB color for smoke shading
    smokeColor: vec3f,
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
    
    // Get base color from uniform
    let baseColor = uniformInput.smokeColor;
    
    // Add complementary tinting to shadows for more visual interest
    let shadowTint = vec3f(0.1, 0.05, 0.2); // Cool blue tint for shadows
    let highlightBoost = vec3f(1.3, 1.2, 1.4); // Warm boost for highlights
    
    // Enhanced gradient stops with better contrast and vibrancy
    let color1 = (baseColor * 0.05) + shadowTint;
    let color2 = (baseColor * 0.2) + shadowTint * 0.5;
    let color3 = baseColor * 0.4;
    let color4 = baseColor * 0.7;
    let color5 = baseColor * 1.0 + vec3f(0.1, 0.1, 0.1);
    let color6 = baseColor * highlightBoost;
    let color7 = mix(baseColor * 1.5, vec3f(1.0, 1.0, 1.0), 0.3);
    let color8 = vec3f(1.2, 1.2, 1.2);
    
    var finalColor: vec3f;
    
    // multi-stop gradient
    if (density < 0.15) {
        let t = density / 0.15;
        finalColor = mix(color1, color2, smoothstep(0.0, 1.0, t));
    } else if (density < 0.3) {
        let t = (density - 0.15) / 0.15;
        finalColor = mix(color2, color3, smoothstep(0.0, 1.0, t));
    } else if (density < 0.45) {
        let t = (density - 0.3) / 0.15;
        finalColor = mix(color3, color4, smoothstep(0.0, 1.0, t));
    } else if (density < 0.6) {
        let t = (density - 0.45) / 0.15;
        finalColor = mix(color4, color5, smoothstep(0.0, 1.0, t));
    } else if (density < 0.75) {
        let t = (density - 0.6) / 0.15;
        finalColor = mix(color5, color6, smoothstep(0.0, 1.0, t));
    } else if (density < 0.9) {
        let t = (density - 0.75) / 0.15;
        finalColor = mix(color6, color7, smoothstep(0.0, 1.0, t));
    } else {
        let t = (density - 0.9) / 0.1;
        finalColor = mix(color7, color8, smoothstep(0.0, 1.0, t));
    }
    
    // Add subtle saturation boost for mid-tones
    let luminance = dot(finalColor, vec3f(0.299, 0.587, 0.114));
    let saturationBoost = 1.0 + (1.0 - abs(luminance - 0.5) * 2.0) * 0.2;
    finalColor = mix(vec3f(luminance), finalColor, saturationBoost);
    
    let alpha = density * density * (3.0 - 2.0 * density); // Smoothstep for better falloff
    
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
        default: {
            output = vec4(1, 1, 1, 1);
        }
    }

    return addNoise(output, input.texCoord);
}