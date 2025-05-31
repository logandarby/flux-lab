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
    // Velocity
    let pi = 3.14159;
    // let angle = atan(value.y / value.x);
    let angle = acos((value.y * value.y) / (dot(value.xy, value.xy)));
    let magnitude = sqrt(dot(value.xy, value.xy));
    // let color = vec3f(cos(angle), cos(angle + 2 * pi / 3), cos(angle - 2 * pi / 3));
    let color = hsv2rgb(vec3f(angle / 8 + pi / 2, 1, 1));
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
    let alpha = density * density * (3.0 - 2.0 * density); // Smoothstep
    
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
  // Box-Muller method for sampling from the normal distribution
  // http://en.wikipedia.org/wiki/Normal_distribution#Generating_values_from_normal_distribution
  // This method requires 2 uniform random inputs and produces 2 
  // Gaussian random outputs.  We'll take a 3rd random variable and use it to
  // switch between the two outputs.
  let PI = 3.141592653589793238462;
  var Z: f32;
  let offsets = uniformInput.offsets;
  let U = rand(co + vec2(offsets.x, offsets.x));
  let V = rand(co + vec2(offsets.y, offsets.y));
  let R = rand(co + vec2(offsets.z, offsets.z));
  // Switch between the two random outputs.
  if(R < 0.5) {
    Z = sqrt(-2.0 * log(U)) * sin(2.0 * PI * V);
  }
  else {
    Z = sqrt(-2.0 * log(U)) * cos(2.0 * PI * V);
  }

  // Apply the stddev and mean.
  Z = Z * uniformInput.stddev + uniformInput.mean;
  return vec4(Z, Z, Z, 0.0);
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0)vec4f {
    let value = textureSample(texture, textureSampler, input.texCoord);

    var output: vec4f;

    switch uniformInput.shaderMode {
        case 0: {
            output = shadeVelocity(value.xy, input);
        }
        case 1: {
            output = shadePressure(value.x, input);
        }
        case 2: {
            output = shadeDensity(value.x, input);
        }
        default: {
            output = vec4(1, 1, 1, 1);
        }
    }
    // Add noise
    return output + gaussrand(input.texCoord.xy);
}