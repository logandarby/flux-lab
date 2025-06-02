/**
 * WGSL Preprocessor
 *
 * A simple preprocessor for WebGPU Shading Language (WGSL) that provides:
 *
 * - **@include directives**: Import and compose shader modules
 *   Example: @include "common/math.wgsl"
 *
 * - **@define directives**: Define compile-time constants (C-style)
 *   Example: @define PI 3.14159, @define MAX_ITERATIONS 100
 *
 * - **Template variables**: Runtime variable substitution
 *   Example: ${WORKGROUP_SIZE}, ${FORMAT}
 *
 * - **Error handling**: Detailed error messages with line context and suggestions
 *   for debugging circular imports, missing shaders, and syntax errors
 *
 * Processing order: @define → @include → define substitution → template variables
 *
 * Usage:
 *   shaderRegistry.register("myShader", shaderSource);
 *   const processed = wgsl("myShader", { variables: { SIZE: 64 } });
 */

import {
  WGSLRegistryError,
  WGSLError,
  WGSLShaderNotFoundError,
  WGSLMaxImportDepthError,
  WGSLInvalidIncludeError,
  WGSLCircularImportError,
  WGSLVariableError,
  WGSLVariableNotFoundError,
  WGSLInvalidDefineError,
  WGSLDuplicateDefineError,
  WGSLInvalidDefineIdentifierError,
} from "./WGSLError";

export interface WGSLProprocessingOptions {
  variables?: Record<string, string | number>;
}

const WGSL_PREPROCESSOR_CONFIG = {
  maximumImportDepth: 10,
  importRegex: /@include\s*"([^"]+)"/g,
  importMatchRegex: /@include\s*"([^"]+)"/,
  variableRegex: /\$\{(\w+)\}/g,
  identifierRegex: /^[A-Za-z_][A-Za-z0-9_]*$/,
  defineMatchRegex: /@define\s+([A-Za-z_][A-Za-z0-9_]*)\s+(.+)$/,
  defineKeyword: "@define",
};

/**
 * Singleton registry for shader management and preprocessing.
 *
 * Used with the `wgsl` function to process shader strings.
 *
 * Example:
 *
 * ```ts
 * import { shaderRegistry } from "@/lib/preprocessor/core/wgsl";
 *
 * shaderRegistry.register("myShader", "...");
 * const shader = wgsl("myShader", {
 *   variables: {
 *     "WORKGROUP_SIZE": 1024,
 *     "OUT_FORMAT": "r32float",
 *   },
 * });
 * ```
 */
class ShaderRegistry {
  private static instance: ShaderRegistry;
  private registry = new Map<string, string>();

  private constructor() {}

  static getInstance(): ShaderRegistry {
    if (!ShaderRegistry.instance) {
      ShaderRegistry.instance = new ShaderRegistry();
    }
    return ShaderRegistry.instance;
  }

  register(name: string, content: string): ShaderRegistry {
    if (!name.trim()) {
      throw new WGSLRegistryError("Shader name cannot be empty", { name });
    }
    if (!content.trim()) {
      throw new WGSLRegistryError("Shader content cannot be empty", {
        name,
        contentLength: content.length,
      });
    }

    this.registry.set(name, content);
    return this;
  }

  registerMultiple(shaders: Record<string, string>): ShaderRegistry {
    const errors: string[] = [];

    Object.entries(shaders).forEach(([name, content]) => {
      try {
        this.register(name, content);
      } catch (error) {
        if (error instanceof WGSLError) {
          errors.push(`${name}: ${error.message}`);
        } else {
          errors.push(
            `${name}: ${error instanceof Error ? error.message : String(error)}`
          );
        }
      }
    });

    if (errors.length > 0) {
      throw new WGSLRegistryError(
        `Failed to register ${errors.length} shader(s)`,
        {
          failedShaders: errors,
          successfulCount: Object.keys(shaders).length - errors.length,
          totalCount: Object.keys(shaders).length,
        }
      );
    }

    return this;
  }

  private get(name: string): string | undefined {
    return this.registry.get(name);
  }

  private getAvailable(): string[] {
    return Array.from(this.registry.keys());
  }

  /**
   * Process a shader by name with includes and variable injection
   */
  process(shaderName: string, options?: WGSLProprocessingOptions): string {
    const shader = this.get(shaderName);
    if (!shader) {
      throw new WGSLShaderNotFoundError(shaderName, this.getAvailable());
    }
    return this.processContent(shader, options, 0, []);
  }

  /**
   * Process shader content directly
   */
  private processContent(
    shader: string,
    options?: WGSLProprocessingOptions,
    depth: number = 0,
    importStack: string[] = []
  ): string {
    if (depth > WGSL_PREPROCESSOR_CONFIG.maximumImportDepth) {
      throw new WGSLMaxImportDepthError(
        WGSL_PREPROCESSOR_CONFIG.maximumImportDepth,
        importStack
      );
    }

    // Process @define directives first
    const { processedShader, defines } = this.processDefines(shader);
    shader = processedShader;

    // Process @include imports
    shader = this.includeImports(shader, options, depth, importStack);

    // Apply defines to the final shader
    shader = this.applyDefines(shader, defines);

    // Process template variables last
    if (options?.variables) {
      shader = this.injectShaderVariables(shader, options.variables);
    }

    return shader;
  }

  /**
   * Process @define directives and return the shader with defines removed and a map of definitions
   */
  private processDefines(shader: string): {
    processedShader: string;
    defines: Map<string, string>;
  } {
    const defines = new Map<string, string>();
    const lines = shader.split("\n");
    const processedLines: string[] = [];

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const defineMatch = line.match(WGSL_PREPROCESSOR_CONFIG.defineMatchRegex);

      if (defineMatch) {
        const [_fullMatch, identifier, value] = defineMatch;
        const lineContext = `line ${i + 1}: "${line.trim()}"`;

        // Validate identifier
        if (!WGSL_PREPROCESSOR_CONFIG.identifierRegex.test(identifier)) {
          throw new WGSLInvalidDefineIdentifierError(identifier, lineContext);
        }

        // Check for duplicates
        if (defines.has(identifier)) {
          throw new WGSLDuplicateDefineError(
            identifier,
            defines.get(identifier)!,
            value.trim(),
            lineContext
          );
        }

        // Store the definition
        defines.set(identifier, value.trim());

        // Remove the @define line from output (replace with empty line to preserve line numbers)
        processedLines.push("");
      } else if (line.includes(WGSL_PREPROCESSOR_CONFIG.defineKeyword)) {
        // Invalid @define syntax
        const lineContext = `line ${i + 1}: "${line.trim()}"`;
        throw new WGSLInvalidDefineError(line.trim(), lineContext);
      } else {
        // Keep the line as-is
        processedLines.push(line);
      }
    }

    return {
      processedShader: processedLines.join("\n"),
      defines,
    };
  }

  /**
   * Apply define substitutions to the shader code
   */
  private applyDefines(shader: string, defines: Map<string, string>): string {
    if (defines.size === 0) return shader;

    // Sort by identifier length (descending) to avoid partial replacements
    const sortedDefines = Array.from(defines.entries()).sort(
      (a, b) => b[0].length - a[0].length
    );

    for (const [identifier, value] of sortedDefines) {
      // Create a regex that matches the identifier as a whole word
      const regex = new RegExp(`\\b${identifier}\\b`, "g");
      shader = shader.replace(regex, value);
    }

    return shader;
  }

  private includeImports(
    shader: string,
    options?: WGSLProprocessingOptions,
    depth: number = 0,
    importStack: string[] = []
  ): string {
    const matches = shader.match(WGSL_PREPROCESSOR_CONFIG.importRegex);
    if (!matches) return shader;

    matches.forEach((match) => {
      const filenameMatch = match.match(
        WGSL_PREPROCESSOR_CONFIG.importMatchRegex
      );
      if (!filenameMatch) {
        throw new WGSLInvalidIncludeError(
          match,
          this.getLineContext(shader, match)
        );
      }

      const filename = filenameMatch[1];

      if (importStack.includes(filename)) {
        throw new WGSLCircularImportError([...importStack, filename]);
      }

      const importContent = this.get(filename);
      if (!importContent) {
        throw new WGSLShaderNotFoundError(filename, this.getAvailable());
      }

      const processedContent = this.processContent(
        importContent,
        options,
        depth + 1,
        [...importStack, filename]
      );
      shader = shader.replace(match, processedContent);
    });

    return shader;
  }

  /**
   * Get line context for better error messages
   */
  private getLineContext(content: string, searchText: string): string {
    const lines = content.split("\n");
    const lineIndex = lines.findIndex((line) => line.includes(searchText));

    if (lineIndex === -1) return "unknown line";

    const lineNumber = lineIndex + 1;
    const line = lines[lineIndex].trim();

    return `line ${lineNumber}: "${line}"`;
  }

  /**
   * Injects variables into WGSL shader template strings
   */
  private injectShaderVariables(
    template: string,
    variables: Record<string, string | number>
  ): string {
    const availableVariables = Object.keys(variables);

    return template.replace(
      WGSL_PREPROCESSOR_CONFIG.variableRegex,
      (match, variableName) => {
        if (variableName in variables) {
          const value = variables[variableName];

          // Validate variable value type
          if (typeof value !== "string" && typeof value !== "number") {
            throw new WGSLVariableError(
              `Invalid variable type for '${variableName}'`,
              {
                variableName,
                actualType: typeof value,
                expectedTypes: "string | number",
                value: String(value),
              }
            );
          }

          return String(value);
        }

        throw new WGSLVariableNotFoundError(
          variableName,
          availableVariables,
          this.getLineContext(template, match)
        );
      }
    );
  }
}

/**
 * Access to the shader registry
 */
export const shaderRegistry = ShaderRegistry.getInstance();

/**
 * Process a registered shader by name
 * Allows for preprocessing of includes
 */
export function wgsl(
  shaderName: string,
  options?: WGSLProprocessingOptions
): string {
  return shaderRegistry.process(shaderName, options);
}
