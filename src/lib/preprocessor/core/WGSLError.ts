/**
 * Base error class for all WGSL preprocessing errors
 */
export abstract class WGSLError extends Error {
  abstract readonly code: string;
  abstract readonly category: "REGISTRY" | "IMPORT" | "VARIABLE" | "SYNTAX";
  public readonly baseMessage: string;

  constructor(
    message: string,
    public readonly context?: Record<string, unknown>
  ) {
    super(message);
    this.baseMessage = message;
    let detailedMessage = message;

    if (context && Object.keys(context).length > 0) {
      detailedMessage += "\n\nContext:";
      for (const [key, value] of Object.entries(context)) {
        detailedMessage += `\n  ${key}: ${String(value)}`;
      }
    }

    const suggestion = this.getSuggestion();
    if (suggestion) {
      detailedMessage += `\n\nSuggestion: ${suggestion}`;
    }

    this.message = detailedMessage;
    this.name = this.constructor.name;
  }

  protected abstract getSuggestion(): string;
}

/**
 * Error thrown when a shader is not found in the registry
 */
export class WGSLShaderNotFoundError extends WGSLError {
  readonly code = "SHADER_NOT_FOUND";
  readonly category = "REGISTRY" as const;

  constructor(shaderName: string, availableShaders: string[]) {
    super(`Shader "${shaderName}" not found in registry`, {
      requestedShader: shaderName,
      availableShaders:
        availableShaders.length > 0 ? availableShaders.join(", ") : "none",
      registrySize: availableShaders.length,
    });
  }

  protected getSuggestion(): string {
    const { requestedShader, availableShaders } = this.context!;

    if (typeof availableShaders === "string" && availableShaders !== "none") {
      const available = availableShaders.split(", ");
      const similar = available.find(
        (shader) =>
          shader
            .toLowerCase()
            .includes((requestedShader as string).toLowerCase()) ||
          (requestedShader as string)
            .toLowerCase()
            .includes(shader.toLowerCase())
      );

      if (similar) {
        return `Did you mean "${similar}"? Register the shader first: shaderRegistry.register("${requestedShader}", shaderContent)`;
      }
    }

    return `Register the shader first: shaderRegistry.register("${requestedShader}", shaderContent)`;
  }
}

/**
 * Error thrown when circular imports are detected
 */
export class WGSLCircularImportError extends WGSLError {
  readonly code = "CIRCULAR_IMPORT";
  readonly category = "IMPORT" as const;

  constructor(importChain: string[]) {
    super(`Circular import detected`, {
      importChain: importChain.join(" → "),
      circularFile: importChain[importChain.length - 1],
      chainLength: importChain.length,
    });
  }

  protected getSuggestion(): string {
    return `Review your @include statements to remove the circular dependency. The file "${this.context!.circularFile}" is being imported in a cycle.`;
  }
}

/**
 * Error thrown when maximum import depth is exceeded
 */
export class WGSLMaxImportDepthError extends WGSLError {
  readonly code = "MAX_IMPORT_DEPTH";
  readonly category = "IMPORT" as const;

  constructor(maxDepth: number, importStack: string[]) {
    super(`Maximum import depth of ${maxDepth} exceeded`, {
      maxDepth,
      currentDepth: importStack.length,
      importStack: importStack.join(" → "),
      deepestFile: importStack[importStack.length - 1],
    });
  }

  protected getSuggestion(): string {
    return `Reduce nesting depth of @include statements or check for circular imports. This might indicate an infinite recursion in your includes.`;
  }
}

/**
 * Error thrown when @include directive has invalid syntax
 */
export class WGSLInvalidIncludeError extends WGSLError {
  readonly code = "INVALID_INCLUDE_SYNTAX";
  readonly category = "SYNTAX" as const;

  constructor(invalidDirective: string, lineContext?: string) {
    super(`Invalid @include directive syntax`, {
      invalidDirective,
      lineContext: lineContext || "unknown",
      expectedFormat: '@include "filename.wgsl"',
    });
  }

  protected getSuggestion(): string {
    return `Use the correct syntax: @include "filename.wgsl". Make sure the filename is enclosed in double quotes.`;
  }
}

/**
 * Error thrown when a variable is not found in the template
 */
export class WGSLVariableNotFoundError extends WGSLError {
  readonly code = "VARIABLE_NOT_FOUND";
  readonly category = "VARIABLE" as const;

  constructor(
    variableName: string,
    availableVariables: string[],
    templateContext?: string
  ) {
    super(`Variable '${variableName}' not found in template`, {
      missingVariable: variableName,
      availableVariables:
        availableVariables.length > 0 ? availableVariables.join(", ") : "none",
      variableCount: availableVariables.length,
      templateContext: templateContext || "unknown",
    });
  }

  protected getSuggestion(): string {
    const { missingVariable, availableVariables } = this.context!;

    if (
      typeof availableVariables === "string" &&
      availableVariables !== "none"
    ) {
      const available = availableVariables.split(", ");
      const similar = available.find(
        (variable) =>
          variable
            .toLowerCase()
            .includes((missingVariable as string).toLowerCase()) ||
          (missingVariable as string)
            .toLowerCase()
            .includes(variable.toLowerCase())
      );

      if (similar) {
        return `Did you mean "${similar}"? Add the variable to your options: { variables: { "${missingVariable}": value } }`;
      }
    }

    return `Add the variable to your options: { variables: { "${missingVariable}": value } }`;
  }
}

/**
 * Error for general registry issues
 */
export class WGSLRegistryError extends WGSLError {
  readonly code = "REGISTRY_ERROR";
  readonly category = "REGISTRY" as const;

  constructor(message: string, context?: Record<string, unknown>) {
    super(message, context);
  }

  protected getSuggestion(): string {
    return "Check your shader registration and ensure all required shaders are properly registered.";
  }
}

/**
 * Error for general variable-related issues
 */
export class WGSLVariableError extends WGSLError {
  readonly code = "VARIABLE_ERROR";
  readonly category = "VARIABLE" as const;

  constructor(message: string, context?: Record<string, unknown>) {
    super(message, context);
  }

  protected getSuggestion(): string {
    return "Check your variable definitions and ensure all required variables are provided with correct types.";
  }
}

/**
 * Error thrown when @define directive has invalid syntax
 */
export class WGSLInvalidDefineError extends WGSLError {
  readonly code = "INVALID_DEFINE_SYNTAX";
  readonly category = "SYNTAX" as const;

  constructor(invalidDirective: string, lineContext?: string) {
    super(`Invalid @define directive syntax`, {
      invalidDirective,
      lineContext: lineContext || "unknown",
      expectedFormat: "@define IDENTIFIER value",
    });
  }

  protected getSuggestion(): string {
    return `Use the correct syntax: @define IDENTIFIER value. Example: @define PI 3.14159`;
  }
}

/**
 * Error thrown when a @define identifier is already defined
 */
export class WGSLDuplicateDefineError extends WGSLError {
  readonly code = "DUPLICATE_DEFINE";
  readonly category = "SYNTAX" as const;

  constructor(
    identifier: string,
    previousValue: string,
    newValue: string,
    lineContext?: string
  ) {
    super(`Identifier '${identifier}' is already defined`, {
      identifier,
      previousValue,
      newValue,
      lineContext: lineContext || "unknown",
    });
  }

  protected getSuggestion(): string {
    const { identifier, previousValue } = this.context!;
    return `'${identifier}' was previously defined as '${previousValue}'. Use a different identifier name or remove the duplicate definition.`;
  }
}

/**
 * Error thrown when a @define identifier name is invalid
 */
export class WGSLInvalidDefineIdentifierError extends WGSLError {
  readonly code = "INVALID_DEFINE_IDENTIFIER";
  readonly category = "SYNTAX" as const;

  constructor(identifier: string, lineContext?: string) {
    super(`Invalid @define identifier '${identifier}'`, {
      identifier,
      lineContext: lineContext || "unknown",
      validPattern:
        "Must start with letter or underscore, followed by letters, digits, or underscores",
    });
  }

  protected getSuggestion(): string {
    const { identifier } = this.context!;
    return `'${identifier}' is not a valid identifier. Use only letters, digits, and underscores, starting with a letter or underscore.`;
  }
}
