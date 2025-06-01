// Public Preprocessor API

export { wgsl, shaderRegistry } from "./core/wgsl";
export {
  WGSLError,
  WGSLRegistryError,
  WGSLShaderNotFoundError,
  WGSLMaxImportDepthError,
  WGSLInvalidIncludeError,
  WGSLCircularImportError,
  WGSLVariableError,
  WGSLVariableNotFoundError,
  WGSLInvalidDefineError,
  WGSLDuplicateDefineError,
  WGSLInvalidDefineIdentifierError,
} from "./core/WGSLError";
