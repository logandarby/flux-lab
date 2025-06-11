/**
 * Error class for audio visualization mediator operations
 */
export class AudioVisualizationError extends Error {
  constructor(message: string) {
    super(`AudioVisualizationMediator: ${message}`);
  }
}
