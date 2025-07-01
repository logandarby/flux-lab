import * as Tone from "tone";

export interface TimbralSynthOptions extends Tone.SynthOptions {
  timbre: Tone.Unit.NormalRange;
}

/**
 * A custom synthesizer that morphs between different oscillator types
 * based on a normalized timbre parameter (0-1).
 *
 * 0 is a sine wave (least timbral information),
 * 1 is a fat sawtooth wave (most timbral information)
 */
export class TimbralSynth extends Tone.Synth<TimbralSynthOptions> {
  readonly name = "TimbralSynth";
  private _timbre: Tone.Unit.NormalRange = 0;

  constructor(options?: Partial<TimbralSynthOptions>) {
    super(options);

    // Set initial timbre if provided
    if (options?.timbre !== undefined) {
      this.timbre = options.timbre;
    }
  }

  static getDefaults(): TimbralSynthOptions {
    return {
      ...Tone.Synth.getDefaults(),
      timbre: 0,
    };
  }

  /**
   * Get the current timbre value
   */
  get timbre(): Tone.Unit.NormalRange {
    return this._timbre;
  }

  /**
   * Set the timbre value and update the oscillator type accordingly
   */
  set timbre(value: Tone.Unit.NormalRange) {
    this._timbre = Math.max(0, Math.min(1, value));
    this.updateOscillatorType();
  }

  /**
   * Updates the oscillator type based on the current timbre value
   */
  private updateOscillatorType(): void {
    const timbre = this._timbre;

    if (timbre < 0.2) {
      // Sine wave region (0.0 - 0.2)
      this.oscillator.type = "sine";
    } else if (timbre < 0.4) {
      // Triangle wave region (0.2 - 0.4)
      this.oscillator.type = "triangle";
    } else if (timbre < 0.6) {
      // Square wave region (0.4 - 0.6)
      this.oscillator.type = "square";
    } else if (timbre < 0.8) {
      // Sawtooth wave region (0.6 - 0.8)
      this.oscillator.type = "sawtooth";
    } else {
      // Fat sawtooth region (0.8 - 1.0)
      this.oscillator.type = "fatsawtooth";
    }
  }

  /**
   * Override triggerAttackRelease to ensure timbre is applied
   */
  triggerAttackRelease(
    note: Tone.Unit.Frequency,
    duration: Tone.Unit.Time,
    time?: Tone.Unit.Time,
    velocity?: Tone.Unit.NormalRange
  ): this {
    // Ensure oscillator type is up to date
    this.updateOscillatorType();
    return super.triggerAttackRelease(note, duration, time, velocity);
  }

  /**
   * Override triggerAttack to ensure timbre is applied
   */
  triggerAttack(
    note: Tone.Unit.Frequency,
    time?: Tone.Unit.Time,
    velocity?: Tone.Unit.NormalRange
  ): this {
    // Ensure oscillator type is up to date
    this.updateOscillatorType();
    return super.triggerAttack(note, time, velocity);
  }
}
