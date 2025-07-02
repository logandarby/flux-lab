import * as Tone from "tone";
import type { OmniOscillatorType } from "tone/build/esm/source/oscillator/OscillatorInterface";

export interface TimbralSynthOptions {
  timbre?: Tone.Unit.NormalRange;
  envelope?: Partial<Tone.EnvelopeOptions>;
}

/**
 * A custom synthesizer that morphs between different oscillator types
 * based on a normalized timbre parameter (0-1).
 *
 * 0 is a sine wave (least timbral information),
 * 1 is a fat sawtooth wave (most timbral information)
 *
 * Uses crossfading between only two adjacent oscillator types for efficiency.
 */
export class TimbralSynth {
  readonly name = "TimbralSynth";
  private _timbre: Tone.Unit.NormalRange = 0;

  // Two synths for efficient crossfading
  private _synthA: Tone.Synth;
  private _synthB: Tone.Synth;

  // Gain controls for crossfading
  private _gainA: Tone.Gain;
  private _gainB: Tone.Gain;

  // Output mixer
  private _output: Tone.Gain;

  // Track current oscillator types to avoid unnecessary updates
  private _currentTypeA: OmniOscillatorType = "sine";
  private _currentTypeB: OmniOscillatorType = "triangle";

  constructor(options?: TimbralSynthOptions) {
    const opts = Object.assign(TimbralSynth.getDefaults(), options);

    // Create two synths for crossfading
    this._synthA = new Tone.Synth({
      oscillator: { type: "sine" },
      envelope: opts.envelope,
    });
    this._synthB = new Tone.Synth({
      oscillator: { type: "triangle" },
      envelope: opts.envelope,
    });

    // Create gain controls for crossfading
    this._gainA = new Tone.Gain(1); // Start with synthA at full volume
    this._gainB = new Tone.Gain(0); // Start with synthB at zero volume

    // Create output mixer
    this._output = new Tone.Gain(0.5); // Slightly higher gain since we only have 2 voices

    // Connect the audio graph
    this._synthA.connect(this._gainA);
    this._synthB.connect(this._gainB);

    this._gainA.connect(this._output);
    this._gainB.connect(this._output);

    // Set initial timbre
    this.timbre = opts.timbre || 0;
  }

  static getDefaults(): Required<TimbralSynthOptions> {
    return {
      timbre: 0,
      envelope: {
        attack: 0.005,
        decay: 0.1,
        sustain: 0.3,
        release: 1,
      },
    };
  }

  /**
   * Get the output node for connecting to other audio nodes
   */
  get output(): Tone.Gain {
    return this._output;
  }

  /**
   * Connect to another audio node or destination
   */
  connect(destination: Tone.InputNode): this {
    this._output.connect(destination);
    return this;
  }

  /**
   * Disconnect from all connected nodes
   */
  disconnect(): this {
    this._output.disconnect();
    return this;
  }

  /**
   * Connect directly to the main audio destination
   */
  toDestination(): this {
    this._output.toDestination();
    return this;
  }

  /**
   * Get the current timbre value
   */
  get timbre(): Tone.Unit.NormalRange {
    return this._timbre;
  }

  /**
   * Set the timbre value and update the oscillator blend accordingly
   */
  set timbre(value: Tone.Unit.NormalRange) {
    this._timbre = Math.max(0, Math.min(1, value));
    this.updateOscillatorBlend();
  }

  /**
   * Updates the oscillator blend based on the current timbre value
   * Uses only two synths for efficient crossfading
   */
  private updateOscillatorBlend(): void {
    const timbre = this._timbre;

    let typeA: OmniOscillatorType;
    let typeB: OmniOscillatorType;
    let blend: number;

    if (timbre <= 0.5) {
      // Blend between sine (0) and triangle (0.5)
      typeA = "sine";
      typeB = "triangle";
      blend = timbre * 2; // 0-1 range
    } else {
      // Blend between triangle (0.5) and sawtooth (1.0)
      typeA = "triangle";
      typeB = "fatsawtooth";
      blend = (timbre - 0.5) * 2; // 0-1 range
    }

    // Update oscillator types only if they've changed
    if (this._currentTypeA !== typeA) {
      this._synthA.oscillator.type = typeA;
      this._currentTypeA = typeA;
    }
    if (this._currentTypeB !== typeB) {
      this._synthB.oscillator.type = typeB;
      this._currentTypeB = typeB;
    }

    // Calculate crossfade gains
    const gainA = 1 - blend;
    const gainB = blend;

    // Apply gain values smoothly
    const now = Tone.now();
    this._gainA.gain.setValueAtTime(gainA, now);
    this._gainB.gain.setValueAtTime(gainB, now);
  }

  /**
   * Trigger attack phase
   */
  triggerAttack(
    note: Tone.Unit.Frequency,
    time?: Tone.Unit.Time,
    velocity?: Tone.Unit.NormalRange
  ): this {
    // Trigger both synths simultaneously
    this._synthA.triggerAttack(note, time, velocity);
    this._synthB.triggerAttack(note, time, velocity);

    return this;
  }

  /**
   * Trigger release phase
   */
  triggerRelease(time?: Tone.Unit.Time): this {
    // Release both synths simultaneously
    this._synthA.triggerRelease(time);
    this._synthB.triggerRelease(time);

    return this;
  }

  /**
   * Trigger attack and release
   */
  triggerAttackRelease(
    note: Tone.Unit.Frequency,
    duration: Tone.Unit.Time,
    time?: Tone.Unit.Time,
    velocity?: Tone.Unit.NormalRange
  ): this {
    // Trigger both synths simultaneously
    this._synthA.triggerAttackRelease(note, duration, time, velocity);
    this._synthB.triggerAttackRelease(note, duration, time, velocity);

    return this;
  }

  /**
   * Set the volume of the synthesizer
   */
  set volume(value: Tone.Unit.Decibels) {
    this._output.gain.value = Tone.dbToGain(value);
  }

  get volume(): Tone.Unit.Decibels {
    return Tone.gainToDb(this._output.gain.value);
  }

  /**
   * Clean up resources
   */
  dispose(): this {
    this._synthA.dispose();
    this._synthB.dispose();

    this._gainA.dispose();
    this._gainB.dispose();

    this._output.dispose();

    return this;
  }
}
