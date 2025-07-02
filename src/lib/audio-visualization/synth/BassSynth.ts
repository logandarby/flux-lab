import * as Tone from "tone";
import type { Voice } from "./CustomPolySynth";

/**
 * A custom bass synthesizer with fatsawtooth oscillator and filter envelope
 * Designed for sustained bass notes with expressive filtering
 */
export class BassSynth extends Tone.ToneAudioNode implements Voice {
  readonly name = "BassSynth";

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private oscillator: Tone.OmniOscillator<any>;
  private envelope: Tone.AmplitudeEnvelope;
  private filter: Tone.Filter;
  private secondaryFilter: Tone.Filter;
  private filterEnvelope: Tone.FrequencyEnvelope;
  private filterLFO: Tone.LFO;
  output: Tone.Gain;
  input: Tone.InputNode | undefined;

  constructor(options: Record<string, unknown> = {}) {
    super(options);

    // Create oscillator with sawtooth wave (closest to fatsawtooth)
    this.oscillator = new Tone.OmniOscillator({
      type: "fatsawtooth",
      frequency: 440,
    });

    // Create amplitude envelope with sustain characteristics
    this.envelope = new Tone.AmplitudeEnvelope({
      attack: 2,
      decay: 0.2,
      sustain: 0.8,
      release: 3.0,
    });

    // Create primary filter with low cutoff for bass
    this.filter = new Tone.Filter({
      type: "lowpass",
      frequency: 400,
      Q: 1,
    });

    // Create secondary filter for real-time control
    this.secondaryFilter = new Tone.Filter({
      type: "lowpass",
      frequency: 800, // Higher default cutoff
      Q: 2,
    });

    // Create filter envelope with long attack/decay/release
    this.filterEnvelope = new Tone.FrequencyEnvelope({
      attack: 10.0,
      decay: 1.5,
      sustain: 0.3,
      release: 3.0,
    });

    // Create LFO for filter frequency modulation
    this.filterLFO = new Tone.LFO({
      frequency: "8n", // Eighth note rate
      type: "sine",
      amplitude: 0.3,
    });

    // Create output gain
    this.output = new Tone.Gain(0.5);

    // Connect the signal chain: oscillator -> primary filter -> secondary filter -> envelope -> output
    this.oscillator.chain(
      this.filter,
      this.secondaryFilter,
      this.envelope,
      this.output
    );

    // Connect filter envelope to primary filter frequency
    this.filterEnvelope.connect(this.filter.frequency);

    // Connect LFO to primary filter frequency (additive modulation)
    this.filterLFO.connect(this.filter.frequency);

    // Start the LFO
    this.filterLFO.start();
  }

  /**
   * Set the secondary filter cutoff frequency based on normalized mouse position
   * @param normalizedX - Mouse X position (0-1)
   */
  setSecondaryFilterCutoff(normalizedX: number): void {
    // Map normalized X (0-1) to filter frequency range (200Hz - 2000Hz)
    const minFreq = 200;
    const maxFreq = 2000;
    const frequency = minFreq + normalizedX * (maxFreq - minFreq);

    this.secondaryFilter.frequency.setValueAtTime(frequency, Tone.now());
  }

  /**
   * Trigger attack and release with specified duration
   */
  triggerAttackRelease(
    note: Tone.Unit.Frequency,
    duration: Tone.Unit.Time,
    time?: Tone.Unit.Time,
    velocity?: Tone.Unit.NormalRange
  ): void {
    const computedTime =
      time !== undefined ? Tone.Time(time).toSeconds() : Tone.now();
    const computedDuration = Tone.Time(duration).toSeconds();

    // Set frequency
    this.oscillator.frequency.setValueAtTime(note, computedTime);

    // Start oscillator if not already running
    if (this.oscillator.state === "stopped") {
      this.oscillator.start(computedTime);
    }

    // Trigger envelopes
    this.envelope.triggerAttackRelease(
      computedDuration,
      computedTime,
      velocity
    );
    this.filterEnvelope.triggerAttackRelease(computedDuration, computedTime);
  }

  /**
   * Trigger attack phase (for sustained notes)
   */
  triggerAttack(
    note: Tone.Unit.Frequency,
    time?: Tone.Unit.Time,
    velocity: Tone.Unit.NormalRange = 1
  ): void {
    const computedTime =
      time !== undefined ? Tone.Time(time).toSeconds() : Tone.now();
    this.oscillator.frequency.setValueAtTime(note, computedTime);
    if (this.oscillator.state === "stopped") {
      this.oscillator.start(computedTime);
    }
    this.envelope.triggerAttack(computedTime, velocity);
    this.filterEnvelope.triggerAttack(computedTime);
  }

  /**
   * Trigger release phase
   */
  triggerRelease(time?: Tone.Unit.Time): void {
    const computedTime =
      time !== undefined ? Tone.Time(time).toSeconds() : Tone.now();
    this.envelope.triggerRelease(computedTime);
    this.filterEnvelope.triggerRelease(computedTime);
  }

  /**
   * Connect to destination (required by Voice interface)
   */
  connect(destination: Tone.InputNode): this {
    this.output.connect(destination);
    return this;
  }

  /**
   * Connect to destination (required by Voice interface)
   */
  toDestination(): this {
    this.output.toDestination();
    return this;
  }

  /**
   * Chain this synth through a series of audio nodes
   * Compatible with Tone.js chaining syntax
   */
  chain(...nodes: Tone.InputNode[]): this {
    this.output.chain(...nodes);
    return this;
  }

  /**
   * Clean up and dispose of all components
   */
  dispose(): this {
    super.dispose();
    this.oscillator.dispose();
    this.envelope.dispose();
    this.filter.dispose();
    this.secondaryFilter.dispose();
    this.filterEnvelope.dispose();
    this.filterLFO.dispose();
    this.output.dispose();
    return this;
  }
}
