import * as Tone from "tone";
import type { InputNode } from "tone";

/**
 * Minimum interface that a voice must implement to be used with CustomPolySynth
 */
export interface Voice {
  triggerAttackRelease(
    note: Tone.Unit.Frequency,
    duration: Tone.Unit.Time,
    time?: Tone.Unit.Time,
    velocity?: Tone.Unit.NormalRange
  ): void;
  dispose(): void;
  connect(destination: Tone.InputNode): this;
  toDestination(): this;
  timbre?: Tone.Unit.NormalRange; // Optional timbre property
}

/**
 * Constructor type for voice creation
 */
export type VoiceConstructor<V extends Voice> = {
  new (options: Record<string, unknown>): V;
};

export interface CustomPolySynthOptions<V extends Voice> {
  maxPolyphony: number;
  voice: VoiceConstructor<V>;
  voiceOptions: Record<string, unknown>;
  context?: Tone.Context;
}

const DEFAULT_OPTIONS: Required<
  Omit<CustomPolySynthOptions<Voice>, "voice" | "context">
> = {
  maxPolyphony: 32,
  voiceOptions: {},
};

interface VoiceInfo<V extends Voice> {
  voice: V;
  lastUsed: number;
}

/**
 * A custom polyphonic synthesizer that can work with any voice type that implements
 * the Voice interface. Uses a FIFO (First In, First Out) voice allocation strategy
 * when maximum polyphony is reached.
 */
export class CustomPolySynth<V extends Voice> extends Tone.ToneAudioNode {
  private readonly options: Required<CustomPolySynthOptions<V>>;
  private voices: VoiceInfo<V>[] = [];
  readonly name = "CustomPolySynth";

  input: InputNode | undefined;
  output = new Tone.Gain();

  constructor(
    voice: VoiceConstructor<V>,
    options: Partial<Omit<CustomPolySynthOptions<V>, "voice">> = {}
  ) {
    super(options);

    this.options = {
      ...DEFAULT_OPTIONS,
      ...options,
      voice,
      voiceOptions: options.voiceOptions || {},
      context: this.context,
    } as Required<CustomPolySynthOptions<V>>;
  }

  /**
   * Gets the next available voice using FIFO strategy
   */
  private getNextVoice(): V {
    // If we haven't reached max polyphony, create a new voice
    if (this.voices.length < this.options.maxPolyphony) {
      const voice = new this.options.voice(this.options.voiceOptions);
      voice.connect(this.output);

      const voiceInfo: VoiceInfo<V> = {
        voice,
        lastUsed: Tone.now(),
      };

      this.voices.push(voiceInfo);
      return voice;
    }

    // Find the oldest voice (FIFO)
    let oldestVoice = this.voices[0];
    for (let i = 1; i < this.voices.length; i++) {
      if (this.voices[i].lastUsed < oldestVoice.lastUsed) {
        oldestVoice = this.voices[i];
      }
    }

    // Update the last used time
    oldestVoice.lastUsed = Tone.now();
    return oldestVoice.voice;
  }

  /**
   * Triggers both attack and release of a note with a specified duration
   * Optionally sets timbre if the voice supports it
   */
  triggerAttackRelease(
    note: Tone.Unit.Frequency,
    duration: Tone.Unit.Time,
    time?: Tone.Unit.Time,
    velocity?: Tone.Unit.NormalRange,
    timbre?: Tone.Unit.NormalRange
  ): this {
    const voice = this.getNextVoice();

    // Set timbre if provided and voice supports it
    if (timbre !== undefined && "timbre" in voice) {
      (voice as Voice & { timbre: Tone.Unit.NormalRange }).timbre = timbre;
    }

    voice.triggerAttackRelease(note, duration, time, velocity);
    return this;
  }

  /**
   * Cleans up and disposes of all voices
   */
  dispose(): this {
    super.dispose();
    this.output.dispose();
    this.voices.forEach((voiceInfo) => voiceInfo.voice.dispose());
    this.voices = [];
    return this;
  }
}
