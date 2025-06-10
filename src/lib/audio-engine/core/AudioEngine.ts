import * as Tone from "tone";

export class AudioEngineError extends Error {
  constructor(message: string) {
    super(`There was a problem with the AudioEngine: ${message}`);
  }
}

export class AudioEngineInitializationError extends AudioEngineError {
  constructor() {
    super(
      "Audio Engine has not been initialized. Please call `initialize()` first."
    );
  }
}

export type NoteVelocity = Tone.Unit.NormalRange;

export interface Note {
  pitch: Tone.Unit.Frequency;
  velocity: NoteVelocity;
}

export class AudioEngine {
  private synth: Tone.PolySynth | null = null;

  /**
   * Initialize the audio engine and start the audio context.
   *
   * WARNING: This method MUST ONLY be called during a user gesture event
   * (such as click, keydown, touchstart, etc.) due to browser autoplay policies.
   */
  public async initialize(): Promise<void> {
    if (this.synth) return;
    await Tone.start();
    this.synth = this.createSynth();
    console.log("Audio Engine has been Initialized");
  }

  public async playNote(note: Note): Promise<void> {
    // console.log(`Playing Note:\n\tPitch - ${note.pitch}\n\tVelocity - ${note.velocity}`);
    await this.initialize();
    if (!this.synth) throw new AudioEngineInitializationError();
    this.synth.triggerAttackRelease(
      note.pitch,
      "32n",
      undefined,
      note.velocity
    );
  }

  private createSynth(): typeof this.synth {
    const synth = new Tone.PolySynth(Tone.Synth, {
      envelope: {
        attack: 0.8,
        decay: 0,
        sustain: 1,
        release: 1.2,
      },
    });
    synth.maxPolyphony = 100;
    const reverb = new Tone.Reverb({
      decay: 2,
      wet: 0.9,
    });
    const chorus = new Tone.Chorus(4, 2.5, 0.5);
    const delay = new Tone.PingPongDelay("4n", 0.5).chain(new Tone.Gain(0.7));
    const master = new Tone.Gain(1.0).toDestination();
    synth.chain(chorus, reverb, master);
    synth.chain(chorus, delay, reverb, master);
    return synth;
  }
}
