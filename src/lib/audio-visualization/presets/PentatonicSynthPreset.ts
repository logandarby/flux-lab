import * as Tone from "tone";
import type { SmokeTextureExports } from "@/lib/simulation/core/SmokeSimulation";
import type { AudioPreset } from "../types";
import { ToneUtils } from "../utils/ToneUtils";

export type NoteVelocity = Tone.Unit.NormalRange;

export interface Note {
  pitch: Tone.Unit.Frequency;
  velocity: NoteVelocity;
}

/**
 * Default pentatonic synthesis preset
 */
export class PentatonicSynthPreset implements AudioPreset {
  private static readonly PENTATONIC_SCALE = [
    261.63,
    293.66,
    329.63,
    392.0,
    440.0, // C4, D4, E4, G4, A4
    523.25,
    587.33,
    659.25,
    783.99,
    880.0, // C5, D5, E5, G5, A5
    1046.5,
    1174.66,
    1318.51,
    1567.98,
    1760.0, // C6, D6, E6, G6, A6
  ];

  private lastPlayedNote: Note | null = null;
  private synth: Tone.PolySynth | null = null;

  constructor() {
    this.initializeSynth();
  }

  async onMouseDown(
    normalizedX: number,
    normalizedY: number,
    textureData: SmokeTextureExports | null
  ): Promise<void> {
    // Ensure ToneJS is initialized
    if (!ToneUtils.isInitialized()) {
      await ToneUtils.initialize();
    }
    await this.playNoteAtPosition(normalizedX, normalizedY, textureData);
  }

  async onMouseMove(
    normalizedX: number,
    normalizedY: number,
    textureData: SmokeTextureExports | null
  ): Promise<void> {
    await this.playNoteAtPosition(normalizedX, normalizedY, textureData);
  }

  async onMouseUp(): Promise<void> {
    this.reset();
  }

  reset(): void {
    this.lastPlayedNote = null;
  }

  private initializeSynth(): void {
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
    const chorus = new Tone.Chorus(4, 2.5, 1);
    const delay = new Tone.PingPongDelay("4n", 0.5).chain(new Tone.Gain(0.7));
    const master = new Tone.Gain(1.0).toDestination();

    synth.chain(chorus, reverb, master);
    synth.chain(chorus, delay, reverb, master);

    this.synth = synth;
  }

  private async playNote(note: Note): Promise<void> {
    if (!this.synth) {
      console.warn("Synth not initialized");
      return;
    }

    this.synth.triggerAttackRelease(
      note.pitch,
      "32n",
      undefined,
      note.velocity
    );
  }

  private async playNoteAtPosition(
    normalizedX: number,
    normalizedY: number,
    textureData: SmokeTextureExports | null
  ): Promise<void> {
    if (!textureData) return;

    // Ensure coordinates are in valid range
    if (
      normalizedX < 0 ||
      normalizedX > 1 ||
      normalizedY < 0 ||
      normalizedY > 1
    ) {
      return;
    }

    const density = textureData.smokeDensity.getAtNormalized(
      normalizedX,
      normalizedY
    );

    // Map normalized Y to pentatonic scale (0 = lowest pitch, 1 = highest pitch)
    const scaleIndex = Math.floor(
      normalizedY * (PentatonicSynthPreset.PENTATONIC_SCALE.length - 1)
    );
    const pitch = PentatonicSynthPreset.PENTATONIC_SCALE[scaleIndex];

    // Map density to velocity (0-1 range, with minimum threshold for audibility)
    const velocity = Math.max(0.5, Math.min(1.0, density * 2)) as NoteVelocity;

    // Create note and check if it's different from the previous one
    const note: Note = { pitch, velocity };

    // Only play if the note is different from the previous one
    if (!this.areNotesEqual(this.lastPlayedNote, note)) {
      // console.log("Playing note", note);
      await this.playNote(note);
      this.lastPlayedNote = note;
    }
  }

  private areNotesEqual(note1: Note | null, note2: Note): boolean {
    if (!note1) return false;
    return note1.pitch === note2.pitch && note1.velocity === note2.velocity;
  }
}
