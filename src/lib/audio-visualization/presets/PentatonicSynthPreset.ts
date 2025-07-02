import * as Tone from "tone";
import type { SmokeTextureExports } from "@/lib/simulation/core/SmokeSimulation";
import type { AudioPreset } from "../types";
import { ToneUtils } from "../utils/ToneUtils";
import { CustomPolySynth } from "../synth/CustomPolySynth";
import { TimbralSynth } from "../synth/TimbralSynth";
import { BassSynth } from "../synth/BassSynth";

export type NoteVelocity = Tone.Unit.NormalRange;

export interface Note {
  pitch: Tone.Unit.Frequency;
  velocity: NoteVelocity;
  timbre: Tone.Unit.NormalRange;
}

/**
 * Default pentatonic synthesis preset with timbral morphing and bass loop
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

  // Bass loop sequence: A, F, C, D in bass range
  private static readonly BASS_LOOP = [
    55.0, // A1
    43.655, // F1
    32.705, // C1
    36.71, // D1
  ];

  private lastPlayedNote: Note | null = null;
  private synth: CustomPolySynth<TimbralSynth> | null = null;
  private bassSynth: BassSynth | null = null;
  private bassLoopIndex = 0;
  private isMouseDown = false;

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
    this.isMouseDown = true;
    this.playBassNote();

    // Control bass synth secondary filter with X position
    if (this.bassSynth) {
      this.bassSynth.setSecondaryFilterCutoff(normalizedX);
    }

    await this.playNoteAtPosition(normalizedX, normalizedY, textureData);
  }

  async onMouseMove(
    normalizedX: number,
    normalizedY: number,
    textureData: SmokeTextureExports | null
  ): Promise<void> {
    if (this.isMouseDown) {
      // Control bass synth secondary filter with X position
      if (this.bassSynth) {
        this.bassSynth.setSecondaryFilterCutoff(normalizedX);
      }

      await this.playNoteAtPosition(normalizedX, normalizedY, textureData);
    }
  }

  async onMouseUp(): Promise<void> {
    this.isMouseDown = false;
    this.releaseBassNote();
    this.reset();
  }

  reset(): void {
    this.lastPlayedNote = null;
  }

  private initializeSynth(): void {
    // Initialize main timbral synth
    const synth = new CustomPolySynth(TimbralSynth, {
      maxPolyphony: 8,
      voiceOptions: {
        envelope: {
          attack: 4,
          decay: 0,
          sustain: 1,
          release: 1.2,
        },
        timbre: 0, // Default timbre value
      },
    });

    // Initialize bass synth
    this.bassSynth = new BassSynth();

    const reverb = new Tone.Reverb({
      decay: 2,
      wet: 0.9,
    });
    const chorus = new Tone.Chorus(4, 10, 1);
    const delay = new Tone.PingPongDelay("4t", 0.5).chain(new Tone.Gain(0.3));
    const master = new Tone.Gain(3.0).toDestination();

    // Connect main synth
    synth.chain(chorus, reverb, master);
    synth.chain(chorus, delay, reverb, master);

    // Connect bass synth with less reverb
    const bassGain = new Tone.Gain(0.15);
    this.bassSynth.chain(bassGain, chorus, reverb);

    bassGain.toDestination();

    this.synth = synth;
  }

  private playBassNote(): void {
    if (!this.bassSynth) return;

    const bassNote = PentatonicSynthPreset.BASS_LOOP[this.bassLoopIndex];
    this.bassSynth.triggerAttack(bassNote);

    // Advance to next note in loop
    this.bassLoopIndex =
      (this.bassLoopIndex + 1) % PentatonicSynthPreset.BASS_LOOP.length;
  }

  private releaseBassNote(): void {
    if (!this.bassSynth) return;
    this.bassSynth.triggerRelease();
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
      note.velocity,
      note.timbre
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

    // Map normalized X to timbre (0 = sine, 1 = fat sawtooth)
    const timbre = normalizedX as Tone.Unit.NormalRange;

    // Create note with timbre information
    const note: Note = { pitch, velocity, timbre };

    // Only play if the note is different from the previous one
    if (!this.areNotesEqual(this.lastPlayedNote, note)) {
      // console.log("Playing note", note);
      await this.playNote(note);
      this.lastPlayedNote = note;
    }
  }

  private areNotesEqual(note1: Note | null, note2: Note): boolean {
    if (!note1) return false;
    return (
      note1.pitch === note2.pitch &&
      note1.velocity === note2.velocity &&
      note1.timbre === note2.timbre
    );
  }
}
