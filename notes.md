# Fluid Simulation

## Idols

- David bruce
- Martin Keary
- Stephane
- Cris Mccandles

## Inspo

- L-Drones https://codepen.io/teropa/full/opjrBE
  - kind of similar to my idea
- Magic sketchpad
  - Inspired feeling
- WebGL fluid simulation https://paveldogreat.github.io/WebGL-Fluid-Simulation/
  - for smoke simulation
- Some cool AV stuff from an artist
  - https://ctpt.co/
- Google experiments have some cool things
  - Patatap is a great artistic experiement
  - Enough is a beautiful interactive audiobook
- Tero. Some cool experiments https://teropa.info/
- The most GORGEOUS particle simulation I've seen. This is the vibe:
  - https://haxiomic.github.io/projects/webgl-fluid-and-particles/
- INCREDIBLE AUDIO RANDOMIZATION ENGINE WOWW (88 keys and generative.fm)
  - https://mynoise.net/NoiseMachines/acousticPianoSoundscapeGenerator.php
  - https://mynoise.net/NoiseMachines/twinBlackLodgesSoundscapeGenerator.php
  - Generative FM Play
    - https://play.generative.fm/browse
    - https://docs.generative.fm/docs/repositories
    - https://github.com/generativefm/play
  - http://www.scribble.audio/
- Also a huge fan of patatap. It's just so fun

- Generative stuff at the end (David bruce on In C)
  - https://www.youtube.com/watch?v=JN0bW3ilqF4

## Fluid Simulation

- For better poisson eq solvers, see Bolz et al. 2003, Goodnight et al. 2003, and Krüger and Westermann 2003.

Process

```
// Apply the first 3 operators in Equation 12.
u = advect(u);
u = diffuse(u);
u = addForces(u); // Now apply the projection operator to the result.
p = computePressure(u);
u = subtractPressureGradient(u, p);
```

## Name Ideas

- Flux Lab
- Smokesichord

## Notes

- Maybe not just smoke, but water, fire, air, earth

  - or maybe split up into four quadrants w/ different intstruments
  - start w/ one instrument -- arp would be easy

- midi keyboard interface to change fundamental note

- some meme shit-- press space and some thunderbolt shit makes a loud noise

- Fonts I like: anova, nevalisa, aqreamig (lowercase)

- magic sketchpad inspo

## Tech Details

- All goes off of some keyboard root fundamental to create an arp

- w/ smoke simulation
- each portion of grid represents some note (low - high is down - up)
  - left to right is some timbral modulation
- velocity & density can go to volume & reverb
  - higher density = louder volume
  - lower velocity = more reverb
- ONLY applies to notes in its vicinity -- some smoothing algorithm
- maybe everything is sustained normally by default
  - vorticity around notes creates an ARP effect

### Possible technology to get audio

- tidal/strudel
  - live coding
- tone.js
