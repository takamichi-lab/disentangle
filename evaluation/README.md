# Fixed paper-evaluation metadata

This directory contains the small metadata needed to reconstruct the DISSE
evaluation grid; it does not contain evaluation waveforms.

- `audio_fixed.csv`: the 96 AudioCaps sources used in the paper, including
  AudioCaps ID, YouTube ID, segment start time, and source caption.
- `rir_fixed.csv`: the 96 fixed shoebox-room geometries and source positions,
  one for each non-empty combination of direction (4), elevation (2),
  distance (2), room size (3), and reverberation (2).
- `summary.txt`: the original fixed-set counts.

Run `disse generate-rirs` to reconstruct the four-channel A-format RIRs. Then
run `disse make-evaluation-manifest --dry-root /path/to/local/dry_clips` to
create the source-major 9,216-row manifest and the seed-42 spatial captions.

Local dry clips may be WAV, FLAC, MP3, OGG, or M4A and must be named by
`audiocap_id` (for example, `104274.wav`). The code does not download YouTube
audio. See `DATA_NOTICE.md` in the repository root.
