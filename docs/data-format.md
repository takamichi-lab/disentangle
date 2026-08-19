# Evaluation data format

## Evaluation layout

```text
data/evaluation/
├── manifest.csv
└── rirs/
    └── auto_*.wav
```

`disse generate-rirs` recreates the 96 RIRs from `evaluation/rir_fixed.csv`.
`disse make-evaluation-manifest` crosses them with the 96 locally supplied dry
clips in `evaluation/audio_fixed.csv`.

The generated `manifest.csv` has one row per source-spatial pair:

| Column | Required | Meaning |
|---|---|---|
| `dry_path` | for audio | Local dry clip path, relative to the manifest directory |
| `rir_path` | for audio | Four-channel A-format RIR path |
| `text` | for text | Source-and-spatial caption supplied to DISSE |
| `source_id` | yes | Identifier shared by items with the same dry source |
| `spatial_id` | yes | Identifier shared by items with the same RIR |

Inference spatializes each pair on demand. A pre-spatialized manifest may
instead provide `audio_path` (FOA WAV) and optional `feature_path`.
`--modality audio` does not require `text`; `--modality text` does not require
audio paths. The default `--modality auto` evaluates both branches for a
manifest and therefore requires both inputs.

For compatibility with precomputed research data, the loader also accepts:

- `foa_path` for `audio_path`;
- `feat_path` for `feature_path`;
- `caption` for `text`;
- `audiocap_id` for `source_id`;
- `space_id` for `spatial_id` (or derives it from `rir_path`).

Example:

```csv
dry_path,rir_path,text,source_id,spatial_id
../../dry/104274.wav,rirs/auto_000000.wav,Rain from the right in a room,104274,auto_000000
```

## Fixed RIR generation

The RIR generator uses Pyroomacoustics 0.8.4 with a 48-kHz shoebox room,
image-source order 10, and four cardioid microphones on a regular tetrahedron
of radius 5 cm centered in the room. The fixed CSV provides room dimensions,
wall absorption, quantized distance/azimuth/elevation, and sample rate. The
generator reconstructs the original unrounded Cartesian source position from
those polar coordinates; `source_pos_xyz` is retained as a rounded consistency
check. RIR WAVs use PCM-16, matching the research pipeline.

## FOA convention

WAV files contain four channels ordered **W, Y, Z, X**. The original
simulation converts the four A-format microphone channels
\(m_0,m_1,m_2,m_3\) using:

\[
\begin{aligned}
W &= (m_0+m_1+m_2+m_3)/2,\\
Y &= (m_0-m_1+m_2-m_3)/2,\\
Z &= (m_0-m_1-m_2+m_3)/2,\\
X &= (m_0+m_1-m_2-m_3)/2.
\end{aligned}
\]

Inference downmixes local dry clips to mono, resamples to 48 kHz, repeats audio
shorter than 10 seconds, and crops audio longer than 10 seconds. After
convolution and A-to-FOA conversion, the W channel is passed to HTS-AT.

## Precomputed intensity features

Each `.pt` feature file is a dictionary with:

- `i_act`: float tensor `[3, 201, 1601]`;
- `i_rea`: float tensor `[3, 201, 1601]`;
- optionally `omni_48k`: float tensor `[480000]`.

When the feature path is absent, `disse infer` derives both intensity tensors
from the FOA WAV using a 16-kHz STFT (`n_fft=400`, `hop_length=100`). When
`omni_48k` is absent, the W channel is read from the corresponding WAV.

## Embedding cache

Inference produces a NumPy `.npz` file containing the selected modality's
embeddings plus two non-pickled label arrays:

| Key | Present when | Shape |
|---|---|---:|
| `audio_source` | audio selected | `[N, 512]` |
| `audio_spatial` | audio selected | `[N, 512]` |
| `text_source` | text selected | `[N, 512]` |
| `text_spatial` | text selected | `[N, 512]` |
| `source_id` | always | `[N]` |
| `spatial_id` | always | `[N]` |

The evaluator accepts string or numeric labels and L2-normalizes all
embeddings before computing distances and retrieval scores. IIDR and
intra-modal retrieval are computed for whichever modalities are present;
cross-modal retrieval is reported only when both are available.
