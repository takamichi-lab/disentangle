# DISSE

Official inference and evaluation code for **DISSE: Learning Disentangled
Source and Spatial Representations from Spatial Audio-Text Contrastive
Learning** (EUSIPCO 2026).

DISSE maps spatial audio and text to separate 512-dimensional embeddings for:

- **source** information: what is sounding;
- **spatial** information: where and in what acoustic environment it sounds.

This repository intentionally contains only inference, preprocessing, and
paper-aligned evaluation code. Training experiments, workstation paths,
W&B logs, cached Python files, and obsolete implementations are excluded.

## Install

Python 3.10-3.12 is supported. Install a PyTorch build appropriate for your
CPU/CUDA environment, then install the inference dependencies:

```bash
python -m pip install -e ".[inference]"
```

The released experiment used PyTorch 2.7.0, torchaudio 2.7.0, and
Transformers 4.52.4. A CUDA GPU is strongly recommended; CPU inference is
supported but slow.

To regenerate the paper's synthetic RIRs, also install the evaluation-data
extra:

```bash
python -m pip install -e ".[inference,evaluation-data]"
```

## Download the epoch-20 checkpoint

```bash
disse download checkpoint
```

The file is downloaded from
[Google Drive](https://drive.google.com/file/d/1fyEWNrfZe6bfsPmHqppof_dHMWu_Skmc/view?usp=drive_link)
and verified against the SHA-256 value in `artifacts.json`. The download is
approximately 619 MiB. Inference expects
`checkpoints/disse_epoch20.pt` by default.

```text
SHA-256  a38ffabd0db88e4d42335ef47895524bb3ebcc01ecd9a8cbf7ff555b40f98398
```

## Run inference on one FOA file

Input audio must be first-order Ambisonics in **W, Y, Z, X** channel order.
Audio is resampled to 48 kHz and repeated/cropped to 10 seconds, matching the
evaluation preprocessing.

One audio-text pair:

```bash
disse infer \
  --audio example_foa.wav \
  --text "A dog barking from the left in a reverberant room" \
  --checkpoint checkpoints/disse_epoch20.pt \
  --output results/example_embeddings.npz
```

The embedding cache contains:

- `audio_source`, `audio_spatial`, `text_source`, `text_spatial`;
- `source_id`, `spatial_id`.

See [the data-format documentation](docs/data-format.md) for the public
manifest format, legacy column aliases, FOA conventions, and precomputed
feature shapes.

## Recreate the paper evaluation

The paper uses the Cartesian product of 96 AudioCaps source clips and 96
synthetic spatial conditions (9,216 items). `evaluation/audio_fixed.csv` and
`evaluation/rir_fixed.csv` identify that exact grid. Raw YouTube-derived audio
is not included.

First, place lawful local copies of the 96 dry 10-second clips in one
directory, named by `audiocap_id`, for example `104274.wav` or `104274.mp3`.
Then run:

```bash
# Recreate the 96 tetrahedral A-format RIRs from fixed room geometry.
disse generate-rirs

# Make the 9,216-row manifest. Captions are regenerated with seed 42.
disse make-evaluation-manifest --dry-root /path/to/local/dry_clips

disse validate-manifest data/evaluation/manifest.csv

# Convolution and FOA/IV preprocessing happen on demand; intermediate
# spatialized waveforms do not need to be stored.
disse infer \
  --manifest data/evaluation/manifest.csv \
  --checkpoint checkpoints/disse_epoch20.pt \
  --batch-size 8 \
  --device cuda \
  --amp \
  --output results/evaluation_embeddings.npz

disse evaluate \
  results/evaluation_embeddings.npz \
  --output results/evaluation_metrics.json
```

This pipeline is distilled from `make_balanced_rirs_by_category.py`,
`make_val_fixed.py`, and `precompute_val.py` in the research repository. See
[the data-format documentation](docs/data-format.md) for the exact microphone,
FOA, and manifest conventions.

## IIDR

Let (d(a,b)=1-\cos(a,b)). For one embedding space, DISSE reports:

\[
\operatorname{IIDR}_{\mathrm{source}} =
\frac{\mathbb{E}[d(x^{(s,p)},x^{(s',p)})]}
     {\mathbb{E}[d(x^{(s,p)},x^{(s,p')})]},
\qquad
\operatorname{IIDR}_{\mathrm{spatial}} =
\frac{\mathbb{E}[d(x^{(s,p)},x^{(s,p')})]}
     {\mathbb{E}[d(x^{(s,p)},x^{(s',p)})]}.
\]

Larger target-factor IIDR and smaller non-target IIDR are desirable. The
implementation is exact but avoids allocating a 9,216 x 9,216 distance matrix.

Expected values from Table I of the paper are:

| Modality | DISSE embedding | IIDR (source) | IIDR (spatial) |
|---|---:|---:|---:|
| Audio | Source | 5.5653 | 0.1797 |
| Audio | Spatial | 0.2817 | 3.5493 |
| Text | Source | 15.5462 | 0.0643 |
| Text | Spatial | 0.1246 | 8.0247 |

`disse evaluate` also computes multi-positive R@K and MedR for the on-task,
off-task, intra-modal, cross-modal, and joint source-and-spatial conditions in
Tables II and III. Exact definitions are in [docs/metrics.md](docs/metrics.md).

## Reproducibility notes

- The paper evaluates 96 source clips crossed with 96 RIRs: 9,216 items.
- Audio is four-channel FOA at 48 kHz and 10 seconds.
- All four embeddings have dimension 512.
- The released checkpoint is the standard DISSE model after epoch 20.
- The model is loaded strictly by default; missing or unexpected checkpoint
  keys are treated as errors.
- The exact metadata-to-language lexicon is an implementation detail not
  specified in the paper. It is documented separately in
  [docs/caption-mapping.md](docs/caption-mapping.md).

## Optional metric smoke test

The IIDR evaluator depends only on NumPy. Its basic operation can therefore be
checked with synthetic embeddings without downloading the model checkpoint:

```bash
disse demo-cache
disse evaluate results/demo_embeddings.npz --iidr-only
```

## Data redistribution

AudioCaps captions and metadata are available from the
[AudioCaps repository](https://github.com/cdjkim/audiocaps). The underlying
clips are identified by YouTube video IDs. Google AudioSet distributes segment
metadata and extracted features rather than the original YouTube audio; see
the [official AudioSet download page](https://research.google.com/audioset/download.html).
Users distributing derived FOA waveforms must independently confirm that they
have the necessary rights. This repository instead performs evaluation from
locally obtained clips and regenerated synthetic RIRs. See
[DATA_NOTICE.md](DATA_NOTICE.md).

## Citation

```bibtex
@inproceedings{ueji2026disse,
  title     = {DISSE: Learning Disentangled Source and Spatial Representations
               from Spatial Audio--Text Contrastive Learning},
  author    = {Ueji, Shotaro and Takamichi, Shinnosuke and Yamaoka, Kouei},
  booktitle = {Proceedings of the European Signal Processing Conference (EUSIPCO)},
  year      = {2026}
}
```
