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

Python 3.10-3.12 is supported. Clone the repository and create an isolated
environment:

```bash
git clone https://github.com/takamichi-lab/disentangle.git
cd disentangle

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

If necessary, install PyTorch 2.7.0 and torchaudio 2.7.0 for your CPU/CUDA
environment first. Then install the inference dependencies:

```bash
python -m pip install -e ".[inference]"
```

If `disse` is not available on `PATH`, use `python -m disse` in the commands
below.

The released experiment used PyTorch 2.7.0, torchaudio 2.7.0, and
Transformers 4.52.4. A CUDA GPU is strongly recommended; CPU inference is
supported but slow.

To generate the released synthetic RIR grid, also install the evaluation-data
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

## Run inference

Input audio must be first-order Ambisonics in **W, Y, Z, X** channel order.
Audio is resampled to 48 kHz and repeated/cropped to 10 seconds, matching the
evaluation preprocessing.

Audio-only inference evaluates only the audio branch:

```bash
disse infer \
  --audio /path/to/your_foa.wav \
  --checkpoint checkpoints/disse_epoch20.pt \
  --output results/audio_embeddings.npz
```

Text-only inference evaluates only the text branch:

```bash
disse infer \
  --text "A dog barking from the left in a reverberant room" \
  --checkpoint checkpoints/disse_epoch20.pt \
  --output results/text_embeddings.npz
```

Pass both inputs to extract all four embeddings in one invocation:

```bash
disse infer \
  --audio /path/to/your_foa.wav \
  --text "A dog barking from the left in a reverberant room" \
  --checkpoint checkpoints/disse_epoch20.pt \
  --output results/paired_embeddings.npz
```

Audio and text are encoded by separate inference methods; one input is never
used to compute the other modality's embeddings. A single-modality command
constructs and moves only that model branch to the selected device. The full
checkpoint was nevertheless learned with cross-modal contrastive objectives.

The embedding cache contains the selected modality's arrays:

- audio: `audio_source`, `audio_spatial`;
- text: `text_source`, `text_spatial`;
- `source_id`, `spatial_id`.

See [the data-format documentation](docs/data-format.md) for the public
manifest format, legacy column aliases, FOA conventions, and precomputed
feature shapes.

## Run the IIDR evaluation

The paper uses the Cartesian product of 96 AudioCaps source clips and 96
synthetic spatial conditions (9,216 items). The released
`evaluation/audio_fixed.csv` and `evaluation/rir_fixed.csv` define a grid that
follows this evaluation protocol. See
[Reproducibility notes](#reproducibility-notes) before comparing the resulting
values with the paper.

Download the AudioCaps test archive and extract the 96 required MP3 files:

```bash
disse download evaluation-audio
```

This downloads the approximately 1.26 GiB archive from
[ss-takashi.sakura.ne.jp](https://ss-takashi.sakura.ne.jp/corpus/audiocaps/test.zip),
verifies its SHA-256 checksum, extracts only the clips listed in
`evaluation/audio_fixed.csv` to `data/evaluation/dry/`, and removes the
downloaded ZIP. The audio is third-party data and is not covered by this
repository's MIT License; see [Data redistribution](#data-redistribution).

Then run:

```bash
# Generate the 96 tetrahedral A-format RIRs from the released room geometry.
disse generate-rirs

# Make the 9,216-row manifest. Captions are regenerated with seed 42.
disse make-evaluation-manifest --dry-root data/evaluation/dry

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
  --iidr-only \
  --output results/evaluation_metrics.json
```

Manifest inference uses both modalities by default. To compute audio IIDR
without evaluating the text branch, add `--modality audio`. Use
`--modality text` for text-only extraction.

Remove `--iidr-only` to additionally compute the retrieval metrics implemented
by this repository. This pipeline is distilled from
`make_balanced_rirs_by_category.py`, `make_val_fixed.py`, and
`precompute_val.py` in the research repository. See
[the data-format documentation](docs/data-format.md) for the microphone, FOA,
and manifest conventions.

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
implementation avoids allocating a 9,216 x 9,216 distance matrix while
computing the same mean pairwise cosine-distance quantities.

`disse evaluate` also computes multi-positive R@K and MedR for the on-task,
off-task, intra-modal, cross-modal, and joint source-and-spatial conditions in
Tables II and III. Exact definitions are in [docs/metrics.md](docs/metrics.md).

## Reproducibility notes

- The paper evaluates 96 source clips crossed with 96 RIRs: 9,216 items.
- This repository provides code for inference and IIDR computation, but does
  not include the exact evaluation manifest and precomputed features used to
  produce Table I. The released data-generation pipeline follows the protocol
  described in the paper; it does not guarantee identical item selection or
  numerical results.
- Results may also vary because of audio decoding, spatialization, feature
  extraction, software versions, and hardware.
- Audio is four-channel FOA at 48 kHz and 10 seconds.
- All four embeddings have dimension 512.
- The released checkpoint is the standard DISSE model after epoch 20.
- Selected model branches are loaded strictly by default. Paired inference
  checks the complete checkpoint; single-modality inference intentionally
  ignores weights belonging only to the unselected branch.
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

`disse download evaluation-audio` retrieves a third-party AudioCaps test
archive from the URL recorded in `artifacts.json`. The archive and extracted
audio are not distributed as part of this Git repository and are not covered
by its MIT License. Users are responsible for complying with the applicable
terms and for confirming that their use of the audio is lawful.

AudioCaps captions and metadata are available from the
[AudioCaps repository](https://github.com/cdjkim/audiocaps). The underlying
clips are identified by YouTube video IDs. Google AudioSet distributes segment
metadata and extracted features rather than the original YouTube audio; see
the [official AudioSet download page](https://research.google.com/audioset/download.html).
Users distributing derived FOA waveforms must independently confirm that they
have the necessary rights. This repository performs evaluation from downloaded
dry clips and regenerated synthetic RIRs. See [DATA_NOTICE.md](DATA_NOTICE.md).

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
