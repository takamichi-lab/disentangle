# Data notice

The MIT License in this repository covers the DISSE source code. It does not
grant rights to third-party audio, captions, datasets, pretrained models, or
other external material.

The files in `evaluation/` identify 96 AudioCaps sources and 96 synthetic
spatial conditions for a 9,216-item grid following the evaluation protocol in
the paper. They are not the exact manifest or precomputed features used for
Table I.

`disse download evaluation-audio` retrieves a third-party AudioCaps test
archive from the URL recorded in `artifacts.json` and extracts the 96 selected
dry clips. Neither the archive nor the extracted audio is covered by this
repository's MIT License. Users must obtain and use audio lawfully and follow
the applicable AudioCaps, AudioSet, YouTube, source-video, and archive-provider
terms.

The epoch-20 checkpoint is distributed separately through Google Drive. Its
use is also subject to the licenses of the model components on which DISSE is
built.
