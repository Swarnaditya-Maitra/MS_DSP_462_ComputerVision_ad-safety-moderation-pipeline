# Local Jupyter Book source

This directory contains the landing page, table of contents, and visual theme
for the Ad Safety technical book. The canonical executable chapter is
`../ad_safety_moderation_pipeline.ipynb`.

The published book is available at
[swarnaditya-maitra.github.io/MS_DSP_462_ComputerVision_ad-safety-moderation-pipeline](https://swarnaditya-maitra.github.io/MS_DSP_462_ComputerVision_ad-safety-moderation-pipeline/).

From the repository root, build the static site with:

```bash
python scripts/build_book.py
```

The build refuses to run when the classifier or any required evaluation file is
missing. It executes the root notebook and saves that file as the canonical
executed copy. For a Jupyter Book build, it stages a temporary chapter instead
of creating a second tracked notebook under `book/`. It uses `jupyter-book` when
available; otherwise it creates an equivalent local static site with
`nbconvert` at `book/_build/html/index.html`.

No measured value is stored in the Markdown source. Results come only from the
saved artifacts under `outputs/evaluation/`.
