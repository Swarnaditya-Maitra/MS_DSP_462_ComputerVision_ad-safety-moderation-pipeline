# Local Jupyter Book source

This directory contains the landing page, table of contents, and visual theme
for the Ad Safety technical book. The canonical executable chapter is
`../ad_safety_moderation_pipeline.ipynb`.

From `Project/`, build the static site with:

```bash
python scripts/build_book.py
```

The build refuses to run when the classifier or any required evaluation file is
missing. It executes the notebook from `Project/`, saves the same executed copy
to the primary path and this book chapter, and then uses `jupyter-book` when available. If
`jupyter-book` is unavailable, it creates an equivalent local static site with
`nbconvert` at `book/_build/html/index.html`.

No measured value is stored in the Markdown source. Results come only from the
saved artifacts under `outputs/evaluation/`.
