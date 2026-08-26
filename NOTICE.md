# Public release scope

This repository is a bounded public release of a computer-vision course pilot. It is not a production moderation service. It includes source code, two small trained policy heads from the validated course run, and a curated set of aggregate evaluation charts.

The public snapshot intentionally excludes:

- raw and normalized dataset images;
- pretrained model caches and any classifier heads other than the two files listed in `models/trained_head_manifest.json`;
- executed notebooks with embedded image outputs, reports, presentation files, narrated video, and intermediate media;
- coursework-specific builders that contain teammate identity or local runtime paths.

The CSV registries remain available so source revisions, URLs, declared licenses, selection rules, and known provenance limits can be inspected before anyone rebuilds the data.

Dataset and model assets retain their original terms. In particular, registry-level license metadata does not replace per-item rights review. Anyone redistributing downloaded images must review the applicable source license, attribution, and share-alike requirements.

The included `joblib` heads use Python's pickle mechanism, so any modified or substituted file could execute code during deserialization. Verify the files with `python scripts/preflight.py --profile core` before running inference. The runtime repeats this integrity check before calling `joblib.load`. The expected byte sizes and SHA-256 digests are recorded in `models/trained_head_manifest.json`. Do not load model artifacts obtained from another source. The training script remains available for an independent local rebuild.

No `LICENSE` file is included. Public visibility does not grant a blanket right to copy, modify, or redistribute this repository. A code license should be added only after the project contributors agree on one.
