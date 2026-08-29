# Project data, attribution, and reuse notice

This is a bounded MS_DSP_462 computer-vision course project, not a production moderation service. Public Git contains the runnable source code, trained policy heads, saved model and evaluation evidence, an executed notebook, the original proposal, the final report, the PowerPoint presentation, the narrated demonstration video, and the dataset files cleared for this release.

## Dataset boundary

[DATASETS.md](DATASETS.md) is the sole detailed inventory of the formal and external diagnostic collections. It records provenance, source revisions, hashes, release decisions, and known rights gaps. Item-level Wikimedia credits and license notices remain in `data/WIKIMEDIA_ATTRIBUTION.md`.

Some third-party inputs remain local because supported redistribution terms or item-level provenance are incomplete. Reconstructing a file from an upstream source does not grant redistribution rights. Project-generated and attributable files included in public Git also retain the terms stated in the canonical data record.

Tracked course records can contain bounded screenshots, thumbnails, or annotations derived from examples that are not released as standalone files. This includes notebook output, evaluation and demo figures, app screenshots, the report, the presentation, and the video. Publishing a course record does not grant permission to extract, reuse, or republish an embedded source image.

Repository-level license metadata does not prove that an upstream publisher held every right needed for every item. Anyone republishing material must review its recorded source, license, attribution, and share-alike obligations. This repository documents known gaps; it does not cure them.

## Contributor-identified deliverables

The notebook, report, presentation, video, book configuration, and their builders identify the course contributors. Publishing this repository makes those names and the coursework publicly visible. Reusers must not imply that a contributor endorses a reuse, product, or moderation decision.

## Model and code safety

Pretrained backbone caches are not tracked. Bootstrap downloads only the revisions recorded in `models/pretrained_model_manifest.json` and verifies their snapshot fingerprints.

The included `joblib` policy heads use Python's pickle mechanism, so a modified or substituted file could execute code during deserialization. Run `python scripts/preflight.py --profile core` before inference. The runtime repeats the size and SHA-256 check before calling `joblib.load`; expected values are in `models/trained_head_manifest.json`.

## Repository license status

No top-level `LICENSE` file is included. Public visibility does not grant a blanket right to copy, modify, or redistribute the code, datasets, model artifacts, or course deliverables. The project contributors must agree on code and team-generated-asset terms before a general reuse license is added. Third-party assets retain their original terms regardless of any future repository license.
