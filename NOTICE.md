# Public release scope

This repository is a source-only release of a bounded computer-vision course pilot. It is not a production moderation service.

The public snapshot intentionally excludes:

- raw and normalized dataset images;
- pretrained model caches and locally trained classifier heads;
- executed notebooks with embedded image outputs;
- reports, presentation files, narrated video, and intermediate media;
- coursework-specific builders that contain teammate identity or local runtime paths.

The CSV registries remain available so source revisions, URLs, declared licenses, selection rules, and known provenance limits can be inspected before anyone rebuilds the data.

Dataset and model assets retain their original terms. In particular, registry-level license metadata does not replace per-item rights review. Anyone redistributing downloaded images must review the applicable source license, attribution, and share-alike requirements.

The training script produces `joblib` model artifacts locally. Because `joblib` is based on Python pickle, untrusted model files can execute code during deserialization. This public release does not distribute trained heads. Rebuild them locally and do not load model artifacts obtained from an untrusted source.

No `LICENSE` file is included. Public visibility does not grant a blanket right to copy, modify, or redistribute this repository. A code license should be added only after the project contributors agree on one.
