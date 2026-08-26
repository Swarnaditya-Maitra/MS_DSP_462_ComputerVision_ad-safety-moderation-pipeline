# Dataset inventory, provenance, and public-release boundary

Audit date: 2026-08-26

This project uses two image collections for different purposes. The 288-image capstone dataset is the formal train, validation, and test dataset. The 27-image Wikimedia collection is a separate external diagnostic for measuring source-style shift. They are not duplicate copies of the same benchmark.

## Exact local inventory

The complete validated local inventory contains 315 images and 16,101,535 bytes:

| Collection | Purpose | Images | Bytes | Registry |
|---|---|---:|---:|---|
| `data/capstone_dataset/` | Formal train, validation, and test data | 288 | 13,047,772 | [`data/capstone_registry.csv`](data/capstone_registry.csv) |
| `data/wikimedia_pilot/` | External diagnostic | 27 | 3,053,763 | [`data/dataset_registry.csv`](data/dataset_registry.csv) |
| Total |  | 315 | 16,101,535 |  |

The inventory audit found:

- 315 of 315 registered files present;
- 315 of 315 local SHA-256 values matching their registries;
- 315 of 315 images decodable with their registered dimensions;
- zero duplicate local SHA-256 values across the two collections;
- zero duplicate raw hashes, local hashes, perceptual dHashes, sample IDs, or paths within the 288-image capstone dataset; and
- zero exact-hash or source-group overlap across capstone train, validation, and test splits.

The deterministic inventory digest is:

```text
05654ca315f2f9167cc3ee88934c35fbeaf59eb5f5bc713a3550cb70ef793134
```

This value is the SHA-256 of the UTF-8 encoding of all 315 records sorted by relative path, with each record formatted as `<relative_path>\t<file_sha256>\n`. It verifies the audited inventory as a set of path-and-content pairs. The per-file hashes remain authoritative.

## Capstone dataset composition

The capstone builder fixes seed `462`, uses source groups for splitting, and targets 48 train, 12 validation, and 12 test examples for each of four policy labels. The resulting 288 rows comprise:

| Source | Policy use | Images | Recorded terms | Public status |
|---|---|---:|---|---|
| `EthanGabis/ADautoGen-DS` | Safe class | 72 | Registry says MIT, but the pinned upstream repository does not publish supporting license metadata or a license file | Local-only pending written permission or a verifiable upstream license |
| `rajshivanshuu/weapons_set1` | Firearms and explosives classes | 144 | Repository-level OSL-3.0 metadata only; per-image origins and rights are absent | Local-only pending item-level provenance and redistribution clearance |
| `project_generated_financial_creatives_v1` | Financial-promotion class | 72 | Project-generated with Pillow; no dataset reuse license is granted | Included as course artifacts at the repository owner's direction; contributor rights remain reserved |

### ADautoGen-DS

The builder pins revision `fa7a7803265fc97926d7fab694bfc1e05c1fc7b4` and source file `data/train-00000-of-00001.parquet`. The official README describes 120 synthetic product advertisements generated with Stable Diffusion v1.5. However, at the pinned and current revision:

- the Hugging Face API reports no dataset-card license metadata;
- the repository has no `LICENSE` file; and
- the README has no license section supporting the registry's MIT assertion.

Public download availability is not a redistribution grant. The selected 72 normalized images therefore remain local-only until the dataset owner supplies verifiable terms.

Official source evidence:

- Pinned repository: https://huggingface.co/datasets/EthanGabis/ADautoGen-DS/tree/fa7a7803265fc97926d7fab694bfc1e05c1fc7b4
- Pinned README: https://huggingface.co/datasets/EthanGabis/ADautoGen-DS/blob/fa7a7803265fc97926d7fab694bfc1e05c1fc7b4/README.md
- Pinned API metadata: https://huggingface.co/api/datasets/EthanGabis/ADautoGen-DS/revision/fa7a7803265fc97926d7fab694bfc1e05c1fc7b4

### weapons_set1

The builder pins revision `d4d9dbc8272958820a3f6757e4c48d8987300271`. The repository card declares OSL-3.0 at repository level, but it provides no per-image creator, original source, copyright notice, attribution, model release, or consent record. The local registry accurately labels this limitation.

The selected 144 images therefore remain local-only until the uploader or original rights holders provide adequate provenance and redistribution clearance. Any future OSL-based release would also have to preserve the applicable license, source-form, and attribution obligations.

Official source evidence:

- Pinned repository: https://huggingface.co/datasets/rajshivanshuu/weapons_set1/tree/d4d9dbc8272958820a3f6757e4c48d8987300271
- Pinned README: https://huggingface.co/datasets/rajshivanshuu/weapons_set1/blob/d4d9dbc8272958820a3f6757e4c48d8987300271/README.md
- Pinned API metadata: https://huggingface.co/api/datasets/rajshivanshuu/weapons_set1/revision/d4d9dbc8272958820a3f6757e4c48d8987300271
- OSL-3.0 text: https://opensource.org/license/osl-3-0

### Project-generated financial creatives

The 72 financial-promotion images are rendered deterministically by [`scripts/build_capstone_dataset.py`](scripts/build_capstone_dataset.py) using Pillow drawing primitives, seed `462`, 24 fictional campaigns, and three variants per campaign. The images say `SIMULATED CREATIVE` and `NOT FINANCIAL ADVICE`. The registry records no third-party visual assets.

At the repository owner's direction, these 72 images are included in the public repository as course artifacts so the saved project evidence can be inspected. Their inclusion and public visibility do not grant permission to copy, modify, or redistribute them. No dataset reuse license is granted, and all contributor rights remain reserved. Anyone seeking reuse must obtain permission from the applicable contributor or rights holder.

## The 216-image local-only boundary

The 72 ADautoGen and 144 weapons images, 216 images in total, remain local-only until their rights are cleared. Their standalone normalized dataset files are excluded from public Git history. Their exact source revisions, source paths, normalized hashes, grouping rules, and selection seed remain recorded in [`data/capstone_registry.csv`](data/capstone_registry.csv).

Some tracked course records, such as notebook outputs, evaluation figures, demo overlays, screenshots, the report, the presentation, and the video, may reproduce or annotate selected source examples. Those bounded course records are not a substitute for the missing image-level rights evidence and do not clear the underlying images for reuse. Anyone republishing or extracting an embedded example must review the recorded source and obtain any permission the intended use requires.

The method remains reproducible through the pinned builder:

```bash
python scripts/build_capstone_dataset.py
python scripts/build_capstone_dataset.py --verify-only
```

The builder can reconstruct the selected local dataset while the pinned sources remain available. Running it does not grant redistribution rights, and upstream availability can change. A rebuilt dataset must be checked against its registry and current source terms before use or sharing.

The included trained heads keep the application runnable without these raw images. The local-only images are required for an exact audit or retraining workflow, not for normal app inference.

## Wikimedia external diagnostic

The Wikimedia collection contains 27 downloaded and normalized images. Manual review retains 26 as the external diagnostic and excludes one title-search collision with no visible explosive content. The relevant per-item creator, source, license, and change notices are in [`data/WIKIMEDIA_ATTRIBUTION.md`](data/WIKIMEDIA_ATTRIBUTION.md).

The 26 relevant items consist of:

- 13 public-domain files;
- 2 CC BY 4.0 files;
- 4 CC BY-SA 4.0 files, including the deleted-source `Bitsquare.png` record;
- 4 CC BY-SA 3.0 files;
- 2 CC BY-SA 2.0 France files; and
- 1 CC BY-SA 2.5 file.

One relevant source page, `File:Bitsquare.png`, was deleted from Commons on 2026-08-26 as promotional spam or out-of-scope content. The local registry preserves its historical CC BY-SA 4.0, creator, source, and hash record. The attribution file describes the heightened provenance caveat and links the official deletion discussion.

## Normalization and attribution

The Wikimedia builder downloads a Commons source or thumbnail, applies EXIF orientation, converts it to RGB, scales it within 768 by 768 pixels, and saves a normalized JPEG at quality 91 with optimization enabled. Each released copy must be identified as a normalized and re-encoded version.

At minimum, each Creative Commons item must retain:

- the creator or other designated attribution party;
- the source file-page URL;
- the specific license name and license URL;
- a notice that the file was normalized and re-encoded; and
- the same license for adaptations when the applicable ShareAlike terms require it.

Public-domain items retain source and creator information for provenance even where attribution is not legally required. Creative Commons terms do not grant trademark, privacy, publicity, personality, patent, or endorsement rights. Those restrictions require separate review.

Official guidance:

- Wikimedia Commons reuse guidance: https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia
- CC BY 4.0: https://creativecommons.org/licenses/by/4.0/
- CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/
- CC BY-SA 3.0: https://creativecommons.org/licenses/by-sa/3.0/
- CC BY-SA 2.5: https://creativecommons.org/licenses/by-sa/2.5/
- CC BY-SA 2.0 France: https://creativecommons.org/licenses/by-sa/2.0/fr/

## Replacing blocked material

A fully self-contained public training repository requires replacing the 216 unresolved third-party images with contributor-owned images or files carrying item-level open licenses. That replacement creates a new dataset version and requires retraining the classifier heads and regenerating all metrics, notebook outputs, reports, and presentation claims. It must not be represented as the saved 288-image course run.

This document records a provenance and release-control assessment, not legal advice. A qualified professional should review any disputed or commercial reuse.
