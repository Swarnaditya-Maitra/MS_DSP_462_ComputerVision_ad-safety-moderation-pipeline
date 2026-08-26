# Ad Safety Moderation Pilot

This local technical book documents a reproducible, multi-technique computer
vision pilot for static display-ad triage. The system combines a frozen visual
policy classifier with optional open-vocabulary object detection, OCR cues, and
an auditable approve, review, or block policy.

```{admonition} Evidence boundary
:class: warning
Every result shown in the executable chapter is loaded from saved evaluation
artifacts. The build stops if evidence is missing. Financial-promotion imagery
does not prove fraud, legality, or regulatory status, and pilot metrics do not
establish production, fairness, or adversarial-robustness performance.
```

## What the book covers

- problem scope, policy boundaries, and ethics;
- dataset provenance, EDA, and leakage controls;
- ViT and CNN comparison under a shared protocol;
- classifier, detector, and OCR policy fusion;
- held-out metrics, confusion analysis, latency, and failure cases;
- local Streamlit and FastAPI demonstration instructions.

```{note}
The canonical notebook is `../ad_safety_moderation_pipeline.ipynb`. The book
builder executes it from the project root and synchronizes the resulting chapter
before creating the static site.
```

Continue to {doc}`ad_safety_moderation_pipeline`.

## Primary references

- Dosovitskiy et al., [An Image is Worth 16x16 Words](https://openreview.net/forum?id=YicbFdNTTy), ICLR 2021.
- He et al., [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html), CVPR 2016.
- Liu et al., [Grounding DINO](https://arxiv.org/abs/2303.05499), ECCV 2024.
- Formal per-image provenance, source groups, revisions, hashes, and license
  evidence are stored in `data/capstone_registry.csv`.
- `data/wikimedia_external_manifest.csv` is a separately labeled diagnostic
  manifest, not part of model training or the formal test set.
