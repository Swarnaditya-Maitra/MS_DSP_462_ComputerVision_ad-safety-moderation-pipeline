#!/usr/bin/env python3
"""Build the evidence-driven Ad Safety capstone notebook.

The builder writes structure and executable analysis cells only. It never
inserts measured values. Evaluation results are loaded from the required files
under outputs/evaluation when the notebook is executed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "ad_safety_moderation_pipeline.ipynb"
EVALUATION_ARTIFACTS = (
    "outputs/evaluation/evaluation_metrics.json",
    "outputs/evaluation/independent_validation.json",
    "outputs/evaluation/per_class_metrics.csv",
    "outputs/evaluation/test_predictions.csv",
    "outputs/evaluation/thresholds.csv",
    "outputs/evaluation/metrics_summary.csv",
    "outputs/evaluation/benchmark_cpu_batch1.csv",
    "outputs/evaluation/split_summary.csv",
    "outputs/evaluation/output_checksums.csv",
    "outputs/evaluation/dataset_distribution.png",
    "outputs/evaluation/dataset_contact_sheet.jpg",
    "outputs/evaluation/confusion_matrix_vit.png",
    "outputs/evaluation/confusion_matrix_resnet50.png",
    "outputs/evaluation/precision_recall_curves.png",
    "outputs/evaluation/threshold_calibration.png",
    "outputs/evaluation/model_comparison.png",
)
RUNTIME_ARTIFACTS = ("models/vit_policy_head.joblib",)


def _text(value: str) -> str:
    return dedent(value).strip() + "\n"


def md(value: str):
    return nbf.v4.new_markdown_cell(_text(value))


def code(value: str):
    return nbf.v4.new_code_cell(_text(value))


def missing_artifacts(project_root: Path = PROJECT_ROOT) -> list[str]:
    required = EVALUATION_ARTIFACTS + RUNTIME_ARTIFACTS
    return [relative for relative in required if not (project_root / relative).is_file()]


def validate_artifacts(project_root: Path = PROJECT_ROOT) -> None:
    missing = missing_artifacts(project_root)
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Measured-evidence gate failed. The notebook will not substitute "
            "placeholder results. Missing required files:\n"
            f"{formatted}\n"
            "Run the project training/evaluation workflow, then rebuild or execute "
            "the notebook from the repository root."
        )


def build_notebook() -> nbf.NotebookNode:
    cells = [
        md(
            """
            # Ad Safety: Reproducible Multi-Technique Moderation Pilot

            **Team:** Swarnaditya Maitra, Vijay Agnihotri, Myetchae Thu, and Bickramjit Basu  
            **Course:** MS_DSP_462 Computer Vision

            This notebook is the executable technical record for a bounded static-ad
            moderation pilot. It combines a frozen visual classifier, optional
            open-vocabulary object evidence, OCR cues, and a deterministic policy
            engine. Every reported measurement is read from saved evaluation
            artifacts. The notebook fails at the evidence gate when those artifacts
            are absent, rather than displaying invented placeholders.

            The pilot does **not** prove fraud, legality, intent, production safety,
            fairness, or adversarial robustness.
            """
        ),
        md(
            """
            ## 1. Objective, questions, and success criteria

            I evaluate five questions:

            1. Does a ViT frozen-backbone policy head outperform the CNN comparison on
               the same leakage-controlled split?
            2. What errors occur for Safe, Firearms, Explosives, and Financial
               Promotion images?
            3. Can detector and OCR cues support auditable review without being
               treated as policy truth?
            4. Do the saved results satisfy proposal targets for macro F1,
               Safe-class precision, and per-class restricted recall, while also
               meeting the separate Safe-ad restricted-threshold crossing guardrail?
            5. Which latency, concurrent-throughput, and MPS resource targets were
               measured, and which remain unassessed?

            Proposal targets are **macro F1 > 0.93**, **Safe-class precision > 0.98**,
            **per-class restricted recall > 0.95**, **latency < 50 ms per asset**,
            strong concurrent-load throughput, and minimal MPS resource use. The
            project separately tracks a **Safe-ad restricted-threshold crossing rate
            < 0.02** as an operational guardrail. The validation recall floor of **>= 0.90**
            used during threshold selection is a tuning rule, not the proposal's
            > 0.95 per-class acceptance target. These are targets, not claims. Result
            cells assess them only when an unambiguous measured field exists.
            """
        ),
        md(
            """
            ## 2. Reproducible setup

            Run this notebook with the working directory set to the repository root. Dependencies
            belong in the project environment described by `requirements.txt`; the
            notebook deliberately contains no package-install cells.
            """
        ),
        code(
            """
            from __future__ import annotations

            import hashlib
            import json
            import os
            import platform
            import random
            import re
            import subprocess
            import sys
            import warnings
            from pathlib import Path

            PROJECT_ROOT = Path.cwd().resolve()
            if not (PROJECT_ROOT / "src" / "ad_safety").is_dir():
                raise RuntimeError(
                    "Start Jupyter from the repository root so Path.cwd() contains "
                    "src/ad_safety, data, models, and outputs."
                )

            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
            os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mpl-cache"))
            os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
            os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
            warnings.filterwarnings("ignore", message="IProgress not found.*")
            sys.path.insert(0, str(PROJECT_ROOT / "src"))

            SEED = 42
            random.seed(SEED)
            os.environ["PYTHONHASHSEED"] = str(SEED)
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            import torch
            import yaml
            from IPython.display import Image as NotebookImage
            from IPython.display import display
            from PIL import Image

            from ad_safety.features import CNN_MODEL_NAME, VIT_MODEL_NAME, select_device
            from ad_safety.policy import PolicyEngine

            np.random.seed(SEED)
            torch.manual_seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED)

            device = select_device("auto")
            environment = {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "device": str(device),
                "mps_available": bool(torch.backends.mps.is_available()),
                "cuda_available": bool(torch.cuda.is_available()),
                "seed": SEED,
            }
            display(pd.DataFrame([environment]))
            """
        ),
        md(
            """
            ## 3. Problem definition, policy boundary, and ethics

            The system triages static JPEG, PNG, or WebP advertisements into four
            visual policy categories. Firearms and explosives can cross a validated
            block threshold. Financial promotion remains a review category because an
            image cannot establish fraud, legality, or jurisdiction. Ambiguous cases
            should reach a person rather than receive an automatic approval.

            The formal capstone benchmark is small and bounded. It combines synthetic
            advertising creatives with externally sourced weapon imagery and separates
            source groups before training. It is not representative of live ad traffic.
            Repository revisions, source groups, license evidence, hashes, synthetic
            status, and audit state remain part of the evidence record. Wikimedia
            examples are reserved for a separately labeled external diagnostic. OCR
            text and detector boxes are supporting cues only. Occlusion maps are
            qualitative sensitivity views, not causal explanations.
            """
        ),
        code(
            """
            POLICY_PATH = PROJECT_ROOT / "configs" / "policy.yaml"
            if not POLICY_PATH.is_file():
                raise FileNotFoundError(f"Missing policy configuration: {POLICY_PATH}")

            policy_config = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
            policy_summary = pd.DataFrame(
                [
                    {
                        "label": label,
                        "display_name": policy_config["label_display_names"].get(label, label),
                        "default_action": policy_config["actions"].get(label, "UNSPECIFIED"),
                        "threshold": policy_config["default_thresholds"].get(label),
                    }
                    for label in policy_config["model_labels"]
                ]
            )
            display(policy_summary)
            configured_limitations = list(policy_config.get("limitations", []))
            legacy_dataset_wording = [
                limitation
                for limitation in configured_limitations
                if "wikimedia benchmark" in str(limitation).casefold()
            ]
            print("Configuration limitations:")
            for limitation in configured_limitations:
                prefix = (
                    "[LEGACY DATASET WORDING; NOT THIS FORMAL EVALUATION] "
                    if limitation in legacy_dataset_wording
                    else ""
                )
                print(f"- {prefix}{limitation}")
            if legacy_dataset_wording:
                print(
                    "Formal evaluation source: data/capstone_registry.csv. "
                    "Wikimedia is reserved for an optional, separately labeled "
                    "external diagnostic."
                )
            """
        ),
        md(
            """
            ## 4. Rebuild controls and project scripts

            Normal report execution reuses the versioned dataset, model artifact, and
            evaluation files. Expensive or networked rebuild steps are opt-in through
            environment variables. This keeps a routine report run deterministic while
            preserving a callable path to the project scripts.
            """
        ),
        code(
            """
            def run_project_script(relative_path: str, *arguments: str) -> None:
                script = PROJECT_ROOT / relative_path
                if not script.is_file():
                    raise FileNotFoundError(f"Required project script does not exist: {relative_path}")
                command = [sys.executable, str(script), *arguments]
                print("Running:", " ".join(command))
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)


            if os.getenv("AD_SAFETY_REBUILD_DATA", "0") == "1":
                run_project_script("scripts/build_capstone_dataset.py")

            if os.getenv("AD_SAFETY_REDOWNLOAD_MODELS", "0") == "1":
                run_project_script("scripts/download_models.py")

            if os.getenv("AD_SAFETY_REBUILD_EVALUATION", "0") == "1":
                combined = PROJECT_ROOT / "scripts" / "train_and_evaluate.py"
                train = PROJECT_ROOT / "scripts" / "train_models.py"
                evaluate = PROJECT_ROOT / "scripts" / "evaluate.py"
                if combined.is_file():
                    run_project_script("scripts/train_and_evaluate.py")
                elif train.is_file() and evaluate.is_file():
                    run_project_script("scripts/train_models.py")
                    run_project_script("scripts/evaluate.py")
                else:
                    raise FileNotFoundError(
                        "AD_SAFETY_REBUILD_EVALUATION=1 requires either "
                        "scripts/train_and_evaluate.py or both scripts/train_models.py "
                        "and scripts/evaluate.py."
                    )

            if os.getenv("AD_SAFETY_RUN_EXTERNAL_DIAGNOSTIC", "0") == "1":
                run_project_script("scripts/evaluate_external_spot_check.py")

            print("Rebuild flags are opt-in; this run will read existing artifacts.")
            """
        ),
        md(
            """
            ## 5. Dataset provenance and exploratory analysis

            The registry is the source of truth. EDA below derives counts and visuals
            from the current registry, so no sample totals are embedded in prose.
            """
        ),
        code(
            """
            REGISTRY_PATH = PROJECT_ROOT / "data" / "capstone_registry.csv"
            if not REGISTRY_PATH.is_file():
                raise FileNotFoundError(
                    f"Missing formal training registry: {REGISTRY_PATH}. "
                    "Run scripts/build_capstone_dataset.py before this notebook."
                )

            registry = pd.read_csv(REGISTRY_PATH)
            DATASET_MANIFEST_PATH = PROJECT_ROOT / "data" / "capstone_dataset" / "dataset_manifest.json"
            if not DATASET_MANIFEST_PATH.is_file():
                raise FileNotFoundError(f"Missing dataset manifest: {DATASET_MANIFEST_PATH}")
            dataset_manifest = json.loads(DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))
            required_registry_columns = {
                "sample_id", "label", "split", "local_path", "source_dataset",
                "source_revision", "source_group", "raw_image_sha256",
                "local_sha256", "perceptual_dhash", "declared_license",
                "is_synthetic", "width", "height", "label_audit_status",
            }
            missing_columns = sorted(required_registry_columns - set(registry.columns))
            if missing_columns:
                raise ValueError(f"Dataset registry is missing columns: {missing_columns}")

            resolved_paths = registry["local_path"].map(lambda value: PROJECT_ROOT / str(value))
            missing_images = [str(path.relative_to(PROJECT_ROOT)) for path in resolved_paths if not path.is_file()]
            if missing_images:
                preview = "\\n".join(f"  - {item}" for item in missing_images[:20])
                raise FileNotFoundError(f"Registry references missing image files:\\n{preview}")

            profile = {
                "registered_images": int(len(registry)),
                "labels": int(registry["label"].nunique()),
                "splits": int(registry["split"].nunique()),
                "source_datasets": int(registry["source_dataset"].nunique()),
                "licenses": int(registry["declared_license"].fillna("UNDECLARED").nunique()),
                "synthetic_images": int(
                    registry["is_synthetic"].astype(str).str.casefold().eq("true").sum()
                ),
                "without_independent_label_audit": int(
                    (~registry["label_audit_status"].fillna("").str.casefold().eq("approved")).sum()
                ),
            }
            display(pd.DataFrame([profile]))
            display(pd.crosstab(registry["label"], registry["split"], margins=True))
            display(
                registry.assign(declared_license=registry["declared_license"].fillna("UNDECLARED"))
                .groupby(["source_dataset", "declared_license", "is_synthetic"], dropna=False)
                .size()
                .rename("images")
                .sort_values(ascending=False)
                .reset_index()
            )
            display(pd.DataFrame(dataset_manifest.get("sources", [])))
            print("Dataset-manifest limitations:")
            for limitation in dataset_manifest.get("known_limitations", []):
                print(f"- {limitation}")
            """
        ),
        code(
            """
            sns.set_theme(style="whitegrid")
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
            split_order = [item for item in ("train", "val", "test") if item in set(registry["split"])]

            sns.countplot(data=registry, x="label", hue="split", hue_order=split_order, ax=axes[0])
            axes[0].set_title("Registered images by label and split")
            axes[0].tick_params(axis="x", rotation=30)
            axes[0].set_xlabel("")

            aspect_ratio = registry["width"].astype(float) / registry["height"].astype(float)
            sns.histplot(aspect_ratio, bins=min(16, max(4, len(registry) // 2)), ax=axes[1])
            axes[1].set_title("Original aspect-ratio distribution")
            axes[1].set_xlabel("width / height")
            plt.tight_layout()
            plt.show()

            labels = sorted(registry["label"].astype(str).unique())
            splits = [item for item in ("train", "val", "test") if item in set(registry["split"])]
            fig, axes = plt.subplots(
                len(labels), len(splits),
                figsize=(4.2 * len(splits), 3.2 * len(labels)),
                squeeze=False,
            )
            for row_index, label in enumerate(labels):
                for column_index, split in enumerate(splits):
                    axis = axes[row_index, column_index]
                    candidates = registry.loc[
                        registry["label"].eq(label) & registry["split"].eq(split)
                    ].sort_values("sample_id")
                    if candidates.empty:
                        axis.text(0.5, 0.5, "No registered sample", ha="center", va="center")
                    else:
                        sample_path = PROJECT_ROOT / str(candidates.iloc[0]["local_path"])
                        with Image.open(sample_path) as sample:
                            axis.imshow(sample.convert("RGB"))
                    axis.set_title(f"{label} | {split}")
                    axis.axis("off")
            plt.suptitle("Deterministic registry sample grid", y=1.01)
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## 6. Leakage controls and split coverage

            Exact local and raw hashes, perceptual hashes, and source groups must not
            cross train, validation, and test. Class coverage and source-group counts
            are shown separately because a technically clean split can still be too
            small or too narrow for reliable generalization.
            """
        ),
        code(
            """
            leakage_rows = []
            for identity_column in (
                "local_sha256", "raw_image_sha256", "perceptual_dhash", "source_group"
            ):
                usable = registry.loc[registry[identity_column].fillna("").astype(str).str.len() > 0]
                split_counts = usable.groupby(identity_column)["split"].nunique()
                for identity in split_counts[split_counts > 1].index:
                    rows = usable.loc[usable[identity_column] == identity]
                    leakage_rows.append(
                        {
                            "identity_type": identity_column,
                            "identity": identity,
                            "splits": ", ".join(sorted(rows["split"].astype(str).unique())),
                            "rows": int(len(rows)),
                        }
                    )

            leakage = pd.DataFrame(leakage_rows)
            if not leakage.empty:
                display(leakage)
                raise RuntimeError(
                    "Cross-split identity leakage detected. Correct the registry and rerun evaluation."
                )
            print("PASS: no exact hash, perceptual-hash, or source-group identity crosses splits.")

            coverage = pd.crosstab(registry["label"], registry["split"])
            display(coverage)
            expected_splits = {"train", "val", "test"}
            coverage_gaps = [
                {"label": label, "missing_splits": ", ".join(sorted(expected_splits - set(group["split"]))) }
                for label, group in registry.groupby("label")
                if expected_splits - set(group["split"])
            ]
            if coverage_gaps:
                print("Coverage gaps that limit class-level evaluation:")
                display(pd.DataFrame(coverage_gaps))

            display(
                registry.groupby(["label", "split"])["source_group"]
                .nunique()
                .rename("distinct_source_groups")
                .reset_index()
            )
            """
        ),
        md(
            """
            ## 7. Model architecture and comparison plan

            The measured comparison uses frozen ImageNet-pretrained features with the
            same downstream policy-head procedure and split. The selected artifact is
            `models/vit_policy_head.joblib`. Grounding DINO and Tesseract remain
            optional evidence paths in `ModerationEngine`; they do not override the
            validated classifier block threshold. Letterboxing, EXIF correction,
            transparency handling, byte limits, and pixel limits live in the project
            preprocessing module.
            """
        ),
        code(
            """
            MODEL_MANIFEST_PATH = PROJECT_ROOT / "models" / "pretrained_model_manifest.json"
            if not MODEL_MANIFEST_PATH.is_file():
                raise FileNotFoundError(f"Missing pretrained model manifest: {MODEL_MANIFEST_PATH}")

            model_manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
            display(pd.DataFrame(model_manifest.get("models", [])))
            print("ViT comparison backbone:", VIT_MODEL_NAME)
            print("CNN comparison backbone:", CNN_MODEL_NAME)
            print("Runtime classifier:", "models/vit_policy_head.joblib")
            """
        ),
        md(
            """
            ## 8. Measured-evidence gate

            Execution stops here unless every contracted evaluation file and the
            deployable classifier artifact exist. This prevents a partially completed
            run from being presented as final evidence.
            """
        ),
        code(
            """
            REQUIRED_EVALUATION_ARTIFACTS = (
                "outputs/evaluation/evaluation_metrics.json",
                "outputs/evaluation/independent_validation.json",
                "outputs/evaluation/per_class_metrics.csv",
                "outputs/evaluation/test_predictions.csv",
                "outputs/evaluation/thresholds.csv",
                "outputs/evaluation/metrics_summary.csv",
                "outputs/evaluation/benchmark_cpu_batch1.csv",
                "outputs/evaluation/split_summary.csv",
                "outputs/evaluation/output_checksums.csv",
                "outputs/evaluation/dataset_distribution.png",
                "outputs/evaluation/dataset_contact_sheet.jpg",
                "outputs/evaluation/confusion_matrix_vit.png",
                "outputs/evaluation/confusion_matrix_resnet50.png",
                "outputs/evaluation/precision_recall_curves.png",
                "outputs/evaluation/threshold_calibration.png",
                "outputs/evaluation/model_comparison.png",
            )
            REQUIRED_RUNTIME_ARTIFACTS = ("models/vit_policy_head.joblib",)

            missing_artifacts = [
                relative
                for relative in REQUIRED_EVALUATION_ARTIFACTS + REQUIRED_RUNTIME_ARTIFACTS
                if not (PROJECT_ROOT / relative).is_file()
            ]
            if missing_artifacts:
                missing_text = "\\n".join(f"  - {path}" for path in missing_artifacts)
                raise FileNotFoundError(
                    "Measured-evidence gate failed. This notebook will not create "
                    "placeholder results. Missing required files:\\n"
                    f"{missing_text}\\n"
                    "Run the project training/evaluation workflow, then restart the notebook."
                )
            print("PASS: all contracted evaluation and runtime artifacts exist.")
            """
        ),
        code(
            """
            EVALUATION_DIR = PROJECT_ROOT / "outputs" / "evaluation"

            def read_json(name: str):
                path = EVALUATION_DIR / name
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path.relative_to(PROJECT_ROOT)}: {exc}") from exc


            def flatten_json(value, prefix: str = "") -> dict[str, object]:
                flattened: dict[str, object] = {}
                if isinstance(value, dict):
                    for key, child in value.items():
                        child_prefix = f"{prefix}.{key}" if prefix else str(key)
                        flattened.update(flatten_json(child, child_prefix))
                elif isinstance(value, list):
                    flattened[prefix] = json.dumps(value, sort_keys=True)
                else:
                    flattened[prefix] = value
                return flattened


            metrics = read_json("evaluation_metrics.json")
            independent_validation = read_json("independent_validation.json")
            per_class_metrics = pd.read_csv(EVALUATION_DIR / "per_class_metrics.csv")
            predictions = pd.read_csv(EVALUATION_DIR / "test_predictions.csv")
            thresholds = pd.read_csv(EVALUATION_DIR / "thresholds.csv")
            model_comparison = pd.read_csv(EVALUATION_DIR / "metrics_summary.csv")
            benchmark = pd.read_csv(EVALUATION_DIR / "benchmark_cpu_batch1.csv")
            split_summary = pd.read_csv(EVALUATION_DIR / "split_summary.csv")
            output_checksums = pd.read_csv(EVALUATION_DIR / "output_checksums.csv")

            if "local_path" not in predictions.columns:
                predictions = predictions.merge(
                    registry[["sample_id", "local_path"]],
                    on="sample_id",
                    how="left",
                    validate="many_to_one",
                )

            primary_model_key = str(metrics.get("protocol", {}).get("primary_model_predeclared", "vit"))
            model_results = metrics.get("models", [])
            primary_result = next(
                (row for row in model_results if str(row.get("key")) == primary_model_key),
                None,
            )
            if primary_result is None:
                raise ValueError(
                    f"evaluation_metrics.json has no result for predeclared primary model {primary_model_key!r}."
                )

            primary_predictions = predictions.loc[predictions["model"].astype(str).eq(primary_model_key)].copy()
            failure_cases = primary_predictions.loc[
                primary_predictions["true_label"].astype(str)
                .ne(primary_predictions["predicted_label"].astype(str))
            ].copy()

            artifact_shapes = pd.DataFrame(
                [
                    {"artifact": "per_class_metrics.csv", "rows": len(per_class_metrics), "columns": len(per_class_metrics.columns)},
                    {"artifact": "test_predictions.csv", "rows": len(predictions), "columns": len(predictions.columns)},
                    {"artifact": "thresholds.csv", "rows": len(thresholds), "columns": len(thresholds.columns)},
                    {"artifact": "metrics_summary.csv", "rows": len(model_comparison), "columns": len(model_comparison.columns)},
                    {"artifact": "benchmark_cpu_batch1.csv", "rows": len(benchmark), "columns": len(benchmark.columns)},
                    {"artifact": "split_summary.csv", "rows": len(split_summary), "columns": len(split_summary.columns)},
                    {"artifact": "derived primary-model failures", "rows": len(failure_cases), "columns": len(failure_cases.columns)},
                ]
            )
            display(artifact_shapes)
            """
        ),
        md(
            """
            ## 9. ViT versus CNN comparison

            The table below is read directly from `metrics_summary.csv`. Model names,
            scores, fit time, and latency columns are not assumed; the complete saved
            schema is displayed so the comparison remains traceable.
            """
        ),
        code(
            """
            if model_comparison.empty:
                raise ValueError("metrics_summary.csv exists but contains no measured rows.")
            display(model_comparison)
            """
        ),
        md(
            """
            ## 10. Test metrics and confusion analysis

            Overall and per-class results remain separate. Macro F1 alone can hide a
            dangerous restricted-class miss rate, while aggregate accuracy can hide
            poor behavior on a small class.
            """
        ),
        code(
            """
            metrics_flat = flatten_json(
                {
                    "run": metrics.get("run", {}),
                    "protocol": metrics.get("protocol", {}),
                    "dataset": metrics.get("dataset", {}),
                    "limitations": metrics.get("limitations", []),
                }
            )
            metrics_table = pd.DataFrame(
                [{"metric_path": key, "measured_value": value} for key, value in metrics_flat.items()]
            )
            if metrics_table.empty:
                raise ValueError("evaluation_metrics.json contains no run or protocol evidence.")
            display(metrics_table)

            primary_test_metrics = dict(primary_result.get("test_metrics", {}))
            primary_scalar_metrics = {
                key: value
                for key, value in primary_test_metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            display(
                pd.DataFrame(
                    [
                        {
                            "model": primary_model_key,
                            "split": "untouched test",
                            **primary_scalar_metrics,
                        }
                    ]
                )
            )

            if per_class_metrics.empty:
                raise ValueError("per_class_metrics.csv exists but contains no measured rows.")
            display(per_class_metrics.sort_values(["model", "split", "label"]))
            display(thresholds.sort_values(["model", "label"]))
            """
        ),
        code(
            """
            labels = list(policy_config["model_labels"])
            for model_key in sorted(predictions["model"].astype(str).unique()):
                model_predictions = predictions.loc[predictions["model"].astype(str).eq(model_key)]
                matrix_table = pd.crosstab(
                    model_predictions["true_label"],
                    model_predictions["predicted_label"],
                    dropna=False,
                ).reindex(index=labels, columns=labels, fill_value=0)
                print(f"Untouched test confusion matrix: {model_key}")
                display(matrix_table)
                image_path = EVALUATION_DIR / f"confusion_matrix_{model_key}.png"
                if not image_path.is_file():
                    raise FileNotFoundError(f"Missing saved confusion plot: {image_path}")
                display(NotebookImage(filename=str(image_path), width=700))
            """
        ),
        code(
            """
            def canonical_metric_name(value: str) -> str:
                return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


            def assessed_target(
                metric: str,
                measured: float,
                target: float,
                relation: str,
                source: str,
                detail: str,
                criterion_type: str,
            ) -> dict[str, object]:
                passed = measured > target if relation == ">" else measured < target
                return {
                    "metric": metric,
                    "measured_value": measured,
                    "target": f"{relation} {target}",
                    "status": "PASS" if passed else "FAIL",
                    "criterion_type": criterion_type,
                    "source": source,
                    "detail": detail,
                }


            threshold_test_metrics = primary_result.get("threshold_test_metrics", {})
            restricted_labels = ["firearms", "explosives", "financial_promotion"]
            missing_restricted = [
                label
                for label in restricted_labels
                if not isinstance(threshold_test_metrics.get(label, {}).get("recall"), (int, float))
            ]
            if missing_restricted:
                raise ValueError(
                    f"Primary-model threshold test recall is missing for: {missing_restricted}"
                )
            restricted_recalls = {
                label: float(threshold_test_metrics[label]["recall"])
                for label in restricted_labels
            }
            primary_threshold_rows = thresholds.loc[
                thresholds["model"].astype(str).eq(primary_model_key)
            ]
            calibration_floor_values = sorted(
                pd.to_numeric(
                    primary_threshold_rows["target_validation_recall"], errors="raise"
                ).dropna().unique()
            )
            if len(calibration_floor_values) != 1:
                raise ValueError(
                    "Expected one validation recall floor for the primary model, found "
                    f"{calibration_floor_values}."
                )
            calibration_recall_floor = float(calibration_floor_values[0])
            argmax_restricted_recalls = {
                label: float(primary_test_metrics["per_class"][label]["recall"])
                for label in restricted_labels
            }
            display(
                pd.DataFrame(
                    [
                        {
                            "interpretation": "multiclass argmax restricted recall",
                            "mean": float(np.mean(list(argmax_restricted_recalls.values()))),
                            "minimum": min(argmax_restricted_recalls.values()),
                            "per_class": json.dumps(argmax_restricted_recalls, sort_keys=True),
                            "target_use": "descriptive aggregate",
                        },
                        {
                            "interpretation": "validation-threshold operating recall",
                            "mean": float(np.mean(list(restricted_recalls.values()))),
                            "minimum": min(restricted_recalls.values()),
                            "per_class": json.dumps(restricted_recalls, sort_keys=True),
                            "threshold_selection_floor": f">= {calibration_recall_floor:.2f} on validation",
                            "proposal_acceptance": "minimum test class compared with > 0.95",
                        },
                    ]
                )
            )

            safe_class_metrics = primary_test_metrics.get("per_class", {}).get("safe", {})
            missing_safe_metrics = [
                name
                for name in ("precision", "recall")
                if not isinstance(safe_class_metrics.get(name), (int, float))
            ]
            if missing_safe_metrics:
                raise ValueError(
                    f"Primary-model Safe-class metrics are missing: {missing_safe_metrics}."
                )
            safe_class_precision = float(safe_class_metrics["precision"])
            safe_argmax_false_flag_rate = float(1.0 - safe_class_metrics["recall"])
            safe_predictions = primary_predictions.loc[
                primary_predictions["true_label"].astype(str).eq("safe")
            ]
            if safe_predictions.empty:
                raise ValueError("Primary-model test predictions contain no Safe examples.")

            def any_threshold_flag(value: str) -> bool:
                payload = json.loads(str(value))
                if not isinstance(payload, dict):
                    raise ValueError("threshold_flags must contain a JSON object.")
                return any(bool(payload.get(label, False)) for label in restricted_labels)


            safe_flags = safe_predictions["threshold_flags"].map(any_threshold_flag)
            safe_false_positives = int(safe_flags.sum())
            safe_threshold_crossing_rate = float(safe_false_positives / len(safe_predictions))
            display(
                pd.DataFrame(
                    [
                        {
                            "measure": "Safe-class precision",
                            "value": safe_class_precision,
                            "definition": "true Safe among argmax predictions labeled Safe",
                            "role": "proposal metric",
                        },
                        {
                            "measure": "Safe-ad argmax false-flag rate",
                            "value": safe_argmax_false_flag_rate,
                            "definition": "1 - Safe recall; true Safe images assigned any non-Safe argmax label",
                            "role": "descriptive diagnostic",
                        },
                        {
                            "measure": "Safe-ad restricted-threshold crossing rate",
                            "value": safe_threshold_crossing_rate,
                            "definition": "true Safe images crossing at least one saved restricted threshold",
                            "role": "separate project operational guardrail",
                        },
                    ]
                )
            )
            acceptance = pd.DataFrame(
                [
                    assessed_target(
                        "primary test macro F1",
                        float(primary_test_metrics["macro_f1"]),
                        0.93,
                        ">",
                        "evaluation_metrics.json",
                        primary_model_key,
                        "proposal acceptance target",
                    ),
                    assessed_target(
                        "Safe-class precision",
                        safe_class_precision,
                        0.98,
                        ">",
                        "evaluation_metrics.json",
                        "multiclass Safe precision on the untouched test split",
                        "proposal acceptance target",
                    ),
                    assessed_target(
                        "minimum restricted-class threshold recall",
                        min(restricted_recalls.values()),
                        0.95,
                        ">",
                        "evaluation_metrics.json",
                        (
                            f"validation selection floor >= {calibration_recall_floor:.2f}; "
                            f"test recalls {json.dumps(restricted_recalls, sort_keys=True)}"
                        ),
                        "proposal acceptance target",
                    ),
                    assessed_target(
                        "Safe-ad restricted-threshold crossing rate",
                        safe_threshold_crossing_rate,
                        0.02,
                        "<",
                        "test_predictions.csv",
                        f"{safe_false_positives}/{len(safe_predictions)} Safe test images crossed a restricted threshold",
                        "separate project operational guardrail",
                    ),
                ]
            )
            display(acceptance)
            print(
                "All measured criteria use the predeclared ViT and untouched test artifacts. "
                "The >=0.90 validation recall floor is a threshold-selection rule, not the "
                "proposal's >0.95 per-class acceptance target."
            )
            """
        ),
        md(
            """
            ## 11. Policy fusion and decision audit

            `PolicyEngine` combines classifier scores with optional OCR and detector
            evidence through noisy-OR fusion. The following are deterministic unit
            checks of configured behavior, not estimates of predictive performance.
            Block decisions still require the corresponding classifier score to cross
            its validated threshold. Financial promotion remains human review.
            """
        ),
        code(
            """
            policy_engine = PolicyEngine(policy_config)
            unit_cases = [
                {
                    "case": "clear_safe_contract",
                    "scores": {"safe": 0.92, "firearms": 0.02, "explosives": 0.02, "financial_promotion": 0.04},
                    "ocr": "",
                    "detections": [],
                    "expected": "APPROVE",
                },
                {
                    "case": "validated_firearms_threshold_contract",
                    "scores": {"safe": 0.05, "firearms": 0.80, "explosives": 0.08, "financial_promotion": 0.07},
                    "ocr": "",
                    "detections": [],
                    "expected": "BLOCK",
                },
                {
                    "case": "financial_review_contract",
                    "scores": {"safe": 0.28, "firearms": 0.04, "explosives": 0.03, "financial_promotion": 0.45},
                    "ocr": "crypto investment",
                    "detections": [],
                    "expected": "REVIEW",
                },
            ]
            unit_rows = []
            for case in unit_cases:
                decision = policy_engine.decide(
                    case["scores"],
                    ocr_text=case["ocr"],
                    detections=case["detections"],
                    model_version="unit-contract-only",
                )
                assert decision.verdict == case["expected"], (case["case"], decision.verdict)
                unit_rows.append(
                    {
                        "case": case["case"],
                        "expected": case["expected"],
                        "observed": decision.verdict,
                        "reason": "; ".join(decision.review_reasons),
                    }
                )
            display(pd.DataFrame(unit_rows))
            """
        ),
        code(
            """
            if predictions.empty:
                raise ValueError("test_predictions.csv exists but contains no held-out predictions.")

            audit_tokens = ("path", "true", "label", "pred", "verdict", "score", "prob", "fused", "reason")
            audit_columns = [
                column for column in predictions.columns
                if any(token in column.casefold() for token in audit_tokens)
            ]
            display(predictions[audit_columns[:16]].head(20))

            verdict_columns = [column for column in predictions.columns if "verdict" in column.casefold()]
            if verdict_columns:
                display(
                    predictions[verdict_columns[0]]
                    .value_counts(dropna=False)
                    .rename_axis("verdict")
                    .reset_index(name="held_out_assets")
                )
            else:
                print(
                    "test_predictions.csv stores classifier and threshold evidence, not detector/OCR-fused "
                    "runtime verdicts. Policy behavior is covered by the unit contract above."
                )
            """
        ),
        md(
            """
            ## 12. Latency and resource evidence

            Latency must name the host, device, warm-up, sample count, and timing scope.
            The saved benchmark covers warm CPU batch-1 preprocessing, frozen backbone,
            and logistic head on decoded in-memory images. It excludes file decoding,
            OCR, object detection, network transfer, and application rendering.
            Therefore, it is a model-path benchmark and cannot establish the proposal's
            end-to-end 50 ms target. The saved run also did not measure concurrent-load
            throughput or MPS resource use.
            """
        ),
        code(
            """
            required_benchmark_columns = {
                "model", "scope", "device", "batch_size", "warmup_runs",
                "measured_runs", "median_ms", "p95_ms",
            }
            missing_benchmark_columns = sorted(required_benchmark_columns - set(benchmark.columns))
            if missing_benchmark_columns:
                raise ValueError(
                    f"benchmark_cpu_batch1.csv is missing columns: {missing_benchmark_columns}"
                )
            display(benchmark)

            primary_benchmark = benchmark.loc[
                benchmark["model"].astype(str).eq(primary_model_key)
            ]
            if len(primary_benchmark) != 1:
                raise ValueError(
                    f"Expected one CPU benchmark row for {primary_model_key}, found {len(primary_benchmark)}."
                )
            primary_p95_ms = float(primary_benchmark.iloc[0]["p95_ms"])
            display(
                pd.DataFrame(
                    [
                        {
                            "model": primary_model_key,
                            "recorded_scope": primary_benchmark.iloc[0]["scope"],
                            "p95_ms": primary_p95_ms,
                            "50_ms_model_path_diagnostic": (
                                "UNDER 50 ms" if primary_p95_ms < 50.0 else "OVER 50 ms"
                            ),
                            "proposal_end_to_end_target": "NOT ASSESSED",
                            "proposal_concurrent_throughput_target": "NOT ASSESSED",
                            "proposal_mps_resource_use_target": "NOT ASSESSED",
                        }
                    ]
                )
            )
            """
        ),
        md(
            """
            ## 13. Failure cases and limitations

            Failure analysis must remain visible even when aggregate metrics look
            strong. False approvals and false blocks carry different harms. The tables
            below are derived mechanically from all primary-model rows in the saved
            test-prediction artifact. They are not a hand-picked gallery.
            """
        ),
        code(
            """
            if failure_cases.empty:
                print("No primary-model multiclass misclassifications were recorded in test_predictions.csv.")
            else:
                display(failure_cases.head(25))

            def threshold_failure_type(row: pd.Series) -> str | None:
                flags = json.loads(str(row["threshold_flags"]))
                true_label = str(row["true_label"])
                if true_label == "safe" and any(bool(flags.get(label, False)) for label in restricted_labels):
                    return "safe image crossed at least one restricted threshold"
                if true_label in restricted_labels and not bool(flags.get(true_label, False)):
                    return "restricted image missed its own class threshold"
                return None


            threshold_failure_labels = primary_predictions.apply(threshold_failure_type, axis=1)
            threshold_failures = primary_predictions.loc[threshold_failure_labels.notna()].copy()
            threshold_failures["failure_type"] = threshold_failure_labels.dropna()
            print("Primary-model threshold failures:")
            display(threshold_failures.head(25))

            visual_failures = pd.concat([failure_cases, threshold_failures], ignore_index=True)
            visual_failures = visual_failures.drop_duplicates(subset=["sample_id"])
            if not visual_failures.empty:
                path_candidates = [
                    column for column in visual_failures.columns
                    if canonical_metric_name(column) in {"path", "image_path", "local_path", "file_path"}
                ]
                if path_candidates:
                    shown = 0
                    for raw_path in visual_failures[path_candidates[0]].dropna().astype(str):
                        candidate = Path(raw_path)
                        if not candidate.is_absolute():
                            candidate = PROJECT_ROOT / candidate
                        if candidate.is_file():
                            print(candidate.name)
                            display(NotebookImage(filename=str(candidate), width=420))
                            shown += 1
                        if shown >= 6:
                            break
            """
        ),
        md(
            """
            ## 13A. Separately labeled external Wikimedia diagnostic

            data/wikimedia_external_manifest.csv is not the formal training registry
            and is not part of the untouched model test set. When the external
            diagnostic outputs exist, this cell reports them as a small, manually
            reviewed spot check with historical and keyword-search selection bias.
            """
        ),
        code(
            """
            external_csv = EVALUATION_DIR / "external_spot_check.csv"
            external_json = EVALUATION_DIR / "external_spot_check.json"
            if external_csv.is_file() and external_json.is_file():
                external_rows = pd.read_csv(external_csv)
                external_summary = json.loads(external_json.read_text(encoding="utf-8"))
                print(external_summary.get("scope", "External diagnostic scope not recorded."))
                display(
                    pd.DataFrame(
                        [
                            {"diagnostic_path": key, "value": value}
                            for key, value in flatten_json(external_summary).items()
                        ]
                    )
                )
                display(external_rows.head(25))
            elif external_csv.exists() or external_json.exists():
                raise FileNotFoundError(
                    "External diagnostic is incomplete. Expected both "
                    "outputs/evaluation/external_spot_check.csv and "
                    "outputs/evaluation/external_spot_check.json."
                )
            else:
                print("External diagnostic not run; no Wikimedia result is included.")
            """
        ),
        md(
            """
            ## 14. Optional single-image runtime smoke test

            Set `AD_SAFETY_RUN_SMOKE=1` before starting Jupyter to exercise the saved
            classifier through `ModerationEngine`. Detector, OCR, and explanation paths
            remain separately opt-in because they add dependencies and latency. This
            smoke test does not replace the held-out evaluation artifacts.
            """
        ),
        code(
            """
            RUN_SMOKE = os.getenv("AD_SAFETY_RUN_SMOKE", "0") == "1"
            if RUN_SMOKE:
                from ad_safety.inference import ModerationEngine

                test_rows = registry.loc[registry["split"].eq("test")]
                if test_rows.empty:
                    raise RuntimeError("Runtime smoke test requires at least one registered test image.")
                sample_path = PROJECT_ROOT / str(test_rows.iloc[0]["local_path"])
                enable_detector = os.getenv("AD_SAFETY_ENABLE_DETECTOR", "0") == "1"
                enable_ocr = os.getenv("AD_SAFETY_ENABLE_OCR", "0") == "1"
                explain = os.getenv("AD_SAFETY_ENABLE_EXPLANATION", "0") == "1"

                engine = ModerationEngine(enable_detector=enable_detector)
                smoke = engine.analyze(
                    sample_path,
                    run_detector=enable_detector,
                    run_ocr=enable_ocr,
                    explain=explain,
                )
                display(pd.DataFrame([smoke.audit_record()["decision"]]))
                display(smoke.annotated_image)
                if smoke.heatmap_image is not None:
                    display(smoke.heatmap_image)
            else:
                print("Smoke test skipped. Set AD_SAFETY_RUN_SMOKE=1 to enable it.")
            """
        ),
        md(
            """
            ## 15. Run the local application

            The classifier artifact `models/vit_policy_head.joblib` must exist before
            either service starts. Detector and OCR are optional supporting paths.
            From the repository root, run one service at a time in a terminal:

            ```bash
            PYTHONPATH=src streamlit run app.py
            ```

            ```bash
            PYTHONPATH=src uvicorn api:app --host 127.0.0.1 --port 8000
            ```

            This notebook does not launch a persistent server. The dashboard is for
            visual demonstration; the API supports programmatic integration and audit
            records.
            """
        ),
        code(
            """
            service_files = pd.DataFrame(
                [
                    {"service": "Streamlit", "entrypoint": "app.py", "exists": (PROJECT_ROOT / "app.py").is_file()},
                    {"service": "FastAPI", "entrypoint": "api.py", "exists": (PROJECT_ROOT / "api.py").is_file()},
                ]
            )
            display(service_files)
            missing_services = service_files.loc[~service_files["exists"], "entrypoint"].tolist()
            if missing_services:
                raise FileNotFoundError(f"Application entrypoints are missing: {missing_services}")
            """
        ),
        md(
            """
            ## 16. Evidence manifest and next steps

            I keep final claims bounded to the saved evaluation manifest. Immediate
            next steps are to complete visual label review, enlarge every split with
            modern ad creatives and hard negatives, add source-held-out evaluation,
            calibrate with more data, audit subgroup and domain behavior, test
            production-like concurrency, and profile MPS memory and utilization in a
            separately identified run. No pilot result should be promoted to a
            production, fairness, fraud-detection, or attack-resistance claim.
            """
        ),
        code(
            """
            manifest_flat = flatten_json(independent_validation)
            display(
                pd.DataFrame(
                    [{"manifest_path": key, "value": value} for key, value in manifest_flat.items()]
                )
            )
            display(output_checksums)

            artifact_hashes = []
            for relative in REQUIRED_EVALUATION_ARTIFACTS:
                path = PROJECT_ROOT / relative
                artifact_hashes.append(
                    {
                        "artifact": relative,
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
            display(pd.DataFrame(artifact_hashes))
            """
        ),
        md(
            """
            ## References

            - Dosovitskiy et al., *An Image is Worth 16x16 Words*, ICLR 2021:
              https://openreview.net/forum?id=YicbFdNTTy
            - He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016:
              https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html
            - Liu et al., *Grounding DINO*, ECCV 2024:
              https://arxiv.org/abs/2303.05499
            - Formal dataset provenance, source groups, revisions, hashes, and license
              evidence are recorded per image in `data/capstone_registry.csv`.
            - `data/wikimedia_external_manifest.csv` is an external diagnostic
              manifest, not a training or formal test registry.
            - Project scope originates in `Capstone Project Idea - Ad Safety.pdf`.
            """
        ),
    ]

    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
            "ad_safety": {
                "artifact_driven": True,
                "project_working_directory": "repository root",
                "builder": "scripts/build_notebook.py",
            },
        }
    )
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help="Fail before writing if evaluation and runtime artifacts are incomplete.",
    )
    args = parser.parse_args()

    if args.require_artifacts:
        validate_artifacts(PROJECT_ROOT)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), output)
    print(f"Notebook written: {output}")


if __name__ == "__main__":
    main()
