# Ad Safety Management Presentation Speaker Script

Target running time: approximately 11:55 at a natural pace.

Use this as a rehearsal guide, not a passage to memorize. Keep the wording conversational, pause on the charts, and let the presenter handoffs sound natural.

## Slide 1: Ad Safety Moderation: Pilot Results and Next Steps

Presenter: Swarnaditya Maitra

Target time: 50 seconds

Good afternoon. We built Ad Safety Studio as a decision-support tool for a reviewer, not as an automatic policy judge. On our formal 48-image test, the model reached 0.979 macro F1, where 1 is best and every class counts equally. On a small external diagnostic set, that score fell to 0.555. That gap is the main story today. The local demo works, records the evidence behind each decision, and can help sort a review queue. It is not ready to make enforcement decisions on its own. I will return to that recommendation at the end. First, Vijay will introduce the team.

## Slide 2: Team and Presentation Plan

Presenter: Vijay Agnihotri

Target time: 25 seconds

We divided the presentation by responsibility, while keeping one shared story. I am Vijay, covering the data and error analysis. Swarnaditya covers the problem, formal results, and recommendation. Myetchae covers policy, architecture, the external check, and risk. Bickramjit covers method, prior research, and the working app. Swarnaditya will now frame the review problem.

## Slide 3: Ambiguous Ads Are Hard to Review Consistently

Presenter: Swarnaditya Maitra

Target time: 40 seconds

Ad review is difficult because one creative can mix several kinds of evidence. A weapon may be obvious, important text may be small, and context can change what an image means. Our goal was not to replace a policy reviewer. We wanted to help a queue move faster, apply the same first-pass process to every image, and send uncertain cases to a person. Myetchae will now set the boundary around what the prototype can actually decide.

## Slide 4: Four Labels Are Not a Full Ad Policy

Presenter: Myetchae Thu

Target time: 45 seconds

We deliberately kept the scope narrow: safe, firearms, explosives, and visible financial-promotion cues. The financial label does not mean fraud or illegality. It means the image appears promotional, so our rules always send it to review. For the other labels, a readable policy file maps evidence to approve, review, or block. We do not cover landing pages, advertiser identity, geography, age rules, alcohol, tobacco, or the rest of the ad-policy surface. This is a triage prototype, not a complete policy engine. Vijay will show the data behind it.

## Slide 5: Balanced Counts, Unbalanced Sources

Presenter: Vijay Agnihotri

Target time: 45 seconds

The pilot contains 288 JPEG images. Every class has 48 training images, 12 validation images, and 12 final test images. We also kept each of the 215 source groups inside one split and checked that file hashes do not overlap. That reduces leakage, meaning the same campaign or exact image cannot appear on both sides of the evaluation. The weakness is source diversity. Safe and financial images are synthetic, while both weapon classes come from one weapons dataset. The counts are balanced, but the visual sources are not. The next slide makes that problem visible.

## Slide 6: The Classes Look Different for Reasons Unrelated to Policy

Presenter: Vijay Agnihotri

Target time: 40 seconds

The contact sheet shows the shortcut risk more clearly than a metric can. Safe images look like generated product scenes. Weapon images are often isolated objects. Financial images repeat synthetic layouts. Even with a clean group split, a classifier can learn the look of the source instead of the policy concept itself. That does not make the formal result useless, but it limits the claim to this pilot domain. Myetchae will now show how the system keeps different kinds of evidence separate.

## Slide 7: How the System Reaches a Decision

Presenter: Myetchae Thu

Target time: 50 seconds

The system runs locally and has three parts. PyTorch and timm run a frozen vision transformer, while Grounding DINO and Tesseract add optional object and text evidence. FastAPI coordinates those services, and Streamlit gives the reviewer one workspace for the image, scores, boxes, text, and timing. A versioned YAML policy then returns approve, review, or block with reason codes and a JSON audit record. Keeping the signals separate lets a reviewer inspect what each component contributed. It does not prove causation, but it is clearer than one unexplained score. Bickramjit will explain how we trained and tested the models.

## Slide 8: We Kept the Test Set Out of Training and Tuning

Presenter: Bickramjit Basu

Target time: 50 seconds

Because the dataset is small, we froze the ViT and ResNet-50 backbones instead of fine-tuning millions of weights. We then trained small logistic classifiers on training embeddings. We chose class thresholds only with the validation split. The 48-image test split was not used to train, choose a model, or tune a threshold, and we ran the final evaluation after those choices were locked. The chart shows the validation-only explosives threshold, about 0.173, with recall and precision both at 1.0 on that small validation sample. I will stay on for one slide to explain how prior research shaped the design.

## Slide 9: What Prior Research Contributed

Presenter: Bickramjit Basu

Target time: 40 seconds

Prior work helped us choose credible building blocks. The Vision Transformer paper supports transferable whole-image features. Grounding DINO supports text-guided object localization, and Tesseract provides visible-text extraction. We did not find an outside benchmark that was directly comparable to our four-label policy task, so the proposal targets are the benchmarks we report against. The papers explain why the components are reasonable choices. Only our saved evaluation supports the numbers in this presentation. Swarnaditya will now take us through those results.

## Slide 10: ViT Wins Accuracy, Not Speed

Presenter: Swarnaditya Maitra

Target time: 70 seconds

On the untouched 48-image test, ViT reached 0.97913 macro F1, compared with 0.87315 for the ResNet-50 baseline. Safe precision was 1.0, so it passed the proposal target above 0.98, and none of the 12 safe test images crossed into a restricted class. The main caveat is recall. Under the operating thresholds, restricted-class recall averaged 0.94444, with a low value of 0.91667. Firearms and explosives were both below the proposal target of greater than 0.95 for every restricted class. The 0.97222 number uses simple highest-score classification, so it is a different metric. Speed also favors ResNet. ViT classifier-path p95 was 71.527 milliseconds, above the 50-millisecond target, while ResNet was 40.064. This benchmark excludes decoding, OCR, detection, network time, and rendering, so full end-to-end p95 is still unknown. Vijay will look at the one formal error.

## Slide 11: One Grenade Was Classified as a Firearm

Presenter: Vijay Agnihotri

Target time: 45 seconds

The formal test had one wrong class label. A grenade image labeled explosives was predicted as firearms, while the other 47 labels were correct. More importantly, its firearm and explosive scores did not cross either class-specific block threshold. In the saved policy run, the case went to REVIEW rather than an automatic block or approval. The optional detector found restricted-object cues, but the class separation was still fragile. This is exactly the kind of edge case that makes the human review step necessary. Myetchae will now show what happened outside the pilot sources.

## Slide 12: External Images Exposed Weak Generalization

Presenter: Myetchae Thu

Target time: 60 seconds

The external check used 26 manually reviewed Wikimedia images for diagnosis, not as a second formal test set. Macro F1 fell to 0.55489. At the current detector threshold, 90 percent of the nonweapon images in this small sample received a false detector signal. The examples show the pattern. A historical safe ad was blocked, one explosives image was classified as safe, and a financial example received a false box even though its policy route remained review. These images have different languages, layouts, ages, and photographic context from our training sources. The sample is small and selected, so this is not a population estimate. It is still enough to show that the formal score does not generalize reliably. Bickramjit will show what the app already does well.

## Slide 13: A Reviewer Can Follow One Case from Upload to Audit

Presenter: Bickramjit Basu

Target time: 50 seconds

The working app keeps one case in one place. A reviewer uploads an image, turns optional evidence on or off, reviews the decision and the raw versus fused scores, and exports an audit record with hashes, model versions, options, and thresholds. This screenshot is a browser-validated firearm case that returned BLOCK. The 2,448-millisecond total shown here is one app run with optional stages. It is not the same measurement as the classifier-only p95 on slide 10. The session charts summarize cases from the current session only. They are not production monitoring or population evidence. Myetchae will connect those limits to the controls we still need.

## Slide 14: What Must Change Before a Pilot

Presenter: Myetchae Thu

Target time: 50 seconds

Before a real pilot, we need controls in three areas. For data, we need diverse real campaigns and independent annotation, not source-linked synthetic shortcuts. For the detector, we need local box labels and recalibration, and we should not let the detector block by itself. For operations, we still need full end-to-end p95, concurrent throughput, and CPU, Apple GPU, and memory measurements. Today's controls reduce the chance of harm, but they do not remove the uncertainty. Our decision is no-go for autonomous use and go only for further human-reviewed validation. Swarnaditya will close with that path.

## Slide 15: What We Would Do Next: A 10-Week Plan

Presenter: Swarnaditya Maitra

Target time: 55 seconds

The 10-week roadmap is our planning recommendation, not a measured delivery promise. In weeks one through three, we would collect at least 1,000 real ads and lock an independent campaign-grouped holdout. In weeks four through six, we would recalibrate and require greater than 0.95 recall for every restricted class, with fewer than 2 percent safe crossings. In weeks seven through ten, we would run a shadow, human-reviewed pilot and test 10 concurrent requests with full end-to-end latency and resource tracking. If any gate fails, we stop and fix it before scaling. The project is complete as a reproducible demo, but the evidence tells us that the responsible next step is stronger validation, not autonomous enforcement. Thank you. We are ready for questions.

## Delivery notes

- The 15-slide deck changes presenter 11 times. Rehearse each handoff so the switches feel deliberate.
- Define macro F1 once on slide 1, then use the shorter term afterward.
- On slide 10, separate highest-score recall from recall at the operating thresholds.
- On slide 12, call the Wikimedia set a small selected diagnostic, not a population estimate.
- On slide 13, separate the single app-case timing from the repeated classifier-only benchmark.
- If time is short, trim examples and transitions before trimming the caveats.
