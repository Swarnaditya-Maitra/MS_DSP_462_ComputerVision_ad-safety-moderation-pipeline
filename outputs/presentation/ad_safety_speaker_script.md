# Ad Safety Presentation Speaker Script

Target running time: approximately 12:20 at a natural pace.

Use this as a rehearsal guide, not a passage to memorize. Keep the wording conversational, pause on the charts, and let the presenter handoffs sound natural.

## Slide 1: Ad Safety Moderation: What the Pilot Showed

Presenter: Swarnaditya Maitra

Target time: 45 seconds

Good afternoon. We built Ad Safety Studio to help a reviewer sort image-based ads, not to replace a policy decision. The model scored 97.9 percent macro F1 on our 48-image held-out test, but only 55.5 percent on a small external check. That gap is the main result. The app works, keeps the evidence beside the decision, and can support a first-pass review. It is not ready for autonomous enforcement. I will introduce the team first, then frame the problem we set out to solve.

## Slide 2: Meet the Team

Presenter: Swarnaditya Maitra

Target time: 35 seconds

There are four of us, and each person owns one continuous part of the story. I am Swarnaditya, covering the opening, problem, and scope. Vijay covers the data, system, and evaluation method. Myetchae covers prior work, results, and the external check. Bickramjit leads the working app, controls, and next steps. During the app demo, I will share the screen and all four of us will explain the part we worked on. Let me start with the review problem.

## Slide 3: Ambiguous Ads Are Hard to Review Consistently

Presenter: Swarnaditya Maitra

Target time: 40 seconds

Ad review is difficult because one creative can mix several kinds of evidence. A weapon may be obvious, important text may be small, and context can change what an image means. Our goal was not to replace a policy reviewer. We wanted to help a queue move faster, apply the same first-pass process to every image, and send uncertain cases to a person. That first-pass goal also sets the boundary for what the prototype can and cannot decide.

## Slide 4: Four Labels Are Not a Full Ad Policy

Presenter: Swarnaditya Maitra

Target time: 45 seconds

We deliberately kept the scope narrow: safe, firearms, explosives, and visible financial-promotion cues. The financial label does not mean fraud or illegality. It means the image appears promotional, so our rules always send it to review. For the other labels, a readable policy file maps evidence to approve, review, or block. We do not cover landing pages, advertiser identity, geography, age rules, alcohol, tobacco, or the rest of the ad-policy surface. This is a triage prototype, not a complete policy engine. That is the boundary of the prototype. Vijay will now show the data and how we protected the evaluation.

## Slide 5: Balanced Counts, Unbalanced Sources

Presenter: Vijay Agnihotri

Target time: 45 seconds

The pilot contains 288 JPEG images. Every class has 48 training images, 12 validation images, and 12 final test images. We also kept each of the 215 source groups inside one split and checked that file hashes do not overlap. That reduces leakage, meaning the same campaign or exact image cannot appear on both sides of the evaluation. The weakness is source diversity. Safe and financial images are synthetic, while both weapon classes come from one weapons dataset. The counts are balanced, but the visual sources are not. The next slide makes that problem visible.

## Slide 6: The Model Can Learn Source Style, Not Policy Meaning

Presenter: Vijay Agnihotri

Target time: 40 seconds

The contact sheet shows the shortcut risk more clearly than a metric can. Safe images look like generated product scenes. Weapon images are often isolated objects. Financial images repeat synthetic layouts. Even with a clean group split, a classifier can learn the look of the source instead of the policy concept itself. That does not make the formal result useless, but it limits the claim to this pilot domain. That risk shaped both our model design and how cautiously we read the results.

## Slide 7: How the System Reaches a Decision

Presenter: Vijay Agnihotri

Target time: 50 seconds

The system runs locally and has three parts. PyTorch and timm run a frozen vision transformer, while Grounding DINO and Tesseract add optional object and text evidence. FastAPI coordinates those services, and Streamlit gives the reviewer one workspace for the image, scores, boxes, text, and timing. A versioned YAML policy then returns approve, review, or block with reason codes and a JSON audit record. Keeping the signals separate lets a reviewer inspect what each component contributed. With those components separated, the next step was to train and test the classifier without letting the final test set influence our choices.

## Slide 8: We Kept the Test Set Out of Training and Tuning

Presenter: Vijay Agnihotri

Target time: 50 seconds

Because the dataset is small, we froze the ViT and ResNet-50 backbones instead of fine-tuning millions of weights. We then trained small logistic classifiers on training embeddings. We chose class thresholds only with the validation split. The 48-image test split was not used to train, choose a model, or tune a threshold, and we ran the final evaluation after those choices were locked. The chart shows the validation-only explosives threshold, about 0.173, with recall and precision both at 100 percent on that small validation sample. That gives us a controlled evaluation. Myetchae will now connect the design to prior work and walk through the results.

## Slide 9: What Prior Research Contributed

Presenter: Myetchae Thu

Target time: 40 seconds

Prior work helped us choose credible building blocks. The Vision Transformer paper supports transferable whole-image features. Grounding DINO supports text-guided object localization, and Tesseract provides visible-text extraction. We did not find an outside benchmark directly comparable to our four-label policy task. Those papers guided our choices; the performance numbers come from our own evaluation. With that distinction clear, we can look at what the evaluation actually found.

## Slide 10: Better Accuracy Came with Recall and Speed Trade-offs

Presenter: Myetchae Thu

Target time: 65 seconds

On the untouched 48-image test, ViT reached 97.9 percent macro F1, compared with 87.3 percent for the ResNet-50 baseline. Safe precision was 100 percent, so it met our target, and none of the 12 safe test images crossed into a restricted class. The caveat is recall. At the operating thresholds, firearms and explosives each reached 91.7 percent recall, below our goal of more than 95 percent for every restricted class. Speed also favors ResNet. ViT classifier-path p95 was 71.5 milliseconds, above the 50-millisecond budget, while ResNet was 40.1. This benchmark excludes decoding, OCR, detection, network time, and rendering, so full end-to-end p95 is still unknown. The next slide shows the one classification error behind those results.

## Slide 11: One Error Is Why a Reviewer Stays in Control

Presenter: Myetchae Thu

Target time: 45 seconds

The formal test had one wrong class label. A grenade image labeled explosives was predicted as firearms, while the other 47 labels were correct. Its firearm and explosive scores did not cross either class-specific block threshold, so the saved policy result was REVIEW rather than an automatic block or approval. The optional detector found restricted-object cues, but the class separation was still fragile. The bigger question is whether the same system holds up when the images stop looking like our pilot sources.

## Slide 12: Outside the Pilot Data, Performance Fell Sharply

Presenter: Myetchae Thu

Target time: 60 seconds

The external check used 26 manually reviewed Wikimedia images for diagnosis, not as a second formal test set. Macro F1 fell to 55.5 percent. At the current detector threshold, 90 percent of the nonweapon images in this small sample received a false detector signal. The examples show the pattern: a historical safe ad was blocked, one explosives image was classified as safe, and a financial example received a false box even though its policy route remained review. The sample is small and selected, so this is not a population estimate. It is still enough to show that the formal score does not generalize reliably. Bickramjit will now move to the working app and the controls needed before a pilot.

## Slide 13: From Upload to Audit in One Workspace

Presenter: Full team (Bickramjit Basu leads; Swarnaditya Maitra shares the screen)

Target time: 75 seconds

Bickramjit: The working app keeps one case in one place. Swarnaditya is sharing the screen and will run the example. Swarnaditya: I will upload the firearm image, keep the optional detector and occlusion view enabled, and start the analysis. Vijay: The results separate classifier scores from fused scores and show how long each stage took, so we can see what changed instead of relying on one final label. Myetchae: The policy turns that evidence into BLOCK, REVIEW, or APPROVE with reason codes, and the audit export records hashes, model versions, options, and thresholds. Bickramjit: The timing shown here is one app run, not the repeated classifier-only benchmark. The session charts summarize only the cases reviewed today. That brings us to the controls we still need.

## Slide 14: What Must Change Before a Pilot

Presenter: Bickramjit Basu

Target time: 50 seconds

Before a real pilot, we need controls in three areas. For data, we need diverse real campaigns and independent annotation, not source-linked synthetic shortcuts. For the detector, we need local box labels and recalibration, and we should not let the detector block by itself. For operations, we still need full end-to-end p95, concurrent throughput, and CPU, Apple GPU, and memory measurements. These controls reduce the chance of harm, but they do not remove the uncertainty. Our decision is no-go for autonomous use and go only for further human-reviewed validation. Those controls lead directly to the gated plan on the final slide.

## Slide 15: What We Would Do Next: A 10-Week Plan

Presenter: Bickramjit Basu

Target time: 55 seconds

This 10-week roadmap is our proposed next step. In weeks one through three, we would collect at least 1,000 real ads and lock an independent campaign-grouped holdout. In weeks four through six, we would recalibrate and require more than 95 percent recall for every restricted class, with fewer than 2 percent safe crossings. In weeks seven through ten, we would run a shadow, human-reviewed pilot and test 10 concurrent requests with full end-to-end latency and resource tracking. If any gate fails, we stop and fix it before scaling. The prototype is reproducible, but stronger validation must come before autonomous use. Thank you. All four of us are ready for questions.

## Delivery notes

- Speaker order: Swarnaditya slides 1-4, Vijay slides 5-8, Myetchae slides 9-12, and Bickramjit slides 13-15.
- Slide 13 is a team demo. Bickramjit leads, Swarnaditya shares the screen, and all four members explain their part.
- Define macro F1 once on slide 1, then use the shorter term afterward.
- On slide 10, separate highest-score recall from recall at the operating thresholds.
- On slide 12, call the Wikimedia set a small selected diagnostic, not a population estimate.
- On slide 13, separate the single app-case timing from the repeated classifier-only benchmark.
- If time is short, trim examples and transitions before trimming the caveats.
