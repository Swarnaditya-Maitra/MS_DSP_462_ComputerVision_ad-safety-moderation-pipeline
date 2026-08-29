# Ad Safety Capstone Narration Script

Voice disclosure: This script uses offline macOS system-generated narration with the Aman voice.

## 01. Ad Safety Moderation: Pilot results and next steps

Presenter section: Swarnaditya Maitra

This video uses offline macOS system-generated narration with the Aman voice. I am presenting a bounded ad safety decision-support prototype. I turn one ad image into an APPROVE, REVIEW, or BLOCK recommendation while keeping the evidence visible. The formal result is strong, but I do not treat this pilot as approval for autonomous deployment.

## 02. Team and presentation plan

Presenter section: Vijay Agnihotri

I divide the story across four presenters. Swarnaditya covers the decision and results. Vijay covers data and failure analysis. Myetchae covers policy, architecture, and audit. Bickramjit covers method, literature, and workflow.

## 03. Ambiguous ads are hard to review consistently

Presenter section: Swarnaditya Maitra

I start from a queue problem. A creative can hide a restricted object, subtle text, or misleading context inside an ordinary image. Manual review is slow and inconsistent. I therefore use computer vision to prioritize risky images and show evidence, while a person remains the final decision maker.

## 04. Four labels are not a full ad policy

Presenter section: Myetchae Thu

My modeled boundary has four image labels: safe, firearms, explosives, and financial promotion. Financial content always routes to REVIEW because an image cannot prove a misleading claim or a legal violation. The prototype does not model explicit content, violence, alcohol, tobacco, geography, age restrictions, landing pages, advertiser identity, or the full policy rulebook.

## 05. Balanced counts, unbalanced sources

Presenter section: Vijay Agnihotri

I built a 288-image pilot dataset. Each class has 48 training images, 12 validation images, and 12 untouched test images. I split 215 source groups so the same campaign or exact source group stays in one partition. This blocks a simple duplicate leak, but balanced counts do not remove source bias.

## 06. The classes look different for reasons unrelated to policy

Presenter section: Vijay Agnihotri

The contact sheet exposes the main weakness. Safe ads are synthetic product creatives. Financial ads are synthetic templates. Both weapon classes come from one weapons source. A classifier can therefore learn photographic style instead of policy meaning. I call this source confounding, and it limits every formal score that follows.

## 07. How the system reaches a decision

Presenter section: Myetchae Thu

My pipeline keeps three evidence streams separate. A frozen vision transformer scores the whole image. Grounding DINO can localize weapon and explosive cues. Tesseract extracts visible words. A versioned policy layer then combines classifier, detector, and text evidence into APPROVE, REVIEW, or BLOCK. This design makes each reason inspectable instead of hiding the final action inside one score.

## 08. We kept the test set out of training and tuning

Presenter section: Bickramjit Basu

I froze both the V-I-T and ResNet-fifty backbones, cached their embeddings, and fit logistic heads only on training data. I selected conservative class thresholds on validation data. The recorded protocol evaluates the 48-image test set once after calibration. I also benchmarked warm batch-one C-P-U inference with one thread so the timing protocol stays reproducible.

## 09. What prior research contributed

Presenter section: Bickramjit Basu

I use published work to justify component choices, not to claim this pilot is production-ready. The V-I-T paper supports transferable image representations. Grounding DINO supports text-conditioned object localization. Tesseract supports visible-text extraction. Only this project's saved evaluation supports the numbers I report.

## 10. ViT wins accuracy, not speed

Presenter section: Swarnaditya Maitra

On the untouched 48-image test, my V-I-T reached macro F one of 0.97913 and accuracy of 0.97917. The frozen ResNet-fifty baseline reached macro F one of 0.87315. Safe precision 1.0 means the proposal's greater-than-0.98 Safe-class precision benchmark passed. The Safe-ad false-flag rate, defined as one minus Safe recall, was zero. Multiclass argmax restricted-class mean recall 0.97222 differs from the threshold-operating mean 0.94444 and minimum 0.91667. The proposal's per-class greater-than-0.95 target failed for firearms and explosives. The 0.90 validation-recall tuning floor in code was only for calibration, not that acceptance target. Warm C-P-U batch-one P ninety-five latency was 71.527 milliseconds for V-I-T and 40.064 for ResNet. The V-I-T classifier path exceeded the 50-millisecond target. Full end-to-end P ninety-five remains unassessed because this timing covers preprocessing and classification only. It excludes file decoding, O-C-R, object detection, network transfer, and app rendering.

## 11. One grenade was classified as a firearm

Presenter section: Vijay Agnihotri

The test had one classification error. An explosives image was predicted as firearms, so 47 of 48 labels were correct. In the saved demo, the fused policy score for firearms is 0.88998. Neither the calibrated firearms threshold nor the explosives threshold fires, so the policy returns REVIEW. The detector still localizes shotgun and grenade-like cues. This is a class mistake, not an automatic approval.

## 12. External images exposed weak generalization

Presenter section: Myetchae Thu

I then tested a manually reviewed 26-image Wikimedia diagnostic set. Classifier macro F one fell to 0.55489, and the detector's nonweapon false-positive rate reached 0.90. This historical shoe ad is labeled safe. The classifier predicts explosives at 0.32411, detector evidence raises the fused policy score to 0.45302, and the policy returns BLOCK. The detector boxes and heatmap concentrate on the shoe. This is direct evidence of domain shift and source confounding. The small diagnostic sample is not a second formal test set.

## 13. A reviewer can follow one case from upload to audit

Presenter section: Bickramjit Basu

The workspace has four steps. I upload one J-P-E-G, P-N-G, or WebP image, configure optional object context, O-C-R, and occlusion, review raw and fused scores with reasons and timing, then export a case-level audit record. The policy returns APPROVE, REVIEW, or BLOCK. I use four saved formal examples next.

## 14. Safe creative routes to APPROVE

Presenter section: Bickramjit Basu

For the safe formal example, the classifier score is 0.99735. O-C-R finds three tokens, the optional detector is off, and the policy returns APPROVE. I can trace the verdict to the saved input and audit record. This confirms the intended path for an ordinary product creative inside the pilot domain.

## 15. Firearm evidence routes to BLOCK

Presenter section: Bickramjit Basu

For the firearm example, the fused policy score is 0.99907. Grounding DINO returns three detections, the evidence overlay localizes weapon cues, and the policy returns BLOCK. The optional detector and heatmap take seconds, so their latency must not be confused with the 71.527-millisecond classifier benchmark.

## 16. Explosive evidence routes to BLOCK

Presenter section: Bickramjit Basu

For the explosives example, the fused policy score is 0.99981 and the detector returns three boxes. The overlay identifies grenade and bomb-like cues among the surrounding objects. The policy returns BLOCK. This is the clean in-domain success case, and it contrasts with the earlier grenade image that crossed into the firearms class.

## 17. Financial promotion routes to REVIEW

Presenter section: Bickramjit Basu

For the financial promotion example, the classifier score is 0.99921 and O-C-R extracts 17 tokens. The policy still returns REVIEW, not BLOCK, because this prototype cannot determine whether a financial claim is lawful or misleading from pixels alone. The text and heatmap give a reviewer useful context without replacing legal or policy judgment.

## 18. What must change before a pilot

Presenter section: Myetchae Thu

The largest risks are source-confounded training data, an uncalibrated detector outside the pilot domain, and incomplete policy coverage. The formal test has only 48 images. The external audit has only 26. Full end-to-end latency and concurrent throughput were not measured. M-P-S resource use was not measured, including C-P-U, G-P-U, and memory use. Before an M-V-P trial, I need diverse real ads, local detector calibration, out-of-distribution checks, audit logs, human escalation, and monitoring for drift and subgroup failures.

## 19. What we would do next: a 10-week plan

Presenter section: Swarnaditya Maitra

My recommendation is only a human-gated M-V-P, not autonomous enforcement. The recommended ten-week roadmap is a planning assumption, not a measured commitment. In weeks one through three, a data lead, policy reviewer, and two annotators build a campaign-grouped real-ad holdout. In weeks four through six, one M-L engineer and a reviewer recalibrate the system against the declared quality gates. In weeks seven through ten, product, M-L-Ops, and reviewers run a shadow pilot. The scale gate tests ten concurrent requests and measures end-to-end p ninety-five plus C-P-U, G-P-U, and memory before broader rollout.

## Source index

Public sources:

- https://support.google.com/adspolicy/answer/6008942
- https://huggingface.co/datasets/EthanGabis/ADautoGen-DS
- https://huggingface.co/datasets/rajshivanshuu/weapons_set1
- https://arxiv.org/abs/2010.11929
- https://arxiv.org/abs/2303.05499
- https://huggingface.co/IDEA-Research/grounding-dino-tiny
- https://github.com/tesseract-ocr/tesseract
- https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia

Saved local evidence:

- `outputs/evaluation/metrics.json`
- `outputs/evaluation/model_comparison.csv`
- `outputs/evaluation/latency.json`
- `outputs/evaluation/external_spot_check.json`
- `outputs/demo_cases/case_summary.json`
- `outputs/presentation/ad_safety_management_presentation.pptx`
