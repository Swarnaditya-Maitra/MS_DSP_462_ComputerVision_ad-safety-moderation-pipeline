import crypto from "node:crypto";
import fs from "node:fs";
import fsp from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PROJECT = path.resolve(HERE, "..");
const OUT = path.join(PROJECT, "outputs", "presentation");
const BUILD = path.join(OUT, ".build");
const RENDERED = path.join(OUT, "rendered");
const PPTX_PATH = path.join(OUT, "ad_safety_management_presentation.pptx");
const SPEAKER_SCRIPT_PATH = path.join(OUT, "ad_safety_speaker_script.md");

function requireDirectory(environmentVariable) {
  const configuredPath = process.env[environmentVariable];
  if (!configuredPath) {
    throw new Error(`${environmentVariable} must point to an existing directory.`);
  }
  const resolvedPath = path.resolve(configuredPath);
  let stats;
  try {
    stats = fs.statSync(resolvedPath);
  } catch {
    throw new Error(`${environmentVariable} does not exist or cannot be read: ${resolvedPath}`);
  }
  if (!stats.isDirectory()) {
    throw new Error(`${environmentVariable} must point to a directory: ${resolvedPath}`);
  }
  return resolvedPath;
}

const RUNTIME_NODE_MODULES = requireDirectory("RUNTIME_NODE_MODULES");
const GRID_SOURCE = requireDirectory("CODEX_GRID_SOURCE");
const GRID_SOURCE_LABEL = "Codex Grid source supplied through CODEX_GRID_SOURCE";
const SELECTED_GRID_FILES = [
  "slide-08.mjs",
  "slide-18.mjs",
  "slide-19.mjs",
  "slide-20.mjs",
  "runtime.mjs",
  "content-tokens.json",
];
const SELECTED_TEMPLATE_IDS = [
  "codex-grid-layout-library#slide-08",
  "codex-grid-layout-library#slide-18",
  "codex-grid-layout-library#slide-19",
  "codex-grid-layout-library#slide-20",
];

const W = 1280;
const H = 720;
// Arial remains metrically stable across Artifact Tool, macOS PowerPoint, and
// LibreOffice. Helvetica Neue was substituted by LibreOffice in the independent
// render path, which made left-edge text appear inconsistent across artifacts.
const FONT = "Arial";
const SAFE_LEFT = 64;
const BLACK = "#000000";
const WHITE = "#FFFFFF";
const PANEL = "#EDEDED";
const LIGHT_BLUE = "#EAF5FB";
const RULE = "#B8BCC4";
const MUTED = "#5B6470";
const BLUE = "#3D8DFF";
const CYAN = "#6DCBF4";
const RED = "#D9485F";

const P = (...parts) => path.join(PROJECT, ...parts);
const ABS = (relativePath) => P(...relativePath.split("/"));

const ASSETS = {
  contact: ABS("outputs/evaluation/dataset_contact_sheet.jpg"),
  distribution: ABS("outputs/evaluation/dataset_distribution.png"),
  threshold: ABS("outputs/evaluation/threshold_calibration.png"),
  comparison: ABS("outputs/evaluation/model_comparison.png"),
  confusion: ABS("outputs/evaluation/confusion_matrix_vit.png"),
  formalFailure: ABS("data/capstone_dataset/test/explosives/explosives-a847a50b677d3214.jpg"),
  externalSafeFalse: ABS("outputs/evaluation/external_annotated/02_safe_train_eb77534129f9.jpg"),
  externalExplosiveMiss: ABS("outputs/evaluation/external_annotated/20_explosives_val_dd020909c102.jpg"),
  externalFinancial: ABS("outputs/evaluation/external_annotated/25_financial_promotion_test_128923d1c757.jpg"),
  appShell: ABS("outputs/app/app_shell.png"),
  appFirearmResult: ABS("outputs/app/app_firearm_result.png"),
};

const PRESENTERS = {
  swarnaditya: "Swarnaditya Maitra",
  vijay: "Vijay Agnihotri",
  myetchae: "Myetchae Thu",
  bickramjit: "Bickramjit Basu",
};

const SPEAKER_SCRIPTS = {
  1: {
    title: "Ad Safety Moderation: Pilot Results and Next Steps",
    presenter: PRESENTERS.swarnaditya,
    seconds: 50,
    text: "Good afternoon. We built Ad Safety Studio as a decision-support tool for a reviewer, not as an automatic policy judge. On our formal 48-image test, the model reached 0.979 macro F1, where 1 is best and every class counts equally. On a small external diagnostic set, that score fell to 0.555. That gap is the main story today. The local demo works, records the evidence behind each decision, and can help sort a review queue. It is not ready to make enforcement decisions on its own. I will return to that recommendation at the end. First, Vijay will introduce the team.",
  },
  2: {
    title: "Team and Presentation Plan",
    presenter: PRESENTERS.vijay,
    seconds: 25,
    text: "We divided the presentation by responsibility, while keeping one shared story. I am Vijay, covering the data and error analysis. Swarnaditya covers the problem, formal results, and recommendation. Myetchae covers policy, architecture, the external check, and risk. Bickramjit covers method, prior research, and the working app. Swarnaditya will now frame the review problem.",
  },
  3: {
    title: "Ambiguous Ads Are Hard to Review Consistently",
    presenter: PRESENTERS.swarnaditya,
    seconds: 40,
    text: "Ad review is difficult because one creative can mix several kinds of evidence. A weapon may be obvious, important text may be small, and context can change what an image means. Our goal was not to replace a policy reviewer. We wanted to help a queue move faster, apply the same first-pass process to every image, and send uncertain cases to a person. Myetchae will now set the boundary around what the prototype can actually decide.",
  },
  4: {
    title: "Four Labels Are Not a Full Ad Policy",
    presenter: PRESENTERS.myetchae,
    seconds: 45,
    text: "We deliberately kept the scope narrow: safe, firearms, explosives, and visible financial-promotion cues. The financial label does not mean fraud or illegality. It means the image appears promotional, so our rules always send it to review. For the other labels, a readable policy file maps evidence to approve, review, or block. We do not cover landing pages, advertiser identity, geography, age rules, alcohol, tobacco, or the rest of the ad-policy surface. This is a triage prototype, not a complete policy engine. Vijay will show the data behind it.",
  },
  5: {
    title: "Balanced Counts, Unbalanced Sources",
    presenter: PRESENTERS.vijay,
    seconds: 45,
    text: "The pilot contains 288 JPEG images. Every class has 48 training images, 12 validation images, and 12 final test images. We also kept each of the 215 source groups inside one split and checked that file hashes do not overlap. That reduces leakage, meaning the same campaign or exact image cannot appear on both sides of the evaluation. The weakness is source diversity. Safe and financial images are synthetic, while both weapon classes come from one weapons dataset. The counts are balanced, but the visual sources are not. The next slide makes that problem visible.",
  },
  6: {
    title: "The Classes Look Different for Reasons Unrelated to Policy",
    presenter: PRESENTERS.vijay,
    seconds: 40,
    text: "The contact sheet shows the shortcut risk more clearly than a metric can. Safe images look like generated product scenes. Weapon images are often isolated objects. Financial images repeat synthetic layouts. Even with a clean group split, a classifier can learn the look of the source instead of the policy concept itself. That does not make the formal result useless, but it limits the claim to this pilot domain. Myetchae will now show how the system keeps different kinds of evidence separate.",
  },
  7: {
    title: "How the System Reaches a Decision",
    presenter: PRESENTERS.myetchae,
    seconds: 50,
    text: "The system runs locally and has three parts. PyTorch and timm run a frozen vision transformer, while Grounding DINO and Tesseract add optional object and text evidence. FastAPI coordinates those services, and Streamlit gives the reviewer one workspace for the image, scores, boxes, text, and timing. A versioned YAML policy then returns approve, review, or block with reason codes and a JSON audit record. Keeping the signals separate lets a reviewer inspect what each component contributed. It does not prove causation, but it is clearer than one unexplained score. Bickramjit will explain how we trained and tested the models.",
  },
  8: {
    title: "We Kept the Test Set Out of Training and Tuning",
    presenter: PRESENTERS.bickramjit,
    seconds: 50,
    text: "Because the dataset is small, we froze the ViT and ResNet-50 backbones instead of fine-tuning millions of weights. We then trained small logistic classifiers on training embeddings. We chose class thresholds only with the validation split. The 48-image test split was not used to train, choose a model, or tune a threshold, and we ran the final evaluation after those choices were locked. The chart shows the validation-only explosives threshold, about 0.173, with recall and precision both at 1.0 on that small validation sample. I will stay on for one slide to explain how prior research shaped the design.",
  },
  9: {
    title: "What Prior Research Contributed",
    presenter: PRESENTERS.bickramjit,
    seconds: 40,
    text: "Prior work helped us choose credible building blocks. The Vision Transformer paper supports transferable whole-image features. Grounding DINO supports text-guided object localization, and Tesseract provides visible-text extraction. We did not find an outside benchmark that was directly comparable to our four-label policy task, so the proposal targets are the benchmarks we report against. The papers explain why the components are reasonable choices. Only our saved evaluation supports the numbers in this presentation. Swarnaditya will now take us through those results.",
  },
  10: {
    title: "ViT Wins Accuracy, Not Speed",
    presenter: PRESENTERS.swarnaditya,
    seconds: 70,
    text: "On the untouched 48-image test, ViT reached 0.97913 macro F1, compared with 0.87315 for the ResNet-50 baseline. Safe precision was 1.0, so it passed the proposal target above 0.98, and none of the 12 safe test images crossed into a restricted class. The main caveat is recall. Under the operating thresholds, restricted-class recall averaged 0.94444, with a low value of 0.91667. Firearms and explosives were both below the proposal target of greater than 0.95 for every restricted class. The 0.97222 number uses simple highest-score classification, so it is a different metric. Speed also favors ResNet. ViT classifier-path p95 was 71.527 milliseconds, above the 50-millisecond target, while ResNet was 40.064. This benchmark excludes decoding, OCR, detection, network time, and rendering, so full end-to-end p95 is still unknown. Vijay will look at the one formal error.",
  },
  11: {
    title: "One Grenade Was Classified as a Firearm",
    presenter: PRESENTERS.vijay,
    seconds: 45,
    text: "The formal test had one wrong class label. A grenade image labeled explosives was predicted as firearms, while the other 47 labels were correct. More importantly, its firearm and explosive scores did not cross either class-specific block threshold. In the saved policy run, the case went to REVIEW rather than an automatic block or approval. The optional detector found restricted-object cues, but the class separation was still fragile. This is exactly the kind of edge case that makes the human review step necessary. Myetchae will now show what happened outside the pilot sources.",
  },
  12: {
    title: "External Images Exposed Weak Generalization",
    presenter: PRESENTERS.myetchae,
    seconds: 60,
    text: "The external check used 26 manually reviewed Wikimedia images for diagnosis, not as a second formal test set. Macro F1 fell to 0.55489. At the current detector threshold, 90 percent of the nonweapon images in this small sample received a false detector signal. The examples show the pattern. A historical safe ad was blocked, one explosives image was classified as safe, and a financial example received a false box even though its policy route remained review. These images have different languages, layouts, ages, and photographic context from our training sources. The sample is small and selected, so this is not a population estimate. It is still enough to show that the formal score does not generalize reliably. Bickramjit will show what the app already does well.",
  },
  13: {
    title: "A Reviewer Can Follow One Case from Upload to Audit",
    presenter: PRESENTERS.bickramjit,
    seconds: 50,
    text: "The working app keeps one case in one place. A reviewer uploads an image, turns optional evidence on or off, reviews the decision and the raw versus fused scores, and exports an audit record with hashes, model versions, options, and thresholds. This screenshot is a browser-validated firearm case that returned BLOCK. The 2,448-millisecond total shown here is one app run with optional stages. It is not the same measurement as the classifier-only p95 on slide 10. The session charts summarize cases from the current session only. They are not production monitoring or population evidence. Myetchae will connect those limits to the controls we still need.",
  },
  14: {
    title: "What Must Change Before a Pilot",
    presenter: PRESENTERS.myetchae,
    seconds: 50,
    text: "Before a real pilot, we need controls in three areas. For data, we need diverse real campaigns and independent annotation, not source-linked synthetic shortcuts. For the detector, we need local box labels and recalibration, and we should not let the detector block by itself. For operations, we still need full end-to-end p95, concurrent throughput, and CPU, Apple GPU, and memory measurements. Today's controls reduce the chance of harm, but they do not remove the uncertainty. Our decision is no-go for autonomous use and go only for further human-reviewed validation. Swarnaditya will close with that path.",
  },
  15: {
    title: "What We Would Do Next: A 10-Week Plan",
    presenter: PRESENTERS.swarnaditya,
    seconds: 55,
    text: "The 10-week roadmap is our planning recommendation, not a measured delivery promise. In weeks one through three, we would collect at least 1,000 real ads and lock an independent campaign-grouped holdout. In weeks four through six, we would recalibrate and require greater than 0.95 recall for every restricted class, with fewer than 2 percent safe crossings. In weeks seven through ten, we would run a shadow, human-reviewed pilot and test 10 concurrent requests with full end-to-end latency and resource tracking. If any gate fails, we stop and fix it before scaling. The project is complete as a reproducible demo, but the evidence tells us that the responsible next step is stronger validation, not autonomous enforcement. Thank you. We are ready for questions.",
  },
};

const slideAudit = [];
const notesAudit = [];

function sha256(bytes) {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

async function fileSha256(filePath) {
  return sha256(await fsp.readFile(filePath));
}

async function ensureBuildRuntime() {
  await fsp.mkdir(BUILD, { recursive: true });
  await fsp.mkdir(RENDERED, { recursive: true });
  const nodeModulesLink = path.join(BUILD, "node_modules");
  if (!fs.existsSync(nodeModulesLink)) {
    await fsp.symlink(RUNTIME_NODE_MODULES, nodeModulesLink, "dir");
  }
  const gridSnapshot = path.join(BUILD, "codex-grid");
  await fsp.mkdir(gridSnapshot, { recursive: true });
  for (const filename of SELECTED_GRID_FILES) {
    await fsp.copyFile(path.join(GRID_SOURCE, filename), path.join(gridSnapshot, filename));
  }

  const registry = JSON.parse(await fsp.readFile(path.join(GRID_SOURCE, "template-registry.json"), "utf8"));
  const templates = Array.isArray(registry) ? registry : registry.templates ?? registry.layouts ?? Object.values(registry);
  const tokens = JSON.parse(await fsp.readFile(path.join(GRID_SOURCE, "content-tokens.json"), "utf8"));
  const selection = {
    source: GRID_SOURCE_LABEL,
    selected_template_ids: SELECTED_TEMPLATE_IDS,
    registry_entries: templates.filter((entry) => SELECTED_TEMPLATE_IDS.includes(entry.templateId)),
    content_tokens: Object.fromEntries(["slide-08", "slide-18", "slide-19", "slide-20"].map((key) => [key, tokens[key]])),
    copied_module_sha256: Object.fromEntries(
      await Promise.all(
        SELECTED_GRID_FILES.map(async (filename) => [filename, await fileSha256(path.join(gridSnapshot, filename))]),
      ),
    ),
  };
  await fsp.writeFile(path.join(OUT, "grid_selection.json"), JSON.stringify(selection, null, 2));
  return gridSnapshot;
}

async function buildCalibrationZoom() {
  const runtimeRequire = createRequire(path.join(BUILD, "runtime-loader.cjs"));
  const sharp = runtimeRequire("sharp");
  const outputPath = path.join(BUILD, "threshold_calibration_vit_explosives.png");
  await sharp(ASSETS.threshold)
    .extract({ left: 900, top: 0, width: 750, height: 720 })
    .png()
    .toFile(outputPath);
  return outputPath;
}

function rich(text, fontSize = 21.33, { bold = false, color = BLACK } = {}) {
  return {
    runs: [
      {
        run: String(text),
        textStyle: {
          fontSize: `${fontSize}px`,
          typeface: FONT,
          color,
          ...(bold ? { bold: true } : {}),
        },
      },
    ],
    paragraphStyle: { lineSpacingPercent: 100000 },
  };
}

function titleToken(text, fontSize = 48) {
  return rich(text, fontSize, { bold: false, color: BLACK });
}

function splitBody(title, body, titleKey = "titleHere") {
  return {
    [titleKey]: rich(title, 24, { bold: true }),
    loremIpsumDolorSitAmetConsecteturAdipiscing: rich(body, 21.33),
  };
}

function metricIntro(topic, body) {
  return {
    topic: rich(topic, 21.33, { bold: true, color: BLUE }),
    loremIpsumDolorSitAmetConsecteturAdipiscing: rich(body, 21.33),
  };
}

function chartCard(title, body) {
  return splitBody(title, body, "titleGoesHere");
}

function mimeFor(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  if (ext === ".jpg" || ext === ".jpeg") return "image/jpeg";
  if (ext === ".webp") return "image/webp";
  return "image/png";
}

async function bytesFor(filePath) {
  if (!fs.existsSync(filePath)) throw new Error(`Missing presentation asset: ${filePath}`);
  return new Uint8Array(await fsp.readFile(filePath));
}

function addTextBox(slide, text, position, options = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    name: options.name,
    position,
    fill: options.fill ?? "none",
    line: options.line ?? { style: "solid", fill: "none", width: 0 },
  });
  box.text = String(text);
  box.text.style = {
    typeface: FONT,
    fontSize: options.fontSize ?? 21.33,
    color: options.color ?? BLACK,
    bold: options.bold ?? false,
    alignment: options.alignment ?? "left",
    verticalAlignment: options.verticalAlignment ?? "top",
  };
  return box;
}

function addPanel(slide, position, { fill = PANEL, line = { style: "solid", fill: RULE, width: 1 }, name } = {}) {
  return slide.shapes.add({ geometry: "rect", name, position, fill, line });
}

function addMetricBox(slide, position, value, label, { fill = PANEL, valueColor = BLACK } = {}) {
  addPanel(slide, position, { fill, line: { style: "solid", fill: RULE, width: 1 } });
  addTextBox(
    slide,
    value,
    { left: position.left + 20, top: position.top + 18, width: position.width - 40, height: 66 },
    { fontSize: 46, color: valueColor, bold: true, verticalAlignment: "middle" },
  );
  addTextBox(
    slide,
    label,
    { left: position.left + 20, top: position.top + 88, width: position.width - 40, height: position.height - 96 },
    { fontSize: 21.33, color: MUTED },
  );
}

async function setHeroImage(slide, filePath, alt, { fit = "contain", position, crop } = {}) {
  const image = slide.images.items[0];
  if (!image) throw new Error(`Codex Grid slide-08 did not create its expected image slot for ${alt}`);
  image.replace({ blob: await bytesFor(filePath), contentType: mimeFor(filePath), alt, fit });
  image.alt = alt;
  image.fit = fit;
  image.geometry = "rect";
  if (position) image.position = position;
  if (crop) image.crop = crop;
  return image;
}

async function addImage(slide, filePath, alt, position, { fit = "cover" } = {}) {
  return slide.images.add({
    blob: await bytesFor(filePath),
    contentType: mimeFor(filePath),
    alt,
    fit,
    position,
    geometry: "rect",
  });
}

function configureGridChart(slide, { categories, series1, series2, direction = "column", max, majorUnit, legend = true, showDataLabels = true, valueFormat }) {
  const chart = slide.charts.items[0];
  if (!chart) throw new Error("Codex Grid slide-20 did not create its expected chart.");
  try {
    chart.categories = categories;
  } catch {
    // Series-level categories remain the authoritative fallback.
  }
  const first = chart.series.getItemAt(0);
  const second = chart.series.getItemAt(1);
  for (const [series, spec] of [
    [first, series1],
    [second, series2],
  ]) {
    series.name = spec.name;
    series.categories = categories;
    series.values = spec.values;
    series.fill = spec.fill;
    if (valueFormat) series.valuesFormatCode = valueFormat;
  }
  chart.title = "";
  chart.position = { left: 42.91, top: 148, width: 537.97, height: 535 };
  chart.hasLegend = legend;
  chart.legend = {
    position: "bottom",
    overlay: false,
    textStyle: { typeface: FONT, fontSize: 21.33, fill: BLACK },
  };
  chart.barOptions.direction = direction;
  chart.barOptions.grouping = "clustered";
  chart.barOptions.gapWidth = 82;
  chart.dataLabels = {
    showValue: showDataLabels,
    position: direction === "bar" ? "outEnd" : "outEnd",
    textStyle: { typeface: FONT, fontSize: 21.33, fill: BLACK, bold: true },
  };
  chart.xAxis = {
    visible: true,
    deleted: false,
    ...(direction === "bar" ? { min: 0, ...(max ? { max } : {}), ...(majorUnit ? { majorUnit } : {}) } : {}),
    majorGridlines: direction === "bar" ? { style: "solid", width: 1, fill: PANEL } : null,
    line: { style: "solid", width: 1, fill: RULE },
    textStyle: { typeface: FONT, fontSize: 21.33, color: BLACK },
    ...(valueFormat ? { numberFormatCode: valueFormat } : {}),
  };
  chart.yAxis = {
    visible: true,
    deleted: false,
    ...(direction === "column" ? { min: 0, ...(max ? { max } : {}), ...(majorUnit ? { majorUnit } : {}) } : {}),
    majorGridlines: direction === "column" ? { style: "solid", width: 1, fill: PANEL } : null,
    line: { style: "solid", width: 1, fill: RULE },
    textStyle: { typeface: FONT, fontSize: 21.33, color: BLACK },
    ...(valueFormat ? { numberFormatCode: valueFormat } : {}),
  };
}

function repairGridTimeline(slide) {
  const line = slide.shapes.items.find((shape) => shape.name === "Google-Shape-2259-p159-4");
  if (line) line.position = { left: 41.33, top: 560.8, width: 1197.33, height: 0.03 };
}

function repairPolicyBoundaryTitle(slide) {
  const title = slide.shapes.items.find((shape) => shape.name === "Title-10-3");
  if (!title) throw new Error("Policy-boundary title shape was not found.");
  title.position = { left: SAFE_LEFT, top: 36.12, width: 1174.66, height: 109.97 };
}

function decorate(slide, index, section, presenter) {
  slide.background.fill = WHITE;
  addPanel(slide, { left: 41.33, top: 18, width: 48, height: 4 }, { fill: BLUE, line: { style: "solid", fill: BLUE, width: 0 } });
  addTextBox(
    slide,
    `${String(index).padStart(2, "0")} · ${section.toUpperCase()}`,
    { left: 915, top: 11, width: 323, height: 21 },
    { fontSize: 13.33, color: MUTED, bold: true, alignment: "right", verticalAlignment: "middle" },
  );
  addTextBox(
    slide,
    `PRESENTER  ${presenter}`,
    { left: SAFE_LEFT, top: 688, width: 537.33, height: 24 },
    { fontSize: 13.33, color: MUTED, bold: true, verticalAlignment: "middle" },
  );
}

function setNotes(slide, index, presenter, keyMessage, sources) {
  const speakerScript = SPEAKER_SCRIPTS[index];
  if (!speakerScript) throw new Error(`Slide ${index} is missing a speaker script.`);
  if (speakerScript.presenter !== presenter) {
    throw new Error(`Slide ${index} speaker script presenter does not match the deck assignment.`);
  }
  const normalizedSources = sources.map((source) => {
    if (/^https?:\/\//i.test(source)) return source;
    if (path.isAbsolute(source)) {
      const relative = path.relative(PROJECT, source);
      if (relative && !relative.startsWith("..") && !path.isAbsolute(relative)) {
        return `Project-relative: ${relative.split(path.sep).join("/")}`;
      }
    }
    return source;
  });
  const notes = [
    `Presenter: ${presenter}`,
    `Target time: ${speakerScript.seconds} seconds`,
    `Key message: ${keyMessage}`,
    "",
    "[Speaker script]",
    speakerScript.text,
    "",
    "[Sources]",
    ...normalizedSources.map((source) => `- ${source}`),
  ].join("\n");
  if (!notes.includes("[Sources]") || normalizedSources.length === 0) throw new Error(`Slide ${index} is missing source notes.`);
  if (normalizedSources.some((source) => source.startsWith(`${PROJECT}${path.sep}`))) {
    throw new Error(`Slide ${index} contains a non-portable project-root source path.`);
  }
  slide.speakerNotes.textFrame.setText(notes);
  slide.speakerNotes.setVisible(true);
  notesAudit.push({
    slide: index,
    presenter,
    source_count: normalizedSources.length,
    speaker_script_seconds: speakerScript.seconds,
    contains_sources_block: true,
    all_local_sources_portable: normalizedSources.every(
      (source) => /^https?:\/\//i.test(source) || source.startsWith("Project-relative: "),
    ),
  });
}

function buildSpeakerScriptMarkdown() {
  const slides = Object.entries(SPEAKER_SCRIPTS)
    .sort(([left], [right]) => Number(left) - Number(right));
  const totalSeconds = slides.reduce((sum, [, entry]) => sum + entry.seconds, 0);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  const lines = [
    "# Ad Safety Management Presentation Speaker Script",
    "",
    `Target running time: approximately ${minutes}:${String(seconds).padStart(2, "0")} at a natural pace.`,
    "",
    "Use this as a rehearsal guide, not a passage to memorize. Keep the wording conversational, pause on the charts, and let the presenter handoffs sound natural.",
    "",
  ];
  for (const [slideNumber, entry] of slides) {
    lines.push(
      `## Slide ${slideNumber}: ${entry.title}`,
      "",
      `Presenter: ${entry.presenter}`,
      "",
      `Target time: ${entry.seconds} seconds`,
      "",
      entry.text,
      "",
    );
  }
  lines.push(
    "## Delivery notes",
    "",
    "- The 15-slide deck changes presenter 11 times. Rehearse each handoff so the switches feel deliberate.",
    "- Define macro F1 once on slide 1, then use the shorter term afterward.",
    "- On slide 10, separate highest-score recall from recall at the operating thresholds.",
    "- On slide 12, call the Wikimedia set a small selected diagnostic, not a population estimate.",
    "- On slide 13, separate the single app-case timing from the repeated classifier-only benchmark.",
    "- If time is short, trim examples and transitions before trimming the caveats.",
    "",
  );
  return lines.join("\n");
}

function registerRegions(slideNumber, regions) {
  slideAudit.push({ slide: slideNumber, regions });
}

function validateSemanticRegions() {
  const issues = [];
  for (const { slide, regions } of slideAudit) {
    for (const region of regions) {
      const { left, top, width, height } = region.frame;
      if (left < -0.01 || top < -0.01 || left + width > W + 0.01 || top + height > H + 0.01) {
        issues.push({ slide, type: "out_of_bounds", region: region.name, frame: region.frame });
      }
    }
    for (let i = 0; i < regions.length; i += 1) {
      for (let j = i + 1; j < regions.length; j += 1) {
        const a = regions[i];
        const b = regions[j];
        if (a.allowOverlap || b.allowOverlap) continue;
        const x = Math.max(0, Math.min(a.frame.left + a.frame.width, b.frame.left + b.frame.width) - Math.max(a.frame.left, b.frame.left));
        const y = Math.max(0, Math.min(a.frame.top + a.frame.height, b.frame.top + b.frame.height) - Math.max(a.frame.top, b.frame.top));
        if (x * y > 1) issues.push({ slide, type: "overlap", regions: [a.name, b.name], overlap_area_px2: x * y });
      }
    }
  }
  return { checked_slides: slideAudit.length, checked_regions: slideAudit.reduce((n, item) => n + item.regions.length, 0), issues, passed: issues.length === 0 };
}

async function writeBlob(filePath, blob) {
  await fsp.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function normalizePptxPackage(filePath) {
  const runtimeRequire = createRequire(path.join(BUILD, "runtime-loader.cjs"));
  const JSZip = runtimeRequire("jszip");
  const zip = await JSZip.loadAsync(await fsp.readFile(filePath));
  const themePaths = [
    "ppt/theme/theme1.xml",
    "ppt/slideMasters/theme/theme2.xml",
    "ppt/notesMasters/theme/theme3.xml",
  ];

  for (const themePath of themePaths) {
    const themeFile = zip.file(themePath);
    if (!themeFile) throw new Error(`Missing PPTX theme file: ${themePath}`);
    const xml = await themeFile.async("string");
    zip.file(themePath, xml.replaceAll("ChatGPT", "Ad Safety Project"));
  }

  const appPath = "docProps/app.xml";
  const appFile = zip.file(appPath);
  if (!appFile) throw new Error(`Missing PPTX metadata file: ${appPath}`);
  const appXml = (await appFile.async("string"))
    .replace(/<ap:PresentationFormat>.*?<\/ap:PresentationFormat>/, "<ap:PresentationFormat>On-screen Show (16:9)</ap:PresentationFormat>")
    .replace(/<ap:Slides>\d+<\/ap:Slides>/, "<ap:Slides>15</ap:Slides>")
    .replace(/<ap:Notes>\d+<\/ap:Notes>/, "<ap:Notes>15</ap:Notes>");
  zip.file(appPath, appXml);

  const corePath = "docProps/core.xml";
  const coreFile = zip.file(corePath);
  if (!coreFile) throw new Error(`Missing PPTX metadata file: ${corePath}`);
  const coreXml = (await coreFile.async("string"))
    .replace(/<dc:title>.*?<\/dc:title>/, "<dc:title>Ad Safety Moderation: Pilot Results and Next Steps</dc:title>");
  zip.file(corePath, coreXml);

  const normalized = await zip.generateAsync({
    type: "nodebuffer",
    compression: "DEFLATE",
    compressionOptions: { level: 9 },
    platform: "UNIX",
  });
  await fsp.writeFile(filePath, normalized);

  const checkZip = await JSZip.loadAsync(normalized);
  const checkedThemeXml = await Promise.all(themePaths.map((themePath) => checkZip.file(themePath).async("string")));
  if (checkedThemeXml.some((xml) => xml.includes("ChatGPT"))) {
    throw new Error("PPTX package retained an unrelated default theme label.");
  }
  const checkedAppXml = await checkZip.file(appPath).async("string");
  if (!checkedAppXml.includes("<ap:Slides>15</ap:Slides>") || !checkedAppXml.includes("<ap:Notes>15</ap:Notes>")) {
    throw new Error("PPTX package metadata does not report all slides and notes.");
  }
}

async function buildDeck() {
  const gridSnapshot = await ensureBuildRuntime();
  const calibrationZoom = await buildCalibrationZoom();
  const artifactPath = path.join(BUILD, "node_modules", "@oai", "artifact-tool", "dist", "artifact_tool.mjs");
  const { FileBlob, Presentation, PresentationFile } = await import(pathToFileURL(artifactPath).href);
  const { buildSlide08 } = await import(pathToFileURL(path.join(gridSnapshot, "slide-08.mjs")).href);
  const { buildSlide18 } = await import(pathToFileURL(path.join(gridSnapshot, "slide-18.mjs")).href);
  const { buildSlide19 } = await import(pathToFileURL(path.join(gridSnapshot, "slide-19.mjs")).href);
  const { buildSlide20 } = await import(pathToFileURL(path.join(gridSnapshot, "slide-20.mjs")).href);

  const presentation = Presentation.create({ slideSize: { width: W, height: H } });

  // 01 Executive Summary
  {
    const slide = buildSlide08(presentation, {
      title: titleToken("MS_DSP_462 · REVIEW QUEUE DECISION SUPPORT", 18.67),
      body1: splitBody("", ""),
      footer1: "1",
    });
    await setHeroImage(slide, ASSETS.appShell, "Browser-validated Ad Safety Studio analysis workspace", {
      fit: "cover",
      position: { left: 658.17, top: 112, width: 581.6, height: 327.15 },
    });
    addTextBox(slide, "Ad Safety Moderation\nPilot results and next steps", { left: 41.33, top: 112, width: 565, height: 220 }, { fontSize: 54, bold: false });
    addTextBox(slide, "0.979 macro F1 on our 48-image test\n0.555 macro F1 on the external check", { left: 41.33, top: 350, width: 565, height: 76 }, { fontSize: 23, color: BLACK, bold: true });
    addPanel(slide, { left: 41.33, top: 485, width: 565, height: 106 }, { fill: LIGHT_BLUE, line: { style: "solid", fill: CYAN, width: 1 } });
    addTextBox(slide, "Recommendation\nKeep this as a human-reviewed demo. It is not ready to make decisions on its own.", { left: 61, top: 497, width: 525, height: 83 }, { fontSize: 20, bold: true });
    addPanel(slide, { left: 658.17, top: 465, width: 581.6, height: 126 }, { fill: LIGHT_BLUE, line: { style: "solid", fill: CYAN, width: 1 } });
    addTextBox(slide, "FASTER FIRST PASS", { left: 674, top: 485, width: 174, height: 28 }, { fontSize: 16, bold: true, color: BLUE, alignment: "center" });
    addTextBox(slide, "CLEARER REVIEW", { left: 860, top: 485, width: 174, height: 28 }, { fontSize: 16, bold: true, color: BLUE, alignment: "center" });
    addTextBox(slide, "HUMAN CONTROL", { left: 1046, top: 485, width: 174, height: 28 }, { fontSize: 16, bold: true, color: BLUE, alignment: "center" });
    addTextBox(slide, "Sorts the queue", { left: 678, top: 527, width: 164, height: 24 }, { fontSize: 14.67, color: MUTED, alignment: "center" });
    addTextBox(slide, "Keeps scores + reasons", { left: 858, top: 527, width: 186, height: 24 }, { fontSize: 14.67, color: MUTED, alignment: "center" });
    addTextBox(slide, "Reviewer makes final call", { left: 1040, top: 527, width: 184, height: 24 }, { fontSize: 14.67, color: MUTED, alignment: "center" });
    decorate(slide, 1, "Executive Summary", PRESENTERS.swarnaditya);
    setNotes(slide, 1, PRESENTERS.swarnaditya, "The demo can help reviewers sort cases, but the external check shows that it is not ready to decide cases on its own.", [
      ABS("outputs/evaluation/metrics.json"),
      ABS("outputs/evaluation/external_spot_check.json"),
      ABS("outputs/app/app_shell.png"),
      ABS("outputs/app/browser_validation.json"),
      "https://support.google.com/adspolicy/answer/6008942",
    ]);
    registerRegions(1, [
      { name: "executive-copy", frame: { left: 41.33, top: 125, width: 565, height: 315 } },
      { name: "decision", frame: { left: 41.33, top: 485, width: 565, height: 106 } },
      { name: "hero", frame: { left: 658.17, top: 112, width: 581.6, height: 327.15 } },
      { name: "interface-summary", frame: { left: 658.17, top: 465, width: 581.6, height: 126 } },
    ]);
  }

  // 02 Team
  {
    const slide = buildSlide20(presentation, {
      title: titleToken("Team and\npresentation plan", 48),
      body1: chartCard("Swarnaditya Maitra", "Opening · problem · formal results · recommendation"),
      body2: chartCard("Vijay Agnihotri", "Team · data · EDA · error analysis"),
      body3: chartCard("Myetchae Thu + Bickramjit Basu", "Policy · architecture · external check · risks | method · research · app"),
      footer1: "2",
    });
    configureGridChart(slide, {
      categories: ["Swarnaditya", "Vijay", "Myetchae", "Bickramjit"],
      series1: { name: "Assigned slides", values: [4, 4, 4, 3], fill: BLUE },
      series2: { name: "", values: [0, 0, 0, 0], fill: CYAN },
      direction: "bar",
      max: 5,
      majorUnit: 1,
      legend: false,
      showDataLabels: false,
    });
    decorate(slide, 2, "Team", PRESENTERS.vijay);
    setNotes(slide, 2, PRESENTERS.vijay, "All four team members present named sections, moving from the problem and data to results, limits, and next steps.", [
      ABS("README.md"),
      ABS("scripts/build_presentation.mjs"),
    ]);
    registerRegions(2, [
      { name: "title-and-chart", frame: { left: 41.33, top: 36.12, width: 580, height: 647 } },
      { name: "presenter-cards", frame: { left: 657.68, top: 41.33, width: 580.99, height: 590.67 } },
    ]);
  }

  // 03 Business Problem
  {
    const slide = buildSlide19(presentation, {
      title: titleToken("Ambiguous ads are hard to review consistently", 46),
      body1: metricIntro("BUSINESS PROBLEM", "An image may contain an obvious object, small text, or context that is easy to miss. Our prototype helps reviewers sort cases consistently and sends uncertain cases to a person."),
      stat1: rich("PRIORITY", 40, { bold: true, color: BLUE }),
      stat2: rich("CONSISTENCY", 38, { bold: true, color: BLUE }),
      stat3: rich("CONTEXT", 42, { bold: true, color: BLUE }),
      body2: rich("Move likely high-risk images to the front of the queue.", 21.33),
      body3: rich("Apply the same first-pass review to every image.", 21.33),
      body4: rich("Send questions that pixels cannot answer to a reviewer.", 21.33),
      footer1: "3",
    });
    decorate(slide, 3, "Business Problem", PRESENTERS.swarnaditya);
    setNotes(slide, 3, PRESENTERS.swarnaditya, "The product goal is triage consistency, not a claim that one image can establish legality or fraud.", [
      "https://support.google.com/adspolicy/answer/6008942",
      "https://github.com/InteractiveAdvertisingBureau/Taxonomies/releases",
      ABS("README.md"),
    ]);
    registerRegions(3, [
      { name: "headline", frame: { left: 41.33, top: 36.12, width: 1197.33, height: 251.31 } },
      { name: "problem-cards", frame: { left: 41.33, top: 317.33, width: 1197.62, height: 312 } },
    ]);
  }

  // 04 Policy Boundary
  {
    const slide = buildSlide18(presentation, {
      title: titleToken("Four labels are not a full ad policy", 48),
      body1: splitBody("IN SCOPE", "Safe · firearms · explosives · financial-promotion cues. The last label means visible promotion, not fraud or illegality."),
      body2: splitBody("DECISION LAYER", "APPROVE · REVIEW · BLOCK. Financial cases always route to review. Restricted classes require calibrated evidence."),
      body3: splitBody("OUT OF SCOPE", "Explicit content, general violence, alcohol, tobacco, landing pages, age, geography, legality, and advertiser identity."),
      label1: rich("MODEL PREDICTS", 22, { bold: true, color: BLUE }),
      label2: rich("POLICY ROUTES", 22, { bold: true, color: BLUE }),
      label3: rich("REVIEWER / FUTURE", 22, { bold: true, color: RED }),
      footer1: "4",
    });
    repairGridTimeline(slide);
    repairPolicyBoundaryTitle(slide);
    decorate(slide, 4, "Policy Boundary", PRESENTERS.myetchae);
    setNotes(slide, 4, PRESENTERS.myetchae, "The current prototype covers four visual labels and three workflow outcomes; it does not cover the full ad policy surface.", [
      ABS("configs/policy.yaml"),
      ABS("README.md"),
      "https://support.google.com/adspolicy/answer/6008942",
      "https://github.com/InteractiveAdvertisingBureau/Taxonomies/releases",
    ]);
    registerRegions(4, [
      { name: "title", frame: { left: SAFE_LEFT, top: 36.12, width: 1174.66, height: 109.97 } },
      { name: "policy-cards", frame: { left: 41.33, top: 147.17, width: 1198.17, height: 482.16 } },
    ]);
  }

  // 05 Data Profile
  {
    const slide = buildSlide20(presentation, {
      title: titleToken("Balanced counts,\nunbalanced sources", 48),
      body1: chartCard("288 JPEG images", "48 / 12 / 12 per class for train · validation · final test."),
      body2: chartCard("215 source groups", "Campaign and source groups stay inside one split; exact hashes do not overlap."),
      body3: chartCard("Source shortcut risk", "ADautoGen supplies Safe; our builder supplies Finance; Weapons Set 1 supplies both weapon classes."),
      footer1: "5",
    });
    configureGridChart(slide, {
      categories: ["Safe", "Firearms", "Explosives", "Financial"],
      series1: { name: "Train", values: [48, 48, 48, 48], fill: BLUE },
      series2: { name: "Validation + test", values: [24, 24, 24, 24], fill: CYAN },
      direction: "column",
      max: 60,
      majorUnit: 20,
      legend: true,
    });
    decorate(slide, 5, "Data Profile", PRESENTERS.vijay);
    setNotes(slide, 5, PRESENTERS.vijay, "The split is leakage-aware, but the label-to-source mapping still creates a shortcut risk.", [
      ABS("data/capstone_registry.csv"),
      ABS("outputs/evaluation/split_summary.csv"),
      "https://huggingface.co/datasets/EthanGabis/ADautoGen-DS",
      "https://huggingface.co/datasets/rajshivanshuu/weapons_set1",
    ]);
    registerRegions(5, [
      { name: "title-and-chart", frame: { left: 41.33, top: 36.12, width: 580, height: 647 } },
      { name: "data-cards", frame: { left: 657.68, top: 41.33, width: 580.99, height: 590.67 } },
    ]);
  }

  // 06 EDA
  {
    const slide = buildSlide08(presentation, {
      title: titleToken("The classes look different\nfor reasons unrelated to policy", 46),
      body1: splitBody("WHAT THE CONTACT SHEET REVEALS", "Safe images look like generated product scenes. Weapon images are often isolated objects. Finance images reuse synthetic layouts.\n\nRisk: the classifier may learn visual source style instead of the intended policy category."),
      footer1: "6",
    });
    await setHeroImage(slide, ASSETS.contact, "Contact sheet contrasting source style by class and split", { fit: "contain" });
    decorate(slide, 6, "EDA", PRESENTERS.vijay);
    setNotes(slide, 6, PRESENTERS.vijay, "The contact sheet makes the confounding visible: source style and class are tightly linked despite group-safe splitting.", [
      ABS("outputs/evaluation/dataset_contact_sheet.jpg"),
      ABS("data/capstone_registry.csv"),
      ABS("outputs/evaluation/split_summary.csv"),
    ]);
    registerRegions(6, [
      { name: "eda-copy", frame: { left: 41.33, top: 36.12, width: 581.33, height: 593.21 } },
      { name: "contact-sheet", frame: { left: 658.17, top: 41.62, width: 581.6, height: 588.14 } },
    ]);
  }

  // 07 Architecture
  {
    const slide = buildSlide18(presentation, {
      title: titleToken("How the system reaches a decision", 48),
      body1: splitBody("1 · ANALYZE", "PyTorch and timm run a frozen ViT. Grounding DINO and Tesseract add optional object and text evidence."),
      body2: splitBody("2 · REVIEW", "FastAPI coordinates the services. Streamlit shows scores, boxes, text, timing, and the uploaded case."),
      body3: splitBody("3 · DECIDE", "A versioned YAML policy returns APPROVE, REVIEW, or BLOCK and saves a JSON audit record. Runs locally on CPU."),
      label1: rich("MODELS", 24, { bold: true, color: BLUE }),
      label2: rich("LOCAL APPLICATION", 22, { bold: true, color: BLUE }),
      label3: rich("POLICY + AUDIT", 22, { bold: true, color: BLUE }),
      footer1: "7",
    });
    repairGridTimeline(slide);
    const cards = slide.shapes.items.filter((shape) => shape.name?.startsWith("Rounded-Rectangle"));
    if (cards.length >= 3) {
      slide.shapes.connect(cards[0], cards[1], { kind: "straight", fromSide: "right", toSide: "left", line: { style: "solid", fill: BLUE, width: 2 }, tail: { type: "arrow", width: "med", length: "med" } });
      slide.shapes.connect(cards[1], cards[2], { kind: "straight", fromSide: "right", toSide: "left", line: { style: "solid", fill: BLUE, width: 2 }, tail: { type: "arrow", width: "med", length: "med" } });
    }
    decorate(slide, 7, "Architecture", PRESENTERS.myetchae);
    setNotes(slide, 7, PRESENTERS.myetchae, "The local stack keeps model, detector, OCR, policy, and audit evidence separate so a reviewer can inspect each contribution.", [
      ABS("README.md"),
      ABS("src/ad_safety/inference.py"),
      ABS("configs/policy.yaml"),
      "https://huggingface.co/IDEA-Research/grounding-dino-tiny",
      "https://github.com/tesseract-ocr/tesseract",
    ]);
    registerRegions(7, [
      { name: "title", frame: { left: 41.33, top: 36.12, width: 1197.33, height: 109.97 } },
      { name: "architecture", frame: { left: 41.33, top: 147.17, width: 1198.17, height: 482.16 } },
    ]);
  }

  // 08 Methodology
  {
    const slide = buildSlide08(presentation, {
      title: titleToken("We kept the test set\nout of training and tuning", 48),
      body1: splitBody("METHOD", "1  Freeze ViT and ResNet-50 because the dataset is small\n\n2  Fit logistic heads on train embeddings\n\n3  Select thresholds on validation only\n\n4  Run one final evaluation on 48 test images\n\n5  Benchmark warm batch-1 CPU"),
      footer1: "8",
    });
    await setHeroImage(
      slide,
      calibrationZoom,
      "Zoomed ViT explosives validation-only threshold calibration curve",
      {
        fit: "contain",
      },
    );
    addTextBox(
      slide,
      "Selected t=0.173 on validation · recall 1.00 · precision 1.00",
      { left: 684, top: 579, width: 530, height: 36 },
      { fontSize: 21.33, bold: true, color: BLUE, fill: WHITE, alignment: "center", verticalAlignment: "middle" },
    );
    decorate(slide, 8, "Methodology", PRESENTERS.bickramjit);
    setNotes(slide, 8, PRESENTERS.bickramjit, "Model selection and threshold calibration use validation data only; the final 48-image test remains untouched until evaluation.", [
      ABS("scripts/train_and_evaluate.py"),
      ABS("outputs/evaluation/threshold_calibration.png"),
      ABS("outputs/evaluation/thresholds.csv"),
      ABS("outputs/evaluation/evaluation_manifest.json"),
      "https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k",
      "https://huggingface.co/timm/resnet50.a1_in1k",
    ]);
    registerRegions(8, [
      { name: "method-copy", frame: { left: 41.33, top: 36.12, width: 581.33, height: 593.21 } },
      { name: "threshold-plot", frame: { left: 658.17, top: 41.62, width: 581.6, height: 588.14 } },
    ]);
  }

  // 09 Literature
  {
    const slide = buildSlide19(presentation, {
      title: titleToken("What prior research contributed", 48),
      body1: metricIntro("LITERATURE + BENCHMARKS", "Prior work informed our model choices. No outside benchmark is directly comparable to this four-label policy task, so we evaluate against the proposal targets and our saved results."),
      stat1: rich("ViT", 48, { bold: true, color: BLUE }),
      stat2: rich("DINO", 48, { bold: true, color: BLUE }),
      stat3: rich("OCR + RULES", 36, { bold: true, color: BLUE }),
      body2: rich("Dosovitskiy et al. (2021): transferable whole-image features.", 21.33),
      body3: rich("Liu et al. (2023): open-vocabulary object localization.", 21.33),
      body4: rich("Tesseract OCR + YAML rules: text evidence and traceable decisions.", 21.33),
      footer1: "9",
    });
    decorate(slide, 9, "Literature", PRESENTERS.bickramjit);
    setNotes(slide, 9, PRESENTERS.bickramjit, "The literature justifies the selected mechanisms, but it does not transfer benchmark performance to our dataset or policy task.", [
      "https://arxiv.org/abs/2010.11929",
      "https://arxiv.org/abs/2303.05499",
      "https://huggingface.co/IDEA-Research/grounding-dino-tiny",
      "https://github.com/tesseract-ocr/tesseract",
      "https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k",
    ]);
    registerRegions(9, [
      { name: "literature-intro", frame: { left: 41.33, top: 36.12, width: 1197.33, height: 251.31 } },
      { name: "literature-cards", frame: { left: 41.33, top: 317.33, width: 1197.62, height: 312 } },
    ]);
  }

  // 10 Formal Results
  {
    const slide = buildSlide08(presentation, {
      title: titleToken("ViT wins accuracy,\nnot speed", 48),
      body1: splitBody("", ""),
      footer1: "10",
    });
    await setHeroImage(slide, ASSETS.comparison, "Formal ViT versus ResNet test metrics and CPU latency plot", { fit: "contain" });
    addMetricBox(slide, { left: 41.33, top: 166, width: 270, height: 136 }, "0.97913", "ViT macro F1 · n=48", { fill: LIGHT_BLUE, valueColor: BLUE });
    addMetricBox(slide, { left: 326, top: 166, width: 296, height: 136 }, "0.87315", "ResNet macro F1 · n=48", { fill: PANEL });
    addMetricBox(
      slide,
      { left: 41.33, top: 320, width: 580.67, height: 151 },
      "71.527 / 40.064 ms",
      "Classifier p95 · ViT / ResNet · batch 1\nViT exceeds 50 ms; full-pipeline timing is unknown",
      { fill: PANEL },
    );
    addTextBox(
      slide,
      "Safe precision 1.000: PASS (>0.98)\nSafe false-flag rate: 0.0%\nRestricted recall, argmax mean: 0.97222\nRestricted recall, threshold mean / minimum: 0.94444 / 0.91667\nResult: two restricted classes miss the >0.95 proposal target",
      { left: 41.33, top: 489, width: 580, height: 149 },
      { fontSize: 21.33, color: MUTED },
    );
    decorate(slide, 10, "Formal Results", PRESENTERS.swarnaditya);
    setNotes(slide, 10, PRESENTERS.swarnaditya, "The ViT leads the ResNet baseline on the untouched 48-image test. Safe precision is 1.0, so the proposal's >0.98 Safe-class precision benchmark passes. Recall semantics differ: 0.97222 is multiclass-argmax restricted class-average recall, while validation-threshold operating recall averages 0.94444 and has a 0.91667 worst class. The threshold code used a validation tuning floor of >=0.90; that is not the proposal acceptance target of per-class restricted recall >0.95, which fails on the untouched test for firearms and explosives. ViT classifier-path p95 is 71.527 ms, above 50 ms; full end-to-end p95 is unassessed.", [
      ABS("Capstone Project Idea - Ad Safety.pdf"),
      ABS("outputs/evaluation/metrics.json"),
      ABS("outputs/evaluation/model_comparison.csv"),
      ABS("outputs/evaluation/model_comparison.png"),
      ABS("outputs/evaluation/latency.json"),
      ABS("outputs/evaluation/thresholds.csv"),
    ]);
    registerRegions(10, [
      { name: "formal-metrics", frame: { left: 41.33, top: 166, width: 580.67, height: 472 } },
      { name: "comparison-plot", frame: { left: 658.17, top: 41.62, width: 581.6, height: 588.14 } },
    ]);
  }

  // 11 Class and Failure Analysis
  {
    const slide = buildSlide20(presentation, {
      title: titleToken("One grenade was classified\nas a firearm", 48),
      body1: chartCard("WHAT HAPPENED", "One explosives image was predicted as firearms. The other 47 labels were correct."),
      body2: chartCard("POLICY RESULT", "Neither block threshold fired, so the saved case went to REVIEW rather than automatic enforcement."),
      body3: chartCard("WHY IT MATTERS", "Lowest class: explosives.\nViT F1 0.957\nResNet F1 0.750"),
      footer1: "11",
    });
    configureGridChart(slide, {
      categories: ["Safe", "Firearms", "Explosives", "Financial"],
      series1: { name: "ViT F1", values: [1.0, 0.96, 0.9565217391, 1.0], fill: BLUE },
      series2: { name: "ResNet F1", values: [0.7826086957, 0.96, 0.75, 1.0], fill: CYAN },
      direction: "column",
      max: 1.1,
      majorUnit: 0.25,
      legend: true,
      showDataLabels: false,
      valueFormat: "0.00",
    });
    await addImage(slide, ASSETS.formalFailure, "Formal test grenade that ViT classified as firearms", { left: 1098, top: 500, width: 110, height: 90 }, { fit: "cover" });
    decorate(slide, 11, "Class and Failure Analysis", PRESENTERS.vijay);
    setNotes(slide, 11, PRESENTERS.vijay, "The only wrong class label went to REVIEW, but it still exposes fragile separation between firearms and explosives.", [
      ABS("outputs/evaluation/per_class_metrics.csv"),
      ABS("outputs/evaluation/failure_cases.csv"),
      ABS("outputs/evaluation/confusion_matrix_vit.png"),
      ASSETS.formalFailure,
    ]);
    registerRegions(11, [
      { name: "title-and-chart", frame: { left: 41.33, top: 36.12, width: 580, height: 647 } },
      { name: "failure-cards", frame: { left: 657.68, top: 41.33, width: 580.99, height: 590.67 } },
    ]);
  }

  // 12 External Generalization Audit
  {
    const slide = buildSlide08(presentation, {
      title: titleToken("External images exposed\nweak generalization", 48),
      body1: splitBody("WIKIMEDIA DIAGNOSTIC · n=26", "0.55489 classifier macro F1\n0.90 detector nonweapon FPR\n\nThe images use different layouts, languages, ages, and photographic contexts. This small selected diagnostic is a warning, not a formal test set or population estimate."),
      footer1: "12",
    });
    const first = await setHeroImage(slide, ASSETS.externalSafeFalse, "Safe historical food advertisement falsely detected as explosive", { fit: "cover", position: { left: 658.17, top: 41.62, width: 226, height: 588.14 } });
    first.position = { left: 658.17, top: 41.62, width: 226, height: 588.14 };
    await addImage(slide, ASSETS.externalExplosiveMiss, "External explosives example classified safe with noisy detector boxes", { left: 900, top: 41.62, width: 339.77, height: 280 }, { fit: "cover" });
    await addImage(slide, ASSETS.externalFinancial, "External financial promotion with a false explosive detector box", { left: 900, top: 349.76, width: 339.77, height: 280 }, { fit: "cover" });
    addTextBox(slide, "SAFE → BLOCK", { left: 669, top: 582, width: 204, height: 36 }, { fontSize: 21.33, bold: true, color: RED, fill: WHITE, verticalAlignment: "middle", alignment: "center" });
    addTextBox(slide, "EXPLOSIVES → SAFE", { left: 911, top: 276, width: 318, height: 36 }, { fontSize: 21.33, bold: true, color: RED, fill: WHITE, verticalAlignment: "middle", alignment: "center" });
    addTextBox(slide, "FINANCE OK · FALSE BOX", { left: 911, top: 584, width: 318, height: 36 }, { fontSize: 21.33, bold: true, color: RED, fill: WHITE, verticalAlignment: "middle", alignment: "center" });
    decorate(slide, 12, "External Generalization Audit", PRESENTERS.myetchae);
    setNotes(slide, 12, PRESENTERS.myetchae, "In the small selected Wikimedia diagnostic, macro F1 falls to 0.55489 and 90 percent of nonweapon images receive a false detector signal.", [
      ABS("outputs/evaluation/external_spot_check.json"),
      ABS("outputs/evaluation/external_spot_check.csv"),
      ABS("outputs/evaluation/external_annotated"),
      "https://www.mediawiki.org/wiki/API:Imageinfo",
      "https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia",
    ]);
    registerRegions(12, [
      { name: "external-copy", frame: { left: 41.33, top: 36.12, width: 581.33, height: 593.21 } },
      { name: "example-a", frame: { left: 658.17, top: 41.62, width: 226, height: 588.14 } },
      { name: "example-b", frame: { left: 900, top: 41.62, width: 339.77, height: 280 } },
      { name: "example-c", frame: { left: 900, top: 349.76, width: 339.77, height: 280 } },
    ]);
  }

  // 13 Demo Workflow
  {
    const slide = buildSlide18(presentation, {
      title: titleToken("A reviewer can follow one case from upload to audit", 44),
      body1: splitBody("", ""),
      body2: splitBody("", ""),
      body3: splitBody("", ""),
      label1: rich("", 24),
      label2: rich("", 24),
      label3: rich("", 24),
      footer1: "13",
    });
    repairGridTimeline(slide);
    await addImage(
      slide,
      ASSETS.appFirearmResult,
      "Browser-validated BLOCK result with policy focus, evidence score, latency, and detector counts",
      { left: 41.33, top: 153, width: 760, height: 398 },
      { fit: "cover" },
    );
    const callouts = [
      ["DECISION STATE", "BLOCK, policy focus, and evidence score stay visible."],
      ["WHERE THE SCORE CAME FROM", "Raw and fused policy evidence remain separate."],
      ["EVIDENCE + TIMING", "Boxes, text, occlusion, and stage latency drill down."],
      ["AUDIT EXPORT", "Options, hashes, versions, and applied thresholds persist."],
    ];
    callouts.forEach(([title, body], index) => {
      const top = 153 + index * 102;
      addPanel(slide, { left: 825, top, width: 413.66, height: 90 }, { fill: index === 0 ? LIGHT_BLUE : PANEL, line: { style: "solid", fill: index === 0 ? CYAN : RULE, width: 1 } });
      addTextBox(slide, title, { left: 845, top: top + 9, width: 373.66, height: 26 }, { fontSize: 21.33, bold: true, color: index === 0 ? BLUE : BLACK });
      addTextBox(slide, body, { left: 845, top: top + 36, width: 373.66, height: 49 }, { fontSize: 21.33, color: MUTED });
    });
    addPanel(slide, { left: 30, top: 551, width: 1209, height: 17 }, { fill: WHITE, line: { style: "solid", fill: WHITE, width: 0 } });
    addPanel(slide, { left: 825, top: 559, width: 413.66, height: 70 }, { fill: WHITE, line: { style: "solid", fill: RED, width: 1 } });
    addTextBox(slide, "Session charts cover this session only.\nThey are not production or population evidence.", { left: 845, top: 568, width: 373.66, height: 52 }, { fontSize: 18, bold: true, color: RED, verticalAlignment: "middle" });
    decorate(slide, 13, "Demo Workflow", PRESENTERS.bickramjit);
    setNotes(slide, 13, PRESENTERS.bickramjit, "The browser-validated workspace keeps the decision, evidence, timing, and exact audit record together while preserving human control over review cases.", [
      ABS("app.py"),
      ABS("api.py"),
      ABS("outputs/app/browser_validation.json"),
      ABS("outputs/app/app_firearm_result.png"),
      ABS("src/ad_safety/inference.py"),
    ]);
    registerRegions(13, [
      { name: "title", frame: { left: 41.33, top: 36.12, width: 1197.33, height: 109.97 } },
      { name: "validated-result", frame: { left: 41.33, top: 153, width: 760, height: 398 } },
      { name: "interface-callouts", frame: { left: 825, top: 153, width: 413.66, height: 396 } },
      { name: "monitoring-caveat", frame: { left: 825, top: 559, width: 413.66, height: 70 } },
    ]);
  }

  // 14 Risks and Controls
  {
    const slide = buildSlide19(presentation, {
      title: titleToken("What must change before a pilot", 48),
      body1: metricIntro("THREE GAPS", "Before a pilot, we need to address source-biased data, noisy detections, and missing full-pipeline performance measurements."),
      stat1: rich("DATA", 46, { bold: true, color: RED }),
      stat2: rich("DETECTOR", 40, { bold: true, color: RED }),
      stat3: rich("OPERATIONS", 38, { bold: true, color: RED }),
      body2: rich("Collect diverse real campaigns and use independent labels instead of source-linked shortcuts.", 21.33),
      body3: rich("A 0.90 nonweapon FPR requires local box labels, recalibration, and no detector-only block.", 21.33),
      body4: rich("Measure end-to-end p95,\nconcurrency, CPU/GPU load,\nMPS acceleration, and memory.", 21.33),
      footer1: "14",
    });
    decorate(slide, 14, "Risks and Controls", PRESENTERS.myetchae);
    setNotes(slide, 14, PRESENTERS.myetchae, "The main risks are source confounding, an uncalibrated detector, and classifier-only latency evidence. Full end-to-end p95, concurrent throughput, CPU and GPU load, MPS acceleration, and memory use were not measured; each needs an explicit control before a trial.", [
      ABS("outputs/evaluation/external_spot_check.json"),
      ABS("outputs/evaluation/latency.json"),
      ABS("data/capstone_registry.csv"),
      ABS("README.md"),
    ]);
    registerRegions(14, [
      { name: "risk-intro", frame: { left: 41.33, top: 36.12, width: 1197.33, height: 251.31 } },
      { name: "risk-cards", frame: { left: 41.33, top: 317.33, width: 1197.62, height: 312 } },
    ]);
  }

  // 15 Next Steps to MVP
  {
    const slide = buildSlide18(presentation, {
      title: titleToken("What we would do next: a 10-week plan", 48),
      body1: splitBody("WEEKS 1–3 · EVIDENCE", "A data lead, policy reviewer, and 2 annotators curate ≥1,000 real ads and lock a campaign-grouped independent holdout."),
      body2: splitBody("WEEKS 4–6 · CALIBRATE", "1 ML engineer and a reviewer tune against two gates: >0.95 recall per restricted class and <2% Safe crossings."),
      body3: splitBody("WEEKS 7–10 · SHADOW PILOT", "A product lead, app/MLOps engineer, and reviewer rotation test 10 concurrent requests, end-to-end p95 latency, and resource use."),
      label1: rich("DATA + POLICY", 24, { bold: true, color: BLUE }),
      label2: rich("ML + QA", 24, { bold: true, color: BLUE }),
      label3: rich("PRODUCT + OPS", 24, { bold: true, color: BLUE }),
      footer1: "15",
    });
    repairGridTimeline(slide);
    decorate(slide, 15, "Next Steps to MVP", PRESENTERS.swarnaditya);
    setNotes(slide, 15, PRESENTERS.swarnaditya, "This is a recommended 10-week plan and staffing assumption, not a measured commitment. A human-gated pilot starts only after evidence, quality, and explicit capacity gates pass.", [
      ABS("README.md"),
      ABS("outputs/evaluation/metrics.json"),
      ABS("outputs/evaluation/external_spot_check.json"),
      "https://support.google.com/adspolicy/answer/6008942",
    ]);
    registerRegions(15, [
      { name: "title", frame: { left: 41.33, top: 36.12, width: 1197.33, height: 109.97 } },
      { name: "mvp-roadmap", frame: { left: 41.33, top: 147.17, width: 1198.17, height: 482.16 } },
    ]);
  }

  if (presentation.slides.items.length !== 15) throw new Error(`Expected 15 slides, found ${presentation.slides.items.length}.`);
  if (
    notesAudit.length !== 15
    || notesAudit.some((entry) => !entry.contains_sources_block || !entry.all_local_sources_portable)
  ) {
    throw new Error("Every slide must contain a [Sources] block with portable local paths.");
  }

  const semanticValidation = validateSemanticRegions();
  if (!semanticValidation.passed) throw new Error(`Semantic layout validation failed: ${JSON.stringify(semanticValidation.issues)}`);

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(path.join(RENDERED, `${stem}.png`), await presentation.export({ slide, format: "png", scale: 1 }));
    const layout = await slide.export({ format: "layout" });
    await fsp.writeFile(path.join(RENDERED, `${stem}.layout.json`), await layout.text());
  }
  await writeBlob(path.join(OUT, "deck_montage.webp"), await presentation.export({ format: "webp", montage: true, scale: 1 }));

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(PPTX_PATH);
  await normalizePptxPackage(PPTX_PATH);

  const reopened = await PresentationFile.importPptx(await FileBlob.load(PPTX_PATH));
  const reopenedCount = reopened.slides.items.length;
  if (reopenedCount !== 15) throw new Error(`PPTX reopen check found ${reopenedCount} slides, expected 15.`);
  await writeBlob(path.join(OUT, "reopened_montage.webp"), await reopened.export({ format: "webp", montage: true, scale: 0.5 }));

  const plan = [
    "Communication job: By the end, evaluators and management stakeholders should understand what the prototype can do, where the evidence breaks down, and why the next step is a human-reviewed validation pilot rather than autonomous enforcement.",
    "Narrative: review problem -> scope -> data and method -> formal result -> external warning -> controls and next steps.",
    "Design system: Codex Grid 1280x720, white canvas, black ink, gray panels, restrained blue accent, Arial.",
    `Selected layouts: ${SELECTED_TEMPLATE_IDS.join(", ")}`,
  ].join("\n");
  await fsp.writeFile(path.join(BUILD, "slide-plan.txt"), plan);
  await fsp.writeFile(SPEAKER_SCRIPT_PATH, buildSpeakerScriptMarkdown());
  await fsp.writeFile(
    path.join(BUILD, "source-notes.txt"),
    notesAudit.map((entry) => `Slide ${entry.slide}: ${entry.presenter}; source_count=${entry.source_count}`).join("\n"),
  );

  const validation = {
    schema_version: "1.0.0",
    output: path.relative(PROJECT, PPTX_PATH).split(path.sep).join("/"),
    slide_size: { width: W, height: H },
    authored_slide_count: 15,
    reopened_slide_count: reopenedCount,
    sources_notes: {
      slides_checked: notesAudit.length,
      all_have_sources_block: true,
      all_local_sources_portable: notesAudit.every((entry) => entry.all_local_sources_portable),
    },
    semantic_overlap_and_bounds_test: semanticValidation,
    selected_grid_templates: SELECTED_TEMPLATE_IDS,
    rendered_png_count: (await fsp.readdir(RENDERED)).filter((name) => name.endsWith(".png")).length,
    cross_renderer_left_edge_clearance: {
      typeface: FONT,
      slide_4_title_left: SAFE_LEFT,
      presenter_footer_left: SAFE_LEFT,
      presenter_footer_top: 688,
    },
    pptx_sha256: await fileSha256(PPTX_PATH),
    mvp_roadmap_plan: {
      status: "recommended plan, not a measured commitment",
      timeline_weeks: 10,
      phase_windows: ["weeks 1-3", "weeks 4-6", "weeks 7-10"],
      resource_requirements: [
        "data lead, policy reviewer, and two annotators",
        "one ML engineer, reviewer, and CPU/GPU test capacity",
        "product lead, app/MLOps engineer, and reviewer rotation",
      ],
      scale_gate: {
        concurrent_requests: 10,
        measurements: ["end-to-end p95", "CPU", "GPU", "RAM"],
      },
    },
    measured_facts_used: {
      formal_vit_macro_f1: 0.97913,
      formal_resnet_macro_f1: 0.87315,
      formal_vit_classifier_path_cpu_p95_ms: 71.527,
      formal_resnet_classifier_path_cpu_p95_ms: 40.064,
      classifier_path_p95_target_ms: 50,
      formal_vit_classifier_path_p95_target_met: false,
      full_end_to_end_p95_assessed: false,
      concurrent_throughput_assessed: false,
      mps_resource_use_assessed: false,
      formal_safe_ad_false_flag_rate_one_minus_safe_recall: 0,
      formal_safe_precision: 1.0,
      proposal_safe_precision_target: ">0.98",
      proposal_safe_precision_target_met: true,
      formal_safe_precision_target_met: true,
      formal_multiclass_argmax_restricted_class_average_recall: 0.97222,
      formal_validation_threshold_operating_recall_mean: 0.94444,
      formal_validation_threshold_operating_recall_worst_class: 0.91667,
      calibration_validation_recall_tuning_floor: ">=0.90",
      formal_per_class_recall_target: ">0.95",
      formal_per_class_recall_target_met: false,
      proposal_per_class_restricted_recall_target: ">0.95",
      proposal_per_class_restricted_recall_target_met: false,
      formal_test_n: 48,
      external_macro_f1: 0.55489,
      external_detector_nonweapon_false_positive_rate: 0.9,
      external_diagnostic_n: 26,
    },
    interface_facts_used: {
      browser_validated_cases: ["APPROVE", "REVIEW", "BLOCK"],
      audit_schema_version: "1.1",
      result_persistence_checked: true,
      stale_option_state_checked: true,
      charts_are_descriptive_interface_evidence_only: true,
      production_monitoring_claimed: false,
    },
  };
  await fsp.writeFile(path.join(OUT, "presentation_validation.json"), JSON.stringify(validation, null, 2));
  console.log(JSON.stringify(validation, null, 2));
}

buildDeck().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
