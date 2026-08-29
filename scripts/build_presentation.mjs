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
const BLACK = "#172033";
const WHITE = "#FFFFFF";
const PANEL = "#F3F5F8";
const LIGHT_BLUE = "#EAF2FF";
const RULE = "#D2D9E4";
const MUTED = "#526071";
const BLUE = "#2367C9";
const CYAN = "#55AFC2";
const TEAL = "#16836F";
const GOLD = "#B77A18";
const VIOLET = "#6B5AB6";
const RED = "#C43D52";

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
  teamDemo: "Full team (Bickramjit Basu leads; Swarnaditya Maitra shares the screen)",
};

const SPEAKER_SCRIPTS = {
  1: {
    title: "Ad Safety Moderation: What the Pilot Showed",
    presenter: PRESENTERS.swarnaditya,
    seconds: 45,
    text: "Good afternoon. We built Ad Safety Studio to help a reviewer sort image-based ads, not to replace a policy decision. The model scored 97.9 percent macro F1 on our 48-image held-out test, but only 55.5 percent on a small external check. That gap is the main result. The app works, keeps the evidence beside the decision, and can support a first-pass review. It is not ready for autonomous enforcement. I will introduce the team first, then frame the problem we set out to solve.",
  },
  2: {
    title: "Meet the Team",
    presenter: PRESENTERS.swarnaditya,
    seconds: 35,
    text: "There are four of us, and each person owns one continuous part of the story. I am Swarnaditya, covering the opening, problem, and scope. Vijay covers the data, system, and evaluation method. Myetchae covers prior work, results, and the external check. Bickramjit leads the working app, controls, and next steps. During the app demo, I will share the screen and all four of us will explain the part we worked on. Let me start with the review problem.",
  },
  3: {
    title: "Ambiguous Ads Are Hard to Review Consistently",
    presenter: PRESENTERS.swarnaditya,
    seconds: 40,
    text: "Ad review is difficult because one creative can mix several kinds of evidence. A weapon may be obvious, important text may be small, and context can change what an image means. Our goal was not to replace a policy reviewer. We wanted to help a queue move faster, apply the same first-pass process to every image, and send uncertain cases to a person. That first-pass goal also sets the boundary for what the prototype can and cannot decide.",
  },
  4: {
    title: "Four Labels Are Not a Full Ad Policy",
    presenter: PRESENTERS.swarnaditya,
    seconds: 45,
    text: "We deliberately kept the scope narrow: safe, firearms, explosives, and visible financial-promotion cues. The financial label does not mean fraud or illegality. It means the image appears promotional, so our rules always send it to review. For the other labels, a readable policy file maps evidence to approve, review, or block. We do not cover landing pages, advertiser identity, geography, age rules, alcohol, tobacco, or the rest of the ad-policy surface. This is a triage prototype, not a complete policy engine. That is the boundary of the prototype. Vijay will now show the data and how we protected the evaluation.",
  },
  5: {
    title: "Balanced Counts, Unbalanced Sources",
    presenter: PRESENTERS.vijay,
    seconds: 45,
    text: "The pilot contains 288 JPEG images. Every class has 48 training images, 12 validation images, and 12 final test images. We also kept each of the 215 source groups inside one split and checked that file hashes do not overlap. That reduces leakage, meaning the same campaign or exact image cannot appear on both sides of the evaluation. The weakness is source diversity. Safe and financial images are synthetic, while both weapon classes come from one weapons dataset. The counts are balanced, but the visual sources are not. The next slide makes that problem visible.",
  },
  6: {
    title: "The Model Can Learn Source Style, Not Policy Meaning",
    presenter: PRESENTERS.vijay,
    seconds: 40,
    text: "The contact sheet shows the shortcut risk more clearly than a metric can. Safe images look like generated product scenes. Weapon images are often isolated objects. Financial images repeat synthetic layouts. Even with a clean group split, a classifier can learn the look of the source instead of the policy concept itself. That does not make the formal result useless, but it limits the claim to this pilot domain. That risk shaped both our model design and how cautiously we read the results.",
  },
  7: {
    title: "How the System Reaches a Decision",
    presenter: PRESENTERS.vijay,
    seconds: 50,
    text: "The system runs locally and has three parts. PyTorch and timm run a frozen vision transformer, while Grounding DINO and Tesseract add optional object and text evidence. FastAPI coordinates those services, and Streamlit gives the reviewer one workspace for the image, scores, boxes, text, and timing. A versioned YAML policy then returns approve, review, or block with reason codes and a JSON audit record. Keeping the signals separate lets a reviewer inspect what each component contributed. With those components separated, the next step was to train and test the classifier without letting the final test set influence our choices.",
  },
  8: {
    title: "We Kept the Test Set Out of Training and Tuning",
    presenter: PRESENTERS.vijay,
    seconds: 50,
    text: "Because the dataset is small, we froze the ViT and ResNet-50 backbones instead of fine-tuning millions of weights. We then trained small logistic classifiers on training embeddings. We chose class thresholds only with the validation split. The 48-image test split was not used to train, choose a model, or tune a threshold, and we ran the final evaluation after those choices were locked. The chart shows the validation-only explosives threshold, about 0.173, with recall and precision both at 100 percent on that small validation sample. That gives us a controlled evaluation. Myetchae will now connect the design to prior work and walk through the results.",
  },
  9: {
    title: "What Prior Research Contributed",
    presenter: PRESENTERS.myetchae,
    seconds: 40,
    text: "Prior work helped us choose credible building blocks. The Vision Transformer paper supports transferable whole-image features. Grounding DINO supports text-guided object localization, and Tesseract provides visible-text extraction. We did not find an outside benchmark directly comparable to our four-label policy task. Those papers guided our choices; the performance numbers come from our own evaluation. With that distinction clear, we can look at what the evaluation actually found.",
  },
  10: {
    title: "Better Accuracy Came with Recall and Speed Trade-offs",
    presenter: PRESENTERS.myetchae,
    seconds: 65,
    text: "On the untouched 48-image test, ViT reached 97.9 percent macro F1, compared with 87.3 percent for the ResNet-50 baseline. Safe precision was 100 percent, so it met our target, and none of the 12 safe test images crossed into a restricted class. The caveat is recall. At the operating thresholds, firearms and explosives each reached 91.7 percent recall, below our goal of more than 95 percent for every restricted class. Speed also favors ResNet. ViT classifier-path p95 was 71.5 milliseconds, above the 50-millisecond budget, while ResNet was 40.1. This benchmark excludes decoding, OCR, detection, network time, and rendering, so full end-to-end p95 is still unknown. The next slide shows the one classification error behind those results.",
  },
  11: {
    title: "One Error Is Why a Reviewer Stays in Control",
    presenter: PRESENTERS.myetchae,
    seconds: 45,
    text: "The formal test had one wrong class label. A grenade image labeled explosives was predicted as firearms, while the other 47 labels were correct. Its firearm and explosive scores did not cross either class-specific block threshold, so the saved policy result was REVIEW rather than an automatic block or approval. The optional detector found restricted-object cues, but the class separation was still fragile. The bigger question is whether the same system holds up when the images stop looking like our pilot sources.",
  },
  12: {
    title: "Outside the Pilot Data, Performance Fell Sharply",
    presenter: PRESENTERS.myetchae,
    seconds: 60,
    text: "The external check used 26 manually reviewed Wikimedia images for diagnosis, not as a second formal test set. Macro F1 fell to 55.5 percent. At the current detector threshold, 90 percent of the nonweapon images in this small sample received a false detector signal. The examples show the pattern: a historical safe ad was blocked, one explosives image was classified as safe, and a financial example received a false box even though its policy route remained review. The sample is small and selected, so this is not a population estimate. It is still enough to show that the formal score does not generalize reliably. Bickramjit will now move to the working app and the controls needed before a pilot.",
  },
  13: {
    title: "From Upload to Audit in One Workspace",
    presenter: PRESENTERS.teamDemo,
    seconds: 75,
    text: "Bickramjit: The working app keeps one case in one place. Swarnaditya is sharing the screen and will run the example. Swarnaditya: I will upload the firearm image, keep the optional detector and occlusion view enabled, and start the analysis. Vijay: The results separate classifier scores from fused scores and show how long each stage took, so we can see what changed instead of relying on one final label. Myetchae: The policy turns that evidence into BLOCK, REVIEW, or APPROVE with reason codes, and the audit export records hashes, model versions, options, and thresholds. Bickramjit: The timing shown here is one app run, not the repeated classifier-only benchmark. The session charts summarize only the cases reviewed today. That brings us to the controls we still need.",
  },
  14: {
    title: "What Must Change Before a Pilot",
    presenter: PRESENTERS.bickramjit,
    seconds: 50,
    text: "Before a real pilot, we need controls in three areas. For data, we need diverse real campaigns and independent annotation, not source-linked synthetic shortcuts. For the detector, we need local box labels and recalibration, and we should not let the detector block by itself. For operations, we still need full end-to-end p95, concurrent throughput, and CPU, Apple GPU, and memory measurements. These controls reduce the chance of harm, but they do not remove the uncertainty. Our decision is no-go for autonomous use and go only for further human-reviewed validation. Those controls lead directly to the gated plan on the final slide.",
  },
  15: {
    title: "What We Would Do Next: A 10-Week Plan",
    presenter: PRESENTERS.bickramjit,
    seconds: 55,
    text: "This 10-week roadmap is our proposed next step. In weeks one through three, we would collect at least 1,000 real ads and lock an independent campaign-grouped holdout. In weeks four through six, we would recalibrate and require more than 95 percent recall for every restricted class, with fewer than 2 percent safe crossings. In weeks seven through ten, we would run a shadow, human-reviewed pilot and test 10 concurrent requests with full end-to-end latency and resource tracking. If any gate fails, we stop and fix it before scaling. The prototype is reproducible, but stronger validation must come before autonomous use. Thank you. All four of us are ready for questions.",
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
    .extract({ left: 920, top: 150, width: 770, height: 540 })
    .png()
    .toFile(outputPath);
  return outputPath;
}

function rich(text, fontSize = 22.67, { bold = false, color = BLACK } = {}) {
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
  return rich(text, fontSize, { bold: true, color: BLACK });
}

function splitBody(title, body, titleKey = "titleHere") {
  return {
    [titleKey]: rich(title, 25.33, { bold: true }),
    loremIpsumDolorSitAmetConsecteturAdipiscing: rich(body, 22.67),
  };
}

function metricIntro(topic, body) {
  return {
    topic: rich(topic, 22.67, { bold: true, color: BLUE }),
    loremIpsumDolorSitAmetConsecteturAdipiscing: rich(body, 22.67),
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

function decorate(slide, index) {
  slide.background.fill = WHITE;
  addPanel(slide, { left: 41.33, top: 18, width: 48, height: 4 }, { fill: BLUE, line: { style: "solid", fill: BLUE, width: 0 } });
  addTextBox(
    slide,
    `${String(index).padStart(2, "0")} / 15`,
    { left: 1085, top: 11, width: 153, height: 21 },
    { fontSize: 13.33, color: MUTED, bold: true, alignment: "right", verticalAlignment: "middle" },
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
    "# Ad Safety Presentation Speaker Script",
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
    "- Speaker order: Swarnaditya slides 1-4, Vijay slides 5-8, Myetchae slides 9-12, and Bickramjit slides 13-15.",
    "- Slide 13 is a team demo. Bickramjit leads, Swarnaditya shares the screen, and all four members explain their part.",
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
    .replace(/<dc:title>.*?<\/dc:title>/, "<dc:title>Ad Safety Moderation: What the Pilot Showed</dc:title>")
    .replace(
      /<dc:creator>.*?<\/dc:creator>/,
      "<dc:creator>Swarnaditya Maitra; Vijay Agnihotri; Myetchae Thu; Bickramjit Basu</dc:creator>",
    )
    .replace(/<lastModifiedBy>.*?<\/lastModifiedBy>/, "<lastModifiedBy>Ad Safety Project Team</lastModifiedBy>");
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
      title: titleToken("AD SAFETY STUDIO", 18.67),
      body1: splitBody("", ""),
      footer1: "",
    });
    await setHeroImage(slide, ASSETS.appShell, "Browser-validated Ad Safety Studio analysis workspace", {
      fit: "contain",
      position: { left: 666, top: 92, width: 574, height: 466 },
    });
    addTextBox(slide, "Ad safety moderation:\nwhat the pilot showed", { left: 41.33, top: 104, width: 580, height: 170 }, { fontSize: 52, bold: true });
    addTextBox(slide, "97.9%", { left: 41.33, top: 304, width: 210, height: 64 }, { fontSize: 48, color: BLUE, bold: true });
    addTextBox(slide, "macro F1 on 48 held-out images", { left: 190, top: 319, width: 420, height: 40 }, { fontSize: 22, color: MUTED });
    addTextBox(slide, "55.5%", { left: 41.33, top: 377, width: 210, height: 64 }, { fontSize: 48, color: RED, bold: true });
    addTextBox(slide, "macro F1 on the external check", { left: 190, top: 392, width: 420, height: 40 }, { fontSize: 22, color: MUTED });
    addPanel(slide, { left: 41.33, top: 480, width: 580, height: 140 }, { fill: LIGHT_BLUE, line: { style: "solid", fill: CYAN, width: 1 } });
    addTextBox(slide, "Our conclusion", { left: 64, top: 499, width: 210, height: 32 }, { fontSize: 24, bold: true, color: BLUE });
    addTextBox(slide, "Keep a reviewer in control. The demo can triage cases, but it should not decide cases on its own.", { left: 64, top: 536, width: 530, height: 68 }, { fontSize: 21.33, bold: true });
    addTextBox(slide, "Upload  →  evidence  →  decision  →  audit", { left: 690, top: 574, width: 526, height: 36 }, { fontSize: 21.33, color: MUTED, alignment: "center" });
    decorate(slide, 1);
    setNotes(slide, 1, PRESENTERS.swarnaditya, "The demo can help reviewers sort cases, but the external check shows that it is not ready to decide cases on its own.", [
      ABS("outputs/evaluation/metrics.json"),
      ABS("outputs/evaluation/external_spot_check.json"),
      ABS("outputs/app/app_shell.png"),
      ABS("outputs/app/browser_validation.json"),
      "https://support.google.com/adspolicy/answer/6008942",
    ]);
    registerRegions(1, [
      { name: "executive-copy", frame: { left: 41.33, top: 104, width: 580, height: 337 } },
      { name: "decision", frame: { left: 41.33, top: 480, width: 580, height: 140 } },
      { name: "hero", frame: { left: 666, top: 92, width: 574, height: 466 } },
      { name: "workflow-caption", frame: { left: 690, top: 574, width: 526, height: 36 } },
    ]);
  }

  // 02 Team
  {
    const slide = presentation.slides.add();
    addTextBox(slide, "Meet the team", { left: 64, top: 58, width: 820, height: 72 }, { fontSize: 52, bold: true });
    addTextBox(
      slide,
      "Four consecutive speaker blocks, with the shared app demonstration opening the final block.",
      { left: 64, top: 136, width: 980, height: 42 },
      { fontSize: 24, color: MUTED },
    );
    addPanel(slide, { left: 92, top: 288, width: 1094, height: 3 }, { fill: RULE, line: { style: "solid", fill: RULE, width: 0 } });
    const teamBlocks = [
      { range: "1-4", name: "Swarnaditya Maitra", role: "Opening, problem, and scope", color: BLUE },
      { range: "5-8", name: "Vijay Agnihotri", role: "Data, system, and method", color: TEAL },
      { range: "9-12", name: "Myetchae Thu", role: "Prior work, results, and external check", color: VIOLET },
      { range: "13-15", name: "Bickramjit Basu", role: "App, controls, and next steps", color: GOLD },
    ];
    teamBlocks.forEach((entry, index) => {
      const left = 64 + index * 296;
      slide.shapes.add({
        geometry: "ellipse",
        position: { left: left + 4, top: 274, width: 30, height: 30 },
        fill: entry.color,
        line: { style: "solid", fill: WHITE, width: 3 },
      });
      addTextBox(slide, entry.range, { left, top: 210, width: 190, height: 52 }, { fontSize: 38, bold: true, color: entry.color });
      addTextBox(slide, entry.name, { left, top: 330, width: 260, height: 54 }, { fontSize: 26, bold: true });
      addTextBox(slide, entry.role, { left, top: 392, width: 250, height: 76 }, { fontSize: 22, color: MUTED });
    });
    addPanel(slide, { left: 64, top: 526, width: 1152, height: 92 }, { fill: LIGHT_BLUE, line: { style: "solid", fill: CYAN, width: 1 } });
    addTextBox(slide, "Team demo", { left: 88, top: 548, width: 190, height: 40 }, { fontSize: 26, bold: true, color: BLUE, verticalAlignment: "middle" });
    addTextBox(
      slide,
      "All four members contribute. Swarnaditya shares the screen while each person explains their part.",
      { left: 280, top: 545, width: 904, height: 48 },
      { fontSize: 22, color: BLACK, verticalAlignment: "middle" },
    );
    decorate(slide, 2);
    setNotes(slide, 2, PRESENTERS.swarnaditya, "The team presents in four consecutive speaker blocks, with the working app demonstration opening the final block.", [
      ABS("README.md"),
      ABS("scripts/build_presentation.mjs"),
    ]);
    registerRegions(2, [
      { name: "title", frame: { left: 64, top: 58, width: 980, height: 120 } },
      { name: "speaker-sequence", frame: { left: 64, top: 200, width: 1152, height: 288 } },
      { name: "team-demo", frame: { left: 64, top: 526, width: 1152, height: 92 } },
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
      footer1: "",
    });
    decorate(slide, 3);
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
      footer1: "",
    });
    repairGridTimeline(slide);
    repairPolicyBoundaryTitle(slide);
    decorate(slide, 4);
    setNotes(slide, 4, PRESENTERS.swarnaditya, "The current prototype covers four visual labels and three workflow outcomes; it does not cover the full ad policy surface.", [
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
      body3: chartCard("Source shortcut risk", "ADautoGen supplies Safe, project-generated creatives supply Finance, and Weapons Set 1 supplies both weapon classes."),
      footer1: "",
    });
    configureGridChart(slide, {
      categories: ["Safe", "Firearms", "Explosives", "Financial"],
      series1: { name: "Train", values: [48, 48, 48, 48], fill: BLUE },
      series2: { name: "Held-out: 12 validation + 12 test", values: [24, 24, 24, 24], fill: CYAN },
      direction: "column",
      max: 60,
      majorUnit: 20,
      legend: true,
    });
    decorate(slide, 5);
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
      title: titleToken("The model can learn source style,\nnot policy meaning", 46),
      body1: splitBody("WHAT THE CONTACT SHEET REVEALS", "Safe images look like generated product scenes. Weapon images are often isolated objects. Finance images reuse synthetic layouts.\n\nRisk: the classifier may learn visual source style instead of the intended policy category."),
      footer1: "",
    });
    await setHeroImage(slide, ASSETS.contact, "Contact sheet contrasting source style by class and split", { fit: "contain" });
    decorate(slide, 6);
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
      footer1: "",
    });
    repairGridTimeline(slide);
    const cards = slide.shapes.items.filter((shape) => shape.name?.startsWith("Rounded-Rectangle"));
    if (cards.length >= 3) {
      slide.shapes.connect(cards[0], cards[1], { kind: "straight", fromSide: "right", toSide: "left", line: { style: "solid", fill: BLUE, width: 2 }, tail: { type: "arrow", width: "med", length: "med" } });
      slide.shapes.connect(cards[1], cards[2], { kind: "straight", fromSide: "right", toSide: "left", line: { style: "solid", fill: BLUE, width: 2 }, tail: { type: "arrow", width: "med", length: "med" } });
    }
    decorate(slide, 7);
    setNotes(slide, 7, PRESENTERS.vijay, "The local stack keeps model, detector, OCR, policy, and audit evidence separate so a reviewer can inspect each contribution.", [
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
      footer1: "",
    });
    await setHeroImage(
      slide,
      calibrationZoom,
      "Zoomed ViT explosives validation-only threshold calibration curve",
      {
        fit: "contain",
        position: { left: 658.17, top: 144, width: 581.6, height: 414 },
      },
    );
    addTextBox(slide, "Explosives validation threshold", { left: 686, top: 74, width: 525, height: 38 }, { fontSize: 25.33, bold: true, alignment: "center" });
    addTextBox(slide, "Blue: precision | Red: recall | Dashed: selected threshold", { left: 676, top: 112, width: 545, height: 30 }, { fontSize: 18.67, color: MUTED, alignment: "center" });
    addTextBox(
      slide,
      "Selected threshold 0.173 | 100% precision and recall",
      { left: 684, top: 579, width: 530, height: 36 },
      { fontSize: 18.67, bold: true, color: BLUE, fill: LIGHT_BLUE, alignment: "center", verticalAlignment: "middle" },
    );
    decorate(slide, 8);
    setNotes(slide, 8, PRESENTERS.vijay, "Model selection and threshold calibration use validation data only; the final 48-image test remains untouched until evaluation.", [
      ABS("scripts/train_and_evaluate.py"),
      ABS("outputs/evaluation/threshold_calibration.png"),
      ABS("outputs/evaluation/thresholds.csv"),
      ABS("outputs/evaluation/evaluation_manifest.json"),
      "https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k",
      "https://huggingface.co/timm/resnet50.a1_in1k",
    ]);
    registerRegions(8, [
      { name: "method-copy", frame: { left: 41.33, top: 36.12, width: 581.33, height: 593.21 } },
      { name: "threshold-plot", frame: { left: 658.17, top: 74, width: 581.6, height: 541 } },
    ]);
  }

  // 09 Literature
  {
    const slide = buildSlide19(presentation, {
      title: titleToken("What prior research contributed", 48),
      body1: metricIntro("WHAT THE RESEARCH CONTRIBUTED", "Prior work guided the model design. Our held-out evaluation provides the performance evidence for this four-label task."),
      stat1: rich("ViT", 48, { bold: true, color: BLUE }),
      stat2: rich("DINO", 48, { bold: true, color: BLUE }),
      stat3: rich("OCR + RULES", 36, { bold: true, color: BLUE }),
      body2: rich("Dosovitskiy et al. (2021): transferable whole-image features.", 21.33),
      body3: rich("Liu et al. (2023): open-vocabulary object localization.", 21.33),
      body4: rich("Tesseract OCR + YAML rules: text evidence and traceable decisions.", 21.33),
      footer1: "",
    });
    decorate(slide, 9);
    setNotes(slide, 9, PRESENTERS.myetchae, "The literature justifies the selected mechanisms, but it does not transfer benchmark performance to our dataset or policy task.", [
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
    const slide = buildSlide20(presentation, {
      title: titleToken("Better accuracy came with\nrecall and speed trade-offs", 46),
      body1: chartCard("SAFE CLASS", "100% precision on the held-out test. Our goal was above 98%."),
      body2: chartCard("RESTRICTED RECALL", "Firearms 91.7% · explosives 91.7%. Our goal was above 95% for each class."),
      body3: chartCard("CLASSIFIER-PATH P95", "ViT 71.5 ms · ResNet 40.1 ms. Full-pipeline timing remains unknown."),
      footer1: "",
    });
    configureGridChart(slide, {
      categories: ["ViT", "ResNet-50"],
      series1: { name: "Held-out macro F1", values: [0.97913, 0.87315], fill: BLUE },
      series2: { name: "", values: [0, 0], fill: CYAN },
      direction: "column",
      max: 1.05,
      majorUnit: 0.2,
      legend: false,
      showDataLabels: false,
      valueFormat: "0.0%",
    });
    addTextBox(slide, "Held-out macro F1 | 48 images", { left: 108, top: 146, width: 468, height: 30 }, { fontSize: 20.67, bold: true, color: BLUE, alignment: "center" });
    addTextBox(slide, "97.9%", { left: 151, top: 205, width: 104, height: 30 }, { fontSize: 20.67, bold: true, color: WHITE, alignment: "center" });
    addTextBox(slide, "87.3%", { left: 370, top: 255, width: 104, height: 30 }, { fontSize: 20.67, bold: true, color: WHITE, alignment: "center" });
    decorate(slide, 10);
    setNotes(slide, 10, PRESENTERS.myetchae, "ViT leads the ResNet baseline on the held-out test, but firearms and explosives fall below our 95 percent recall goal, and ViT exceeds the 50 ms classifier-path latency budget. Full end-to-end p95 remains unmeasured.", [
      ABS("Capstone Project Idea - Ad Safety.pdf"),
      ABS("outputs/evaluation/metrics.json"),
      ABS("outputs/evaluation/model_comparison.csv"),
      ABS("outputs/evaluation/model_comparison.png"),
      ABS("outputs/evaluation/latency.json"),
      ABS("outputs/evaluation/thresholds.csv"),
    ]);
    registerRegions(10, [
      { name: "title-and-chart", frame: { left: 41.33, top: 36.12, width: 580, height: 647 } },
      { name: "result-summary", frame: { left: 657.68, top: 41.33, width: 580.99, height: 590.67 } },
    ]);
  }

  // 11 Class and Failure Analysis
  {
    const slide = buildSlide08(presentation, {
      title: titleToken("One error is why a reviewer\nstays in control", 46),
      body1: splitBody(
        "THE FORMAL ERROR",
        "A grenade labeled explosives was classified as a firearm.\n\nNeither score crossed its class-specific block threshold, so the policy returned REVIEW.\n\nThe optional detector still found restricted-object cues.",
      ),
      footer1: "",
    });
    await setHeroImage(slide, ASSETS.formalFailure, "Formal test grenade that ViT classified as firearms", {
      fit: "contain",
      position: { left: 700, top: 118, width: 470, height: 470 },
    });
    addTextBox(slide, "47 / 48", { left: 64, top: 498, width: 190, height: 62 }, { fontSize: 46, bold: true, color: BLUE });
    addTextBox(slide, "labels correct", { left: 220, top: 514, width: 210, height: 34 }, { fontSize: 22, color: MUTED });
    addTextBox(slide, "Truth: explosives | Model: firearms | Policy: REVIEW", { left: 670, top: 592, width: 540, height: 32 }, { fontSize: 18.67, bold: true, color: RED, alignment: "center" });
    decorate(slide, 11);
    setNotes(slide, 11, PRESENTERS.myetchae, "The only wrong class label went to REVIEW, but it still exposes fragile separation between firearms and explosives.", [
      ABS("outputs/evaluation/per_class_metrics.csv"),
      ABS("outputs/evaluation/failure_cases.csv"),
      ABS("outputs/evaluation/confusion_matrix_vit.png"),
      ASSETS.formalFailure,
    ]);
    registerRegions(11, [
      { name: "failure-copy", frame: { left: 41.33, top: 36.12, width: 580, height: 552 } },
      { name: "failure-image", frame: { left: 670, top: 118, width: 540, height: 512 } },
    ]);
  }

  // 12 External Generalization Audit
  {
    const slide = buildSlide08(presentation, {
      title: titleToken("Outside the pilot data,\nperformance fell sharply", 46),
      body1: splitBody("OUT-OF-DOMAIN CHECK | 26 IMAGES", "55.5% classifier macro F1\n90.0% nonweapon false detections\n\nThe images use different layouts, languages, ages, and photographic contexts. This small selected check is a warning, not a formal test set or population estimate."),
      footer1: "",
    });
    const first = await setHeroImage(slide, ASSETS.externalSafeFalse, "Safe historical food advertisement falsely detected as explosive", { fit: "cover", position: { left: 658.17, top: 41.62, width: 226, height: 588.14 } });
    first.position = { left: 658.17, top: 41.62, width: 226, height: 588.14 };
    await addImage(slide, ASSETS.externalExplosiveMiss, "External explosives example classified safe with noisy detector boxes", { left: 900, top: 41.62, width: 339.77, height: 280 }, { fit: "cover" });
    await addImage(slide, ASSETS.externalFinancial, "External financial promotion with a false explosive detector box", { left: 900, top: 349.76, width: 339.77, height: 280 }, { fit: "cover" });
    addTextBox(slide, "SAFE → BLOCK", { left: 669, top: 582, width: 204, height: 36 }, { fontSize: 21.33, bold: true, color: RED, fill: WHITE, verticalAlignment: "middle", alignment: "center" });
    addTextBox(slide, "EXPLOSIVES → SAFE", { left: 911, top: 276, width: 318, height: 36 }, { fontSize: 21.33, bold: true, color: RED, fill: WHITE, verticalAlignment: "middle", alignment: "center" });
    addTextBox(slide, "FINANCE OK · FALSE BOX", { left: 911, top: 584, width: 318, height: 36 }, { fontSize: 21.33, bold: true, color: RED, fill: WHITE, verticalAlignment: "middle", alignment: "center" });
    decorate(slide, 12);
    setNotes(slide, 12, PRESENTERS.myetchae, "In the small selected Wikimedia diagnostic, macro F1 falls to 55.5 percent and 90 percent of nonweapon images receive a false detector signal.", [
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
    const slide = presentation.slides.add();
    addTextBox(slide, "From upload to audit in one workspace", { left: 64, top: 58, width: 900, height: 72 }, { fontSize: 46, bold: true });
    addPanel(slide, { left: 1040, top: 65, width: 176, height: 38 }, { fill: LIGHT_BLUE, line: { style: "solid", fill: CYAN, width: 1 } });
    addTextBox(slide, "TEAM DEMO", { left: 1050, top: 72, width: 156, height: 24 }, { fontSize: 18.67, bold: true, color: BLUE, alignment: "center", verticalAlignment: "middle" });
    await addImage(
      slide,
      ASSETS.appFirearmResult,
      "Browser-validated BLOCK result with policy focus, evidence score, latency, and detector counts",
      { left: 64, top: 150, width: 868, height: 472 },
      { fit: "contain" },
    );
    const demoSteps = [
      ["1  Run the case", "Swarnaditya shares the screen."],
      ["2  Review evidence", "Vijay and Myetchae explain scores and policy."],
      ["3  Export the record", "Bickramjit explains the audit trail."],
    ];
    demoSteps.forEach(([title, body], index) => {
      const top = 164 + index * 138;
      addTextBox(slide, title, { left: 974, top, width: 242, height: 36 }, { fontSize: 22.67, bold: true, color: index === 0 ? BLUE : BLACK });
      addTextBox(slide, body, { left: 974, top: top + 42, width: 242, height: 64 }, { fontSize: 19.5, color: MUTED });
      if (index < demoSteps.length - 1) addPanel(slide, { left: 974, top: top + 119, width: 242, height: 2 }, { fill: RULE, line: { style: "solid", fill: RULE, width: 0 } });
    });
    addTextBox(slide, "Session charts summarize today's cases, not production performance.", { left: 974, top: 572, width: 242, height: 52 }, { fontSize: 18.67, bold: true, color: RED });
    decorate(slide, 13);
    setNotes(slide, 13, PRESENTERS.teamDemo, "All four team members demonstrate the workflow while the decision, evidence, timing, and audit record remain visible together.", [
      ABS("app.py"),
      ABS("api.py"),
      ABS("outputs/app/browser_validation.json"),
      ABS("outputs/app/app_firearm_result.png"),
      ABS("src/ad_safety/inference.py"),
    ]);
    registerRegions(13, [
      { name: "title", frame: { left: 64, top: 58, width: 1152, height: 72 } },
      { name: "validated-result", frame: { left: 64, top: 150, width: 868, height: 472 } },
      { name: "demo-roles", frame: { left: 974, top: 164, width: 242, height: 460 } },
    ]);
  }

  // 14 Risks and Controls
  {
    const slide = buildSlide19(presentation, {
      title: titleToken("What must change before a pilot", 48),
      body1: metricIntro("", ""),
      stat1: rich("DATA", 46, { bold: true, color: RED }),
      stat2: rich("DETECTOR", 40, { bold: true, color: RED }),
      stat3: rich("OPERATIONS", 38, { bold: true, color: RED }),
      body2: rich("Collect diverse real campaigns and use independent labels instead of source-linked shortcuts.", 21.33),
      body3: rich("A 90% false-positive rate on nonweapon images requires local box labels, recalibration, and no detector-only block.", 21.33),
      body4: rich("Measure end-to-end p95,\nconcurrency, CPU/GPU load,\nMPS acceleration, and memory.", 21.33),
      footer1: "",
    });
    addTextBox(slide, "THREE GAPS", { left: 41.33, top: 124, width: 220, height: 30 }, { fontSize: 24, bold: true, color: BLUE });
    addTextBox(
      slide,
      "Before a pilot, we need to address source-biased data, noisy detections, and missing full-pipeline performance measurements.",
      { left: 41.33, top: 154, width: 1170, height: 54 },
      { fontSize: 22.67 },
    );
    decorate(slide, 14);
    setNotes(slide, 14, PRESENTERS.bickramjit, "The main risks are source confounding, an uncalibrated detector, and classifier-only latency evidence. Full end-to-end p95, concurrent throughput, CPU and GPU load, MPS acceleration, and memory use were not measured; each needs an explicit control before a trial.", [
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
      footer1: "",
    });
    repairGridTimeline(slide);
    decorate(slide, 15);
    setNotes(slide, 15, PRESENTERS.bickramjit, "This 10-week plan uses explicit evidence, quality, and capacity gates before any human-reviewed pilot.", [
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
    cross_renderer_clearance: {
      typeface: FONT,
      slide_4_title_left: SAFE_LEFT,
      slide_counter_top: 11,
      slide_counter_right: 42,
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
