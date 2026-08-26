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
    `Key message: ${keyMessage}`,
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
    contains_sources_block: true,
    all_local_sources_portable: normalizedSources.every(
      (source) => /^https?:\/\//i.test(source) || source.startsWith("Project-relative: "),
    ),
  });
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
      title: titleToken("MS_DSP_462 · MANAGEMENT REVIEW", 18.67),
      body1: splitBody("", ""),
      footer1: "1",
    });
    await setHeroImage(slide, ASSETS.appShell, "Browser-validated Ad Safety Studio analysis workspace", {
      fit: "cover",
      position: { left: 658.17, top: 112, width: 581.6, height: 327.15 },
    });
    addTextBox(slide, "Ad safety\nat a decision\npoint", { left: 41.33, top: 112, width: 565, height: 228 }, { fontSize: 68, bold: false });
    addTextBox(slide, "0.979 formal macro F1\n0.555 external macro F1", { left: 41.33, top: 365, width: 565, height: 82 }, { fontSize: 28, color: BLACK, bold: true });
    addPanel(slide, { left: 41.33, top: 485, width: 565, height: 106 }, { fill: LIGHT_BLUE, line: { style: "solid", fill: CYAN, width: 1 } });
    addTextBox(slide, "Decision\nAdvance the bounded demo. Do not approve autonomous deployment.", { left: 61, top: 501, width: 525, height: 77 }, { fontSize: 21.33, bold: true });
    addPanel(slide, { left: 658.17, top: 465, width: 581.6, height: 126 }, { fill: LIGHT_BLUE, line: { style: "solid", fill: CYAN, width: 1 } });
    addTextBox(slide, "LOCAL PILOT", { left: 678, top: 485, width: 164, height: 28 }, { fontSize: 17.33, bold: true, color: BLUE, alignment: "center" });
    addTextBox(slide, "AUDIT-READY", { left: 864, top: 485, width: 164, height: 28 }, { fontSize: 17.33, bold: true, color: BLUE, alignment: "center" });
    addTextBox(slide, "HUMAN-GATED", { left: 1050, top: 485, width: 164, height: 28 }, { fontSize: 17.33, bold: true, color: BLUE, alignment: "center" });
    addTextBox(slide, "No cloud upload", { left: 678, top: 527, width: 164, height: 24 }, { fontSize: 14.67, color: MUTED, alignment: "center" });
    addTextBox(slide, "Exact run record", { left: 864, top: 527, width: 164, height: 24 }, { fontSize: 14.67, color: MUTED, alignment: "center" });
    addTextBox(slide, "Review stays human", { left: 1050, top: 527, width: 164, height: 24 }, { fontSize: 14.67, color: MUTED, alignment: "center" });
    decorate(slide, 1, "Executive Summary", PRESENTERS.swarnaditya);
    setNotes(slide, 1, PRESENTERS.swarnaditya, "The formal pilot is strong, but the external audit shows that the present evidence is not production-grade.", [
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
      title: titleToken("Four presenters,\none evidence chain", 48),
      body1: chartCard("Swarnaditya Maitra", "Executive summary · problem · formal results · MVP gate"),
      body2: chartCard("Vijay Agnihotri", "Team · data profile · EDA · class and failure analysis"),
      body3: chartCard("Myetchae Thu + Bickramjit Basu", "Policy · architecture · audit · risks | method · literature · demo"),
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
    setNotes(slide, 2, PRESENTERS.vijay, "Every team member owns a visible presentation section; the sequence follows one evidence chain from problem to MVP gate.", [
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
      title: titleToken("Manual review fails on ambiguous creatives", 48),
      body1: metricIntro("BUSINESS PROBLEM", "An ad image can contain obvious objects, subtle text, or misleading context. A review queue needs fast, consistent triage while preserving a human gate for uncertain cases."),
      stat1: rich("SPEED", 42, { bold: true, color: BLUE }),
      stat2: rich("CONSISTENCY", 38, { bold: true, color: BLUE }),
      stat3: rich("CONTEXT", 42, { bold: true, color: BLUE }),
      body2: rich("Prioritize risky creatives before they reach broad distribution.", 21.33),
      body3: rich("Apply the same evidence path to every uploaded image.", 21.33),
      body4: rich("Escalate questions that pixels alone cannot answer.", 21.33),
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
      label1: rich("MODELLED", 24, { bold: true, color: BLUE }),
      label2: rich("TRIAGED", 24, { bold: true, color: BLUE }),
      label3: rich("HUMAN / FUTURE", 24, { bold: true, color: RED }),
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
      body1: chartCard("48 / 12 / 12 per class", "Train · validation · untouched test. Total pilot size: 288 images."),
      body2: chartCard("215 source groups", "Campaign and source groups stay inside one split; exact hashes do not overlap."),
      body3: chartCard("Source-confounded", "Safe and finance are synthetic; both weapon classes come from Weapons Set 1."),
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
      title: titleToken("Source style\npredicts labels", 48),
      body1: splitBody("WHAT THE CONTACT SHEET REVEALS", "Safe: generated product scenes\nWeapons: isolated objects\nFinance: repeated synthetic templates\n\nRisk: the classifier may learn where the image came from instead of what policy concept it contains."),
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
      title: titleToken("Separate evidence streams before policy", 48),
      body1: splitBody("1 · SEE", "Frozen ViT scores the whole image. Grounding DINO Tiny proposes weapon and explosive object evidence."),
      body2: splitBody("2 · READ", "Tesseract extracts visible text. Classifier, detector, and OCR evidence remain separately inspectable."),
      body3: splitBody("3 · DECIDE", "A versioned YAML policy returns APPROVE, REVIEW, or BLOCK with reasons and an audit trail."),
      label1: rich("INPUT + VISION", 24, { bold: true, color: BLUE }),
      label2: rich("EVIDENCE", 24, { bold: true, color: BLUE }),
      label3: rich("VERDICT", 24, { bold: true, color: BLUE }),
      footer1: "7",
    });
    repairGridTimeline(slide);
    const cards = slide.shapes.items.filter((shape) => shape.name?.startsWith("Rounded-Rectangle"));
    if (cards.length >= 3) {
      slide.shapes.connect(cards[0], cards[1], { kind: "straight", fromSide: "right", toSide: "left", line: { style: "solid", fill: BLUE, width: 2 }, tail: { type: "arrow", width: "med", length: "med" } });
      slide.shapes.connect(cards[1], cards[2], { kind: "straight", fromSide: "right", toSide: "left", line: { style: "solid", fill: BLUE, width: 2 }, tail: { type: "arrow", width: "med", length: "med" } });
    }
    decorate(slide, 7, "Architecture", PRESENTERS.myetchae);
    setNotes(slide, 7, PRESENTERS.myetchae, "The architecture avoids collapsing evidence into one opaque score; the policy layer can explain which signal changed a verdict.", [
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
      title: titleToken("Test evidence\nstayed untouched", 48),
      body1: splitBody("METHOD", "1  Freeze ViT and ResNet-50 backbones\n\n2  Fit logistic heads on train embeddings\n\n3  Tune thresholds on validation only\n\n4  Open the 48-image test once\n\n5  Benchmark warm batch-1 CPU"),
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
      "RECALL red · PRECISION blue · selected t=0.173",
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
      title: titleToken("Research supports components, not claims", 48),
      body1: metricIntro("LITERATURE BOUNDARY", "Papers and model cards explain why each component is plausible. Only this project’s saved evaluation can support claims about this pilot."),
      stat1: rich("ViT", 48, { bold: true, color: BLUE }),
      stat2: rich("DINO", 48, { bold: true, color: BLUE }),
      stat3: rich("OCR + RULES", 36, { bold: true, color: BLUE }),
      body2: rich("Patch attention supplies a transferable whole-image representation.", 21.33),
      body3: rich("Text-conditioned detection localizes open-vocabulary object cues.", 21.33),
      body4: rich("Visible words add evidence; explicit policy rules keep the final action inspectable.", 21.33),
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
      "Classifier p95 · ViT / ResNet · batch 1\nViT >50 ms; end-to-end p95 unassessed",
      { fill: PANEL },
    );
    addTextBox(
      slide,
      "Safe precision 1.000: PASS (>0.98)\nSafe false-flag rate: 0 (1 − recall)\nArgmax restricted recall: 0.97222\nThreshold recall mean / min: 0.94444 / 0.91667\n>0.95 per-class target: FAIL · calibration floor ≥0.90",
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
      title: titleToken("One grenade crossed\nthe class boundary", 48),
      body1: chartCard("FORMAL ERROR", "One explosives image was predicted as firearms. All other 47 labels were correct."),
      body2: chartCard("NO AUTO-BLOCK", "Neither class-specific block threshold fired for that image; a restricted error still needs review."),
      body3: chartCard("WEAKEST CLASS", "ViT F1 0.957 · ResNet F1 0.750.\nSeparation remains fragile."),
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
    await addImage(slide, ASSETS.formalFailure, "Formal test grenade that ViT classified as firearms", { left: 1080, top: 500, width: 132, height: 90 }, { fit: "cover" });
    decorate(slide, 11, "Class and Failure Analysis", PRESENTERS.vijay);
    setNotes(slide, 11, PRESENTERS.vijay, "The single formal failure is restricted-to-restricted, but it exposes weak explosives separation and did not meet either automatic block threshold.", [
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
      title: titleToken("External audit\nbreaks confidence", 48),
      body1: splitBody("WIKIMEDIA DIAGNOSTIC · n=26", "0.55489 classifier macro F1\n0.90 detector nonweapon FPR\n\nHistorical layouts, languages, and photographic context differ sharply from the pilot sources. This is a diagnostic spot check, not a formal test set."),
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
    setNotes(slide, 12, PRESENTERS.myetchae, "The 26-image Wikimedia diagnostic exposes domain shift: classifier macro F1 falls to 0.55489 and the detector flags 90 percent of nonweapon images.", [
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
      title: titleToken("One workspace carries a case from upload to audit", 46),
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
      ["SCORE PROVENANCE", "Raw and fused policy evidence remain separate."],
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
    addTextBox(slide, "Session analytics describe review,\nnot production or population evidence.", { left: 845, top: 568, width: 373.66, height: 52 }, { fontSize: 21.33, bold: true, color: RED, verticalAlignment: "middle" });
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
      title: titleToken("Controls needed before an MVP trial", 48),
      body1: metricIntro("GO / NO-GO", "Current controls limit damage, but they do not remove the uncertainty created by synthetic sources, detector noise, and incomplete end-to-end latency evidence."),
      stat1: rich("DATA", 46, { bold: true, color: RED }),
      stat2: rich("DETECTOR", 40, { bold: true, color: RED }),
      stat3: rich("OPERATIONS", 38, { bold: true, color: RED }),
      body2: rich("Replace source-linked shortcuts with diverse, independently annotated real campaigns.", 21.33),
      body3: rich("A 0.90 nonweapon FPR requires local box labels, recalibration, and no automatic detector block.", 21.33),
      body4: rich("Unmeasured: end-to-end p95,\nconcurrency, and MPS use.\nRetain telemetry and escalation.", 21.33),
      footer1: "14",
    });
    decorate(slide, 14, "Risks and Controls", PRESENTERS.myetchae);
    setNotes(slide, 14, PRESENTERS.myetchae, "The main risks are source confounding, an uncalibrated detector, and classifier-only latency evidence. Full end-to-end p95, concurrent throughput, and MPS CPU/GPU/RAM use were not measured; each needs an explicit control before a trial.", [
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
      title: titleToken("A recommended 10-week path to MVP", 48),
      body1: splitBody("WEEKS 1–3 · EVIDENCE", "Resource: data lead, policy reviewer, 2 annotators. Curate ≥1,000 real ads; lock a campaign-grouped independent holdout."),
      body2: splitBody("WEEKS 4–6 · CALIBRATE", "Resource: 1 ML engineer, reviewer, CPU/GPU test capacity. Gate: >0.95 recall per restricted class and <2% Safe crossings."),
      body3: splitBody("WEEKS 7–10 · PILOT", "Resource: product lead, app/MLOps engineer, reviewer rotation. Scale gate: 10 concurrent requests; measure E2E p95 and CPU/GPU/RAM."),
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

  const reopened = await PresentationFile.importPptx(await FileBlob.load(PPTX_PATH));
  const reopenedCount = reopened.slides.items.length;
  if (reopenedCount !== 15) throw new Error(`PPTX reopen check found ${reopenedCount} slides, expected 15.`);
  await writeBlob(path.join(OUT, "reopened_montage.webp"), await reopened.export({ format: "webp", montage: true, scale: 0.5 }));

  const plan = [
    "Communication job: By the end, evaluators and management stakeholders should understand that the prototype demonstrates credible ad triage on a bounded pilot, while MVP approval requires broader real-ad evidence, detector recalibration, and a full-pipeline operating decision.",
    "Narrative: promise -> boundary -> evidence -> formal result -> external break -> controlled MVP gate.",
    "Design system: Codex Grid 1280x720, white canvas, black ink, gray panels, restrained blue accent, Arial.",
    `Selected layouts: ${SELECTED_TEMPLATE_IDS.join(", ")}`,
  ].join("\n");
  await fsp.writeFile(path.join(BUILD, "slide-plan.txt"), plan);
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
