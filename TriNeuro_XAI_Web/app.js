const $ = (id) => document.getElementById(id);
const BLACK_IMAGE =
  "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1200 760'%3E%3Crect width='1200' height='760' fill='%23000000'/%3E%3C/svg%3E";

const state = {
  result: null,
  history: JSON.parse(localStorage.getItem("trineuroHistory") || "[]"),
};

const fileFields = [
  ["primary", "Primary MRI", "primaryFile"],
  ["t1n", "T1n", "t1nFile"],
  ["t1c", "T1c", "t1cFile"],
  ["t2w", "T2w", "t2wFile"],
  ["t2f", "T2F / FLAIR", "t2fFile"],
];

function apiEndpoint() {
  return "http://localhost:8000/predict";
}

function setView(viewId) {
  document.querySelectorAll(".view").forEach((view) => view.classList.toggle("active", view.id === viewId));
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === viewId);
  });
  if (viewId === "visualizationView") renderAllSlices();
}

function activeModalities() {
  return fileFields
    .filter(([, , id]) => $(id)?.files?.[0])
    .map(([key, label, id]) => ({
      key,
      label,
      filename: $(id).files[0].name,
    }));
}

function formatPct(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "--";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function saveHistory(item) {
  const existing = state.history.filter((h) => h.patientId !== item.patientId);
  state.history = [item, ...existing].slice(0, 24);
  localStorage.setItem("trineuroHistory", JSON.stringify(state.history));
  renderHistory();
}

function renderHistory() {
  const list = $("historyList");
  const select = $("historySelect");
  list.innerHTML = "";
  select.innerHTML = `<option value="">No previous patient selected</option>`;

  if (!state.history.length) {
    list.innerHTML = `<div class="history-item"><strong>No saved patients</strong><small>Completed analyses are saved in this browser.</small></div>`;
    return;
  }

  state.history.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.patientId;
    option.textContent = `${item.patientId} - ${item.finalLabel}`;
    select.appendChild(option);

    const node = document.createElement("button");
    node.className = "history-item";
    node.type = "button";
    node.innerHTML = `
      <strong>${item.patientId}</strong>
      <small>${item.finalLabel} - ${item.date}</small>
      <small>${item.modalities.map((m) => m.label).join(", ") || "No modality metadata"}</small>
    `;
    node.addEventListener("click", () => {
      $("patientId").value = item.patientId;
      $("historySelect").value = item.patientId;
      setReportText(item.report);
      renderModalityList(item.modalities);
      setView("analysisView");
    });
    list.appendChild(node);
  });
}

function selectedPriorContext() {
  const selected = $("historySelect").value;
  if (!selected) return null;
  return state.history.find((item) => item.patientId === selected) || null;
}

function renderModalityList(modalities) {
  const list = $("modalityList");
  if (!list) return;
  list.innerHTML = "";
  if (!modalities.length) {
    list.innerHTML = "<li>No MRI uploaded yet.</li>";
    return;
  }
  modalities.forEach((m) => {
    const li = document.createElement("li");
    li.textContent = `${m.label}: ${m.filename}`;
    list.appendChild(li);
  });
}

function drawBlackSlice() {
  drawBlackCanvas("mriCanvas", "sliceLabel");
}

function drawBlackXaiSlice() {
  drawBlackCanvas("xaiCanvas", "xaiSliceLabel");
}

function drawBlackCanvas(canvasId, labelId) {
  const canvas = $(canvasId);
  if (!canvas) return;
  resizeCanvasForDisplay(canvas);
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, w, h);
  if (labelId && $(labelId)) $(labelId).textContent = "";
}

function resizeCanvasForDisplay(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function currentSliceImages(kind = "mri") {
  if (!state.result) return null;
  const plane = $("planeSelect").value;
  if (kind === "attention") return state.result.attentionSlices?.[plane] || null;
  if (kind === "gradcam") return state.result.gradcamSlices?.[plane] || null;
  if (kind === "segmentation") return state.result.segmentationSlices?.[plane] || null;
  return state.result.mriSlices?.[plane] || null;
}

function drawImageToCanvas(src, label, canvasId = "mriCanvas", labelId = "sliceLabel") {
  const img = new Image();
  img.onload = () => {
    const canvas = $(canvasId);
    resizeCanvasForDisplay(canvas);
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
    const w = img.width * scale;
    const h = img.height * scale;
    ctx.drawImage(img, (canvas.width - w) / 2, (canvas.height - h) / 2, w, h);
    if (labelId && $(labelId)) $(labelId).textContent = label;
  };
  img.src = src;
}

function renderReturnedSlice() {
  const images = currentSliceImages("mri");
  if (!images || !images.length) {
    drawBlackSlice();
    return;
  }
  const slider = $("sliceSlider");
  slider.max = String(images.length - 1);
  if (Number(slider.value) > images.length - 1) slider.value = String(Math.floor(images.length / 2));
  const idx = Number(slider.value);
  drawImageToCanvas(images[idx], `${$("planeSelect").value} slice ${idx + 1}/${images.length}`);
}

function renderXaiSlice() {
  if (!state.result) {
    drawBlackXaiSlice();
    $("xaiCaption").textContent = "Run the API to generate the segmentation mask from the uploaded MRI.";
    return;
  }

  const images = currentSliceImages("segmentation");
  if (images && images.length) {
    const idx = Math.min(Number($("sliceSlider").value), images.length - 1);
    drawImageToCanvas(
      images[idx],
      `${$("planeSelect").value} slice ${idx + 1}/${images.length}`,
      "xaiCanvas",
      "xaiSliceLabel",
    );
    $("xaiCaption").textContent = "Segmentation overlay generated on the same slice grid as the MRI preview.";
    return;
  }

  if (state.result.xaiPanelImage) {
    drawBlackXaiSlice();
    $("xaiCaption").textContent = "No segmentation mask was generated for this case.";
    return;
  }

  drawBlackXaiSlice();
  $("xaiCaption").textContent = "No segmentation mask was generated for this case.";
}

function renderAllSlices() {
  renderReturnedSlice();
  renderXaiSlice();
}

function safeFilenamePart(value) {
  return String(value || "UPLOADED_CASE")
    .trim()
    .replace(/[^A-Za-z0-9_-]+/g, "_")
    .replace(/^_+|_+$/g, "") || "UPLOADED_CASE";
}

function syncReportBoxWidth() {
  const report = $("reportPreview");
  const card = document.querySelector(".report-card");
  if (!report || !card) return;
  const cardStyle = window.getComputedStyle(card);
  const available =
    card.clientWidth -
    parseFloat(cardStyle.paddingLeft || "0") -
    parseFloat(cardStyle.paddingRight || "0");
  if (available > 240) {
    report.style.width = `${available}px`;
    report.style.maxWidth = `${available}px`;
  }
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatReportForDisplay(report) {
  const sectionTitles = new Set(["TECHNIQUE:", "COMPARISON:", "FINDINGS:", "IMPRESSION:", "RECOMMENDATIONS:"]);
  return String(report || "")
    .split(/\r?\n/)
    .map((line) => {
      const trimmed = line.trim();
      const escaped = escapeHtml(line);
      if (!trimmed) return "";
      if (/^(Patient|Volume):/i.test(trimmed)) {
        return escaped.replace(/^([^:]+:)/, '<strong class="report-meta-title">$1</strong>');
      }
      if (sectionTitles.has(trimmed.toUpperCase())) {
        return `<strong class="report-section-title">${escapeHtml(trimmed)}</strong>`;
      }
      return escaped;
    })
    .join("\n");
}

function setReportText(report) {
  const text = String(report || "");
  if ($("reportText")) $("reportText").value = text;
  if ($("reportPreview")) $("reportPreview").innerHTML = formatReportForDisplay(text);
}

function setReportLoading(isLoading) {
  const layout = document.querySelector(".report-layout");
  if (!layout) return;
  layout.classList.toggle("loading", Boolean(isLoading));
  const loading = $("reportLoading");
  if (loading) loading.setAttribute("aria-hidden", isLoading ? "false" : "true");
}

function downloadReport() {
  const reportText = $("reportText")?.value || "";
  const patientId = safeFilenamePart($("patientId")?.value || state.result?.clinical?.patient_id || "UPLOADED_CASE");
  const now = new Date();
  const stamp = [
    now.getFullYear(),
    String(now.getMonth() + 1).padStart(2, "0"),
    String(now.getDate()).padStart(2, "0"),
  ].join("");
  const blob = new Blob([reportText], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `TriNeuro_Report_${patientId}_${stamp}.txt`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function appendPriorContext(report, prior) {
  if (!prior) return report;
  return `${report}\n\nPRIOR PATIENT CONTEXT USED:\n- Previous patient/case: ${prior.patientId}\n- Previous automated label: ${prior.finalLabel}\n`;
}

function stripReportHeader(report) {
  return String(report || "").replace(
    /^TRINEURO (?:XAI|AI) MULTI-AGENT REPORT\s*\nMRI MODALITIES USED:\s*\n(?:- .*\n)*\s*/i,
    "",
  );
}

function buildFallbackReport(finalLabel, modalities, clinical = {}) {
  const volume = clinical.segmented_lesion_volume_cm3 ?? clinical.tumor_volume_cm3 ?? "pending";
  const diameter = clinical.max_diameter_mm ?? "pending";
  const axialSpan = clinical.axial_span_mm ?? "pending";
  const location = clinical.hemisphere ?? "pending";
  return `Patient: ${$("patientId").value || "UPLOADED_CASE"}

TECHNIQUE:
Automated TriNeuro AI processing was performed using the uploaded MRI modality set. The workflow includes tumor screening, tumor-type classification when routed to the tumor branch, Grad-CAM++ explainability, seg_model.pth segmentation, and structured report generation.

FINDINGS:
- Automated final class: ${finalLabel}.
- Segmented lesion volume: ${volume} cm3.
- Maximum diameter: ${diameter} mm.
- Axial span: ${axialSpan} mm.
- Location/laterality: ${location}.

IMPRESSION:
- TriNeuro AI generated an automated decision-support interpretation based on classification and segmentation-derived features.
- Grad-CAM++ is provided to visualize the classifier focus region.

RECOMMENDATIONS:
This report is generated by the report workflow and must be reviewed by a radiologist before clinical use.`;
}

function normalizeApiResult(payload, modalities) {
  const tumor = payload.tumor_result || {};
  const type = payload.tumor_type_result || null;
  const clinical = payload.clinical_findings || {};
  const clinicalLabel = String(clinical.classification_label || "").toLowerCase();
  const reportLooksNoTumor = /no (measurable )?(intracranial )?(lesion|tumou?r|mass)|no evidence of/i.test(payload.report || "");
  const isNoTumor = clinicalLabel === "no tumor" || tumor.label === "no_tumor" || reportLooksNoTumor;
  const tumorLabel = isNoTumor ? "no tumor" : tumor.label || clinical.classification_label || "pending";
  const typeLabel = isNoTumor ? "not required" : type?.label || (tumorLabel === "tumor" ? "tumor type pending" : "not required");
  const finalLabel = isNoTumor ? "no tumor" : type?.label || tumorLabel || clinical.classification_label || "unknown";
  const baseReport = payload.report || buildFallbackReport(finalLabel, modalities, clinical);
  const report = appendPriorContext(stripReportHeader(baseReport), selectedPriorContext());
  const segmentationSource = isNoTumor ? "not needed" : payload.segmentation_source || "No segmentation source returned";
  const preprocessingInfo = payload.preprocessing_info || {};

  return {
    tumorLabel,
    typeLabel,
    finalLabel,
    segmentationStatus: segmentationSource === "not needed" ? "Not needed" : payload.segmentation_source ? "Available" : "Not generated",
    segmentationSource,
    preprocessingInfo,
    report,
    reportSource: payload.report_source || "api",
    xaiPanelImage: payload.xai_panel_png || null,
    fullPanelImage: payload.xai_full_panel_png || payload.full_panel_png || payload.xai_panel_png || null,
    mriColumnImage: payload.mri_column_png || null,
    mriSlices: payload.mri_slices || null,
    attentionSlices: payload.attention_slices || null,
    gradcamSlices: payload.gradcam_slices || null,
    segmentationSlices: payload.segmentation_slices || null,
    clinical,
  };
}

function reportSourceLabel(source) {
  const labels = {
    openai_gpt_4_1: "OpenAI gpt-4.1",
    structured_fallback_missing_openai_key: "Fallback: missing OpenAI key",
    structured_fallback_after_openai_error: "Fallback: OpenAI error",
    structured_fallback_after_empty_openai_report: "Fallback: empty OpenAI report",
    structured_fast_mode: "Fast structured report",
    structured_report: "Structured report",
    structured_fallback: "Structured fallback",
  };
  return labels[source] || source || "Report source unknown";
}

function applyResult(result, modalities) {
  state.result = result;
  $("tumorLabel").textContent = result.tumorLabel.replaceAll("_", " ");
  $("typeLabel").textContent = result.typeLabel.replaceAll("_", " ");
  $("segmentationStatus").textContent = result.segmentationStatus;
  $("segmentationSource").textContent = result.segmentationSource;
  setReportText(result.report);
  setReportLoading(false);
  syncReportBoxWidth();
  renderModalityList(modalities);

  $("fullPanelImage").src = result.fullPanelImage || BLACK_IMAGE;
  renderAllSlices();

  saveHistory({
    patientId: $("patientId").value || `CASE_${Date.now()}`,
    finalLabel: result.finalLabel,
    report: result.report,
    modalities,
    date: new Date().toLocaleString(),
  });
}

async function runPipeline() {
  const modalities = activeModalities();
  const primary = $("primaryFile").files[0];
  if (!primary) {
    $("statusText").textContent = "Please upload at least the primary MRI.";
    return;
  }

  $("runButton").disabled = true;
  const progressMessages = [
    "Uploading MRI and preparing fast previews...",
    "Running tumor screening model...",
    "Checking tumor branch and segmentation if needed...",
    "Building report and compressed visualization previews...",
  ];
  let progressIndex = 0;
  $("statusText").textContent = progressMessages[progressIndex];
  $("apiStatusLabel").textContent = "Running";
  setReportText("");
  setReportLoading(true);
  syncReportBoxWidth();
  renderModalityList(modalities);
  state.result = null;
  $("fullPanelImage").src = BLACK_IMAGE;
  drawBlackSlice();
  drawBlackXaiSlice();

  const progressTimer = window.setInterval(() => {
    progressIndex = Math.min(progressIndex + 1, progressMessages.length - 1);
    $("statusText").textContent = progressMessages[progressIndex];
  }, 3500);

  const form = new FormData();
  form.append("primary", primary);
  form.append("patient_id", $("patientId").value || "UPLOADED_CASE");
  form.append("force_tumor_branch", "false");
  form.append("fast_mode", "true");
  form.append("include_full_panel", "true");
  form.append("include_xai_slices", "false");
  if ($("t1nFile").files[0]) form.append("t1n", $("t1nFile").files[0]);
  if ($("t1cFile").files[0]) form.append("t1c", $("t1cFile").files[0]);
  if ($("t2wFile").files[0]) form.append("t2w", $("t2wFile").files[0]);
  if ($("t2fFile").files[0]) form.append("t2f", $("t2fFile").files[0]);

  try {
    const response = await fetch(apiEndpoint(), { method: "POST", body: form });
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || `HTTP ${response.status}`);
    }
    const payload = await response.json();
    const result = normalizeApiResult(payload, modalities);
    applyResult(result, modalities);
    const preprocessed = Object.values(result.preprocessingInfo || {}).filter((item) => item.status === "applied");
    const seconds = payload.performance?.slice_previews || payload.performance?.report || null;
    const timeText = seconds ? ` (${seconds}s)` : "";
    $("statusText").textContent = preprocessed.length
      ? `Pipeline completed${timeText}. IXI preprocessing applied to ${preprocessed.length} non-BraTS file(s).`
      : `Pipeline completed${timeText}. Report and fast visualizations are ready.`;
    $("apiStatusLabel").textContent = "API Connected";
  } catch (error) {
    setReportLoading(false);
    $("statusText").textContent = `API call failed: ${error.message.slice(0, 220)}`;
    $("apiStatusLabel").textContent = "API Error";
    console.warn(error);
  } finally {
    window.clearInterval(progressTimer);
    $("runButton").disabled = false;
  }
}

function resetCaseInputs() {
  state.result = null;
  fileFields.forEach(([, , id]) => {
    const input = $(id);
    if (input) input.value = "";
  });
  $("patientId").value = "";
  $("historySelect").value = "";
  $("tumorLabel").textContent = "--";
  $("typeLabel").textContent = "--";
  $("segmentationStatus").textContent = "--";
  $("segmentationSource").textContent = "Source --";
  setReportText("The final report will appear here after running the TriNeuro AI pipeline.");
  setReportLoading(false);
  syncReportBoxWidth();
  $("fullPanelImage").src = BLACK_IMAGE;
  $("xaiCaption").textContent = "Run the API to generate the segmentation mask from the uploaded MRI.";
  $("sliceSlider").value = "52";
  renderModalityList([]);
  drawBlackSlice();
  drawBlackXaiSlice();
  $("statusText").textContent = "Inputs cleared. Upload a new primary MRI to start.";
}

function bindEvents() {
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.addEventListener("click", () => setView(button.dataset.view));
  });

  $("runButton").addEventListener("click", runPipeline);
  $("downloadReport").addEventListener("click", downloadReport);
  $("resetInputs").addEventListener("click", resetCaseInputs);
  $("clearHistory").addEventListener("click", () => {
    state.history = [];
    localStorage.removeItem("trineuroHistory");
    renderHistory();
  });
  $("sliceSlider").addEventListener("input", renderAllSlices);
  $("planeSelect").addEventListener("change", () => {
    $("sliceSlider").value = "15";
    renderAllSlices();
  });
  fileFields.forEach(([, , id]) => {
    $(id).addEventListener("change", () => renderModalityList(activeModalities()));
  });
}

bindEvents();
renderHistory();
$("fullPanelImage").src = BLACK_IMAGE;
drawBlackSlice();
drawBlackXaiSlice();
syncReportBoxWidth();
window.addEventListener("resize", () => {
  syncReportBoxWidth();
  renderAllSlices();
});
