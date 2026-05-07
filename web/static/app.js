const bootstrap = window.__BOOTSTRAP__ || {};
const demoSequences = bootstrap.demoSequences || [];

let uploadSessionId = crypto.randomUUID();
let sequenceSessionId = crypto.randomUUID();
let currentScenario = demoSequences[0] || null;
let currentFrameIndex = 0;
let playTimer = null;
let evalActiveSplit = "valid";
let evalCache = { valid: null, test: null };

const uploadInput = document.getElementById("image-input");
const uploadPreview = document.getElementById("upload-preview");
const posePreview = document.getElementById("pose-preview");
const poseStatus = document.getElementById("pose-status");
const uploadCommand = document.getElementById("upload-command");
const uploadCommandDesc = document.getElementById("upload-command-desc");
const uploadLabel = document.getElementById("upload-label");
const uploadConfidence = document.getElementById("upload-confidence");
const uploadState = document.getElementById("upload-state");
const uploadLatency = document.getElementById("upload-latency");
const uploadWindow = document.getElementById("upload-window");
const uploadReason = document.getElementById("upload-reason");
const modelBadge = document.getElementById("model-badge");

const sequenceImage = document.getElementById("sequence-image");
const sequenceExpected = document.getElementById("sequence-expected");
const sequenceCommand = document.getElementById("sequence-command");
const sequenceState = document.getElementById("sequence-state");
const sequenceReason = document.getElementById("sequence-reason");
const sequenceTimeline = document.getElementById("sequence-timeline");
const playSequenceButton = document.getElementById("play-sequence");
const nextFrameButton = document.getElementById("next-frame");
const modulesPanel = document.getElementById("workflow-modules");
const modulesContent = document.getElementById("modules-content");
const reloadConfigButton = document.getElementById("reload-config");
const toggleModulesButton = document.getElementById("toggle-modules");
const evalTabValid = document.getElementById("eval-tab-valid");
const evalTabTest = document.getElementById("eval-tab-test");
const evalRefresh = document.getElementById("eval-refresh");
const evalMaxSamples = document.getElementById("eval-max-samples");
const evalMeasureLatency = document.getElementById("eval-measure-latency");
const evalExportFormat = document.getElementById("eval-export-format");
const evalExportButton = document.getElementById("eval-export");
const evalSummary = document.getElementById("eval-summary");
const confusionMatrix = document.getElementById("confusion-matrix");
const perClassTable = document.getElementById("per-class-table");
const topConfusions = document.getElementById("top-confusions");
const hardClasses = document.getElementById("hard-classes");

function renderWindow(container, items) {
  container.innerHTML = "";
  if (!items || items.length === 0) {
    container.innerHTML = '<span class="tag">None</span>';
    return;
  }
  items.forEach((item) => {
    const node = document.createElement("span");
    node.className = "tag";
    node.textContent = `${item.command} | ${Math.round(item.confidence * 100)}%`;
    container.appendChild(node);
  });
}

function applyPrediction(prefix, payload) {
  const commandNode = document.getElementById(`${prefix}-command`);
  const stateNode = document.getElementById(`${prefix}-state`);
  const reasonNode = document.getElementById(`${prefix}-reason`);
  commandNode.textContent = payload.intent.command;
  stateNode.textContent = payload.intent.state;
  reasonNode.textContent = payload.intent.reason;

  if (prefix === "upload") {
    uploadCommandDesc.textContent = payload.intent.description;
    uploadLabel.textContent = payload.prediction.label;
    uploadConfidence.textContent = `Confidence ${Math.round(payload.prediction.confidence * 100)}%`;
    uploadLatency.textContent = `Latency ${payload.latency_ms} ms`;
    renderWindow(uploadWindow, payload.intent.window);
    if (payload.pose_overlay) {
      posePreview.src = payload.pose_overlay;
      posePreview.style.display = "block";
    }
    if (poseStatus) {
      poseStatus.textContent = payload.pose_detected ? "Pose detected" : "Pose not detected";
    }
  }
}

function renderTimeline() {
  sequenceTimeline.innerHTML = "";
  if (!currentScenario) {
    return;
  }
  currentScenario.frames.forEach((frame, index) => {
    const node = document.createElement("span");
    node.className = `timeline-item ${index === currentFrameIndex ? "active" : ""}`;
    node.textContent = `${index + 1}. ${frame.command}`;
    sequenceTimeline.appendChild(node);
  });
}

async function resetSession(sessionId) {
  await fetch("/api/session/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
}

async function loadModelInfo() {
  try {
    const response = await fetch("/api/model-info");
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload?.error || `HTTP ${response.status}`);
    }
    const payload = await response.json();
    if (modelBadge) {
      modelBadge.textContent = `Current model: ${payload.model_name} | valid ${(payload.valid_accuracy * 100).toFixed(1)}% | test ${(payload.test_accuracy * 100).toFixed(1)}%`;
    }
  } catch (error) {
    if (modelBadge) {
      modelBadge.textContent = "Current model: unavailable";
    }
  }
}

function renderModules(payload) {
  if (!modulesContent) {
    return;
  }
  modulesContent.innerHTML = "";
  const system = payload.system || {};
  const items = [
    { name: "Dataset", spec: system.dataset },
    { name: "Predictor", spec: system.predictor },
    { name: "Pose Overlay", spec: system.pose_overlay },
    { name: "Intent Engine", spec: system.intent_engine },
    { name: "Evaluator", spec: system.evaluator },
  ];
  items.forEach((item) => {
    if (!item.spec) {
      return;
    }
    const target = item.spec.target || "-";
    const node = document.createElement("div");
    node.className = "module-item";
    node.innerHTML = `<strong>${item.name}</strong><code>${target}</code>`;
    modulesContent.appendChild(node);
  });
}

async function loadModules() {
  try {
    const response = await fetch("/api/modules");
    const payload = await response.json();
    renderModules(payload);
  } catch (error) {
    if (modulesContent) {
      modulesContent.innerHTML = '<div class="module-item"><strong>Modules</strong><code>unavailable</code></div>';
    }
  }
}

function setEvalTab(split) {
  evalActiveSplit = split;
  if (evalTabValid && evalTabTest) {
    evalTabValid.classList.toggle("active", split === "valid");
    evalTabTest.classList.toggle("active", split === "test");
  }
  renderEvalActive();
}

function heatColor(value) {
  const v = Math.max(0, Math.min(1, value));
  const base = [207, 92, 47];
  const bg = [255, 255, 255];
  const mix = 0.12 + 0.78 * v;
  const r = Math.round(bg[0] * (1 - mix) + base[0] * mix);
  const g = Math.round(bg[1] * (1 - mix) + base[1] * mix);
  const b = Math.round(bg[2] * (1 - mix) + base[2] * mix);
  return `rgb(${r}, ${g}, ${b})`;
}

function renderConfusion(details) {
  if (!confusionMatrix) {
    return;
  }
  confusionMatrix.innerHTML = "";
  const labels = details.labels || [];
  const cm = details.confusion_matrix_normalized || [];
  const n = labels.length;
  if (!n || !cm.length) {
    confusionMatrix.textContent = "No data";
    return;
  }

  const grid = document.createElement("div");
  grid.className = "cm-grid";
  grid.style.gridTemplateColumns = `repeat(${n}, minmax(28px, 1fr))`;

  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      const v = cm[i]?.[j] ?? 0;
      const cell = document.createElement("div");
      cell.className = "cm-cell";
      cell.style.background = heatColor(v);
      cell.title = `${labels[i]} → ${labels[j]}: ${(v * 100).toFixed(1)}%`;
      cell.textContent = `${Math.round(v * 100)}`;
      grid.appendChild(cell);
    }
  }
  confusionMatrix.appendChild(grid);

  const labelBox = document.createElement("div");
  labelBox.className = "cm-labels";
  labelBox.innerHTML = `<div><strong>Labels</strong></div>${labels
    .map((name, idx) => `<div>${idx + 1}. ${name}</div>`)
    .join("")}`;
  confusionMatrix.appendChild(labelBox);
}

function renderMiniList(container, items, formatter) {
  if (!container) {
    return;
  }
  container.innerHTML = "";
  if (!items || items.length === 0) {
    container.innerHTML = '<div class="mini-item"><strong>None</strong><small>-</small></div>';
    return;
  }
  items.forEach((item) => {
    const node = document.createElement("div");
    node.className = "mini-item";
    node.innerHTML = formatter(item);
    container.appendChild(node);
  });
}

function renderPerClass(details) {
  if (!perClassTable) {
    return;
  }
  const perClass = details.per_class || {};
  const labels = details.labels || Object.keys(perClass);
  const rows = labels
    .map((name) => ({ label: name, ...(perClass[name] || {}) }))
    .sort((a, b) => (a.f1 ?? 0) - (b.f1 ?? 0));

  const table = document.createElement("table");
  table.innerHTML = `
    <thead>
      <tr>
        <th>Label</th>
        <th>Support</th>
        <th>Precision</th>
        <th>Recall</th>
        <th>F1</th>
      </tr>
    </thead>
    <tbody>
      ${rows
        .map(
          (r) => `
        <tr>
          <td>${r.label}</td>
          <td>${r.support ?? 0}</td>
          <td>${(r.precision ?? 0).toFixed(4)}</td>
          <td>${(r.recall ?? 0).toFixed(4)}</td>
          <td>${(r.f1 ?? 0).toFixed(4)}</td>
        </tr>
      `
        )
        .join("")}
    </tbody>
  `;
  perClassTable.innerHTML = "";
  perClassTable.appendChild(table);
}

function renderSummaryCards(details) {
  if (!evalSummary) {
    return;
  }
  const latency = details.latency_ms || null;
  const cards = [
    {
      title: "Accuracy",
      value: `${((details.accuracy ?? 0) * 100).toFixed(1)}%`,
      note: `Top-3 ${(details.top3_accuracy ?? 0).toFixed(4)}`,
    },
    {
      title: "F1 Macro",
      value: `${(details.f1_macro ?? 0).toFixed(4)}`,
      note: `F1 Weighted ${(details.f1_weighted ?? 0).toFixed(4)}`,
    },
    {
      title: "Unknown Rate",
      value: `${((details.unknown_rate ?? 0) * 100).toFixed(1)}%`,
      note: `Margin/Confidence gate`,
    },
    {
      title: "Latency",
      value: latency ? `${latency.p50_ms} ms` : "-",
      note: latency ? `P95 ${latency.p95_ms} ms; FPS ${latency.throughput_fps}` : "Latency disabled",
    },
  ];
  evalSummary.innerHTML = cards
    .map(
      (c) => `
    <div class="eval-summary-card">
      <span class="stat-label">${c.title}</span>
      <strong>${c.value}</strong>
      <small>${c.note}</small>
    </div>
  `
    )
    .join("");
}

function renderEvalActive() {
  const payload = evalCache[evalActiveSplit];
  if (!payload || !payload.details) {
    if (evalSummary) {
      evalSummary.innerHTML = "";
    }
    if (confusionMatrix) {
      confusionMatrix.textContent = "Loading...";
    }
    if (perClassTable) {
      perClassTable.innerHTML = "";
    }
    return;
  }
  const details = payload.details;
  renderSummaryCards(details);
  renderConfusion(details);
  renderPerClass(details);
  renderMiniList(topConfusions, details.top_confusions || [], (item) => {
    return `<strong>${item.true} → ${item.pred}</strong><small>${item.count} (${Math.round((item.rate ?? 0) * 100)}%)</small>`;
  });
  renderMiniList(hardClasses, details.hard_classes || [], (item) => {
    return `<strong>${item.label}</strong><small>F1 ${item.f1} | Recall ${item.recall} | Support ${item.support}</small>`;
  });
}

async function fetchEvaluation(split) {
  const maxSamples = evalMaxSamples ? evalMaxSamples.value : "";
  const measureLatency = evalMeasureLatency ? evalMeasureLatency.value : "1";
  const qs = new URLSearchParams();
  if (maxSamples) {
    qs.set("max_samples", maxSamples);
  }
  qs.set("measure_latency", measureLatency);
  const url = `/api/evaluate/${split}?${qs.toString()}`;
  const response = await fetch(url);
  const payload = await response.json();
  evalCache[split] = payload;
  if (split === "valid") {
    document.getElementById("valid-accuracy").textContent = payload.accuracy;
    document.getElementById("valid-precision").textContent = payload.precision;
    document.getElementById("valid-recall").textContent = payload.recall;
    document.getElementById("valid-f1").textContent = payload.f1;
  }
  if (split === "test") {
    document.getElementById("test-accuracy").textContent = payload.accuracy;
    document.getElementById("test-precision").textContent = payload.precision;
    document.getElementById("test-recall").textContent = payload.recall;
    document.getElementById("test-f1").textContent = payload.f1;
  }
}

function buildEvalQuery() {
  const maxSamples = evalMaxSamples ? evalMaxSamples.value : "";
  const measureLatency = evalMeasureLatency ? evalMeasureLatency.value : "1";
  const qs = new URLSearchParams();
  if (maxSamples) {
    qs.set("max_samples", maxSamples);
  }
  qs.set("measure_latency", measureLatency);
  return qs.toString();
}

async function exportEvaluation() {
  const fmt = evalExportFormat ? evalExportFormat.value : "json";
  const qs = buildEvalQuery();
  const url = `/api/evaluate/${evalActiveSplit}/export?format=${encodeURIComponent(fmt)}&${qs}`;
  const a = document.createElement("a");
  a.href = url;
  a.download = "";
  document.body.appendChild(a);
  a.click();
  a.remove();
}

async function predictUploadedFile(file) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("session_id", uploadSessionId);

  const response = await fetch("/api/predict/upload", {
    method: "POST",
    body: formData,
  });
  const payload = await response.json();
  applyPrediction("upload", payload);
}

async function playCurrentFrame() {
  if (!currentScenario || currentFrameIndex >= currentScenario.frames.length) {
    stopPlayback();
    return;
  }

  const frame = currentScenario.frames[currentFrameIndex];
  const response = await fetch("/api/predict/sample", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image: frame.image,
      session_id: sequenceSessionId,
    }),
  });
  const payload = await response.json();

  sequenceImage.src = `/dataset-image/${frame.image}`;
  sequenceExpected.textContent = frame.command;
  sequenceCommand.textContent = payload.intent.command;
  sequenceState.textContent = payload.intent.state;
  sequenceReason.textContent = `${payload.intent.reason}; raw label: ${payload.prediction.label}; confidence ${Math.round(payload.prediction.confidence * 100)}%.`;
  renderTimeline();
  currentFrameIndex += 1;
}

function stopPlayback() {
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
  }
  playSequenceButton.textContent = "Start Auto Play";
}

async function startPlayback() {
  if (!currentScenario) {
    return;
  }
  if (playTimer) {
    stopPlayback();
    return;
  }
  await resetSession(sequenceSessionId);
  currentFrameIndex = 0;
  renderTimeline();
  await playCurrentFrame();
  playTimer = setInterval(async () => {
    if (currentFrameIndex >= currentScenario.frames.length) {
      stopPlayback();
      return;
    }
    await playCurrentFrame();
  }, 1200);
  playSequenceButton.textContent = "Stop";
}

function bindScenarioButtons() {
  document.querySelectorAll(".scenario-button").forEach((button) => {
    button.addEventListener("click", async () => {
      document.querySelectorAll(".scenario-button").forEach((node) => node.classList.remove("active"));
      button.classList.add("active");
      currentScenario = demoSequences.find((item) => item.id === button.dataset.scenarioId);
      currentFrameIndex = 0;
      sequenceSessionId = crypto.randomUUID();
      await resetSession(sequenceSessionId);
      stopPlayback();
      renderTimeline();
      const firstFrame = currentScenario?.frames?.[0];
      if (firstFrame) {
        sequenceImage.src = `/dataset-image/${firstFrame.image}`;
        sequenceExpected.textContent = firstFrame.command;
        sequenceCommand.textContent = "-";
        sequenceState.textContent = "IDLE";
        sequenceReason.textContent = currentScenario.description;
      }
    });
  });
}

uploadInput.addEventListener("change", async (event) => {
  const [file] = event.target.files;
  if (!file) {
    return;
  }

  const previewUrl = URL.createObjectURL(file);
  uploadPreview.src = previewUrl;
  uploadPreview.style.display = "block";
  await predictUploadedFile(file);
});

document.getElementById("reset-upload-session").addEventListener("click", async () => {
  uploadSessionId = crypto.randomUUID();
  await resetSession(uploadSessionId);
  uploadCommand.textContent = "Waiting";
  uploadCommandDesc.textContent = "Upload an image to run inference";
  uploadLabel.textContent = "-";
  uploadConfidence.textContent = "Confidence -";
  uploadState.textContent = "IDLE";
  uploadLatency.textContent = "Latency -";
  uploadReason.textContent = "Temporal state reset.";
  if (posePreview) {
    posePreview.removeAttribute("src");
  }
  if (poseStatus) {
    poseStatus.textContent = "Waiting for upload";
  }
  renderWindow(uploadWindow, []);
});

playSequenceButton.addEventListener("click", startPlayback);
nextFrameButton.addEventListener("click", async () => {
  if (!currentScenario) {
    return;
  }
  if (currentFrameIndex === 0) {
    await resetSession(sequenceSessionId);
  }
  await playCurrentFrame();
});

bindScenarioButtons();
renderTimeline();
loadModelInfo();
loadModules();

if (currentScenario?.frames?.[0]) {
  const firstFrame = currentScenario.frames[0];
  sequenceImage.src = `/dataset-image/${firstFrame.image}`;
  sequenceExpected.textContent = firstFrame.command;
  sequenceCommand.textContent = "-";
  sequenceState.textContent = "IDLE";
  sequenceReason.textContent = currentScenario.description;
}

async function refreshEvaluations() {
  await fetchEvaluation("valid");
  await fetchEvaluation("test");
  renderEvalActive();
}

if (evalTabValid) {
  evalTabValid.addEventListener("click", () => setEvalTab("valid"));
}
if (evalTabTest) {
  evalTabTest.addEventListener("click", () => setEvalTab("test"));
}
if (evalRefresh) {
  evalRefresh.addEventListener("click", refreshEvaluations);
}
if (evalExportButton) {
  evalExportButton.addEventListener("click", exportEvaluation);
}
if (toggleModulesButton && modulesPanel) {
  toggleModulesButton.addEventListener("click", async () => {
    const next = modulesPanel.style.display === "none";
    modulesPanel.style.display = next ? "block" : "none";
    if (next) {
      await loadModules();
    }
  });
}
if (reloadConfigButton) {
  reloadConfigButton.addEventListener("click", async () => {
    try {
      const response = await fetch("/api/admin/reload", { method: "POST" });
      if (!response.ok) {
        const payload = await response.json();
        alert(payload.error || "reload failed");
        return;
      }
      await loadModules();
      await loadModelInfo();
      await refreshEvaluations();
    } catch (error) {
      alert("reload failed");
    }
  });
}

refreshEvaluations();
setEvalTab("valid");
