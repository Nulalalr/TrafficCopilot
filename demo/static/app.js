const bootstrap = window.__BOOTSTRAP__ || {};
const demoSequences = bootstrap.demoSequences || [];

let uploadSessionId = crypto.randomUUID();
let sequenceSessionId = crypto.randomUUID();
let currentScenario = demoSequences[0] || null;
let currentFrameIndex = 0;
let playTimer = null;

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

if (currentScenario?.frames?.[0]) {
  const firstFrame = currentScenario.frames[0];
  sequenceImage.src = `/dataset-image/${firstFrame.image}`;
  sequenceExpected.textContent = firstFrame.command;
  sequenceCommand.textContent = "-";
  sequenceState.textContent = "IDLE";
  sequenceReason.textContent = currentScenario.description;
}

//评估
fetch('/api/evaluate/valid')
  .then(response => response.json())
  .then(data => {
    // 更新网页内容
    document.getElementById('valid-accuracy').textContent = data.accuracy;
    document.getElementById('valid-precision').textContent = data.precision;
    document.getElementById('valid-recall').textContent = data.recall;
    document.getElementById('valid-f1').textContent = data.f1;
  });

fetch('/api/evaluate/test')
  .then(response => response.json())
  .then(data => {
    // 更新网页内容
    document.getElementById('test-accuracy').textContent = data.accuracy;
    document.getElementById('test-precision').textContent = data.precision;
    document.getElementById('test-recall').textContent = data.recall;
    document.getElementById('test-f1').textContent = data.f1;
  });