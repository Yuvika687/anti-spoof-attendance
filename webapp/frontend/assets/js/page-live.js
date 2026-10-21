import { api } from "./api.js";
import { pct, statusClass } from "./utils.js";

const STAGES = [
  { key: "frame_ms", name: "Frame" },
  { key: "detect_ms", name: "Detect" },
  { key: "crop_ms", name: "Crop" },
  { key: "classify_ms", name: "Classify" },
  { key: "decision_ms", name: "Decision" },
];

export async function initLivePage(root) {
  root.innerHTML = `
    <h1 class="page-title">Live</h1>
    <p class="page-sub">Real webcam or curated sample images, run through the real detection &rarr; crop &rarr; classification &rarr; decision pipeline. Nothing here is cached.</p>

    <div class="live-layout">
      <div>
        <div class="cam-panel">
          <div class="cam-wrap" id="cam-wrap">
            <video id="cam-video" autoplay muted playsinline></video>
            <img id="cam-still" class="cam-still" style="display:none" alt="">
            <canvas id="cam-canvas" style="display:none"></canvas>
            <svg class="cam-overlay" id="cam-overlay" viewBox="0 0 100 100" preserveAspectRatio="none"></svg>
            <div class="state-badge-big" id="state-badge" style="display:none">
              <span class="state-dot" id="state-dot"></span>
              <span id="state-label">&mdash;</span>
              <span class="prob mono" id="state-prob"></span>
            </div>
            <div class="cam-mode-toggle">
              <div class="btn-group">
                <button id="mode-camera-btn" class="active">Camera</button>
                <button id="mode-sample-btn">Sample images</button>
              </div>
            </div>
            <div class="cam-controls">
              <button class="btn" id="cam-start-btn">&#9679; Start</button>
              <button class="btn" id="cam-stop-btn" style="display:none">&#9632; Stop</button>
            </div>
          </div>
        </div>

        <div class="panel" style="margin-top:12px">
          <h3>P(spoof)</h3>
          <div class="gauge-track" id="gauge-track">
            <div class="gauge-band" id="gauge-band"></div>
            <div class="gauge-fill" id="gauge-fill" style="width:0%"></div>
            <div class="gauge-threshold" id="gauge-threshold-marker"></div>
          </div>
          <div class="gauge-labels"><span>0.0</span><span id="gauge-band-label">loading band&hellip;</span><span>1.0</span></div>
        </div>

        <div class="panel" style="margin-top:12px;padding:12px 16px">
          <h3 style="margin-bottom:10px">Pipeline</h3>
          <div class="pipeline-strip" id="pipeline-strip" style="padding:0"></div>
        </div>

        <div class="panel" style="margin-top:12px;display:none" id="sample-panel">
          <h3>Sample images</h3>
          <div class="panel-sub">Each tile re-runs the real pipeline on a real test-set image.</div>
          <div class="sample-grid" id="sample-grid"></div>
        </div>
      </div>

      <div>
        <div class="panel">
          <h3>Explain</h3>
          <div class="panel-sub">Grad-CAM on one frozen frame &mdash; ~20-90ms depending on backbone, too slow to run every frame, so it's on demand only.</div>
          <button class="btn btn-primary" id="explain-btn" style="width:100%">Explain this frame</button>
          <div class="explain-panel" id="explain-panel">
            <div class="explain-images">
              <img id="explain-crop" alt="frozen crop">
              <img id="explain-overlay" alt="grad-cam overlay">
            </div>
            <table style="margin-top:10px"><tbody>
              <tr><td>Prediction</td><td class="mono" id="explain-label">&mdash;</td></tr>
              <tr><td>P(spoof)</td><td class="mono" id="explain-prob">&mdash;</td></tr>
            </tbody></table>
          </div>
        </div>

        <div class="panel" style="margin-top:12px">
          <h3>Recent Decisions</h3>
          <div id="live-feed"><div class="empty-state">No decisions yet this session.</div></div>
        </div>
      </div>
    </div>
  `;

  renderPipelineStrip();

  const video = document.getElementById("cam-video");
  const camStill = document.getElementById("cam-still");
  const canvas = document.getElementById("cam-canvas");
  const overlay = document.getElementById("cam-overlay");
  const camWrap = document.getElementById("cam-wrap");

  let mode = "camera"; // "camera" | "sample"
  let liveRunning = false;
  let mediaStream = null;
  let lastFrameBlob = null; // for Explain, in camera mode
  let activeSampleId = null; // for Explain, in sample mode
  let lastLabel = null;
  let gband = null;
  let threshold = 0.5;

  // ---- gray-zone band + threshold, for the gauge ----
  try {
    const info = await api.modelInfo();
    threshold = info.current_threshold ?? 0.5;
    gband = info.gray_zone_band;
    renderGaugeBand();
  } catch (e) { /* header already shows backend-unreachable */ }

  function renderGaugeBand() {
    const thresholdMarker = document.getElementById("gauge-threshold-marker");
    thresholdMarker.style.left = `${threshold * 100}%`;
    const bandLabel = document.getElementById("gauge-band-label");
    if (gband) {
      const [low, high] = gband;
      const band = document.getElementById("gauge-band");
      band.style.left = `${low * 100}%`;
      band.style.width = `${(high - low) * 100}%`;
      bandLabel.textContent = `gray zone [${low.toFixed(3)}, ${high.toFixed(3)}]  threshold=${threshold}`;
    } else {
      bandLabel.textContent = `threshold=${threshold} (gray zone unavailable)`;
    }
  }

  function updateGaugeFill(prob) {
    document.getElementById("gauge-fill").style.width = `${Math.min(100, Math.max(0, prob * 100))}%`;
  }

  function renderPipelineStrip(timings) {
    const strip = document.getElementById("pipeline-strip");
    strip.innerHTML = STAGES.map((s, i) => {
      const t = timings ? timings[s.key] : null;
      const lit = timings ? "lit" : "";
      const arrow = i < STAGES.length - 1 ? '<span class="pipeline-arrow">&rarr;</span>' : "";
      return `<div class="pipeline-stage ${lit}"><div class="name">${s.name}</div><div class="time mono">${t !== null && t !== undefined ? t.toFixed(1) + "ms" : "&mdash;"}</div></div>${arrow}`;
    }).join("");
  }

  function setBadge(label, probSpoof, message) {
    const badge = document.getElementById("state-badge");
    const dot = document.getElementById("state-dot");
    const text = document.getElementById("state-label");
    const prob = document.getElementById("state-prob");
    badge.style.display = "flex";
    dot.className = "state-dot " + statusClass(label);
    text.textContent = label + (message ? ` — ${message}` : "");
    prob.textContent = `P(spoof)=${probSpoof.toFixed(3)}`;
    updateGaugeFill(probSpoof);
  }

  function drawBox(bbox, label, frameW, frameH, mirrored) {
    overlay.innerHTML = "";
    if (!bbox) return;
    const color = label === "LIVE" ? getComputedStyle(document.documentElement).getPropertyValue("--live")
      : label === "UNCERTAIN" ? getComputedStyle(document.documentElement).getPropertyValue("--uncertain")
      : getComputedStyle(document.documentElement).getPropertyValue("--spoof");
    const x = mirrored ? 100 - ((bbox.x + bbox.width) / frameW) * 100 : (bbox.x / frameW) * 100;
    const y = (bbox.y / frameH) * 100;
    const w = (bbox.width / frameW) * 100;
    const h = (bbox.height / frameH) * 100;
    const svgNS = "http://www.w3.org/2000/svg";
    const rect = document.createElementNS(svgNS, "rect");
    rect.setAttribute("x", x); rect.setAttribute("y", y);
    rect.setAttribute("width", w); rect.setAttribute("height", h);
    rect.setAttribute("fill", "none"); rect.setAttribute("stroke", color);
    rect.setAttribute("stroke-width", "0.6"); rect.setAttribute("vector-effect", "non-scaling-stroke");
    overlay.appendChild(rect);
  }

  function addFeedItem(label, confidence) {
    const feed = document.getElementById("live-feed");
    if (feed.querySelector(".empty-state")) feed.innerHTML = "";
    const item = document.createElement("div");
    item.className = "feed-item";
    const ts = new Date().toLocaleTimeString();
    item.innerHTML = `<span class="ts mono">${ts}</span><span class="badge ${statusClass(label)}">${label}</span><span class="conf">${(confidence * 100).toFixed(1)}%</span>`;
    feed.insertBefore(item, feed.firstChild);
    while (feed.children.length > 10) feed.removeChild(feed.lastChild);
  }

  async function refreshFeedFromLog() {
    try {
      const log = await api.decisionLog(10);
      const feed = document.getElementById("live-feed");
      if (!log.entries.length) { feed.innerHTML = '<div class="empty-state">No decisions yet this session.</div>'; return; }
      feed.innerHTML = log.entries.map((e) => {
        const label = e.decision === "LIVE_ACCEPTED" ? "LIVE" : e.decision === "UNCERTAIN" ? "UNCERTAIN" : "SPOOF";
        return `<div class="feed-item"><span class="ts mono">${e.timestamp.split("T")[1] || e.timestamp}</span><span class="badge ${statusClass(label)}">${label}</span><span class="conf">${(parseFloat(e.confidence) * 100).toFixed(1)}%</span></div>`;
      }).join("");
    } catch (e) { /* non-fatal */ }
  }
  refreshFeedFromLog();

  // ---- camera mode ----
  async function startCamera() {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 960 } });
    } catch (err) {
      alert("Could not access the camera: " + err.message);
      return;
    }
    video.srcObject = mediaStream;
    video.style.display = "block";
    camStill.style.display = "none";
    document.getElementById("cam-start-btn").style.display = "none";
    document.getElementById("cam-stop-btn").style.display = "inline-block";
    liveRunning = true;
    detectLoop();
  }

  // The overlay SVG maps bbox coordinates as percentages of the RAW frame
  // (video.videoWidth/videoHeight). If .cam-wrap's displayed aspect ratio
  // doesn't match the real stream's, object-fit:cover crops the visible
  // video differently than those percentages assume, and the drawn box
  // drifts off the real face (reported: box drawn on the neck, below the
  // face). Fix: match the container's aspect ratio to the real stream's
  // every time metadata is (re-)available -- registered once here, not
  // inside startCamera(), so repeated Start/Stop cycles don't stack
  // duplicate listeners on the same persistent <video> element.
  video.addEventListener("loadedmetadata", () => {
    if (video.videoWidth && video.videoHeight) {
      camWrap.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
    }
  });

  function stopCamera() {
    liveRunning = false;
    if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());
    document.getElementById("cam-start-btn").style.display = "inline-block";
    document.getElementById("cam-stop-btn").style.display = "none";
  }

  async function detectLoop() {
    if (!liveRunning || mode !== "camera") return;
    const ctx = canvas.getContext("2d");
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
      lastFrameBlob = blob;
      try {
        const result = await api.detectFrame(blob);
        if (!result.face_detected) {
          document.getElementById("state-badge").style.display = "none";
          overlay.innerHTML = "";
        } else {
          setBadge(result.label, result.raw_label === "SPOOF" ? result.raw_confidence : 1 - result.raw_confidence, result.message);
          drawBox(result.bbox, result.label, canvas.width, canvas.height, true);
          if (result.label !== lastLabel) { addFeedItem(result.label, result.confidence); lastLabel = result.label; }
        }
        renderPipelineStrip(result.timings);
      } catch (e) { /* transient network hiccup, keep looping */ }

      if (liveRunning && mode === "camera") setTimeout(detectLoop, 60);
    }, "image/jpeg");
  }

  document.getElementById("cam-start-btn").addEventListener("click", startCamera);
  document.getElementById("cam-stop-btn").addEventListener("click", stopCamera);

  // ---- sample-image mode ----
  async function loadSampleGrid() {
    const grid = document.getElementById("sample-grid");
    try {
      const { images } = await api.sampleImages();
      grid.innerHTML = images.map((img) => `<div class="sample-thumb" data-id="${img.id}">${img.label}</div>`).join("");
      grid.querySelectorAll(".sample-thumb").forEach((t) => t.addEventListener("click", () => runSample(t.dataset.id)));
    } catch (e) {
      grid.innerHTML = '<div class="empty-state">Could not load sample images.</div>';
    }
  }

  // Same aspect-ratio fix as the camera stream, for the static sample
  // image (its container uses the same object-fit:cover rule).
  camStill.addEventListener("load", () => {
    if (camStill.naturalWidth && camStill.naturalHeight) {
      camWrap.style.aspectRatio = `${camStill.naturalWidth} / ${camStill.naturalHeight}`;
    }
  });

  async function runSample(id) {
    activeSampleId = id;
    document.querySelectorAll(".sample-thumb").forEach((t) => t.classList.toggle("active", t.dataset.id === id));
    camStill.src = api.sampleImageUrl(id);
    camStill.style.display = "block";
    const result = await api.sampleDetect(id);
    if (!result.face_detected) {
      document.getElementById("state-badge").style.display = "none";
      renderPipelineStrip(result.timings);
      return;
    }
    setBadge(result.label, result.raw_label === "SPOOF" ? result.raw_confidence : 1 - result.raw_confidence, result.message);
    renderPipelineStrip(result.timings);
    addFeedItem(result.label, result.confidence);
    lastLabel = result.label;
    // bbox overlay is skipped in sample mode (no live frame to draw on top of);
    // the state badge + P(spoof) gauge already carry the real result.
    overlay.innerHTML = "";
  }

  function setMode(newMode) {
    mode = newMode;
    document.getElementById("mode-camera-btn").classList.toggle("active", mode === "camera");
    document.getElementById("mode-sample-btn").classList.toggle("active", mode === "sample");
    document.getElementById("sample-panel").style.display = mode === "sample" ? "block" : "none";
    if (mode === "sample") {
      stopCamera();
      video.style.display = "none";
      camStill.style.display = "none";
      document.querySelector(".cam-controls").style.display = "none";
      if (!document.getElementById("sample-grid").children.length) loadSampleGrid();
    } else {
      camStill.style.display = "none";
      document.querySelector(".cam-controls").style.display = "flex";
      document.getElementById("state-badge").style.display = "none";
      overlay.innerHTML = "";
    }
  }
  document.getElementById("mode-camera-btn").addEventListener("click", () => setMode("camera"));
  document.getElementById("mode-sample-btn").addEventListener("click", () => setMode("sample"));

  // ---- explain ----
  document.getElementById("explain-btn").addEventListener("click", async () => {
    const btn = document.getElementById("explain-btn");
    btn.disabled = true;
    const prevText = btn.textContent;
    btn.textContent = "Analyzing…";
    try {
      let result;
      let cropSrc;
      if (mode === "sample" && activeSampleId) {
        result = await api.sampleExplain(activeSampleId);
        cropSrc = api.sampleImageUrl(activeSampleId);
      } else if (mode === "camera" && lastFrameBlob) {
        result = await api.explainFrame(lastFrameBlob);
        cropSrc = URL.createObjectURL(lastFrameBlob);
      } else {
        alert("Start the camera or pick a sample image first.");
        btn.disabled = false; btn.textContent = prevText; return;
      }
      if (!result.face_detected) {
        alert("No face detected in this frame.");
      } else if (result.error) {
        alert("Explain failed: " + result.error);
      } else {
        document.getElementById("explain-overlay").src = result.overlay_image_base64;
        document.getElementById("explain-crop").src = cropSrc || result.overlay_image_base64;
        document.getElementById("explain-label").textContent = result.label;
        document.getElementById("explain-prob").textContent = result.prob_spoof.toFixed(4);
        document.getElementById("explain-panel").classList.add("show");
      }
    } catch (err) {
      alert("Explain request failed: " + err);
    }
    btn.disabled = false;
    btn.textContent = prevText;
  });
}
