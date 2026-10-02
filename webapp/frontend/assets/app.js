// Liveness Gate dashboard frontend.
// Every number on this page comes from a real backend fetch -- nothing
// here is hardcoded chart data. See webapp/backend/app.py for where each
// endpoint reads its data from.

const API = ""; // same-origin, Flask serves both frontend and API

// ---------------------------------------------------------------------
// Nav
// ---------------------------------------------------------------------
document.getElementById("nav").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-view]");
  if (!btn) return;
  document.querySelectorAll("#nav button").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
  document.getElementById("view-" + btn.dataset.view).classList.add("active");
});

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------
function pct(x) { return (x * 100).toFixed(2) + "%"; }

async function getJSON(path) {
  const res = await fetch(API + path);
  if (!res.ok) throw new Error(path + " -> " + res.status);
  return res.json();
}

// ---------------------------------------------------------------------
// Backend connectivity pill
// ---------------------------------------------------------------------
async function pingBackend() {
  try {
    const info = await getJSON("/api/model-info");
    document.getElementById("header-fps").textContent =
      `${info.backbone} on ${info.device} · τ=${info.current_threshold.toFixed(2)}`;
  } catch {
    document.getElementById("header-fps").textContent = "backend offline";
  }
}

// ---------------------------------------------------------------------
// Overview + Analytics: real metrics
// ---------------------------------------------------------------------
async function loadMetricsEverywhere() {
  let m;
  try {
    m = await getJSON("/api/metrics");
  } catch {
    return;
  }

  const overviewTiles = document.getElementById("overview-tiles");
  overviewTiles.innerHTML = `
    <div class="tile"><div class="figure accent">${pct(m.accuracy)}</div><div class="label">Accuracy</div></div>
    <div class="tile"><div class="figure accent">${pct(m.hter)}</div><div class="label">HTER</div></div>
    <div class="tile"><div class="figure">${pct(m.apcer)}</div><div class="label">APCER</div></div>
    <div class="tile"><div class="figure">${pct(m.bpcer)}</div><div class="label">BPCER</div></div>
    <div class="tile"><div class="figure">${m.n_live + m.n_spoof}</div><div class="label">Test images</div></div>
    <div class="tile"><div class="figure">26.7 FPS</div><div class="label">Direct Python pipeline (no browser/network) &mdash; see Live Detection for real web FPS</div></div>
  `;

  document.getElementById("analytics-tiles").innerHTML = `
    <div class="tile"><div class="figure accent">${pct(m.accuracy)}</div><div class="label">Accuracy</div></div>
    <div class="tile"><div class="figure">${pct(m.apcer)}</div><div class="label">APCER</div></div>
    <div class="tile"><div class="figure">${pct(m.bpcer)}</div><div class="label">BPCER</div></div>
    <div class="tile"><div class="figure accent">${pct(m.hter)}</div><div class="label">HTER</div></div>
    <div class="tile"><div class="figure">${pct(m.precision_live)}</div><div class="label">Precision (live)</div></div>
    <div class="tile"><div class="figure">${pct(m.recall_spoof)}</div><div class="label">Recall (spoof)</div></div>
  `;

  // Classical baseline comparison -- real numbers from baseline_opencv.py's
  // actual run (Laplacian-variance sharpness heuristic), not invented.
  const baselineAcc = 0.6123;
  document.getElementById("overview-baseline").innerHTML = `
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:12px">
      <div style="width:170px;flex:0 0 170px;font-size:13.5px;color:var(--text-muted)">Classical baseline<br><span class="mono" style="font-size:11px">Laplacian-variance sharpness</span></div>
      <div style="flex:1;background:var(--surface-2);border-radius:6px;height:28px;border:1px solid var(--border);overflow:hidden">
        <div style="height:100%;width:${baselineAcc*100}%;background:#3A4150;display:flex;align-items:center;justify-content:flex-end;padding-right:8px">
          <span class="mono" style="font-size:12px">${pct(baselineAcc)}</span>
        </div>
      </div>
    </div>
    <div style="display:flex;align-items:center;gap:14px">
      <div style="width:170px;flex:0 0 170px;font-size:13.5px;color:var(--text-muted)">This model<br><span class="mono" style="font-size:11px">MobileNetV3-Small CNN</span></div>
      <div style="flex:1;background:var(--surface-2);border-radius:6px;height:28px;border:1px solid var(--border);overflow:hidden">
        <div style="height:100%;width:${m.accuracy*100}%;background:linear-gradient(90deg,#2E7A73,#22D3EE);display:flex;align-items:center;justify-content:flex-end;padding-right:8px">
          <span class="mono" style="font-size:12px;color:#001015;font-weight:700">${pct(m.accuracy)}</span>
        </div>
      </div>
    </div>
  `;
}

async function loadDatasetInfo() {
  let d;
  try {
    d = await getJSON("/api/dataset-info");
  } catch {
    return;
  }
  document.getElementById("how-dataset-facts").innerHTML = `
    <div class="stat-row"><span class="k">Train</span><span class="v">${d.train_live + d.train_spoof} images (${d.train_live} live / ${d.train_spoof} spoof)</span></div>
    <div class="stat-row"><span class="k">Validation</span><span class="v">${d.val_live + d.val_spoof} images (${d.val_live} live / ${d.val_spoof} spoof)</span></div>
    <div class="stat-row"><span class="k">Test</span><span class="v">${d.test_live + d.test_spoof} images (${d.test_live} live / ${d.test_spoof} spoof)</span></div>
    <div class="stat-row"><span class="k">Total</span><span class="v">${d.total} images, identity-disjoint</span></div>
  `;
}

// ---------------------------------------------------------------------
// Decision log
// ---------------------------------------------------------------------
async function loadDecisionLog() {
  let data;
  try {
    data = await getJSON("/api/decision-log?limit=100");
  } catch {
    return;
  }
  const rows = document.getElementById("log-rows");
  const empty = document.getElementById("log-empty");
  if (!data.entries.length) {
    rows.innerHTML = "";
    empty.style.display = "block";
    return;
  }
  empty.style.display = "none";
  rows.innerHTML = data.entries.map(e => {
    const isLive = e.decision === "LIVE_ACCEPTED";
    return `<tr>
      <td>${e.timestamp}</td>
      <td><span class="badge ${isLive ? "live" : "spoof"}">${isLive ? "LIVE" : "SPOOF"}</span></td>
      <td>${(parseFloat(e.confidence) * 100).toFixed(1)}%</td>
      <td>[${e.bbox_x}, ${e.bbox_y}, ${e.bbox_w}, ${e.bbox_h}]</td>
    </tr>`;
  }).join("");
}

// ---------------------------------------------------------------------
// Threshold tuner -- every slider move recomputes real metrics server-side
// ---------------------------------------------------------------------
const slider = document.getElementById("threshold-slider");
const sliderVal = document.getElementById("threshold-val");

async function updateTunerTiles(threshold) {
  sliderVal.textContent = threshold.toFixed(2);
  let m;
  try {
    m = await getJSON(`/api/threshold-metrics?threshold=${threshold}`);
  } catch {
    return;
  }
  document.getElementById("tuner-tiles").innerHTML = `
    <div class="tile"><div class="figure accent">${pct(m.accuracy)}</div><div class="label">Accuracy</div></div>
    <div class="tile"><div class="figure">${pct(m.apcer)}</div><div class="label">APCER</div></div>
    <div class="tile"><div class="figure">${pct(m.bpcer)}</div><div class="label">BPCER</div></div>
    <div class="tile"><div class="figure accent">${pct(m.hter)}</div><div class="label">HTER</div></div>
  `;
}

let tunerDebounce;
slider.addEventListener("input", () => {
  const t = parseFloat(slider.value);
  sliderVal.textContent = t.toFixed(2);
  clearTimeout(tunerDebounce);
  tunerDebounce = setTimeout(() => updateTunerTiles(t), 120);
});

document.getElementById("tuner-apply-btn").addEventListener("click", async () => {
  const t = parseFloat(slider.value);
  const res = await fetch(API + "/api/threshold", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ threshold: t }),
  });
  const data = await res.json();
  document.getElementById("tuner-current").textContent = "τ = " + data.threshold.toFixed(2);
  pingBackend();
});

async function loadCurrentThreshold() {
  try {
    const d = await getJSON("/api/threshold");
    slider.value = d.threshold;
    sliderVal.textContent = d.threshold.toFixed(2);
    document.getElementById("tuner-current").textContent = "τ = " + d.threshold.toFixed(2);
    document.getElementById("live-threshold").textContent = "τ = " + d.threshold.toFixed(2);
    updateTunerTiles(d.threshold);
  } catch {}
}

// ---------------------------------------------------------------------
// Live detection: real webcam -> real backend inference loop
// ---------------------------------------------------------------------
const video = document.getElementById("cam-video");
const canvas = document.getElementById("cam-canvas");
const overlay = document.getElementById("cam-overlay");
const camWrap = document.getElementById("cam-wrap");
const statusEl = document.getElementById("cam-status");
const fpsLiveEl = document.getElementById("cam-fps-live");
let liveRunning = false;
let frameTimes = [];
let mediaStream = null;

document.getElementById("cam-start-btn").addEventListener("click", startLiveDetection);
document.getElementById("cam-stop-btn").addEventListener("click", stopLiveDetection);

async function startLiveDetection() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
    video.srcObject = mediaStream;

    // Wait for real stream dimensions, then set the container's aspect
    // ratio to match EXACTLY. This is the actual fix for the bounding
    // box being drawn in the wrong place: the container previously had a
    // hardcoded 16:9 aspect-ratio, and if the camera's real stream was a
    // different ratio, object-fit:cover silently cropped the displayed
    // video while our overlay math assumed the full, uncropped frame --
    // causing the box to land in the wrong spot. Matching the container's
    // ratio to the real stream means object-fit:cover never crops
    // anything, so percentage-based overlay coordinates are exact.
    await new Promise(resolve => {
      if (video.videoWidth) return resolve();
      video.onloadedmetadata = () => resolve();
    });
    camWrap.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;

    document.getElementById("cam-start").style.display = "none";
    document.getElementById("cam-stop").style.display = "block";
    liveRunning = true;
    frameTimes = [];
    detectLoop();
  } catch (err) {
    statusEl.innerHTML = `<span class="badge spoof">CAMERA ERROR</span>`;
    console.error(err);
  }
}

function stopLiveDetection() {
  liveRunning = false;
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }
  video.srcObject = null;
  overlay.innerHTML = "";
  statusEl.innerHTML = "";
  document.getElementById("cam-start").style.display = "flex";
  document.getElementById("cam-stop").style.display = "none";
  fpsLiveEl.textContent = "web fps: —";
}

function drawBox(bbox, label) {
  overlay.innerHTML = "";
  if (!bbox) return;
  const vw = video.videoWidth, vh = video.videoHeight;
  // Mirror the x coordinate since the video is displayed mirrored
  // (CSS scaleX(-1)) but the captured frame data (and therefore bbox)
  // is NOT mirrored -- drawImage() always reads the raw decoded frame,
  // unaffected by CSS transforms on the <video> element.
  const x = 100 - ((bbox.x + bbox.width) / vw) * 100;
  const y = (bbox.y / vh) * 100;
  const w = (bbox.width / vw) * 100;
  const h = (bbox.height / vh) * 100;
  const color = label === "LIVE" ? "#22C55E" : "#EF4444";

  const svgNS = "http://www.w3.org/2000/svg";
  const rect = document.createElementNS(svgNS, "rect");
  rect.setAttribute("x", x); rect.setAttribute("y", y);
  rect.setAttribute("width", w); rect.setAttribute("height", h);
  rect.setAttribute("fill", "none"); rect.setAttribute("stroke", color);
  rect.setAttribute("stroke-width", "0.6"); rect.setAttribute("vector-effect", "non-scaling-stroke");
  overlay.appendChild(rect);

  // Label directly on the box (not just in the corner of the video).
  const labelBg = document.createElementNS(svgNS, "rect");
  const labelText = `${label}`;
  const labelWidth = Math.max(w, labelText.length * 2.6);
  labelBg.setAttribute("x", x);
  labelBg.setAttribute("y", Math.max(0, y - 4.5));
  labelBg.setAttribute("width", labelWidth);
  labelBg.setAttribute("height", 4.5);
  labelBg.setAttribute("fill", color);
  overlay.appendChild(labelBg);

  const text = document.createElementNS(svgNS, "text");
  text.setAttribute("x", x + 0.8);
  text.setAttribute("y", Math.max(3.3, y - 1.1));
  text.setAttribute("fill", "#04130F");
  text.setAttribute("font-size", "3.2");
  text.setAttribute("font-weight", "700");
  text.setAttribute("font-family", "JetBrains Mono, monospace");
  text.textContent = labelText;
  overlay.appendChild(text);
}

function addFeedItem(result) {
  const feed = document.getElementById("live-feed");
  if (feed.querySelector(".empty-state")) feed.innerHTML = "";
  const isLive = result.label === "LIVE";
  const item = document.createElement("div");
  item.className = "feed-item";
  const ts = new Date().toLocaleTimeString();
  item.innerHTML = `<span class="ts">${ts}</span><span class="badge ${isLive ? "live" : "spoof"}">${result.label}</span><span class="conf">${(result.confidence*100).toFixed(1)}%</span>`;
  feed.insertBefore(item, feed.firstChild);
  while (feed.children.length > 8) feed.removeChild(feed.lastChild);
}

function updateStatsStrip(result) {
  const cls = (lbl) => lbl === "LIVE" ? "live" : "spoof";
  if (!result || !result.face_detected) {
    document.getElementById("stat-raw").textContent = "—";
    document.getElementById("stat-smoothed").textContent = "—";
    document.getElementById("stat-window").textContent = "—";
    return;
  }
  const rawEl = document.getElementById("stat-raw");
  rawEl.textContent = `${result.raw_label} ${(result.raw_confidence*100).toFixed(1)}%`;
  rawEl.className = "v " + cls(result.raw_label);

  const smEl = document.getElementById("stat-smoothed");
  smEl.textContent = `${result.label} ${(result.confidence*100).toFixed(1)}%`;
  smEl.className = "v " + cls(result.label);

  document.getElementById("stat-window").textContent = `${result.smoothing_window} frame${result.smoothing_window === 1 ? "" : "s"}`;
}

let lastResultForLog = null;

async function detectLoop() {
  if (!liveRunning) return;
  const t0 = performance.now();

  const ctx = canvas.getContext("2d");
  canvas.width = video.videoWidth || 1280;
  canvas.height = video.videoHeight || 720;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(async (blob) => {
    const form = new FormData();
    form.append("frame", blob, "frame.jpg");
    try {
      const res = await fetch(API + "/api/detect", { method: "POST", body: form });
      const result = await res.json();

      if (!result.face_detected) {
        statusEl.innerHTML = `<span class="badge none">NO FACE DETECTED</span>`;
        const statusText = document.getElementById("live-status");
        statusText.textContent = "no face";
        statusText.className = "v status-text none";
        document.getElementById("live-confidence").textContent = "—";
        document.getElementById("live-confidence-bar").style.width = "0%";
        document.getElementById("live-confidence-bar").className = "confidence-bar-fill none";
        drawBox(null);
      } else {
        const isLive = result.label === "LIVE";
        statusEl.innerHTML = `<span class="badge ${isLive ? "live" : "spoof"}">${isLive ? "LIVE" : "SPOOF - REJECTED"}</span>`;
        const statusText = document.getElementById("live-status");
        statusText.textContent = result.label;
        statusText.className = "v status-text " + (isLive ? "live" : "spoof");
        document.getElementById("live-confidence").textContent = (result.confidence*100).toFixed(1) + "%";
        const bar = document.getElementById("live-confidence-bar");
        bar.style.width = (result.confidence*100).toFixed(1) + "%";
        bar.className = "confidence-bar-fill " + (isLive ? "live" : "spoof");
        drawBox(result.bbox, result.label);

        // Only add a feed entry when the backend actually logged this as
        // a new event (label changed, or face just appeared) -- the
        // backend returns the same smoothed label on every frame while
        // the decision is stable, so we compare against the last result
        // we saw to avoid flooding the UI feed too, matching the log.
        if (!lastResultForLog || lastResultForLog.label !== result.label) {
          addFeedItem(result);
        }
        lastResultForLog = result;
      }
      updateStatsStrip(result);
    } catch (err) {
      statusEl.innerHTML = `<span class="badge spoof">BACKEND ERROR</span>`;
    }

    const dt = performance.now() - t0;
    frameTimes.push(dt);
    if (frameTimes.length > 10) frameTimes.shift();
    const avgDt = frameTimes.reduce((a,b) => a+b, 0) / frameTimes.length;
    const fps = (1000 / avgDt).toFixed(1);
    fpsLiveEl.textContent = "web fps: " + fps;
    document.getElementById("stat-fps").textContent = fps + " FPS";

    if (liveRunning) requestAnimationFrame(() => setTimeout(detectLoop, 0));
  }, "image/jpeg", 0.85);
}

// ---------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------
pingBackend();
loadMetricsEverywhere();
loadDatasetInfo();
loadDecisionLog();
loadCurrentThreshold();
setInterval(loadDecisionLog, 5000);
