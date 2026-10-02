import { api } from "./api.js";
import { pct, ms } from "./utils.js";
import { lineChart, coverageErrorChart, COLOR } from "./charts.js";

const KPI_MEANING = {
  accuracy: "Fraction of all test samples correctly classified.",
  apcer: "Attacks wrongly accepted as live.",
  bpcer: "Genuine faces wrongly rejected as spoof.",
  hter: "Average of APCER and BPCER.",
};

export async function initResultsPage(root) {
  root.innerHTML = `
    <h1 class="page-title">Results</h1>
    <p class="page-sub">Headline numbers for the deployed model (v2 MobileNetV3-Small), on the canonical protocol.</p>

    <div class="grid grid-4" id="kpi-row"></div>

    <div class="grid grid-2" style="margin-top:16px">
      <div class="panel">
        <h3>Confusion Matrix</h3>
        <div class="panel-sub" id="cm-protocol"></div>
        <div class="cm-grid" id="cm-grid"></div>
      </div>
      <div class="panel">
        <h3>Training Curves</h3>
        <div class="panel-sub">Per-epoch loss and accuracy, v2's real training run.</div>
        <div class="chart-box short"><canvas id="chart-loss"></canvas></div>
        <div class="chart-box short" style="margin-top:8px"><canvas id="chart-acc"></canvas></div>
      </div>
    </div>

    <div class="panel" style="margin-top:16px">
      <h3>Threshold Tuner</h3>
      <div class="panel-sub">Recomputed live from the real stored test-set probabilities -- not an approximation.</div>
      <input type="range" id="threshold-slider" min="0" max="1" step="0.01" value="0.5" style="width:100%">
      <div class="grid grid-4" style="margin-top:10px" id="threshold-readout"></div>
    </div>

    <div class="panel" style="margin-top:16px">
      <h3>Three-Way Decision (Gray Zone)</h3>
      <div class="panel-sub" id="gz-protocol"></div>
      <div class="grid grid-2">
        <div class="chart-box"><canvas id="chart-gz"></canvas></div>
        <div>
          <table><tbody id="gz-table"></tbody></table>
        </div>
      </div>
    </div>
  `;

  let summary;
  try {
    summary = await api.summary();
  } catch (e) {
    root.innerHTML += '<div class="panel" style="margin-top:16px">Could not load results/summary.json.</div>';
    return;
  }

  renderKPIs(summary);
  renderConfusionMatrix(summary);
  renderTrainingCurves();
  renderGrayZone(summary);
  wireThresholdTuner(summary);
}

function renderKPIs(summary) {
  const h = summary.headline;
  const row = document.getElementById("kpi-row");
  row.innerHTML = ["accuracy", "apcer", "bpcer", "hter"].map((key) => `
    <div class="kpi">
      <div class="figure">${pct(h[key])}</div>
      <div class="label">${key.toUpperCase()}</div>
      <div class="meaning">${KPI_MEANING[key]}</div>
      <div class="protocol">n=${h.n}</div>
    </div>
  `).join("");
}

function renderConfusionMatrix(summary) {
  const h = summary.headline;
  const m = summary.v1_vs_v2.v2; // has n_live/n_spoof alongside apcer/bpcer
  const nLive = m.n_live, nSpoof = m.n_spoof;
  const liveCorrect = Math.round(nLive * (1 - m.bpcer));
  const liveWrong = nLive - liveCorrect;
  const spoofWrong = Math.round(nSpoof * m.apcer);
  const spoofCorrect = nSpoof - spoofWrong;

  document.getElementById("cm-protocol").textContent = summary.headline.protocol.split(";")[0] + `; n=${h.n}`;
  document.getElementById("cm-grid").innerHTML = `
    <div class="cm-cell head"></div>
    <div class="cm-cell head">Predicted LIVE</div>
    <div class="cm-cell head">Predicted SPOOF</div>
    <div class="cm-cell head">Actual LIVE</div>
    <div class="cm-cell correct"><div class="n mono">${liveCorrect}</div></div>
    <div class="cm-cell wrong"><div class="n mono">${liveWrong}</div></div>
    <div class="cm-cell head">Actual SPOOF</div>
    <div class="cm-cell wrong"><div class="n mono">${spoofWrong}</div></div>
    <div class="cm-cell correct"><div class="n mono">${spoofCorrect}</div></div>
  `;
}

async function renderTrainingCurves() {
  try {
    const hist = await api.trainingHistory();
    const epochs = hist.train_loss.map((_, i) => i + 1);
    lineChart(document.getElementById("chart-loss"), epochs, [
      { label: "train loss", data: hist.train_loss, borderColor: COLOR.accent, backgroundColor: COLOR.accent },
      { label: "val loss", data: hist.val_loss, borderColor: COLOR.spoof, backgroundColor: COLOR.spoof },
    ]);
    lineChart(document.getElementById("chart-acc"), epochs, [
      { label: "train acc", data: hist.train_acc, borderColor: COLOR.accent, backgroundColor: COLOR.accent },
      { label: "val acc", data: hist.val_acc, borderColor: COLOR.live, backgroundColor: COLOR.live },
    ]);
  } catch (e) {
    document.querySelector(".chart-box.short").parentElement.innerHTML += '<div class="empty-state">Training history unavailable.</div>';
  }
}

async function wireThresholdTuner(summary) {
  const slider = document.getElementById("threshold-slider");
  const current = summary.headline; // start view matches threshold=0.5 headline
  slider.value = 0.5;
  renderThresholdReadout({ accuracy: current.accuracy, apcer: current.apcer, bpcer: current.bpcer, hter: current.hter });

  let debounce;
  slider.addEventListener("input", () => {
    clearTimeout(debounce);
    debounce = setTimeout(async () => {
      try {
        const m = await api.thresholdMetrics(parseFloat(slider.value));
        renderThresholdReadout(m);
      } catch (e) { /* ignore transient errors */ }
    }, 120);
  });
}

function renderThresholdReadout(m) {
  const row = document.getElementById("threshold-readout");
  row.innerHTML = ["accuracy", "apcer", "bpcer", "hter"].map((key) => `
    <div class="kpi"><div class="figure">${pct(m[key])}</div><div class="label">${key.toUpperCase()}</div></div>
  `).join("");
}

function renderGrayZone(summary) {
  const gz = summary.gray_zone;
  document.getElementById("gz-protocol").textContent = gz.protocol || "";

  if (gz.coverage_vs_error_curve && gz.test_gray_zone_decided_only) {
    coverageErrorChart(document.getElementById("chart-gz"), gz.coverage_vs_error_curve, {
      coverage: gz.test_gray_zone_decided_only.coverage,
      hter: gz.test_gray_zone_decided_only.hter,
    });
  }

  const faCaught = gz.false_accepts_rescued_to_uncertain;
  const faTotal = gz.false_accepts_total;
  const frCaught = gz.false_rejects_rescued_to_uncertain;
  const frTotal = gz.false_rejects_total;
  const decided = gz.test_gray_zone_decided_only || {};
  const baseline = gz.test_baseline_plain_threshold || {};

  document.getElementById("gz-table").innerHTML = `
    <tr><td>Band</td><td class="mono">[${gz.band_low?.toFixed(4)}, ${gz.band_high?.toFixed(4)}]</td></tr>
    <tr><td>Coverage</td><td class="mono">${pct(decided.coverage)}</td></tr>
    <tr><td>False accepts caught by UNCERTAIN</td><td class="mono">${faCaught} of ${faTotal}</td></tr>
    <tr><td>False rejects caught by UNCERTAIN</td><td class="mono">${frCaught} of ${frTotal}</td></tr>
    <tr><td>HTER, baseline (100% coverage)</td><td class="mono">${pct(baseline.hter)}</td></tr>
    <tr><td>HTER, decided subset only</td><td class="mono">${pct(decided.hter)}</td></tr>
  `;
}
