import { api } from "./api.js";
import { pct, ms, fmtNumber } from "./utils.js";
import { scatterChart, COLOR } from "./charts.js";

const BACKBONE_NAME = { mobilenet_v3_small: "MobileNetV3-Small", resnet18: "ResNet-18", efficientnet_b0: "EfficientNet-B0" };
const DEPLOYED = "mobilenet_v3_small";

export async function initModelsPage(root) {
  root.innerHTML = `
    <h1 class="page-title">Models</h1>
    <p class="page-sub">All three measured on the real v2 (face-crop) training/eval pipeline, identical recipe.</p>

    <div class="panel">
      <h3>Comparison</h3>
      <table id="model-table"></table>
    </div>

    <div class="grid grid-2" style="margin-top:16px">
      <div class="panel">
        <h3>HTER vs Latency</h3>
        <div class="btn-group" style="margin-bottom:10px">
          <button id="scatter-onnx-btn" class="active">CPU (ONNX)</button>
          <button id="scatter-mps-btn">MPS</button>
        </div>
        <div class="chart-box"><canvas id="chart-scatter"></canvas></div>
        <div class="panel-sub">Bubble size = parameter count.</div>
      </div>
      <div class="panel">
        <h3>Why MobileNetV3-Small</h3>
        <p id="why-note" style="font-size:13px;line-height:1.6"></p>
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

  const bc = summary.backbone_comparison;
  const latency = bc.latency_and_size;
  if (!latency) {
    root.innerHTML += '<div class="panel" style="margin-top:16px">Latency/size data not available -- run benchmark_backbones.py and benchmark_onnx.py.</div>';
    return;
  }

  renderTable(bc, latency);
  let scatterInstance = renderScatter(bc, latency, "cpu_onnx_ms");
  document.getElementById("scatter-onnx-btn").addEventListener("click", (e) => {
    toggleScatterBtns(e.target);
    scatterInstance.destroy();
    scatterInstance = renderScatter(bc, latency, "cpu_onnx_ms");
  });
  document.getElementById("scatter-mps-btn").addEventListener("click", (e) => {
    toggleScatterBtns(e.target);
    scatterInstance.destroy();
    scatterInstance = renderScatter(bc, latency, "latency_mps_ms_median");
  });

  renderWhyNote(bc, latency);
}

function toggleScatterBtns(clicked) {
  document.querySelectorAll("#chart-scatter").forEach(() => {});
  document.getElementById("scatter-onnx-btn").classList.toggle("active", clicked.id === "scatter-onnx-btn");
  document.getElementById("scatter-mps-btn").classList.toggle("active", clicked.id === "scatter-mps-btn");
}

function renderTable(bc, latency) {
  const backbones = ["mobilenet_v3_small", "resnet18", "efficientnet_b0"];
  const rows = backbones.map((b) => {
    const m = bc[b];
    const l = latency[b] || {};
    const deployed = b === DEPLOYED;
    return `
      <tr style="${deployed ? "background:rgba(56,189,248,0.06)" : ""}">
        <td>${BACKBONE_NAME[b]}${deployed ? ' <span class="badge live" style="margin-left:6px">deployed</span>' : ""}</td>
        <td class="mono">${fmtNumber(l.param_count)}</td>
        <td class="mono">${l.model_file_size_mb ? l.model_file_size_mb.toFixed(2) + "MB" : "—"}</td>
        <td class="mono">${ms(l.latency_cpu_ms_median)}</td>
        <td class="mono">${ms(l.cpu_onnx_ms)}</td>
        <td class="mono">${ms(l.latency_mps_ms_median)}</td>
        <td class="mono">${pct(m.hter)}</td>
        <td class="mono">${pct(m.apcer)}</td>
        <td class="mono">${pct(m.bpcer)}</td>
      </tr>`;
  }).join("");

  document.getElementById("model-table").innerHTML = `
    <tr><th>Backbone</th><th>Params</th><th>File size</th><th>CPU (PyTorch)</th><th>CPU (ONNX)</th><th>MPS</th><th>HTER</th><th>APCER</th><th>BPCER</th></tr>
    ${rows}
  `;
}

function renderScatter(bc, latency, latencyKey) {
  const backbones = ["mobilenet_v3_small", "resnet18", "efficientnet_b0"];
  const points = backbones.map((b) => {
    const l = latency[b] || {};
    const m = bc[b];
    const r = Math.max(6, Math.sqrt(l.param_count || 0) / 90);
    return {
      x: l[latencyKey], y: m.hter * 100, r,
      label: `${BACKBONE_NAME[b]}: ${ms(l[latencyKey])}, HTER ${pct(m.hter)}, ${fmtNumber(l.param_count)} params`,
      color: b === DEPLOYED ? COLOR.accent : COLOR.muted,
    };
  });
  return scatterChart(document.getElementById("chart-scatter"), points, {
    xLabel: latencyKey === "cpu_onnx_ms" ? "CPU (ONNX) latency, ms" : "MPS latency, ms",
    yLabel: "HTER (%)",
  });
}

function renderWhyNote(bc, latency) {
  const mb = latency.mobilenet_v3_small;
  const r18 = latency.resnet18;
  const eff = latency.efficientnet_b0;
  const mbHter = bc.mobilenet_v3_small.hter;
  const r18Hter = bc.resnet18.hter;
  const effHter = bc.efficientnet_b0.hter;

  const sizeRatio = (r18.model_file_size_mb / mb.model_file_size_mb).toFixed(1);
  const onnxFastest = mb.cpu_onnx_ms < r18.cpu_onnx_ms && mb.cpu_onnx_ms < eff.cpu_onnx_ms;
  const mostAccurate = mbHter < r18Hter && mbHter < effHter;

  let note = `MobileNetV3-Small is the smallest checkpoint measured (${mb.model_file_size_mb.toFixed(2)}MB, ${sizeRatio}x smaller than ResNet-18's ${r18.model_file_size_mb.toFixed(2)}MB)`;
  if (onnxFastest) {
    note += ` and the fastest of the three under ONNX Runtime on CPU (${ms(mb.cpu_onnx_ms)} vs ${ms(r18.cpu_onnx_ms)} for ResNet-18 and ${ms(eff.cpu_onnx_ms)} for EfficientNet-B0)`;
  }
  note += ". ";

  if (!mostAccurate) {
    const best = effHter < r18Hter ? "EfficientNet-B0" : "ResNet-18";
    const bestHter = effHter < r18Hter ? effHter : r18Hter;
    note += `It is not the most accurate of the three on this test set -- ${best} measured a lower HTER (${pct(bestHter)} vs MobileNetV3-Small's ${pct(mbHter)}). `;
    note += `The size/CPU-latency advantage is the actual basis for keeping it deployed, not accuracy.`;
  } else {
    note += `It also measured the lowest HTER of the three on this test set (${pct(mbHter)}).`;
  }

  document.getElementById("why-note").textContent = note;
}
