import { api } from "./api.js";
import { pct } from "./utils.js";
import { horizontalBarChart } from "./charts.js";
import { openModal } from "./main.js";

// Same source image (test_5966_000293, a real PC-replay false accept),
// Grad-CAM'd under both checkpoints -- a true same-image before/after,
// not two different examples.
const BEFORE_AFTER = {
  v1: { file: "001_false_accept_test_5966_000293.png.png", prob: 0.011784904636442661 },
  v2: { file: "000_false_accept_test_5966_000293.png.png", prob: 0.4250963032245636 },
};

export async function initRobustnessPage(root) {
  root.innerHTML = `
    <h1 class="page-title">Robustness</h1>
    <p class="page-sub">The train/deploy crop mismatch found mid-project, the fix, and what it changed.</p>

    <div class="panel">
      <h3>Finding &amp; Fix: v1 vs v2 on Deployment Crops</h3>
      <div class="panel-sub" id="finding-protocol"></div>
      <div class="finding-grid">
        <div>
          <table><tbody id="finding-table"></tbody></table>
        </div>
        <div>
          <div class="panel-sub">Same real false-accept image (PC replay), Grad-CAM under each checkpoint &mdash; v1's attention lands on the collar/neck, v2's lands on the face.</div>
          <div class="grid grid-2">
            <div>
              <img id="ba-v1" style="width:100%;border-radius:4px;border:1px solid var(--border)">
              <div class="panel-sub" style="margin-top:4px">v1 &middot; P(spoof)=<span id="ba-v1-prob" class="mono"></span></div>
            </div>
            <div>
              <img id="ba-v2" style="width:100%;border-radius:4px;border:1px solid var(--border)">
              <div class="panel-sub" style="margin-top:4px">v2 &middot; P(spoof)=<span id="ba-v2-prob" class="mono"></span></div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="grid grid-3" style="margin-top:16px">
      <div class="panel"><h3>APCER by Attack Type</h3><div class="chart-box"><canvas id="chart-spoof-type"></canvas></div></div>
      <div class="panel"><h3>APCER by Illumination</h3><div class="chart-box"><canvas id="chart-illum"></canvas></div></div>
      <div class="panel"><h3>APCER by Environment</h3><div class="chart-box"><canvas id="chart-env"></canvas></div></div>
    </div>

    <div class="panel" style="margin-top:16px">
      <h3>Masked vs Unmasked (synthetic masks, v2)</h3>
      <div class="panel-sub" id="masked-protocol"></div>
      <div class="grid grid-4" id="masked-kpis"></div>
      <div class="panel-sub" style="margin-top:12px">Example masked test images (white / light blue / black, nose-to-chin):</div>
      <div class="thumb-grid" id="masked-thumbs" style="margin-top:8px"></div>
    </div>

    <div class="panel" style="margin-top:16px">
      <h3>Grad-CAM Gallery (v2)</h3>
      <div class="filter-row" id="gallery-filters"></div>
      <div class="panel-sub" id="gallery-count"></div>
      <div class="thumb-grid" id="gallery-thumbs"></div>
    </div>

    <div id="real-world-section"></div>
  `;

  let summary;
  try {
    summary = await api.summary();
  } catch (e) {
    return;
  }

  renderFinding(summary);
  renderBreakdownCharts(summary);
  renderMasked(summary);
  renderGallery();
  renderRealWorldIfPresent(summary);
}

function renderFinding(summary) {
  const v1v2 = summary.v1_vs_v2;
  document.getElementById("finding-protocol").textContent = v1v2.protocol;
  document.getElementById("finding-table").innerHTML = `
    <tr><th>Metric</th><th>v1 (raw frame)</th><th>v2 (face crop)</th></tr>
    ${["accuracy", "apcer", "bpcer", "hter"].map((k) => `
      <tr><td>${k.toUpperCase()}</td><td class="mono">${pct(v1v2.v1[k])}</td><td class="mono">${pct(v1v2.v2[k])}</td></tr>
    `).join("")}
  `;
  document.getElementById("ba-v1").src = api.gradcamV1ImageUrl(BEFORE_AFTER.v1.file);
  document.getElementById("ba-v1-prob").textContent = BEFORE_AFTER.v1.prob.toFixed(4);
  document.getElementById("ba-v2").src = api.gradcamImageUrl(BEFORE_AFTER.v2.file);
  document.getElementById("ba-v2-prob").textContent = BEFORE_AFTER.v2.prob.toFixed(4);
}

function renderBreakdownCharts(summary) {
  const b = summary.breakdowns.mobilenet_v3_small;
  if (!b) return;

  const spoofRows = b.spoof_type_breakdown;
  const pcIdx = spoofRows.findIndex((r) => r.name === "PC");
  horizontalBarChart(document.getElementById("chart-spoof-type"),
    spoofRows.map((r) => r.name), spoofRows.map((r) => r.apcer * 100),
    { highlightIndex: pcIdx, counts: spoofRows.map((r) => r.n) });

  const illumRows = b.illumination_breakdown;
  horizontalBarChart(document.getElementById("chart-illum"),
    illumRows.map((r) => r.name), illumRows.map((r) => r.apcer * 100),
    { counts: illumRows.map((r) => r.n) });

  const envRows = b.environment_breakdown;
  horizontalBarChart(document.getElementById("chart-env"),
    envRows.map((r) => r.name), envRows.map((r) => r.apcer * 100),
    { counts: envRows.map((r) => r.n) });
}

function renderMasked(summary) {
  const me = summary.masked_eval;
  document.getElementById("masked-protocol").textContent = me.protocol;
  const masked = me.masked_metrics || {};
  const unmasked = me.unmasked_baseline_metrics || {};
  document.getElementById("masked-kpis").innerHTML = ["accuracy", "apcer", "bpcer", "hter"].map((k) => `
    <div class="kpi">
      <div class="figure">${pct(masked[k])}</div>
      <div class="label">${k.toUpperCase()} masked</div>
      <div class="protocol">unmasked: ${pct(unmasked[k])}</div>
    </div>
  `).join("");

  api.maskedExamples().then(({ examples }) => {
    const grid = document.getElementById("masked-thumbs");
    grid.innerHTML = examples.map((ex) => `
      <div class="thumb" data-file="${ex.file}">
        <img src="${api.maskedImageUrl(ex.file)}" loading="lazy">
        <div class="cap">${ex.color || ""} &middot; ${ex.true_label || ""}</div>
      </div>
    `).join("");
    grid.querySelectorAll(".thumb").forEach((t) => t.addEventListener("click", () => {
      openModal(`<img src="${api.maskedImageUrl(t.dataset.file)}"><div class="panel-sub">${t.dataset.file}</div>`);
    }));
  }).catch(() => {});
}

let galleryData = [];
let activeCategoryFilter = "all";
let activeTypeFilter = "all";

async function renderGallery() {
  try {
    galleryData = await api.gradcamGallery();
  } catch (e) {
    return;
  }
  const categories = ["all", ...new Set(galleryData.map((g) => g.category))];
  const types = ["all", ...new Set(galleryData.map((g) => g.spoof_type).filter(Boolean))];

  const filterRow = document.getElementById("gallery-filters");
  filterRow.innerHTML =
    categories.map((c) => `<span class="filter-chip cat-chip ${c === "all" ? "active" : ""}" data-cat="${c}">${c}</span>`).join("") +
    types.map((t) => `<span class="filter-chip type-chip" data-type="${t}">${t}</span>`).join("");

  filterRow.querySelectorAll(".cat-chip").forEach((chip) => chip.addEventListener("click", () => {
    activeCategoryFilter = chip.dataset.cat;
    filterRow.querySelectorAll(".cat-chip").forEach((c) => c.classList.toggle("active", c === chip));
    renderGalleryThumbs();
  }));
  filterRow.querySelectorAll(".type-chip").forEach((chip) => chip.addEventListener("click", () => {
    activeTypeFilter = chip.dataset.type;
    filterRow.querySelectorAll(".type-chip").forEach((c) => c.classList.toggle("active", c === chip));
    renderGalleryThumbs();
  }));

  renderGalleryThumbs();
}

function renderGalleryThumbs() {
  const filtered = galleryData.filter((g) =>
    (activeCategoryFilter === "all" || g.category === activeCategoryFilter) &&
    (activeTypeFilter === "all" || activeTypeFilter === undefined || g.spoof_type === activeTypeFilter));
  const SHOWN_CAP = 24;
  const shown = filtered.slice(0, SHOWN_CAP);
  const grid = document.getElementById("gallery-thumbs");
  grid.innerHTML = shown.map((g) => `
    <div class="thumb" data-file="${g.file}">
      <img src="${api.gradcamImageUrl(g.file)}" loading="lazy">
      <div class="cap">${g.spoof_type || g.true_label} &middot; P=${g.spoof_probability.toFixed(2)}</div>
    </div>
  `).join("") || '<div class="empty-state">No tiles match this filter.</div>';
  document.getElementById("gallery-count").textContent =
    filtered.length > SHOWN_CAP ? `Showing ${SHOWN_CAP} of ${filtered.length} -- use the filters to narrow further.` : `${filtered.length} tile${filtered.length === 1 ? "" : "s"}.`;

  grid.querySelectorAll(".thumb").forEach((t) => t.addEventListener("click", () => {
    const g = filtered.find((x) => x.file === t.dataset.file);
    openModal(`
      <img src="${api.gradcamImageUrl(g.file)}">
      <table><tbody>
        <tr><td>True label</td><td class="mono">${g.true_label}</td></tr>
        <tr><td>Predicted</td><td class="mono">${g.predicted_label}</td></tr>
        <tr><td>Spoof type</td><td class="mono">${g.spoof_type}</td></tr>
        <tr><td>P(spoof)</td><td class="mono">${g.spoof_probability.toFixed(4)}</td></tr>
        <tr><td>Category</td><td class="mono">${g.category}</td></tr>
      </tbody></table>
    `);
  }));
}

function renderRealWorldIfPresent(summary) {
  // No placeholder: this section only renders if real-world results
  // actually exist somewhere in summary.json. They don't yet (TASK J
  // only prepared the recording instructions and evaluation script), so
  // this intentionally renders nothing.
  if (summary.real_world) {
    const rw = summary.real_world;
    document.getElementById("real-world-section").innerHTML = `
      <div class="panel" style="margin-top:16px">
        <h3>Real-World Results</h3>
        <div class="panel-sub">${rw.protocol || ""}</div>
        <div class="grid grid-4">${["accuracy", "apcer", "bpcer", "hter"].map((k) => `
          <div class="kpi"><div class="figure">${pct(rw[k])}</div><div class="label">${k.toUpperCase()}</div></div>
        `).join("")}</div>
      </div>
    `;
  }
}
