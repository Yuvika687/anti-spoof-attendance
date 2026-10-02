// charts.js -- Chart.js instances themed to match the console design
// system. Colors are read from the real CSS custom properties (single
// source of truth with style.css), not duplicated as separate constants.
const cs = getComputedStyle(document.documentElement);
const COLOR = {
  text: cs.getPropertyValue("--text").trim(),
  muted: cs.getPropertyValue("--text-muted").trim(),
  border: cs.getPropertyValue("--border").trim(),
  live: cs.getPropertyValue("--live").trim(),
  spoof: cs.getPropertyValue("--spoof").trim(),
  uncertain: cs.getPropertyValue("--uncertain").trim(),
  accent: cs.getPropertyValue("--accent").trim(),
};
export { COLOR };

Chart.defaults.color = COLOR.muted;
Chart.defaults.borderColor = COLOR.border;
Chart.defaults.font.family = "'Inter', system-ui, sans-serif";

function baseGrid() {
  return { color: COLOR.border, drawTicks: false };
}

export function horizontalBarChart(canvas, labels, values, { color = COLOR.accent, highlightIndex = null, highlightColor = COLOR.spoof, suffix = "%", counts = null } = {}) {
  const colors = labels.map((_, i) => (highlightIndex !== null && i === highlightIndex ? highlightColor : color));
  return new Chart(canvas, {
    type: "bar",
    data: {
      labels,
      datasets: [{ data: values, backgroundColor: colors, borderRadius: 2, barThickness: 16 }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const n = counts ? ` (n=${counts[ctx.dataIndex]})` : "";
              return `${ctx.parsed.x.toFixed(2)}${suffix}${n}`;
            },
          },
        },
        datalabels: false,
      },
      scales: {
        x: { grid: baseGrid(), ticks: { font: { family: "'JetBrains Mono', monospace", size: 10.5 } } },
        y: { grid: { display: false }, ticks: { font: { size: 11.5 } } },
      },
    },
  });
}

export function lineChart(canvas, labels, datasets) {
  return new Chart(canvas, {
    type: "line",
    data: {
      labels,
      datasets: datasets.map((d) => ({
        ...d,
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.15,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { font: { size: 11.5 } }, boxWidth: 14 } },
      scales: {
        x: { grid: baseGrid(), ticks: { font: { family: "'JetBrains Mono', monospace", size: 10 } } },
        y: { grid: baseGrid(), ticks: { font: { family: "'JetBrains Mono', monospace", size: 10 } } },
      },
    },
  });
}

export function scatterChart(canvas, points, { xLabel = "", yLabel = "" } = {}) {
  return new Chart(canvas, {
    type: "bubble",
    data: {
      datasets: [
        {
          data: points,
          backgroundColor: points.map((p) => p.color || COLOR.accent),
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ctx.raw.label || "",
          },
        },
      },
      scales: {
        x: { title: { display: true, text: xLabel, color: COLOR.muted, font: { size: 11 } }, grid: baseGrid(), ticks: { font: { family: "'JetBrains Mono', monospace", size: 10 } } },
        y: { title: { display: true, text: yLabel, color: COLOR.muted, font: { size: 11 } }, grid: baseGrid(), ticks: { font: { family: "'JetBrains Mono', monospace", size: 10 } } },
      },
    },
  });
}

export function coverageErrorChart(canvas, curve, chosenPoint) {
  return new Chart(canvas, {
    type: "line",
    data: {
      datasets: [
        {
          label: "coverage vs HTER",
          data: curve.map((c) => ({ x: c.coverage * 100, y: c.hter * 100 })),
          borderColor: COLOR.accent,
          backgroundColor: COLOR.accent,
          borderWidth: 2,
          pointRadius: 0,
          showLine: true,
        },
        {
          label: "chosen band",
          data: [{ x: chosenPoint.coverage * 100, y: chosenPoint.hter * 100 }],
          borderColor: COLOR.spoof,
          backgroundColor: COLOR.spoof,
          pointRadius: 5,
          showLine: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { font: { size: 11 } }, boxWidth: 12 } },
      scales: {
        x: { type: "linear", title: { display: true, text: "coverage (%)", color: COLOR.muted, font: { size: 11 } }, grid: baseGrid(), ticks: { font: { family: "'JetBrains Mono', monospace", size: 10 } } },
        y: { title: { display: true, text: "HTER on decided (%)", color: COLOR.muted, font: { size: 11 } }, grid: baseGrid(), ticks: { font: { family: "'JetBrains Mono', monospace", size: 10 } } },
      },
    },
  });
}
