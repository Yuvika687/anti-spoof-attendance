// api.js -- every backend call the UI makes, in one place. No page module
// fetches raw endpoints directly; everything goes through here so the
// base URL and error handling live in exactly one spot.
export const API = window.location.origin;

async function getJSON(path) {
  const res = await fetch(API + path);
  if (!res.ok) throw new Error(`${path} -> HTTP ${res.status}`);
  return res.json();
}

async function postJSON(path, body) {
  const opts = { method: "POST" };
  if (body) {
    opts.body = body;
  }
  const res = await fetch(API + path, opts);
  return res.json();
}

export const api = {
  modelInfo: () => getJSON("/api/model-info"),
  summary: () => getJSON("/api/summary"),
  metrics: () => getJSON("/api/metrics"),
  trainingHistory: () => getJSON("/api/training-history"),
  decisionLog: (limit = 10) => getJSON(`/api/decision-log?limit=${limit}`),
  thresholdMetrics: (threshold) => getJSON(`/api/threshold-metrics?threshold=${threshold}`),
  getThreshold: () => getJSON("/api/threshold"),
  setThreshold: (threshold) =>
    fetch(API + "/api/threshold", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ threshold }),
    }).then((r) => r.json()),
  datasetInfo: () => getJSON("/api/dataset-info"),

  gradcamGallery: () => getJSON("/api/gradcam/gallery"),
  gradcamImageUrl: (file) => `${API}/api/gradcam/image/${encodeURIComponent(file)}`,
  gradcamV1ImageUrl: (file) => `${API}/api/gradcam/v1/image/${encodeURIComponent(file)}`,
  gradcamV1Gallery: () => getJSON("/api/gradcam/v1/gallery"),

  maskedExamples: () => getJSON("/api/masked/examples"),
  maskedImageUrl: (file) => `${API}/api/masked/image/${encodeURIComponent(file)}`,

  plotUrl: (name) => `${API}/api/plots/${name}`,

  sampleImages: () => getJSON("/api/sample-images"),
  sampleImageUrl: (id) => `${API}/api/sample-images/${id}/image`,
  sampleDetect: (id) => postJSON(`/api/sample-images/${id}/detect`),
  sampleExplain: (id) => postJSON(`/api/sample-images/${id}/explain`),

  detectFrame: (blob) => {
    const form = new FormData();
    form.append("frame", blob, "frame.jpg");
    return postJSON("/api/detect", form);
  },
  explainFrame: (blob) => {
    const form = new FormData();
    form.append("frame", blob, "frame.jpg");
    return postJSON("/api/explain", form);
  },
};
