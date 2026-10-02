// utils.js -- small formatting helpers shared across page modules.
export function pct(x, digits = 2) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return (x * 100).toFixed(digits) + "%";
}

export function ms(x, digits = 1) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return x.toFixed(digits) + "ms";
}

export function statusClass(label) {
  if (label === "LIVE") return "live";
  if (label === "SPOOF") return "spoof";
  if (label === "UNCERTAIN") return "uncertain";
  return "none";
}

export function fmtNumber(n) {
  if (n === null || n === undefined) return "—";
  return n.toLocaleString();
}

export function el(tag, className, html) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (html !== undefined) e.innerHTML = html;
  return e;
}
