// Grantha Boundary Annotator -- vertical split-line editor.
//
// Data model: `annotation` mirrors the backend schema:
//   { page_id, image_path, dpi, image_width_px, image_height_px, line_source,
//     lines: [ { line_id, y0, y1, boundaries_px: [...], boundary_source: [...],
//                akshara_labels: [...] } ] }
// boundaries_px[0]/boundaries_px[last] are the line's own left/right edges
// -- by default the full page width (0 and image_width_px), but they CAN be
// dragged inward (or trimmed via Scissor) to exclude scanner noise at the
// page margins from the line entirely (see getLineX0/getLineX1). They can
// never be individually deleted, though (a line always needs at least one
// left and one right edge) -- trimming works by promoting the next interior
// boundary to take the edge's place instead (see trimLineEdge).

const state = {
  pageId: "page_0002",
  image: null,
  annotation: null,
  zoom: 1,
  panX: 0,
  panY: 0,
  selectedLine: -1,
  selectedBoundary: -1, // index into boundaries_px of the selected line
  dragging: false,
  dragMoved: false,
  spaceDown: false,
  panning: false,
  panStart: null,
  dirty: false,
  showSeams: false,
  seams: null, // { line_id: [y_per_x, ...], ... } once fetched
  splitMode: false,
  splitPoints: [], // [{x,y}, ...] image-space points placed while tracing a new separator
  newLineMode: false, // false | 'top' | 'bottom' -- which edge is currently being traced
  newLineTopPoints: [],
  newLineBottomPoints: [],
  extendingLineIdx: null, // index into annotation.lines when the current newLineMode trace is "Extend line" (modifies that line) instead of "New line" (creates a new one) -- same top(cyan)/bottom(pink) tracing UI either way
  // Pencil tool for Split line / New line tracing: click-and-drag streams a
  // dense run of points (freehand); a plain click (no real movement) just
  // adds one point as before ("continuation as usual"); holding Shift while
  // dragging collapses the in-progress stroke to a single straight segment
  // from where the drag started to the current cursor -- see the shared
  // mousedown/mousemove/mouseup handling and activePencilPoints().
  freehandDragging: false,
  freehandMoved: false, // did this mousedown->mouseup actually move (drag) vs. a plain click?
  freehandStrokePointCount: 0, // points added by the in-progress stroke so far -- lets Shift "redo" them as one segment
  freehandStart: null, // {x,y} image-space point where the current drag began
  freehandJustDragged: false, // set on mouseup after a real drag, so the click event right after it doesn't also add a point
  scissorMode: false, // armed/disarmed by the Scissor button; stays armed across multiple cuts
  scissorDragging: false,
  scissorPoints: [], // image-space points collected during one in-progress cut stroke
  showAnnotation: false, // off by default -- page loads blank; toggle "Show annotation" on to see it
  showModelOutput: false,
  modelOutput: null, // { lines: [...] } from /model-guide, shown as a read-only overlay
  showYolo: null, // null | "original" | "finetuned" -- which YOLO26n-seg preview (if any) is toggled on
  yoloOutput: null, // { variant, instances: [{polygon_px, class_name, confidence}, ...] } from /yolo-guide
  trainingPoll: { boundary: null, line_segmenter: null }, // interval handles
  versions: [], // cached list from /versions, newest first
  activeVersionId: null, // which version_id is currently loaded, for highlighting
};

// A line's top/bottom edge is either flat (line.y0/line.y1) or, once split at
// least once, a full-page-width curved seam (line.top_seam_px/bottom_seam_px,
// one y-value per column) -- these two helpers are the single place that
// distinction is resolved, so rendering/hit-testing/splitting all agree.
function getTopY(line, x) {
  return line.top_seam_px ? line.top_seam_px[clampCol(x, line.top_seam_px.length)] : line.y0;
}
function getBottomY(line, x) {
  return line.bottom_seam_px ? line.bottom_seam_px[clampCol(x, line.bottom_seam_px.length)] : line.y1;
}
function clampCol(x, length) {
  return Math.max(0, Math.min(length - 1, Math.round(x)));
}

// A line's horizontal extent is boundaries_px[0]..boundaries_px[last] --
// these used to always be pinned to 0/image width (the whole page), but a
// line can now be trimmed narrower (see trimLineEdge) to exclude scanner
// noise at the page margins, so rendering/hit-testing must respect this
// instead of assuming full page width.
function getLineX0(line) {
  return line.boundaries_px[0];
}
function getLineX1(line) {
  return line.boundaries_px[line.boundaries_px.length - 1];
}

// A boundary is either a straight vertical cut at boundaries_px[bIdx], or --
// when the model generated one (line.boundary_seams_px[bIdx], one x-value
// per row from line.y0 to line.y1) -- a slightly curved seam nudged away
// from ink. Only meaningful for a flat (non-curved-band) line: the seam was
// generated against the line's own straight y0/y1, not a per-column top/
// bottom curve, so it's ignored (falls back to straight) once the line
// itself has been split -- see the isCurved guards at each call site.
function getBoundaryXAt(line, bIdx, y) {
  const seam = line.boundary_seams_px && line.boundary_seams_px[bIdx];
  if (!seam) return line.boundaries_px[bIdx];
  return seam[clampCol(y - line.y0, seam.length)];
}

const HIT_RADIUS_IMG_PX = 6; // boundary hit-test tolerance, in image pixels

const PENCIL_DRAG_THRESHOLD_IMG_PX = 3; // minimum movement before a mousedown+move counts as a drag (vs. a plain click)
const PENCIL_SAMPLE_SPACING_IMG_PX = 2; // minimum spacing between consecutive freehand-sampled points

// The point array a Split-line/New-line pencil stroke should append to, for
// whichever of those two modes is currently active (or null if neither).
function activePencilPoints() {
  if (state.splitMode) return state.splitPoints;
  if (state.newLineMode === "top") return state.newLineTopPoints;
  if (state.newLineMode === "bottom") return state.newLineBottomPoints;
  return null;
}

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const canvasWrap = document.getElementById("canvas-wrap");
const statusEl = document.getElementById("status");
const lineSourceBadge = document.getElementById("line-source-badge");
const lineListEl = document.getElementById("line-list");

function setStatus(msg, kind) {
  statusEl.textContent = msg;
  statusEl.className = "status" + (kind ? " " + kind : "");
}

function resizeCanvasToWrap() {
  const rect = canvasWrap.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
}

function imageToScreen(x, y) {
  return [x * state.zoom + state.panX, y * state.zoom + state.panY];
}

function screenToImage(x, y) {
  return [(x - state.panX) / state.zoom, (y - state.panY) / state.zoom];
}

function fitToView() {
  if (!state.image) return;
  const rect = canvasWrap.getBoundingClientRect();
  const scale = Math.min(rect.width / state.image.width, rect.height / state.image.height) * 0.95;
  state.zoom = scale;
  state.panX = (rect.width - state.image.width * scale) / 2;
  state.panY = 20;
}

async function loadPage(pageId) {
  state.pageId = pageId;
  const pageResp = await fetch(`/api/pages/${pageId}`);
  if (!pageResp.ok) {
    setStatus(`Failed to load page: ${pageResp.status}`, "err");
    return;
  }
  const pageInfo = await pageResp.json();
  state.pageInfo = pageInfo;

  const img = new Image();
  img.src = pageInfo.image_url;
  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = reject;
  });
  state.image = img;

  const annResp = await fetch(`/api/pages/${pageId}/annotations`);
  // A page with no saved annotation yet 404s -- no auto CV-assist init, the
  // page loads blank and "Initialize CV proposals" is the explicit action.
  state.annotation = annResp.ok ? await annResp.json() : null;

  state.selectedLine = -1;
  state.selectedBoundary = -1;
  state.dirty = false;
  state.seams = null;
  state.modelOutput = null;
  state.showModelOutput = false;
  document.getElementById("btn-toggle-model-output").classList.remove("active");
  state.yoloOutput = null;
  state.showYolo = null;
  document.getElementById("btn-yolo-original").classList.remove("active");
  document.getElementById("btn-yolo-finetuned").classList.remove("active");
  state.splitMode = false;
  state.splitPoints = [];
  document.getElementById("btn-split-line").classList.remove("active");
  state.newLineMode = false;
  state.newLineTopPoints = [];
  state.newLineBottomPoints = [];
  state.extendingLineIdx = null;
  resetFreehandDragState();
  document.getElementById("btn-new-line").classList.remove("active");
  document.getElementById("btn-extend-line").classList.remove("active");
  state.scissorMode = false;
  state.scissorDragging = false;
  state.scissorPoints = [];
  document.getElementById("btn-scissor").classList.remove("active");
  state.showAnnotation = false; // every fresh page load starts blank; toggle "Show annotation" to reveal it
  document.getElementById("btn-toggle-annotation").classList.remove("active");

  state.activeVersionId = null; // current file may predate versioning or not match any saved snapshot exactly
  await loadVersionsList();

  resizeCanvasToWrap();
  fitToView();
  updateLineSourceBadge();
  renderLineList();
  render();
  updateAnnotationDependentButtons();
  setStatus(state.annotation ? "Loaded." : "No annotation yet -- click \"Initialize CV proposals\" to start.", "ok");
  resumeTrainingPollIfActive();
}

// Buttons that need an in-memory annotation to operate on -- disabled while
// state.annotation is null (a page that's never been saved/initialized).
// "Initialize CV proposals" itself, page-select, and Save/data-view are
// exempt (Save is separately gated by state.dirty elsewhere).
const ANNOTATION_DEPENDENT_BUTTON_IDS = [
  "btn-toggle-annotation", "btn-toggle-seams", "btn-toggle-model-output", "btn-adopt-model-output",
  "btn-split-line", "btn-new-line", "btn-extend-line", "btn-delete-line", "btn-reset-line", "btn-scissor",
  "btn-guide-model", "btn-reset-proposals", "btn-load-saved", "btn-view-data", "btn-clear-all",
];
function updateAnnotationDependentButtons() {
  const disabled = !state.annotation;
  for (const id of ANNOTATION_DEPENDENT_BUTTON_IDS) {
    document.getElementById(id).disabled = disabled;
  }
  document.getElementById("btn-init-cv").disabled = !!state.annotation;
}

async function resumeTrainingPollIfActive() {
  // In case the page was refreshed mid-training (a job survives on the
  // server, keyed by model_type, independent of any browser tab).
  for (const modelType of Object.keys(MODEL_LABELS)) {
    const resp = await fetch(`/api/pages/${state.pageId}/retrain/status?model_type=${modelType}`);
    const s = await resp.json();
    if (s.running) {
      setTrainButtonsDisabled(true);
      pollTrainingStatus(modelType);
    }
  }
}

function updateLineSourceBadge() {
  lineSourceBadge.textContent = state.annotation ? state.annotation.line_source : "—";
}

function renderLineList() {
  lineListEl.innerHTML = "";
  if (!state.annotation) return;
  state.annotation.lines.forEach((line, idx) => {
    const li = document.createElement("li");
    const nSegments = Math.max(0, line.boundaries_px.length - 1);
    li.textContent = `${line.line_id}  (${nSegments} seg)`;
    if (idx === state.selectedLine) li.classList.add("active");
    li.addEventListener("click", () => {
      state.selectedLine = idx;
      state.selectedBoundary = -1;
      centerOnLine(idx);
      renderLineList();
      render();
    });
    lineListEl.appendChild(li);
  });
}

// ---- Data versions (right panel) ----

const versionListEl = document.getElementById("version-list");

function formatVersionTime(isoString) {
  if (!isoString) return "unknown time";
  const d = new Date(isoString);
  return d.toLocaleString(undefined, {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
}

async function loadVersionsList() {
  const resp = await fetch(`/api/pages/${state.pageId}/versions`);
  if (!resp.ok) return;
  const data = await resp.json();
  state.versions = data.versions;
  renderVersionList();
}

function renderVersionList() {
  versionListEl.innerHTML = "";
  state.versions.forEach((v) => {
    const li = document.createElement("li");
    if (v.version_id === state.activeVersionId) li.classList.add("active");
    const time = document.createElement("span");
    time.className = "v-time";
    time.textContent = formatVersionTime(v.created_at);
    const meta = document.createElement("span");
    meta.className = "v-meta";
    meta.textContent = `${v.n_lines} lines, ${v.n_user_corrected} corrected (${v.line_source || "?"})`;
    li.appendChild(time);
    li.appendChild(meta);
    li.addEventListener("click", () => activateVersion(v.version_id));
    versionListEl.appendChild(li);
  });
}

async function activateVersion(versionId) {
  if (state.dirty && !confirm("Discard unsaved edits and load this version instead?")) return;
  setStatus("Loading version...");
  const resp = await fetch(`/api/pages/${state.pageId}/versions/${versionId}`);
  if (!resp.ok) {
    setStatus(`Failed to load version: ${resp.status}`, "err");
    return;
  }
  state.annotation = await resp.json();
  state.activeVersionId = versionId;
  state.selectedLine = -1;
  state.selectedBoundary = -1;
  state.dirty = false;
  state.seams = null;
  updateLineSourceBadge();
  renderLineList();
  renderVersionList();
  render();
  setStatus("Version loaded -- edit and Save to fork forward from here.", "ok");
}

function centerOnLine(idx) {
  const line = state.annotation.lines[idx];
  if (!line) return;
  const rect = canvasWrap.getBoundingClientRect();
  const midY = (line.y0 + line.y1) / 2;
  state.panY = rect.height / 2 - midY * state.zoom;
}

function render() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!state.image) return;

  ctx.save();
  ctx.translate(state.panX, state.panY);
  ctx.scale(state.zoom, state.zoom);
  ctx.drawImage(state.image, 0, 0);
  ctx.restore();

  if (!state.annotation) return; // blank page, no annotation yet -- "Initialize CV proposals" to start

  const w = state.image.width;

  // Each overlay -- your own annotation, the seam-carving trace, and the
  // model's output -- is independent; any combination can be on at once.
  if (state.showSeams && state.seams) {
    ctx.strokeStyle = getCssVar("--seam-trace");
    ctx.lineWidth = 1.5;
    for (const lineId in state.seams) {
      const seam = state.seams[lineId];
      ctx.beginPath();
      for (let x = 0; x < seam.length; x++) {
        const [sx, sy] = imageToScreen(x, seam[x]);
        if (x === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      }
      ctx.stroke();
    }
  }

  if (state.showModelOutput && state.modelOutput) {
    const modelColor = "#4da3ff";
    state.modelOutput.lines.forEach((line) => {
      const isCurved = !!(line.top_seam_px || line.bottom_seam_px);
      ctx.strokeStyle = modelColor;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 3]);

      const modelX0 = getLineX0(line);
      const modelX1 = getLineX1(line);
      if (!isCurved) {
        const [sx0, sy0] = imageToScreen(modelX0, line.y0);
        const [sx1, sy1] = imageToScreen(modelX1, line.y1);
        ctx.strokeRect(sx0, sy0, sx1 - sx0, sy1 - sy0);
      } else {
        // Flow-matching-generated curved seam -- same sampled-curve
        // rendering as your own split lines, just dashed and blue to mark
        // it as the model's output rather than your annotation.
        const step = Math.max(1, Math.round(4 / state.zoom));
        ctx.beginPath();
        for (let x = modelX0; x <= modelX1; x += step) {
          const xx = Math.min(x, modelX1);
          const [sx, sy] = imageToScreen(xx, getTopY(line, xx));
          if (x === modelX0) ctx.moveTo(sx, sy);
          else ctx.lineTo(sx, sy);
        }
        ctx.stroke();
        ctx.beginPath();
        for (let x = modelX0; x <= modelX1; x += step) {
          const xx = Math.min(x, modelX1);
          const [sx, sy] = imageToScreen(xx, getBottomY(line, xx));
          if (x === modelX0) ctx.moveTo(sx, sy);
          else ctx.lineTo(sx, sy);
        }
        ctx.stroke();
      }
      ctx.setLineDash([]);

      (line.boundaries_px || []).forEach((bx, bIdx) => {
        if (bIdx === 0 || bIdx === line.boundaries_px.length - 1) return; // skip structural edges
        ctx.strokeStyle = modelColor;
        ctx.lineWidth = 1.5;
        drawBoundaryPath(line, bIdx, bx, isCurved);
      });
    });
  }

  if (state.showYolo && state.yoloOutput) {
    // orange = original/stock COCO-pretrained (a sanity check, not a real
    // line predictor -- see backend/services/yolo_runtime.py), purple =
    // our own fine-tuned checkpoint.
    const yoloColor = state.showYolo === "original" ? "#ff9f43" : "#a55eea";
    ctx.strokeStyle = yoloColor;
    ctx.fillStyle = yoloColor;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([]);
    state.yoloOutput.instances.forEach((inst) => {
      const poly = inst.polygon_px;
      if (!poly || poly.length < 3) return;
      ctx.beginPath();
      poly.forEach(([px, py], i) => {
        const [sx, sy] = imageToScreen(px, py);
        if (i === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      });
      ctx.closePath();
      ctx.stroke();
      const [lx, ly] = imageToScreen(poly[0][0], poly[0][1]);
      const confStr = inst.confidence != null ? ` ${(inst.confidence * 100).toFixed(0)}%` : "";
      ctx.font = "11px sans-serif";
      ctx.fillText(`${inst.class_name}${confStr}`, lx + 3, ly - 3);
    });
  }

  // Your own annotation (line bands + boundaries) -- only this part is
  // gated behind "Show annotation"; seams and model output above are not.
  if (state.showAnnotation) {
    state.annotation.lines.forEach((line, lineIdx) => {
      const isSelected = lineIdx === state.selectedLine;
      const isCurved = !!(line.top_seam_px || line.bottom_seam_px);

      // Line band -- a flat rectangle normally, or a filled/stroked curve
      // path sampled every few pixels when the line has a curved seam edge
      // (from a split), so a "mountain"-shaped separator actually looks like one.
      ctx.fillStyle = isSelected ? "rgba(77,163,255,0.12)" : "rgba(80,200,120,0.06)";
      ctx.strokeStyle = isSelected ? "rgba(77,163,255,0.6)" : "rgba(80,200,120,0.4)";
      ctx.lineWidth = 1;

      const lineX0 = getLineX0(line);
      const lineX1 = getLineX1(line);
      if (!isCurved) {
        const [sx0, sy0] = imageToScreen(lineX0, line.y0);
        const [sx1, sy1] = imageToScreen(lineX1, line.y1);
        ctx.fillRect(sx0, sy0, sx1 - sx0, sy1 - sy0);
        ctx.strokeRect(sx0, sy0, sx1 - sx0, sy1 - sy0);
      } else {
        const step = Math.max(1, Math.round(4 / state.zoom));
        ctx.beginPath();
        for (let x = lineX0; x <= lineX1; x += step) {
          const xx = Math.min(x, lineX1);
          const [sx, sy] = imageToScreen(xx, getTopY(line, xx));
          if (x === lineX0) ctx.moveTo(sx, sy);
          else ctx.lineTo(sx, sy);
        }
        for (let x = lineX1; x >= lineX0; x -= step) {
          const xx = Math.max(x, lineX0);
          const [sx, sy] = imageToScreen(xx, getBottomY(line, xx));
          ctx.lineTo(sx, sy);
        }
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }

      // Boundaries -- vertical markers span the line's local top/bottom at
      // that column, so they follow a curved line correctly too.
      line.boundaries_px.forEach((bx, bIdx) => {
        const isEdge = bIdx === 0 || bIdx === line.boundaries_px.length - 1;
        const isBoundarySelected = isSelected && bIdx === state.selectedBoundary;
        const source = line.boundary_source[bIdx];

        let color;
        if (isBoundarySelected) color = getCssVar("--boundary-selected");
        else if (isEdge) color = "rgba(150,150,150,0.5)";
        else if (source === "user") color = getCssVar("--boundary-user");
        else color = getCssVar("--boundary-proposed");

        ctx.strokeStyle = color;
        ctx.lineWidth = isBoundarySelected ? 2.5 : 1.5;
        drawBoundaryPath(line, bIdx, bx, isCurved);
      });
    });
  }

  // In-progress split/new-line traces stay visible regardless of any
  // overlay toggle -- otherwise you'd be placing points blind.
  if (state.splitMode && state.splitPoints.length > 0) {
    drawPointTrace(state.splitPoints, "#ffdd00");
  }

  if (state.newLineMode) {
    if (state.newLineTopPoints.length > 0) drawPointTrace(state.newLineTopPoints, "#00e5ff");
    if (state.newLineBottomPoints.length > 0) drawPointTrace(state.newLineBottomPoints, "#ff66cc");
  }

  if (state.scissorMode && state.scissorPoints.length > 1) {
    drawFreehandTrace(state.scissorPoints, "#ff4040");
  }
}

// Unlike drawPointTrace (used by Split/New line), a scissor stroke is a
// freehand cutting gesture, not a sparse set of points to fit a smooth
// curve through -- draw it in the exact order it was dragged, no sorting.
function drawFreehandTrace(points, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  points.forEach((p, i) => {
    const [sx, sy] = imageToScreen(p.x, p.y);
    if (i === 0) ctx.moveTo(sx, sy);
    else ctx.lineTo(sx, sy);
  });
  ctx.stroke();
}

// Draws one akshara boundary: a curved, ink-avoiding seam when the model
// generated one for this position (and the line's own band is still flat --
// see getBoundaryXAt), otherwise the plain straight vertical cut.
function drawBoundaryPath(line, bIdx, bx, lineIsCurved) {
  const seam = !lineIsCurved && line.boundary_seams_px && line.boundary_seams_px[bIdx];
  const yTop = getTopY(line, bx);
  const yBot = getBottomY(line, bx);

  if (!seam) {
    const [x, sy0] = imageToScreen(bx, yTop);
    const [, sy1] = imageToScreen(bx, yBot);
    ctx.beginPath();
    ctx.moveTo(x, sy0);
    ctx.lineTo(x, sy1);
    ctx.stroke();
    return;
  }

  const step = Math.max(1, Math.round(4 / state.zoom));
  ctx.beginPath();
  for (let y = yTop; y <= yBot; y += step) {
    const [sx, sy] = imageToScreen(getBoundaryXAt(line, bIdx, y), y);
    if (y === yTop) ctx.moveTo(sx, sy);
    else ctx.lineTo(sx, sy);
  }
  const [sxEnd, syEnd] = imageToScreen(getBoundaryXAt(line, bIdx, yBot), yBot);
  ctx.lineTo(sxEnd, syEnd);
  ctx.stroke();
}

function drawPointTrace(points, color) {
  const sorted = [...points].sort((a, b) => a.x - b.x);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  sorted.forEach((p, i) => {
    const [sx, sy] = imageToScreen(p.x, p.y);
    if (i === 0) ctx.moveTo(sx, sy);
    else ctx.lineTo(sx, sy);
  });
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = color;
  sorted.forEach((p) => {
    const [sx, sy] = imageToScreen(p.x, p.y);
    ctx.beginPath();
    ctx.arc(sx, sy, 4, 0, Math.PI * 2);
    ctx.fill();
  });
}

function getCssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// ---- Hit testing ----

function findLineAtPoint(imgX, imgY) {
  if (!state.annotation) return -1;
  return state.annotation.lines.findIndex(
    (l) =>
      imgX >= getLineX0(l) &&
      imgX <= getLineX1(l) &&
      imgY >= getTopY(l, imgX) &&
      imgY <= getBottomY(l, imgX)
  );
}

function findNearestBoundary(lineIdx, imgX) {
  const line = state.annotation.lines[lineIdx];
  if (!line) return -1;
  let best = -1;
  let bestDist = Infinity;
  line.boundaries_px.forEach((bx, idx) => {
    const d = Math.abs(bx - imgX);
    if (d < bestDist) {
      bestDist = d;
      best = idx;
    }
  });
  const hitRadiusScreen = HIT_RADIUS_IMG_PX;
  return bestDist <= hitRadiusScreen ? best : -1;
}

// ---- Mutations (always mark boundary_source as "user") ----

function addBoundary(lineIdx, imgX) {
  const line = state.annotation.lines[lineIdx];
  const x = Math.round(Math.max(1, Math.min(state.image.width - 1, imgX)));

  // Don't add if too close to an existing boundary.
  if (findNearestBoundary(lineIdx, x) !== -1) return;

  let insertAt = line.boundaries_px.findIndex((b) => b > x);
  if (insertAt === -1) insertAt = line.boundaries_px.length;

  line.boundaries_px.splice(insertAt, 0, x);
  line.boundary_source.splice(insertAt, 0, "user");
  if (line.akshara_labels) line.akshara_labels.splice(insertAt > 0 ? insertAt - 1 : 0, 0, null);
  // Indices shift on insert -- any model-generated seams for this line no
  // longer line up with their boundaries_px positions, so drop them all
  // (falls back to straight rendering; regenerate via Guide from model).
  delete line.boundary_seams_px;

  state.selectedLine = lineIdx;
  state.selectedBoundary = insertAt;
  markDirty();
  renderLineList();
  render();
}

function removeBoundary(lineIdx, bIdx) {
  const line = state.annotation.lines[lineIdx];
  if (bIdx <= 0 || bIdx >= line.boundaries_px.length - 1) return; // never remove edges
  line.boundaries_px.splice(bIdx, 1);
  line.boundary_source.splice(bIdx, 1);
  if (line.akshara_labels && line.akshara_labels.length > 0) {
    line.akshara_labels.splice(Math.min(bIdx, line.akshara_labels.length - 1), 1);
  }
  delete line.boundary_seams_px; // indices shifted -- see addBoundary
  state.selectedBoundary = -1;
  markDirty();
  renderLineList();
  render();
}

function moveBoundary(lineIdx, bIdx, newImgX) {
  const line = state.annotation.lines[lineIdx];
  if (bIdx < 0 || bIdx >= line.boundaries_px.length) return;
  // The two edge boundaries (0 and last) can now move too -- that's how a
  // line's own left/right extent gets trimmed inward (see getLineX0/X1 and
  // trimLineEdge) to exclude scanner noise at the page margins. An edge
  // still can't be dragged past its neighbor or off the image entirely.
  const isLeftEdge = bIdx === 0;
  const isRightEdge = bIdx === line.boundaries_px.length - 1;
  const lo = isLeftEdge ? 0 : line.boundaries_px[bIdx - 1] + 1;
  const hi = isRightEdge ? state.image.width : line.boundaries_px[bIdx + 1] - 1;
  const x = Math.round(Math.max(lo, Math.min(hi, newImgX)));
  line.boundaries_px[bIdx] = x;
  line.boundary_source[bIdx] = "user";
  // The generated seam (if any) was anchored to the OLD x position -- drop
  // just this boundary's seam so it falls back to a straight line at the
  // new position rather than rendering stale/misaligned.
  if (line.boundary_seams_px) line.boundary_seams_px[bIdx] = null;
  markDirty();
}

function markDirty() {
  state.dirty = true;
  setStatus("Unsaved changes", "");
}

// ---- Save / load-proposals / model-guide ----

async function save() {
  if (!state.annotation) return;
  setStatus("Saving...");
  const resp = await fetch(`/api/pages/${state.pageId}/annotations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(state.annotation),
  });
  if (resp.ok) {
    state.dirty = false;
    setStatus("Saved.", "ok");
    await loadVersionsList();
    if (state.versions.length > 0) {
      state.activeVersionId = state.versions[0].version_id; // newest = just-saved
      renderVersionList();
    }
    const opt = document.querySelector(`#page-select option[value="${state.pageId}"]`);
    if (opt && !opt.textContent.includes("✓")) opt.textContent = `${state.pageId} ✓`; // now eligible for training
  } else {
    const detail = await resp.text();
    setStatus(`Save failed: ${resp.status}`, "err");
    console.error(detail);
  }
}

// Shared by "Reset to proposals" (an already-annotated page, discards
// edits) and "Initialize CV proposals" (a blank page with no annotation at
// all yet -- no confirm needed, there's nothing to discard). Neither one
// auto-saves: matches the existing convention that Save is always the
// explicit, separate action that actually persists to disk (and is what
// makes a page eligible for training -- see list_annotated_pages).
async function loadCvProposals({ confirmDiscard }) {
  if (confirmDiscard && state.dirty && !confirm("Discard unsaved edits and reload CV-assist proposals?")) return;
  setStatus("Loading proposals...");
  const resp = await fetch(`/api/pages/${state.pageId}/proposals`);
  if (!resp.ok) {
    setStatus(`Failed to fetch proposals: ${resp.status}`, "err");
    return;
  }
  const data = await resp.json();
  const lines = data.lines.map((l) => ({
    line_id: l.line_id,
    y0: l.y0,
    y1: l.y1,
    boundaries_px: l.boundaries_px,
    boundary_source: l.boundaries_px.map(() => "proposed"),
    akshara_labels: l.boundaries_px.slice(1).map(() => null),
  }));
  if (state.annotation) {
    state.annotation.lines = lines;
    state.annotation.line_source = "cv_assist_v1";
  } else {
    state.annotation = {
      page_id: state.pageId,
      image_path: `data/images/${state.pageId}.png`,
      dpi: state.pageInfo.dpi,
      image_width_px: state.pageInfo.width,
      image_height_px: state.pageInfo.height,
      line_source: "cv_assist_v1",
      lines,
    };
  }
  state.dirty = true;
  state.seams = null; // line bands changed -- stale seam overlay would no longer align
  updateLineSourceBadge();
  renderLineList();
  render();
  updateAnnotationDependentButtons();
  setStatus("Loaded CV proposals (unsaved) -- Save to persist.", "");
}

async function resetToProposals() {
  return loadCvProposals({ confirmDiscard: true });
}

async function initCvProposals() {
  return loadCvProposals({ confirmDiscard: false });
}

async function toggleSeams() {
  state.showSeams = !state.showSeams;
  const btn = document.getElementById("btn-toggle-seams");
  btn.classList.toggle("active", state.showSeams);

  if (state.showSeams && !state.seams) {
    setStatus("Loading seams...");
    const resp = await fetch(`/api/pages/${state.pageId}/seams`);
    if (!resp.ok) {
      setStatus(`Seam extraction failed: ${resp.status}`, "err");
      state.showSeams = false;
      btn.classList.remove("active");
      return;
    }
    const data = await resp.json();
    state.seams = {};
    for (const line of data.lines) {
      state.seams[line.line_id] = line.seam_px;
    }
    setStatus("Seams loaded.", "ok");
  }
  render();
}

async function fetchModelOutput() {
  const resp = await fetch(`/api/pages/${state.pageId}/model-guide`, { method: "POST" });
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${resp.status}`);
  }
  return resp.json();
}

async function toggleModelOutput() {
  state.showModelOutput = !state.showModelOutput;
  const btn = document.getElementById("btn-toggle-model-output");
  btn.classList.toggle("active", state.showModelOutput);

  if (state.showModelOutput) {
    setStatus("Requesting model output...");
    try {
      state.modelOutput = await fetchModelOutput();
      setStatus("Model output loaded (blue dashed = model, doesn't affect your annotation).", "ok");
    } catch (err) {
      setStatus(`Model output unavailable: ${err.message}`, "err");
      state.showModelOutput = false;
      btn.classList.remove("active");
    }
  }
  render();
}

async function refreshModelOutputIfShown() {
  if (!state.showModelOutput) return;
  try {
    state.modelOutput = await fetchModelOutput();
    render();
  } catch (err) {
    setStatus(`Model output refresh failed: ${err.message}`, "err");
  }
}

function adoptModelOutput() {
  if (!state.modelOutput) {
    setStatus("Nothing to adopt yet -- click \"Show model output\" first to preview it.", "err");
    return;
  }
  // Deep-copy so later edits to state.annotation don't also mutate the
  // still-displayed preview in state.modelOutput.
  state.annotation.lines = JSON.parse(JSON.stringify(state.modelOutput.lines));
  state.annotation.line_source = state.modelOutput.line_source;
  state.dirty = true;
  state.selectedLine = -1;
  state.selectedBoundary = -1;
  state.seams = null; // line bands changed -- stale seam overlay would no longer align
  updateLineSourceBadge();
  renderLineList();
  render();
  setStatus("Model output adopted into your annotation (unsaved) -- Save to persist it as a new version.", "ok");
}

// ---- YOLO26n-seg preview (read-only, separate model/pipeline from the
// custom boundary/line models above -- see backend/services/yolo_runtime.py
// and model/train_yolo.py). "original" = stock COCO-pretrained checkpoint
// (a sanity check that it loads/runs, not a real line predictor -- COCO
// has no "line" class); "finetuned" = our own checkpoint trained on this
// project's line annotations. No "adopt" here: YOLO's polygon-per-instance
// output doesn't map onto this tool's boundaries_px/line schema.

async function fetchYoloOutput(variant) {
  const resp = await fetch(`/api/pages/${state.pageId}/yolo-guide?variant=${variant}`);
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${resp.status}`);
  }
  return resp.json();
}

async function toggleYoloOutput(variant) {
  const btnId = variant === "original" ? "btn-yolo-original" : "btn-yolo-finetuned";
  const otherBtnId = variant === "original" ? "btn-yolo-finetuned" : "btn-yolo-original";

  if (state.showYolo === variant) {
    state.showYolo = null;
    state.yoloOutput = null;
    document.getElementById(btnId).classList.remove("active");
    render();
    return;
  }

  setStatus(`Requesting ${variant} YOLO output...`);
  try {
    state.yoloOutput = await fetchYoloOutput(variant);
    state.showYolo = variant;
    document.getElementById(btnId).classList.add("active");
    document.getElementById(otherBtnId).classList.remove("active");
    const n = state.yoloOutput.instances.length;
    if (variant === "original" && n === 0) {
      // Not a bug: the stock checkpoint only knows COCO's 80 everyday-object
      // classes (person, car, book, dog, ...) -- it has never seen "line" as
      // a concept, so on a manuscript scan there's usually nothing in those
      // 80 classes to fire on above the confidence threshold. This variant
      // is a "does the checkpoint load and run" sanity check, not a real
      // line predictor -- see "Check finetuned YOLO" for the real one.
      setStatus("Original (COCO-pretrained) YOLO ran fine but found 0 instances -- expected, it doesn't know \"line\" as a class yet. This just confirms the checkpoint loads/runs; train it first, then compare against Check finetuned YOLO.", "");
    } else {
      setStatus(`${variant === "original" ? "Original" : "Fine-tuned"} YOLO output loaded (${n} instance${n === 1 ? "" : "s"}).`, "ok");
    }
  } catch (err) {
    setStatus(`YOLO output unavailable: ${err.message}`, "err");
    state.showYolo = null;
    state.yoloOutput = null;
    document.getElementById(btnId).classList.remove("active");
  }
  render();
}

// ---- Training (background job on the server; polled for progress) ----

const MODEL_LABELS = { boundary: "boundary", line_segmenter: "line", yolo_seg: "YOLO" };

// Train boundaries/lines always continue from the current checkpoint now
// (resume=true) instead of restarting from a random init each time -- but
// still run a FULL training pass, not the smaller "quick fine-tune" epoch
// count /retrain otherwise defaults to whenever resume=true (that smaller
// default is specifically for the separate "Update model" button) -- so
// the full epoch count has to be passed explicitly here to override it.
// Mirrors backend/api/routes_model.py's DEFAULT_EPOCHS. yolo_seg is
// transfer learning (starts from COCO-pretrained weights, never a random
// init -- see model/train_yolo.py), so it needs far fewer epochs than the
// from-scratch custom models to converge.
const FULL_TRAIN_EPOCHS = { boundary: 100, line_segmenter: 300, yolo_seg: 100 };

function setTrainButtonsDisabled(disabled) {
  document.getElementById("btn-train-boundary").disabled = disabled;
  document.getElementById("btn-train-lines").disabled = disabled;
  document.getElementById("btn-update-model").disabled = disabled;
  document.getElementById("btn-train-yolo").disabled = disabled;
}

async function trainModel(modelType, { resume = false, epochs = null } = {}) {
  const label = MODEL_LABELS[modelType];
  setStatus(`Starting ${label} model training...`);
  let url = `/api/pages/${state.pageId}/retrain?model_type=${modelType}&resume=${resume}`;
  if (epochs != null) url += `&epochs=${epochs}`;
  const resp = await fetch(url, { method: "POST" });
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    setStatus(`Training failed to start: ${body.detail || resp.status}`, "err");
    throw new Error(body.detail || String(resp.status));
  }
  const started = await resp.json();
  const verb = started.resumed ? "Fine-tuning" : "Training";
  setStatus(`${verb} ${label} model...`, "");
  // Only one training job (any model type) can run at a time -- running
  // both concurrently was observed to badly starve each other on this GPU --
  // so disable all training buttons until this one finishes.
  setTrainButtonsDisabled(true);
  return pollTrainingStatus(modelType, { resumed: started.resumed });
}

function pollTrainingStatus(modelType, { resumed = false } = {}) {
  const label = MODEL_LABELS[modelType];
  const verb = resumed ? "Fine-tuning" : "Training";
  const progressEl = document.getElementById("train-progress");

  if (state.trainingPoll[modelType]) return Promise.resolve(); // already polling

  return new Promise((resolve) => {
    state.trainingPoll[modelType] = setInterval(async () => {
      const resp = await fetch(`/api/pages/${state.pageId}/retrain/status?model_type=${modelType}`);
      const s = await resp.json();

      if (s.running) {
        const lossStr = s.loss !== null && s.loss !== undefined ? s.loss.toFixed(4) : "...";
        progressEl.textContent = `${verb} ${label}: epoch ${s.epoch}/${s.total_epochs} loss=${lossStr}`;
        return;
      }

      clearInterval(state.trainingPoll[modelType]);
      state.trainingPoll[modelType] = null;
      setTrainButtonsDisabled(false);

      if (s.error) {
        progressEl.textContent = "";
        setStatus(`${label} training failed -- see console.`, "err");
        console.error(s.error);
      } else if (s.done) {
        progressEl.textContent = `${label} ${resumed ? "update" : "training"} done (loss=${s.loss?.toFixed(4)})`;
        setStatus(`${label} model ${resumed ? "updated" : "retrained"}.`, "ok");
        refreshModelOutputIfShown();
      }
      resolve();
    }, 2000);
  });
}

async function updateModelFromCheckpoint() {
  // Fine-tunes both models in sequence (only one training job can run at a
  // time on this GPU), each starting from its currently-served checkpoint
  // rather than a random init -- a quick incremental update instead of a
  // full retrain. (The curved seam-generator models -- character/line
  // separator -- aren't included: model_runtime.py currently generates
  // seams with a classical DP method instead of those checkpoints, see
  // USE_CLASSICAL_SEAM_GENERATION, so training them has no visible effect
  // right now.)
  setTrainButtonsDisabled(true);
  try {
    await trainModel("boundary", { resume: true });
    await trainModel("line_segmenter", { resume: true });
    setStatus("Model updated from current weights.", "ok");
  } catch (e) {
    console.error(e);
  } finally {
    setTrainButtonsDisabled(false);
  }
}

async function guideFromModel() {
  setStatus("Requesting model guidance...");
  const resp = await fetch(`/api/pages/${state.pageId}/model-guide`, { method: "POST" });
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    setStatus(`Model guide unavailable: ${body.detail || resp.status}`, "err");
    return;
  }
  const data = await resp.json();
  state.annotation.lines = data.lines;
  state.annotation.line_source = data.line_source;
  state.dirty = true;
  state.seams = null; // line bands changed -- stale seam overlay would no longer align
  updateLineSourceBadge();
  renderLineList();
  render();
  setStatus("Loaded model-guided proposals (unsaved).", "");
}

async function loadSavedData() {
  if (state.dirty && !confirm("Discard in-session edits and reload the last saved annotation from disk?")) return;
  setStatus("Loading saved annotation...");
  const resp = await fetch(`/api/pages/${state.pageId}/annotations`);
  if (!resp.ok) {
    setStatus(`Load failed: ${resp.status}`, "err");
    return;
  }
  state.annotation = await resp.json();
  state.selectedLine = -1;
  state.selectedBoundary = -1;
  state.dirty = false;
  state.seams = null;
  updateLineSourceBadge();
  renderLineList();
  render();
  updateAnnotationDependentButtons();
  setStatus("Loaded saved annotation.", "ok");
}

function viewSavedData() {
  window.open(`/data/annotations/${state.pageId}.json`, "_blank");
}

// ---- Line splitting (curved separators, for merged-line / wavy-baseline fixes) ----

function startSplit() {
  if (state.selectedLine === -1) {
    setStatus("Select a line first, then click Split line.", "err");
    return;
  }
  if (state.scissorMode) cancelScissor();
  state.splitMode = true;
  state.splitPoints = [];
  document.getElementById("btn-split-line").classList.add("active");
  setStatus("Split mode: click points to trace the new separator, Enter to confirm, Esc to cancel.", "");
  render();
}

// Esc mid-drag (mouse button still down) would otherwise leave a stray
// freehandDragging=true that keeps streaming points nowhere useful once the
// mode's point array is gone -- clear it whenever a trace is cancelled/done.
function resetFreehandDragState() {
  state.freehandDragging = false;
  state.freehandMoved = false;
  state.freehandStrokePointCount = 0;
  state.freehandJustDragged = false;
}

function cancelSplit() {
  state.splitMode = false;
  state.splitPoints = [];
  resetFreehandDragState();
  document.getElementById("btn-split-line").classList.remove("active");
  setStatus("Split cancelled.", "");
  render();
}

function addSplitPoint(imgX, imgY) {
  state.splitPoints.push({ x: imgX, y: imgY });
  render();
}

function interpolateSeam(points, width) {
  const sorted = [...points].sort((a, b) => a.x - b.x);
  const seam = new Array(width);

  if (sorted.length === 1) {
    seam.fill(sorted[0].y);
    return seam;
  }

  // Catmull-Rom spline through the clicked points, evaluated directly as a
  // function of x (valid since x is strictly increasing across the sorted
  // points) -- passes smoothly through every point with continuous
  // curvature, instead of the sharp straight-line segments plain linear
  // interpolation would give between sparse clicks. A manuscript baseline
  // is not always a straight line ("people write like mountain"), so a few
  // clicks should trace a natural curve, not a polyline.
  const catmullRom = (p0, p1, p2, p3, t) => {
    const t2 = t * t;
    const t3 = t2 * t;
    return 0.5 * (
      2 * p1 +
      (-p0 + p2) * t +
      (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
      (-p0 + 3 * p1 - 3 * p2 + p3) * t3
    );
  };

  for (let x = 0; x < width; x++) {
    if (x <= sorted[0].x) {
      seam[x] = sorted[0].y;
      continue;
    }
    if (x >= sorted[sorted.length - 1].x) {
      seam[x] = sorted[sorted.length - 1].y;
      continue;
    }
    for (let i = 0; i < sorted.length - 1; i++) {
      if (x >= sorted[i].x && x <= sorted[i + 1].x) {
        const p0 = sorted[i - 1] || sorted[i];
        const p1 = sorted[i];
        const p2 = sorted[i + 1];
        const p3 = sorted[i + 2] || sorted[i + 1];
        const span = p2.x - p1.x || 1;
        const t = (x - p1.x) / span;
        seam[x] = catmullRom(p0.y, p1.y, p2.y, p3.y, t);
        break;
      }
    }
  }
  return seam;
}

function renumberLines() {
  state.annotation.lines.sort((a, b) => a.y0 - b.y0);
  state.annotation.lines.forEach((l, idx) => {
    l.line_id = `L${String(idx + 1).padStart(2, "0")}`;
  });
}

function confirmSplit() {
  if (state.splitPoints.length === 0) {
    setStatus("Place at least one point before confirming (Esc to cancel instead).", "err");
    return;
  }
  const width = state.image.width;
  const lineIdx = state.selectedLine;
  const line = state.annotation.lines[lineIdx];

  const seam = interpolateSeam(state.splitPoints, width);
  // Clamp the new seam within the original line's own band at each column
  // (so a wild click can't invert the split or escape the line's extent),
  // and round to whole pixels -- y0/y1/seam arrays are used as array-slice
  // indices downstream (dataset cropping, seam-window extraction), which
  // require integers.
  for (let x = 0; x < width; x++) {
    const top = getTopY(line, x);
    const bottom = getBottomY(line, x);
    seam[x] = Math.round(Math.max(top + 2, Math.min(bottom - 2, seam[x])));
  }

  const topSeam = line.top_seam_px
    ? [...line.top_seam_px]
    : new Array(width).fill(Math.round(line.y0));
  const bottomSeam = line.bottom_seam_px
    ? [...line.bottom_seam_px]
    : new Array(width).fill(Math.round(line.y1));

  const upperLine = {
    line_id: "TMP_A",
    y0: Math.round(Math.min(...topSeam)),
    y1: Math.round(Math.max(...seam)),
    top_seam_px: topSeam,
    bottom_seam_px: seam,
    boundaries_px: [0, width],
    boundary_source: ["proposed", "proposed"],
    akshara_labels: [],
  };
  const lowerLine = {
    line_id: "TMP_B",
    y0: Math.round(Math.min(...seam)),
    y1: Math.round(Math.max(...bottomSeam)),
    top_seam_px: [...seam],
    bottom_seam_px: bottomSeam,
    boundaries_px: [0, width],
    boundary_source: ["proposed", "proposed"],
    akshara_labels: [],
  };

  state.annotation.lines.splice(lineIdx, 1, upperLine, lowerLine);
  renumberLines();

  state.splitMode = false;
  state.splitPoints = [];
  resetFreehandDragState();
  document.getElementById("btn-split-line").classList.remove("active");
  state.selectedLine = state.annotation.lines.findIndex((l) => l === upperLine);
  state.selectedBoundary = -1;
  state.seams = null; // line bands changed -- stale seam overlay would no longer align
  markDirty();
  renderLineList();
  render();
  setStatus("Line split into two -- review/correct each half's akshara boundaries.", "ok");
}

// ---- Delete line / draw a brand new line / reset one line to proposal ----

function deleteSelectedLine() {
  if (state.selectedLine === -1) {
    setStatus("Select a line first, then Delete line.", "err");
    return;
  }
  const line = state.annotation.lines[state.selectedLine];
  if (!confirm(`Delete ${line.line_id} and all its boundaries? This can't be undone (unless you Load saved before saving).`)) {
    return;
  }
  state.annotation.lines.splice(state.selectedLine, 1);
  renumberLines();
  state.selectedLine = -1;
  state.selectedBoundary = -1;
  state.seams = null;
  markDirty();
  renderLineList();
  render();
  setStatus("Line deleted.", "ok");
}

function clearAllAnnotation() {
  if (!state.annotation) return;
  if (!confirm(`Clear ALL ${state.annotation.lines.length} lines and boundaries on this page? This can't be undone (unless you Load saved before saving).`)) {
    return;
  }
  state.annotation.lines = [];
  state.selectedLine = -1;
  state.selectedBoundary = -1;
  state.seams = null;
  state.modelOutput = null;
  markDirty();
  renderLineList();
  render();
  setStatus("Cleared all annotation on this page (unsaved).", "ok");
}

function startNewLine() {
  if (state.scissorMode) cancelScissor();
  state.extendingLineIdx = null;
  state.newLineMode = "top";
  state.newLineTopPoints = [];
  state.newLineBottomPoints = [];
  document.getElementById("btn-new-line").classList.add("active");
  setStatus("New line: click points for the TOP edge (cyan), Tab for bottom edge, Enter to confirm, Esc to cancel.", "");
  render();
}

// Same top(cyan)/bottom(pink) point-tracing UI as New line, but requires a
// selected EXISTING line and, on confirm, extends that line's top/bottom
// curve and left/right edges instead of creating a new one -- seeded with
// the line's own current curve so "extending" means adding a few points
// beyond either end, not retracing the whole thing from scratch.
function startExtendLine() {
  if (state.selectedLine === -1) {
    setStatus("Select a line first, then click Extend line.", "err");
    return;
  }
  if (state.scissorMode) cancelScissor();
  const line = state.annotation.lines[state.selectedLine];
  state.extendingLineIdx = state.selectedLine;
  state.newLineMode = "top";
  const SEED_SAMPLES = 12;
  const x0 = getLineX0(line);
  const x1 = getLineX1(line);
  const seedXs = [];
  for (let i = 0; i < SEED_SAMPLES; i++) {
    seedXs.push(Math.round(x0 + ((x1 - x0) * i) / (SEED_SAMPLES - 1)));
  }
  state.newLineTopPoints = seedXs.map((x) => ({ x, y: getTopY(line, x) }));
  state.newLineBottomPoints = seedXs.map((x) => ({ x, y: getBottomY(line, x) }));
  document.getElementById("btn-extend-line").classList.add("active");
  setStatus(
    `Extend line: seeded with ${line.line_id}'s current top edge (cyan) -- add points beyond either end to extend it, Tab for bottom edge (pink), Enter to confirm, Esc to cancel.`,
    ""
  );
  render();
}

function cancelNewLine() {
  const wasExtending = state.extendingLineIdx != null;
  const btnId = wasExtending ? "btn-extend-line" : "btn-new-line";
  state.newLineMode = false;
  state.newLineTopPoints = [];
  state.newLineBottomPoints = [];
  state.extendingLineIdx = null;
  resetFreehandDragState();
  document.getElementById(btnId).classList.remove("active");
  setStatus(wasExtending ? "Extend line cancelled." : "New line cancelled.", "");
  render();
}

function addNewLinePoint(imgX, imgY) {
  if (state.newLineMode === "top") state.newLineTopPoints.push({ x: imgX, y: imgY });
  else if (state.newLineMode === "bottom") state.newLineBottomPoints.push({ x: imgX, y: imgY });
  render();
}

function advanceNewLineEdge() {
  if (state.newLineMode !== "top") return;
  if (state.newLineTopPoints.length === 0) {
    setStatus("Place at least one point on the top edge first.", "err");
    return;
  }
  state.newLineMode = "bottom";
  const verb = state.extendingLineIdx != null ? "Extend line" : "New line";
  setStatus(`${verb}: click points for the BOTTOM edge (pink), Enter to confirm, Esc to cancel.`, "");
  render();
}

function confirmNewLine() {
  if (state.newLineMode === "top") {
    advanceNewLineEdge();
    return; // first Enter moves top -> bottom phase; a second Enter finalizes
  }
  if (state.newLineTopPoints.length === 0 || state.newLineBottomPoints.length === 0) {
    setStatus("Need points on both the top and bottom edge.", "err");
    return;
  }
  const width = state.image.width;
  const topSeam = interpolateSeam(state.newLineTopPoints, width).map((v) => Math.round(v));
  const bottomSeam = interpolateSeam(state.newLineBottomPoints, width).map((v) => Math.round(v));
  for (let x = 0; x < width; x++) {
    if (bottomSeam[x] <= topSeam[x]) bottomSeam[x] = topSeam[x] + 2; // keep a valid (top < bottom) band everywhere
  }

  if (state.extendingLineIdx != null) {
    const line = state.annotation.lines[state.extendingLineIdx];
    const tracedXs = state.newLineTopPoints.map((p) => p.x).concat(state.newLineBottomPoints.map((p) => p.x));
    // Extend outward only -- union with the line's existing extent, never
    // shrink it (Scissor already covers trimming it back in).
    const x0 = Math.max(0, Math.min(getLineX0(line), Math.round(Math.min(...tracedXs))));
    const x1 = Math.min(width - 1, Math.max(getLineX1(line), Math.round(Math.max(...tracedXs))));

    line.top_seam_px = topSeam;
    line.bottom_seam_px = bottomSeam;
    line.y0 = Math.round(Math.min(...topSeam));
    line.y1 = Math.round(Math.max(...bottomSeam));
    line.boundaries_px[0] = x0;
    line.boundaries_px[line.boundaries_px.length - 1] = x1;
    line.boundary_source[0] = "user";
    line.boundary_source[line.boundary_source.length - 1] = "user";
    delete line.boundary_seams_px; // stale curved akshara-boundary seams no longer match the new top/bottom curve

    state.newLineMode = false;
    state.newLineTopPoints = [];
    state.newLineBottomPoints = [];
    state.extendingLineIdx = null;
    resetFreehandDragState();
    document.getElementById("btn-extend-line").classList.remove("active");
    state.selectedBoundary = -1;
    state.seams = null;
    markDirty();
    renderLineList();
    render();
    setStatus(`${line.line_id} extended.`, "ok");
    return;
  }

  const newLine = {
    line_id: "TMP_NEW",
    y0: Math.round(Math.min(...topSeam)),
    y1: Math.round(Math.max(...bottomSeam)),
    top_seam_px: topSeam,
    bottom_seam_px: bottomSeam,
    boundaries_px: [0, width],
    boundary_source: ["proposed", "proposed"],
    akshara_labels: [],
  };
  state.annotation.lines.push(newLine);
  renumberLines();

  state.newLineMode = false;
  state.newLineTopPoints = [];
  state.newLineBottomPoints = [];
  resetFreehandDragState();
  document.getElementById("btn-new-line").classList.remove("active");
  state.selectedLine = state.annotation.lines.findIndex((l) => l === newLine);
  state.selectedBoundary = -1;
  state.seams = null;
  markDirty();
  renderLineList();
  render();
  setStatus("New line added (blank) -- click inside it to add akshara boundaries.", "ok");
}

async function resetSelectedLineToProposal() {
  if (state.selectedLine === -1) {
    setStatus("Select a line first, then Reset line.", "err");
    return;
  }
  if (!confirm("Reset this one line to the CV-assist proposal? Your corrections on it will be lost.")) return;

  setStatus("Fetching proposals...");
  const resp = await fetch(`/api/pages/${state.pageId}/proposals`);
  if (!resp.ok) {
    setStatus(`Failed to fetch proposals: ${resp.status}`, "err");
    return;
  }
  const data = await resp.json();

  const line = state.annotation.lines[state.selectedLine];
  const midY = (line.y0 + line.y1) / 2;

  // Prefer a proposed line whose band actually contains this line's midpoint;
  // fall back to the proposal with the closest midpoint (covers a line you
  // drew/split yourself, which cv_assist never proposed at that exact range).
  let best = null;
  let bestDist = Infinity;
  for (const p of data.lines) {
    if (midY >= p.y0 && midY <= p.y1) {
      best = p;
      break;
    }
    const dist = Math.abs((p.y0 + p.y1) / 2 - midY);
    if (dist < bestDist) {
      bestDist = dist;
      best = p;
    }
  }
  if (!best) {
    setStatus("No matching CV-assist proposal found for this line.", "err");
    return;
  }

  line.y0 = best.y0;
  line.y1 = best.y1;
  delete line.top_seam_px;
  delete line.bottom_seam_px;
  line.boundaries_px = best.boundaries_px;
  line.boundary_source = best.boundaries_px.map(() => "proposed");
  line.akshara_labels = best.boundaries_px.slice(1).map(() => null);

  state.selectedBoundary = -1;
  state.seams = null;
  markDirty();
  renderLineList();
  render();
  setStatus("Line reset to CV-assist proposal.", "ok");
}

function toggleShowAnnotation() {
  state.showAnnotation = !state.showAnnotation;
  document.getElementById("btn-toggle-annotation").classList.toggle("active", state.showAnnotation);
  render();
}

// ---- Scissor (freehand cut-to-delete) ----
//
// A drag across the page deletes whatever it cuts through: a stroke that
// crosses a specific boundary marker deletes just that boundary; a stroke
// that passes through a line's band without lining up on any marker
// deletes the whole line (mirrors Delete line, minus the confirm prompt --
// this tool exists specifically for sweeping through a run of garbage
// lines at once, e.g. scanner noise picked up along a page edge, and a
// confirm dialog per line would defeat that). No selection required first.
// Unlike Split/New line, the tool stays armed after each cut (auto-commits
// on mouseup) so you can keep cutting -- toggle it off via the button or Esc.

function startScissor() {
  if (state.splitMode) cancelSplit();
  if (state.newLineMode) cancelNewLine();
  state.scissorMode = true;
  state.scissorPoints = [];
  document.getElementById("btn-scissor").classList.add("active");
  setStatus("Scissor: short snip near a boundary deletes it; wide drag near a line's edge trims off the noisy margin. Click Scissor or Esc to turn off.", "");
  render();
}

function cancelScissor() {
  state.scissorMode = false;
  state.scissorDragging = false;
  state.scissorPoints = [];
  document.getElementById("btn-scissor").classList.remove("active");
  render();
}

// A boundary marker spans the SAME y-range as its own line (top to bottom),
// so a broad stroke swept across a line would also technically cross every
// one of that line's boundaries -- geometric line-crossing can't tell "cut
// through this whole line" apart from "snip this one tick mark". Instead,
// the stroke's own shape decides intent: a short, narrow drag (barely wider
// than one boundary's hit tolerance) reads as a precise snip -- delete the
// nearest boundary, if any, in each line it touches. A wide sweep reads as
// "trim this line's edge" -- scanner noise sits at the page margins, not in
// the middle of a real line, so a wide cut near a line's left/right edge
// trims that edge inward to the nearest INTERIOR boundary (the "boundary
// corner" nearest your cut becomes the new edge), discarding the
// edge-to-that-boundary strip. It does NOT delete the whole line -- use the
// Delete line button for a line that's entirely garbage.
const SCISSOR_SNIP_MAX_WIDTH_PX = 25;

// Trims line[lineIdx]'s nearer edge (left or right, whichever the stroke's
// center is closer to) to the nearest INTERIOR boundary -- everything from
// the old edge (previously the page margin) out to that boundary is
// discarded. Returns false (no-op) if the line has no interior boundary to
// trim to (already as tight as it can go).
function trimLineEdge(lineIdx, strokePoints) {
  const line = state.annotation.lines[lineIdx];
  const n = line.boundaries_px.length;
  if (n < 3) return false; // no interior boundary to trim to

  const xs = strokePoints.map((p) => p.x);
  const centerX = (Math.max(...xs) + Math.min(...xs)) / 2;
  const midpoint = (line.boundaries_px[0] + line.boundaries_px[n - 1]) / 2;
  const trimLeft = centerX < midpoint;

  let nearestIdx = -1;
  let nearestDist = Infinity;
  for (let idx = 1; idx < n - 1; idx++) { // interior only -- can't trim all the way to the other edge
    const d = Math.abs(line.boundaries_px[idx] - centerX);
    if (d < nearestDist) {
      nearestDist = d;
      nearestIdx = idx;
    }
  }
  if (nearestIdx === -1) return false;

  if (trimLeft) {
    line.boundaries_px = line.boundaries_px.slice(nearestIdx);
    line.boundary_source = line.boundary_source.slice(nearestIdx);
    if (line.akshara_labels) line.akshara_labels = line.akshara_labels.slice(nearestIdx);
    line.boundary_source[0] = "user";
  } else {
    line.boundaries_px = line.boundaries_px.slice(0, nearestIdx + 1);
    line.boundary_source = line.boundary_source.slice(0, nearestIdx + 1);
    if (line.akshara_labels) line.akshara_labels = line.akshara_labels.slice(0, nearestIdx);
    line.boundary_source[line.boundary_source.length - 1] = "user";
  }
  delete line.boundary_seams_px; // indices shifted -- see addBoundary
  return true;
}

function executeScissorCut(strokePoints) {
  if (strokePoints.length < 2) return; // a plain click, not a drag -- nothing to cut

  const xs = strokePoints.map((p) => p.x);
  const strokeWidth = Math.max(...xs) - Math.min(...xs);
  const centerX = (Math.max(...xs) + Math.min(...xs)) / 2;

  const touchedLines = new Set();
  strokePoints.forEach((p) => {
    const idx = findLineAtPoint(p.x, p.y);
    if (idx !== -1) touchedLines.add(idx);
  });

  if (touchedLines.size === 0) {
    setStatus("Cut missed -- drag directly across a line or boundary.", "err");
    return;
  }

  if (strokeWidth <= SCISSOR_SNIP_MAX_WIDTH_PX) {
    let boundaryCount = 0;
    for (const lineIdx of touchedLines) {
      const line = state.annotation.lines[lineIdx];
      const bIdx = findNearestBoundary(lineIdx, centerX);
      if (bIdx > 0 && bIdx < line.boundaries_px.length - 1) { // never snip a structural edge
        removeBoundary(lineIdx, bIdx);
        boundaryCount++;
      }
    }
    if (boundaryCount === 0) {
      setStatus("Cut missed -- no boundary near that snip.", "err");
      return;
    }
    state.seams = null;
    render();
    setStatus(`Cut: deleted ${boundaryCount} boundary(ies).`, "ok");
    return;
  }

  let trimmedCount = 0;
  for (const lineIdx of touchedLines) {
    if (trimLineEdge(lineIdx, strokePoints)) trimmedCount++;
  }
  if (trimmedCount === 0) {
    setStatus("Cut missed -- no interior boundary on that side to trim to.", "err");
    return;
  }
  state.selectedBoundary = -1;
  state.seams = null;
  markDirty();
  renderLineList();
  render();
  setStatus(`Cut: trimmed the edge of ${trimmedCount} line(s).`, "ok");
}

// ---- Mouse / keyboard interaction ----

canvas.addEventListener("mousedown", (e) => {
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left;
  const sy = e.clientY - rect.top;

  if (state.spaceDown || e.button === 1) {
    state.panning = true;
    state.panStart = { x: e.clientX, y: e.clientY, panX: state.panX, panY: state.panY };
    canvasWrap.classList.add("panning");
    return;
  }

  if (state.scissorMode && e.button === 0) {
    const [imgX, imgY] = screenToImage(sx, sy);
    state.scissorDragging = true;
    state.scissorPoints = [{ x: imgX, y: imgY }];
    render();
    return;
  }

  if ((state.splitMode || state.newLineMode) && e.button === 0) {
    const [imgX, imgY] = screenToImage(sx, sy);
    state.freehandDragging = true;
    state.freehandMoved = false;
    state.freehandStrokePointCount = 0;
    state.freehandStart = { x: imgX, y: imgY };
    return; // the point is added by the click handler (plain click) or streamed in mousemove (drag)
  }

  if (state.splitMode || state.newLineMode || state.scissorMode || !state.showAnnotation) return; // point placement / read-only handled elsewhere

  if (e.button !== 0) return; // left-click only for editing

  const [imgX, imgY] = screenToImage(sx, sy);
  const lineIdx = findLineAtPoint(imgX, imgY);
  if (lineIdx === -1) {
    state.selectedLine = -1;
    state.selectedBoundary = -1;
    render();
    return;
  }

  const bIdx = findNearestBoundary(lineIdx, imgX);
  state.selectedLine = lineIdx;

  if (bIdx !== -1) {
    state.selectedBoundary = bIdx;
    state.dragging = true;
    state.dragMoved = false;
  } else {
    state.selectedBoundary = -1;
  }
  renderLineList();
  render();
});

window.addEventListener("mousemove", (e) => {
  if (state.panning && state.panStart) {
    state.panX = state.panStart.panX + (e.clientX - state.panStart.x);
    state.panY = state.panStart.panY + (e.clientY - state.panStart.y);
    render();
    return;
  }

  if (state.scissorDragging) {
    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    const [imgX, imgY] = screenToImage(sx, sy);
    state.scissorPoints.push({ x: imgX, y: imgY });
    render();
    return;
  }

  if (state.freehandDragging) {
    const points = activePencilPoints();
    if (!points) { state.freehandDragging = false; return; } // mode changed mid-drag (e.g. Esc) -- bail out
    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    const [imgX, imgY] = screenToImage(sx, sy);

    if (!state.freehandMoved) {
      const d = Math.hypot(imgX - state.freehandStart.x, imgY - state.freehandStart.y);
      if (d >= PENCIL_DRAG_THRESHOLD_IMG_PX) state.freehandMoved = true;
    }
    if (state.freehandMoved) {
      if (e.shiftKey) {
        // Straight line: collapse whatever this stroke added so far into
        // one segment running from the drag's start point to the cursor --
        // recomputed every sample so the "line" follows the mouse live.
        if (state.freehandStrokePointCount > 0) {
          points.splice(points.length - state.freehandStrokePointCount, state.freehandStrokePointCount);
        }
        points.push({ x: imgX, y: imgY });
        state.freehandStrokePointCount = 1;
      } else {
        const last = points.length > 0 ? points[points.length - 1] : state.freehandStart;
        if (Math.hypot(imgX - last.x, imgY - last.y) >= PENCIL_SAMPLE_SPACING_IMG_PX) {
          points.push({ x: imgX, y: imgY });
          state.freehandStrokePointCount++;
        }
      }
      render();
    }
    return;
  }

  if (state.dragging && state.selectedLine !== -1 && state.selectedBoundary !== -1) {
    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    const [imgX] = screenToImage(sx, sy);
    moveBoundary(state.selectedLine, state.selectedBoundary, imgX);
    state.dragMoved = true;
    render();
  }
});

window.addEventListener("mouseup", (e) => {
  if (state.panning) {
    state.panning = false;
    canvasWrap.classList.remove("panning");
    return;
  }
  if (state.scissorDragging) {
    state.scissorDragging = false;
    const points = state.scissorPoints;
    state.scissorPoints = [];
    executeScissorCut(points);
    return;
  }
  if (state.freehandDragging) {
    state.freehandDragging = false;
    if (state.freehandMoved) state.freehandJustDragged = true; // suppress the click event that follows a real drag
    state.freehandStrokePointCount = 0;
    return;
  }
  if (state.dragging) {
    const wasClickNotDrag = !state.dragMoved;
    state.dragging = false;
    if (wasClickNotDrag && state.selectedLine !== -1 && state.selectedBoundary !== -1) {
      // Plain click on an existing boundary: just select it (already done on mousedown).
    }
    renderLineList();
  }
});

canvas.addEventListener("click", (e) => {
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left;
  const sy = e.clientY - rect.top;
  const [imgX, imgY] = screenToImage(sx, sy);

  if (state.scissorMode) return; // handled entirely in mousedown/mousemove/mouseup
  if (state.splitMode || state.newLineMode) {
    if (state.freehandJustDragged) {
      // A real drag already streamed its points in mousemove -- don't also
      // add the release position as a duplicate point.
      state.freehandJustDragged = false;
      return;
    }
    if (state.splitMode) addSplitPoint(imgX, imgY);
    else addNewLinePoint(imgX, imgY);
    return;
  }
  if (!state.showAnnotation) return; // read-only view, no editing

  // Only handle "add boundary" here; dragging/selecting handled in mousedown/mousemove.
  if (state.dragMoved) {
    state.dragMoved = false;
    return;
  }
  const lineIdx = findLineAtPoint(imgX, imgY);
  if (lineIdx === -1) return;

  const bIdx = findNearestBoundary(lineIdx, imgX);
  if (bIdx === -1) {
    addBoundary(lineIdx, imgX);
  }
});

canvas.addEventListener("dblclick", (e) => {
  e.preventDefault();
  removeAtEvent(e);
});

canvas.addEventListener("contextmenu", (e) => {
  e.preventDefault();
  removeAtEvent(e);
});

function removeAtEvent(e) {
  if (state.splitMode || state.newLineMode || state.scissorMode || !state.showAnnotation) return;
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left;
  const sy = e.clientY - rect.top;
  const [imgX, imgY] = screenToImage(sx, sy);
  const lineIdx = findLineAtPoint(imgX, imgY);
  if (lineIdx === -1) return;
  const bIdx = findNearestBoundary(lineIdx, imgX);
  if (bIdx !== -1) removeBoundary(lineIdx, bIdx);
}

canvasWrap.addEventListener("wheel", (e) => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left;
  const sy = e.clientY - rect.top;
  const [imgXBefore, imgYBefore] = screenToImage(sx, sy);

  const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
  state.zoom = Math.max(0.1, Math.min(8, state.zoom * factor));

  const [sxAfter, syAfter] = imageToScreen(imgXBefore, imgYBefore);
  state.panX += sx - sxAfter;
  state.panY += sy - syAfter;
  render();
}, { passive: false });

window.addEventListener("keydown", (e) => {
  // Unlike Split/New line, Scissor doesn't swallow other keys -- each cut
  // auto-commits on mouseup (no pending edit to protect), so pan/save/etc.
  // still work while the tool stays armed. Esc just turns the tool off.
  if (state.scissorMode && e.key === "Escape") {
    e.preventDefault();
    cancelScissor();
    return;
  }

  if (state.splitMode) {
    if (e.key === "Escape") {
      e.preventDefault();
      cancelSplit();
      return;
    }
    if (e.key === "Enter") {
      e.preventDefault();
      confirmSplit();
      return;
    }
    return; // swallow other keys while placing split points
  }

  if (state.newLineMode) {
    if (e.key === "Escape") {
      e.preventDefault();
      cancelNewLine();
      return;
    }
    if (e.key === "Tab") {
      e.preventDefault();
      advanceNewLineEdge();
      return;
    }
    if (e.key === "Enter") {
      e.preventDefault();
      confirmNewLine();
      return;
    }
    return; // swallow other keys while placing new-line points
  }

  if (e.code === "Space") {
    state.spaceDown = true;
    canvasWrap.classList.add("panning");
    e.preventDefault();
    return;
  }

  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
    e.preventDefault();
    save();
    return;
  }

  if (e.key.toLowerCase() === "n") {
    if (state.annotation && state.annotation.lines.length > 0) {
      state.selectedLine = Math.min(state.selectedLine + 1, state.annotation.lines.length - 1);
      state.selectedBoundary = -1;
      centerOnLine(state.selectedLine);
      renderLineList();
      render();
    }
    return;
  }
  if (e.key.toLowerCase() === "p") {
    if (state.annotation && state.annotation.lines.length > 0) {
      state.selectedLine = Math.max(state.selectedLine - 1, 0);
      state.selectedBoundary = -1;
      centerOnLine(state.selectedLine);
      renderLineList();
      render();
    }
    return;
  }

  if (["ArrowLeft", "ArrowRight"].includes(e.key)) {
    if (state.selectedLine === -1 || state.selectedBoundary === -1) return;
    e.preventDefault();
    const step = e.shiftKey ? 5 : 1;
    const line = state.annotation.lines[state.selectedLine];
    const cur = line.boundaries_px[state.selectedBoundary];
    const delta = e.key === "ArrowLeft" ? -step : step;
    moveBoundary(state.selectedLine, state.selectedBoundary, cur + delta);
    render();
    return;
  }

  if (e.key === "Delete" || e.key === "Backspace") {
    if (state.selectedLine !== -1 && state.selectedBoundary !== -1) {
      e.preventDefault();
      removeBoundary(state.selectedLine, state.selectedBoundary);
    } else if (state.selectedLine !== -1) {
      e.preventDefault();
      deleteSelectedLine();
    }
  }
});

window.addEventListener("keyup", (e) => {
  if (e.code === "Space") {
    state.spaceDown = false;
    canvasWrap.classList.remove("panning");
  }
});

window.addEventListener("resize", () => {
  resizeCanvasToWrap();
  render();
});

window.addEventListener("beforeunload", (e) => {
  if (state.dirty) {
    e.preventDefault();
    e.returnValue = "";
  }
});

document.getElementById("btn-toggle-seams").addEventListener("click", toggleSeams);
document.getElementById("btn-toggle-model-output").addEventListener("click", toggleModelOutput);
document.getElementById("btn-adopt-model-output").addEventListener("click", adoptModelOutput);
document.getElementById("btn-train-boundary").addEventListener("click", () =>
  trainModel("boundary", { resume: true, epochs: FULL_TRAIN_EPOCHS.boundary })
);
document.getElementById("btn-train-lines").addEventListener("click", () =>
  trainModel("line_segmenter", { resume: true, epochs: FULL_TRAIN_EPOCHS.line_segmenter })
);
document.getElementById("btn-update-model").addEventListener("click", () => updateModelFromCheckpoint());
document.getElementById("btn-train-yolo").addEventListener("click", () =>
  trainModel("yolo_seg", { resume: true, epochs: FULL_TRAIN_EPOCHS.yolo_seg })
);
document.getElementById("btn-yolo-original").addEventListener("click", () => toggleYoloOutput("original"));
document.getElementById("btn-yolo-finetuned").addEventListener("click", () => toggleYoloOutput("finetuned"));
document.getElementById("btn-split-line").addEventListener("click", () => {
  if (state.splitMode) cancelSplit();
  else startSplit();
});
document.getElementById("btn-new-line").addEventListener("click", () => {
  const alreadyPlainNewLine = state.newLineMode && state.extendingLineIdx == null;
  if (state.newLineMode) cancelNewLine();
  if (!alreadyPlainNewLine) startNewLine();
});
document.getElementById("btn-extend-line").addEventListener("click", () => {
  const alreadyExtending = state.newLineMode && state.extendingLineIdx != null;
  if (state.newLineMode) cancelNewLine();
  if (!alreadyExtending) startExtendLine();
});
document.getElementById("btn-delete-line").addEventListener("click", deleteSelectedLine);
document.getElementById("btn-reset-line").addEventListener("click", resetSelectedLineToProposal);
document.getElementById("btn-scissor").addEventListener("click", () => {
  if (state.scissorMode) cancelScissor();
  else startScissor();
});
document.getElementById("btn-toggle-annotation").addEventListener("click", toggleShowAnnotation);
document.getElementById("btn-save").addEventListener("click", save);
document.getElementById("btn-reset-proposals").addEventListener("click", resetToProposals);
document.getElementById("btn-clear-all").addEventListener("click", clearAllAnnotation);
document.getElementById("btn-load-saved").addEventListener("click", loadSavedData);
document.getElementById("btn-view-data").addEventListener("click", viewSavedData);
document.getElementById("btn-guide-model").addEventListener("click", guideFromModel);
document.getElementById("btn-init-cv").addEventListener("click", initCvProposals);
document.getElementById("page-select").addEventListener("change", (e) => loadPage(e.target.value));

// The page dropdown is built from whatever's actually rendered in
// data/images/ (all 209 source pages, see scripts/extract_all_pages.py) --
// no longer a couple of hardcoded <option>s. Pages with a saved annotation
// (the only ones training ever uses -- see list_annotated_pages) are
// marked with a checkmark so it's obvious at a glance which are done.
async function populatePageSelect() {
  const select = document.getElementById("page-select");
  const resp = await fetch("/api/pages");
  const data = await resp.json();
  select.innerHTML = "";
  for (const p of data.pages) {
    const opt = document.createElement("option");
    opt.value = p.page_id;
    opt.textContent = p.has_annotation ? `${p.page_id} ✓` : p.page_id;
    select.appendChild(opt);
  }
  if (data.pages.some((p) => p.page_id === state.pageId)) {
    select.value = state.pageId;
  } else if (data.pages.length > 0) {
    state.pageId = data.pages[0].page_id;
  }
}

resizeCanvasToWrap();
populatePageSelect().then(() => loadPage(state.pageId));
