// In-browser inference for the CUB-200 classifier with ONNX Runtime Web (WebAssembly backend).
//
// Pipeline (identical to birdcls.data.eval_transform used in training/evaluation):
//   image -> resize shorter side to meta.resize (512) -> centre crop meta.img_size (448)
//         -> RGB floats in [0, 1] -> (x - mean) / std -> 1 x 3 x 448 x 448 tensor (NCHW)
//         -> model.onnx -> 200 logits -> softmax -> top-5
"use strict";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
// Multi-threaded WASM needs cross-origin isolation, which static hosting usually lacks; use what is available.
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

const EXAMPLES = ["American_Goldfinch", "Blue_Jay", "Cardinal", "Painted_Bunting", "Pileated_Woodpecker",
                  "Ruby_throated_Hummingbird"];
const $ = (id) => document.getElementById(id);
let session = null, meta = null, pending = null;

function pct(x, digits = 1) { return (100 * x).toFixed(digits) + "%"; }

function showStats() {
  const m = meta.metrics || {};
  const items = [
    [m.test_top1 != null ? pct(m.test_top1, 2) : "-", "test Top-1 accuracy"],
    [m.test_avg_per_class != null ? pct(m.test_avg_per_class, 2) : "-", "average per-class accuracy"],
    ["200", "species"],
    [meta.params_M + " M", "parameters"],
    [meta.img_size + " px", "input size"],
  ];
  $("stats").innerHTML = items.map(([v, l]) => `<div class="stat"><b>${v}</b><span>${l}</span></div>`).join("");
  $("modelinfo").textContent = `ConvNeXt-Tiny (${meta.params_M} M parameters, run ${meta.run})`;
}

// Image -> normalised Float32 tensor, following torchvision Resize(shorter side) + CenterCrop.
function preprocess(img) {
  const S = meta.img_size, R = meta.resize;
  const w = img.naturalWidth, h = img.naturalHeight;
  const scale = R / Math.min(w, h);          // after resizing, the shorter side is R pixels
  const side = S / scale;                    // the S x S crop expressed in original-image pixels
  const sx = (w - side) / 2, sy = (h - side) / 2;
  const canvas = document.createElement("canvas");
  canvas.width = S; canvas.height = S;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";        // smooth downscaling, close to PIL's antialiased bilinear resize
  ctx.drawImage(img, sx, sy, side, side, 0, 0, S, S);
  const px = ctx.getImageData(0, 0, S, S).data;   // RGBA, row-major
  const plane = S * S, out = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) {
      out[c * plane + i] = (px[4 * i + c] / 255 - meta.mean[c]) / meta.std[c];
    }
  }
  return new ort.Tensor("float32", out, [1, 3, S, S]);
}

function softmax(logits) {
  const mx = Math.max(...logits);
  const e = Array.from(logits, (v) => Math.exp(v - mx));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
}

async function classify(img) {
  if (!session) { pending = img; return; }   // model still loading: run as soon as it is ready
  $("status").textContent = "Classifying...";
  await new Promise((r) => setTimeout(r, 20)); // let the status text render before the heavy work
  const t0 = performance.now();
  const output = await session.run({ image: preprocess(img) });
  const ms = performance.now() - t0;
  const probs = softmax(output.logits.data);
  const top = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 5);

  $("best").textContent = `${meta.class_names[top[0][1]]} (${pct(top[0][0])})`;
  $("note").textContent = top[0][0] < 0.3
    ? "Low confidence: similar-looking species compete, see the other candidates." : "";
  $("bars").innerHTML = top.map(([p, i]) => `
    <div class="row"><div class="lab"><span>${meta.class_names[i]}</span><span>${pct(p)}</span></div>
      <div class="track"><div class="fill" style="width:${(100 * p).toFixed(1)}%"></div></div></div>`).join("");
  $("status").textContent = `Done in ${(ms / 1000).toFixed(2)} s on this device.`;
}

function showImage(src) {
  const img = new Image();
  img.onload = () => {
    $("drop").replaceChildren(img);
    classify(img).catch((e) => { $("status").textContent = "Error: " + e.message; });
  };
  img.onerror = () => { $("status").textContent = "Could not read that file as an image."; };
  img.alt = "Selected bird photo";
  img.src = src;
}

function setupInput() {
  const drop = $("drop"), file = $("file");
  const openPicker = () => file.click();
  drop.addEventListener("click", openPicker);
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openPicker(); } });
  file.addEventListener("change", () => { if (file.files[0]) showImage(URL.createObjectURL(file.files[0])); });
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault(); drop.classList.remove("over");
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) showImage(URL.createObjectURL(f));
  });
  $("examples").innerHTML = EXAMPLES.map((n) =>
    `<img src="examples/${n}.jpg" alt="${n.replace(/_/g, " ")}" title="True species: ${n.replace(/_/g, " ")}" tabindex="0">`).join("");
  $("examples").querySelectorAll("img").forEach((el) => {
    el.addEventListener("click", () => showImage(el.src));
    el.addEventListener("keydown", (e) => { if (e.key === "Enter") showImage(el.src); });
  });
}

async function main() {
  setupInput();
  meta = await (await fetch("meta.json")).json();
  showStats();
  session = await ort.InferenceSession.create("model.onnx", { executionProviders: ["wasm"] });
  $("backend").textContent = `WebAssembly, ${ort.env.wasm.numThreads} thread(s)`;
  $("status").textContent = "Model ready. Choose a photo or click an example.";
  if (pending) { const img = pending; pending = null; await classify(img); }
}

main().catch((e) => { $("status").textContent = "Failed to load the model: " + e.message; console.error(e); });
