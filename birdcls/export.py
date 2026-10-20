"""Export a trained run for the Hugging Face Space (hf_space/model.pt).

    python -m birdcls.export                                   # best row of outputs/results.csv (by val)
    python -m birdcls.export --run convnext_tiny__B_finegrained --tta
    python -m birdcls.export --run convnext_tiny__tl_L6_hires__full --onnx    # + static web app files

model.pt stores the model name, weights, class names, input size and TTA flag, so hf_space/app.py can
rebuild the model and reproduce the exact evaluation preprocessing.

--onnx also writes hf_static/model.onnx + meta.json for the free *static* Hugging Face Space, where the model runs
in the visitor's browser with ONNX Runtime Web (no server, no Python). The ONNX model is checked against PyTorch
on the whole test set (same preprocessing), so the web app's accuracy is the reported one.
"""
import argparse
import json

import pandas as pd
import torch

from . import config as C
from .data import class_names_from, read_annotations
from .models import build_model, normalization


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run name (default: best validation accuracy in outputs/results.csv)")
    ap.add_argument("--tta", action="store_true", help="enable TTA in the app (only with --run)")
    ap.add_argument("--onnx", action="store_true", help="also export model.onnx for the static web app")
    args = ap.parse_args()

    if args.run:
        run, tta = args.run, args.tta
    else:
        results = pd.read_csv(C.OUTPUT_DIR / "results.csv", index_col="run")
        best = results.val_top1.idxmax()
        run, tta = best.replace(" +TTA", ""), bool(results.loc[best, "tta"])
        print(f"best by validation accuracy: {best} (val {results.loc[best, 'val_top1']:.4f}, "
              f"test {results.loc[best, 'test_top1']:.4f})")

    ckpt = torch.load(C.run_dir(run) / "best.pt", map_location="cpu")
    full = read_annotations(C.TRAIN_TXT)
    class_names = class_names_from(full, int(full.label.max()) + 1)

    mean, std = (ckpt["mean"], ckpt["std"]) if "mean" in ckpt else normalization(ckpt["model"])

    # test metrics of the exported variant, so the app can show them (empty if not evaluated yet)
    metrics = {}
    csv = C.OUTPUT_DIR / "results.csv"
    label = run + (" +TTA" if tta else "")
    if csv.exists():
        results = pd.read_csv(csv, index_col="run")
        if label in results.index:
            metrics = {k: float(results.loc[label, k]) for k in ("test_top1", "test_avg_per_class")}

    C.DEPLOY_DIR.mkdir(exist_ok=True)
    out = C.DEPLOY_DIR / "model.pt"
    torch.save({"model": ckpt["model"], "state_dict": ckpt["state_dict"], "class_names": class_names,
                "img_size": ckpt["img_size"], "mean": list(mean), "std": list(std), "tta": tta, "run": run,
                "metrics": metrics}, out)
    print(f"exported {run} ({ckpt['model']}, {ckpt['img_size']}px, TTA={tta}, test {metrics}) -> {out} "
          f"({out.stat().st_size / 2**20:.0f} MB)")

    if args.onnx:
        export_onnx(ckpt, class_names, list(mean), list(std), run, metrics)


def export_onnx(ckpt, class_names, mean, std, run, metrics):
    """Export the network to ONNX (fixed 1 x 3 x S x S input, logits output) and verify it on the test set."""
    import numpy as np
    import onnxruntime as ort

    from .data import load_data, make_loader

    size = ckpt["img_size"]
    model = build_model(ckpt["model"], len(class_names), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()                                    # inference graph: no dropout / stochastic depth

    C.STATIC_DIR.mkdir(exist_ok=True)
    path = C.STATIC_DIR / "model.onnx"
    torch.onnx.export(model, torch.randn(1, 3, size, size), path, input_names=["image"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    # meta.json: everything the browser needs to reproduce the evaluation preprocessing + label the output
    meta = {"run": run, "model": ckpt["model"], "img_size": size, "resize": int(size * 256 / 224),
            "mean": mean, "std": std, "class_names": class_names, "metrics": metrics,
            "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1)}
    (C.STATIC_DIR / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"ONNX -> {path} ({path.stat().st_size / 2**20:.0f} MB)")

    # Verification: PyTorch (CPU) vs ONNX Runtime on all 1,204 test images, identical preprocessing
    data = load_data()
    loader = make_loader(data.test, size, 16, train=False, pin_memory=False, norm=(mean, std))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    agree = correct = 0
    max_diff = 0.0
    with torch.no_grad():
        for x, y in loader:
            ref = model(x).numpy()
            out = np.concatenate([sess.run(None, {"image": x[i:i + 1].numpy()})[0] for i in range(len(x))])
            max_diff = max(max_diff, float(np.abs(out - ref).max()))
            agree += int((out.argmax(1) == ref.argmax(1)).sum())
            correct += int((out.argmax(1) == y.numpy()).sum())
    n = len(data.test)
    print(f"ONNX check on {n} test images: same top-1 as PyTorch {agree}/{n}, ONNX test top-1 {correct / n:.4f}, "
          f"max |logit diff| {max_diff:.2e}")


if __name__ == "__main__":
    main()
