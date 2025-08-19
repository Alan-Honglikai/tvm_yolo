import os
import sys
import argparse
import warnings
import torch
import tvm
from tvm import relax
import numpy as np
import cv2

# ------------------------------------------------------------------
# Monkey-patch：避免 torch.export 警告
# ------------------------------------------------------------------
def _noop_simplefilter(*args, **kwargs):
    return None

if not hasattr(warnings, "_old_simplefilter"):
    warnings._old_simplefilter = warnings.simplefilter
    warnings.simplefilter = _noop_simplefilter

# 類別列表：0-9 + A-Z
CLASS_NAMES = [str(i) for i in range(10)] + [chr(ord("A") + i) for i in range(26)]

def load_yolov5_core_model(project_dir: str):
    yolov5_path = os.path.abspath(os.path.join(project_dir, "yolov5"))
    model_path = os.path.join(project_dir, "models", "best.pt")
    if not os.path.isdir(yolov5_path):
        raise FileNotFoundError(f"yolov5 repo not found: {yolov5_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"model not found: {model_path}")
    if yolov5_path not in sys.path:
        sys.path.insert(0, yolov5_path)
    model = torch.hub.load(yolov5_path, "custom", path=model_path, source="local", force_reload=False)
    core = getattr(model, "model", model)
    core.eval()
    return core

def export_to_relax(core_model: torch.nn.Module, input_shape=(1, 3, 640, 640)):
    from torch.export import export as torch_export
    from tvm.relax.frontend.torch import from_exported_program

    example_args = (torch.rand(input_shape, dtype=torch.float32),)
    exported_program = torch_export(core_model, args=example_args)
    mod: tvm.IRModule = from_exported_program(
        exported_program,
        keep_params_as_input=False,
        unwrap_unit_return_tuple=False,
        no_bind_return_tuple=False,
    )

    # detach params
    params = None
    try:
        from tvm.relax.frontend import detach_params
        mod, params = detach_params(mod)
        print("[INFO] detach_params: OK")
    except Exception as e:
        print(f"[WARN] detach_params skipped: {e}")

    try:
        mod = relax.transform.LegalizeOps()(mod)
        print("[INFO] Applied LegalizeOps")
    except Exception as e:
        print(f"[WARN] LegalizeOps skipped: {e}")

    return mod, params

def optimize_relax(mod: tvm.IRModule, enable=False):
    if enable:
        try:
        # pipeline optimize
            from tvm import transform
            with transform.PassContext(opt_level=3):  # level range from 0 ~ 3
                # TODO: Optimization Analysis
                mod = relax.transform.FoldConstant()(mod)
                mod = relax.transform.DeadCodeElimination()(mod)
                mod = relax.transform.FuseOps(fuse_opt_level=1)(mod)
                mod = relax.transform.FuseTIR()(mod)
                mod = relax.transform.RewriteCUDAGraph()(mod)
                mod = relax.transform.OptimizeLayoutTransform()(mod)
            print("[INFO] Applied relax transform")
        except Exception as e:
            print(f"[WARN] relax transform error: {e}")
    return mod

def generate_output(mod: tvm.IRModule):
    with open("output.log", "w", encoding="utf-8") as f:
        f.write(str(mod))
    print("[INFO] Relax IR saved to output.log")
    return

def build_vm(mod: tvm.IRModule, params=None, prefer_cuda=True, cuda_arch=None):
    def _try_build(target_str, host_target=None):
        try:
            tgt = tvm.target.Target(target_str, host=host_target) if host_target else tvm.target.Target(target_str)
            ex = relax.build(mod, target=tgt, params=params)
            dev = tvm.cuda(0) if "cuda" in target_str else tvm.cpu(0)
            vm = relax.VirtualMachine(ex, dev)
            return vm, vm["main"], dev, str(tgt)
        except Exception as e:
            return e

    if prefer_cuda:
        cuda_target = f"cuda -arch={cuda_arch}" if cuda_arch else "cuda"
        res = _try_build(cuda_target, host_target=tvm.target.Target("llvm"))
        if not isinstance(res, Exception):
            vm, main_func, dev, used = res
            print(f"[INFO] Built with CUDA target: {used}")
            return vm, main_func, dev
        else:
            print(f"[WARN] CUDA build failed: {res}\n[FALLBACK] Trying CPU...")

    cpu_res = _try_build("llvm")
    if not isinstance(cpu_res, Exception):
        vm, main_func, dev, used = cpu_res
        print(f"[INFO] Built with CPU (llvm) target: {used}")
        return vm, main_func, dev
    raise cpu_res

def preprocess(img_path: str, imgsz: int):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    orig = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (imgsz, imgsz))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return tvm.nd.array(img), orig

def postprocess_and_draw(orig_img, output, class_names=CLASS_NAMES, conf_thresh=0.3, nms_thresh=0.4, imgsz: int = 640):
    h, w, _ = orig_img.shape
    output_np = output.numpy() if hasattr(output, "numpy") else np.array(output)
    if output_np.ndim == 3 and output_np.shape[0] == 1:
        output_np = output_np[0]

    boxes, confidences, class_ids = [], [], []
    for det in output_np:
        det = np.array(det)
        if det.size < 6:
            continue
        cx, cy, bw, bh = det[:4]
        obj_conf = det[4]
        class_scores = det[5:]
        class_id = int(np.argmax(class_scores)) if class_scores.size > 0 else 0
        conf = float(obj_conf * (class_scores[class_id] if class_scores.size > 0 else 1.0))
        if conf < conf_thresh:
            continue

        x1 = int((cx - bw / 2) * w / imgsz)
        y1 = int((cy - bh / 2) * h / imgsz)
        x2 = int((cx + bw / 2) * w / imgsz)
        y2 = int((cy + bh / 2) * h / imgsz)

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        confidences.append(conf)
        class_ids.append(class_id)

    if len(boxes) == 0:
        print("[INFO] No boxes detected")
        return orig_img

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
    if len(indices) == 0:
        print("[INFO] NMS filtered all boxes")
        return orig_img

    for i in np.array(indices).flatten():
        x, y, w_box, h_box = boxes[int(i)]
        class_id = class_ids[int(i)]
        label_text = class_names[class_id] if class_id < len(class_names) else str(class_id)
        label = f"{label_text}: {confidences[int(i)]:.2f}"
        print(f"[DEBUG] Box {i}: x={x}, y={y}, w={w_box}, h={h_box}, class={label_text}, conf={confidences[int(i)]:.2f}")
        cv2.rectangle(orig_img, (x, y), (x + w_box, y + h_box), (255, 0, 0), 3)
        cv2.putText(orig_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return orig_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--img", default="input.jpg")
    parser.add_argument("--out", default="output.jpg")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--enable_optimizing", type=bool, default=False)
    args = parser.parse_args()

    core_model = load_yolov5_core_model(args.project_dir)
    mod, params = export_to_relax(core_model, input_shape=(1, 3, args.imgsz, args.imgsz))
    mod = optimize_relax(mod, args.enable_optimizing)
    generate_output(mod)
    vm, main_func, dev = build_vm(mod, params)

    if args.no_run:  # 簡化保留原行為
        print("[INFO] Skipped inference (--no-run)")
        return

    img_path = args.img if os.path.isabs(args.img) else os.path.join(args.project_dir, args.img)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(args.project_dir, args.out)

    input_tensor, orig_img = preprocess(img_path, args.imgsz)

    # -----------------------------
    # Fix 1: PyTorch 真 forward (避免 FakeTensor)
    # -----------------------------
    try:
        pt_model = load_yolov5_core_model(args.project_dir)  # fresh instance
        pt_model.eval()
        input_tensor_torch = torch.from_numpy(input_tensor.numpy()).float()
        with torch.no_grad():
            pt_out = pt_model(input_tensor_torch)
            pt_out_tensor = pt_out[0] if isinstance(pt_out, (list, tuple)) else pt_out
            print("[DEBUG] PyTorch forward output shape:", pt_out_tensor.shape)
            try:
                if isinstance(pt_out_tensor, torch.Tensor):
                    print("[DEBUG] First 5 PyTorch detections:\n", pt_out_tensor[0][:5].cpu().numpy())
                else:
                    print("[DEBUG] PyTorch forward returned non-tensor:", type(pt_out_tensor))
            except Exception as e:
                print("[WARN] Could not print numeric PyTorch output:", e)
    except Exception as e:
        print("[WARN] PyTorch true-forward check failed (continuing):", e)

    # -----------------------------
    # Fix 2: 處理 TVM Relax VM output（簡化，保證展開 NDArray）
    # -----------------------------
    res = main_func(input_tensor)

    # 如果是 list/tuple，取第一個元素
    if isinstance(res, (list, tuple)):
        res0 = res[0]
    else:
        res0 = res

    # 如果是 NDArray，轉成 numpy
    if isinstance(res0, tvm.nd.NDArray):
        output_np = res0.numpy()
    else:
        # 若是多層包裝（例如 list 包 NDArray），再展開
        try:
            output_np = np.array(res0)
            if output_np.ndim == 1 and isinstance(output_np[0], tvm.nd.NDArray):
                output_np = output_np[0].numpy()
        except Exception:
            output_np = np.array(res0)

    # Debug 確認 shape
    print("[DEBUG] Relax VM forward output shape:", getattr(output_np, "shape", None))
    print("[DEBUG] Relax VM first few detections:\n", output_np[0][:5] if output_np.ndim >= 2 else output_np[:5])


    # Debug prints
    try:
        if isinstance(output_np, np.ndarray) and output_np.ndim == 3 and output_np.shape[0] == 1:
            print("[DEBUG] Relax VM forward output shape:", output_np.shape)
            print("[DEBUG] First 5 Relax VM detections:\n", output_np[0][:5])
        else:
            print("[DEBUG] Relax VM forward output shape:", getattr(output_np, "shape", None))
            try:
                print("[DEBUG] Relax VM sample:\n", output_np[:5])
            except Exception:
                pass
    except Exception as e:
        print("[WARN] Debug printing of Relax output failed:", e)

    # -----------------------------
    # 後處理 & 輸出圖片
    # -----------------------------
    output_img = postprocess_and_draw(orig_img, output_np, imgsz=args.imgsz)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, output_img)
    print(f"[OK] Saved: {out_path}")

if __name__ == "__main__":
    main()
