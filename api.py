import io, os, uuid, numpy as np
from typing import Literal, Dict, Tuple
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

# --- DICOM support ---
try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
except Exception:
    pydicom = None

import cv2

APP_TITLE = "Dental X-ray AI – Dentist-Friendly"
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI(title=APP_TITLE)

# ---------------- CORS ----------------
ALLOWED_ORIGINS = [
    "https://preview--dental-scan-analyzer.lovable.app",
    "https://dental-scan-analyzer.lovable.app",
    "https://bcfe2fae-0c0f-4493-96b6-4f09ed047686.lovableproject.com",
]
ALLOW_ORIGIN_REGEX = r"^https:\/\/([a-z0-9-]+\.)?(lovable(app|project)\.com)$"

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOW_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---- Per-class thresholds ----
PER_CLASS_THRESH: Dict[str, float] = {
    "caries": 0.40,
    "bone loss": 0.45,
    "periapical lesion": 0.50,
    "impaction": 0.35,
}

# ---- Helpers ----
def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", " ")

def dicom_to_pil(raw: bytes) -> Image.Image:
    if pydicom is None:
        raise RuntimeError("DICOM not supported. Install pydicom.")
    ds = pydicom.dcmread(io.BytesIO(raw))
    arr = ds.pixel_array
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    arr = arr.astype(np.float32)
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)

def load_upload(file: UploadFile) -> Image.Image:
    name = (file.filename or "").lower()
    raw = file.file.read()
    if name.endswith(".dcm") or file.content_type in {"application/dicom","application/dicom+json"}:
        return dicom_to_pil(raw)
    return Image.open(io.BytesIO(raw)).convert("RGB")

def maybe_enhance(pil: Image.Image, enable: bool) -> Image.Image:
    if not enable:
        return pil
    arr = np.array(pil)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    rgb = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)

def mode_params(mode: str):
    return {
        "balanced":     {"iou": 0.50},
        "high_recall":  {"iou": 0.55},
        "high_precision":{"iou": 0.45},
    }.get(mode, {"iou": 0.50})

def parse_color(val: str, default: Tuple[int,int,int]) -> Tuple[int,int,int]:
    """Parse 'r,g,b' string into tuple."""
    try:
        parts = [int(x.strip()) for x in val.split(",")]
        if len(parts) == 3:
            return tuple(parts)
    except Exception:
        pass
    return default

# ---- Load model ----
try:
    MODEL = YOLO("best.pt")
except Exception as e:
    print("❌ Model load error:", e)
    MODEL = None

def _draw_dashed_rect(img, pt1, pt2, color, thickness=2, dash_len=12, gap_len=8):
    x1, y1 = pt1; x2, y2 = pt2
    for x in range(x1, x2, dash_len + gap_len):
        cv2.line(img, (x, y1), (min(x + dash_len, x2), y1), color, thickness)
    for x in range(x1, x2, dash_len + gap_len):
        cv2.line(img, (x, y2), (min(x + dash_len, x2), y2), color, thickness)
    for y in range(y1, y2, dash_len + gap_len):
        cv2.line(img, (x1, y), (x1, min(y + dash_len, y2)), color, thickness)
    for y in range(y1, y2, dash_len + gap_len):
        cv2.line(img, (x2, y), (x2, min(y + dash_len, y2)), color, thickness)

def _text_scale(img_w):
    if img_w >= 3000: return 1.0, 2
    if img_w >= 2000: return 0.8, 2
    if img_w >= 1200: return 0.7, 2
    return 0.6, 1

# ---- Detect Endpoint ----
@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    mode: Literal["balanced","high_recall","high_precision"] = Query("balanced"),
    imgsz: int = Query(1024, ge=512, le=3072),
    enhance: bool = Query(True),
    use_augment: bool = Query(False),
    min_show: float = Query(0.12, ge=0.0, le=1.0),
    topk_fallback: int = Query(3, ge=0, le=10),
    show_levels: str = Query("high,medium,low"),
    color_high: str = Query("0,255,0", description="RGB for high confidence boxes"),
    color_medium: str = Query("0,165,255", description="RGB for medium confidence boxes"),
    color_low: str = Query("180,180,180", description="RGB for low confidence boxes"),
):
    """
    Run detection and draw bounding boxes.
    Box colors configurable via ?color_high=R,G,B etc.
    """
    if MODEL is None:
        raise HTTPException(500, "Model not loaded. Ensure best.pt is present.")

    try:
        pil = load_upload(file)
    except Exception as e:
        raise HTTPException(400, f"Could not read image: {e}")

    pil = maybe_enhance(pil, enable=enhance)
    base = np.array(pil)
    H, W = base.shape[:2]

    iou = mode_params(mode)["iou"]
    try:
        res = MODEL.predict(
            source=pil,
            imgsz=imgsz,
            conf=0.001,
            iou=iou,
            device="cpu",
            agnostic_nms=True,
            augment=use_augment,
            verbose=False,
            max_det=300,
        )[0]
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")

    names = res.names or {}
    def id2name(ci:int) -> str:
        return names.get(int(ci), str(int(ci)))

    cand, top_scores = [], []
    if res.boxes is not None and len(res.boxes):
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clsi = res.boxes.cls.cpu().numpy().astype(int)
        top_scores = sorted([float(c) for c in confs], reverse=True)[:10]
        for box, sc, ci in zip(xyxy, confs, clsi):
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            cls_name = _normalize_name(id2name(ci))
            thr = PER_CLASS_THRESH.get(cls_name, 0.40)
            cand.append({
                "raw": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": float(sc),
                        "class_id": int(ci), "class_name": id2name(ci), "cls_norm": cls_name,
                        "thr": float(thr)}
            })

    high, med, low = [], [], []
    for c in cand:
        sc, thr = c["raw"]["conf"], c["raw"]["thr"]
        if sc >= max(0.8, thr + 0.20): high.append(c)
        elif sc >= thr: med.append(c)
        elif sc >= min_show: low.append(c)

    lowered = False
    if not high and not med and topk_fallback > 0 and cand:
        tmp = sorted([z for z in cand if z["raw"]["conf"] >= min_show],
                     key=lambda z: z["raw"]["conf"], reverse=True)[:topk_fallback]
        low = tmp
        lowered = True

    show_set = {s.strip().lower() for s in show_levels.split(",") if s.strip()}
    color_high = parse_color(color_high, (0,255,0))
    color_medium = parse_color(color_medium, (0,165,255))
    color_low = parse_color(color_low, (180,180,180))

    canvas = base.copy()
    scale, thick = _text_scale(W)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_box(b, color, solid=True):
        x1,y1,x2,y2 = b["raw"]["x1"], b["raw"]["y1"], b["raw"]["x2"], b["raw"]["y2"]
        label = f'{b["raw"]["class_name"]} {b["raw"]["conf"]:.2f}'
        if solid:
            cv2.rectangle(canvas, (x1,y1), (x2,y2), color, thickness=2)
        else:
            _draw_dashed_rect(canvas, (x1,y1), (x2,y2), color, thickness=2)
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        ytxt = max(0, y1 - 5)
        cv2.rectangle(canvas, (x1, ytxt - th - 4), (x1 + tw + 6, ytxt), color, -1)
        cv2.putText(canvas, label, (x1 + 3, ytxt - 4), font, scale, (0,0,0), thick, cv2.LINE_AA)

    if "high" in show_set:
        for b in high: draw_box(b, color_high, solid=True)
    if "medium" in show_set:
        for b in med:  draw_box(b, color_medium, solid=True)
    if "low" in show_set:
        for b in low:  draw_box(b, color_low, solid=False)

    filename = f"{uuid.uuid4()}.png"
    outpath = os.path.join(STATIC_DIR, filename)
    cv2.imwrite(outpath, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    def to_out(b):
        x1,y1,x2,y2 = b["raw"]["x1"], b["raw"]["y1"], b["raw"]["x2"], b["raw"]["y2"]
        conf, cname, cid = b["raw"]["conf"], b["raw"]["class_name"], b["raw"]["class_id"]
        return {
            "class_id": cid,
            "class_name": cname,
            "confidence": round(conf, 3),
            "box": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
            "box_norm": {"x1": x1/W, "y1": y1/H, "x2": x2/W, "y2": y2/H},
        }

    resp = {
        "annotated_image_url": f"/static/{filename}",
        "buckets": {
            "high": [to_out(b) for b in high],
            "medium": [to_out(b) for b in med],
            "low": [to_out(b) for b in low],
        },
        "counts": {"high": len(high), "medium": len(med), "low": len(low)},
        "meta": {
            "mode": mode, "imgsz": imgsz, "enhance": enhance, "augment": use_augment,
            "iou": iou, "min_show": min_show, "topk_fallback": topk_fallback,
            "per_class_thresholds": PER_CLASS_THRESH,
            "show_levels": list(show_set),
            "colors": {"high": color_high, "medium": color_medium, "low": color_low},
        },
        "debug": {"top_raw_conf_scores": top_scores, "fallback_used": lowered}
    }
    return JSONResponse(resp)

@app.get("/")
def health():
    return {"status": "ok", "message": APP_TITLE}

@app.get("/version")
def version():
    return {"version": "2025-08-18-04"}
