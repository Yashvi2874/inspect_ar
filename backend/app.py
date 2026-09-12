"""
ZENITH INSPECT-AR — HTTP API

A Flask server exposing the inspection pipeline over HTTP, so something other
than the desktop OpenCV window can drive it: a web dashboard, a UAV ground
station, or a batch job.

Ported from the team's hackathon API (zenith_practice/backend/app.py) and
rebuilt on IntegratedInspectionPipeline rather than on individually wired
models, so it inherits the pipeline's fixes instead of repeating its bugs.

Run it:
    cd inspect_ar
    set PYTHONIOENCODING=utf-8
    python backend/app.py                 # http://127.0.0.1:5000

Endpoints
    GET  /api/health     service and model status
    POST /api/calibrate  image -> px-to-mm ratio from an ArUco marker
    POST /api/analyze    image -> damage detections (+ dimensions if calibrated)
    POST /api/anomaly    image -> unsupervised anomaly regions + heatmap
    POST /api/text       image -> serial/part numbers read off the component
    GET  /api/models     which checkpoints are present on disk

Every endpoint takes a multipart upload under the field name `file`.
"""

import base64
import io
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from models.aruco_calibration import ArucoCalibrator  # noqa: E402

app = Flask(__name__)
CORS(app)

# 50 MB ceiling. Flask returns 413 past it rather than buffering the whole body.
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

MODELS_DIR = BACKEND_DIR / "models"
DETECTOR_CHECKPOINT = MODELS_DIR / "best_model.pth"
AUTOENCODER_CHECKPOINT = MODELS_DIR / "anomaly_autoencoder.pth"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}

# Heavy models are loaded on first use, not at import. Importing this module
# stays instant, which keeps it testable and lets /api/health answer before a
# 165 MB checkpoint has been read from disk.
_pipeline = None
_anomaly = None
_text_reader = None
_calibrator = ArucoCalibrator()


# --------------------------------------------------------------------------
# Lazy model access
# --------------------------------------------------------------------------
def get_pipeline():
    """The supervised pipeline: Faster R-CNN damage detection + dimensions."""
    global _pipeline
    if _pipeline is None:
        from integrated_inspection_pipeline import IntegratedInspectionPipeline
        _pipeline = IntegratedInspectionPipeline(
            use_defect_detector=True,
            use_yolo=False,     # stock COCO weights have no damage classes
            use_vgg16=False,    # no trained Keras checkpoint exists
            use_blip=False,     # ~1 GB download, and it segfaults on some setups
            use_dimensions=True,
        )
    return _pipeline


def get_text_reader():
    """OCR for part markings. Loaded on first use; weights are ~64 MB."""
    global _text_reader
    if _text_reader is None:
        from models.text_reader import TextReader
        _text_reader = TextReader()
    return _text_reader


def get_anomaly_detector():
    """The unsupervised autoencoder: reconstruction error + heatmap."""
    global _anomaly
    if _anomaly is None:
        from models.anomaly_autoencoder import AnomalyDetector
        _anomaly = AnomalyDetector(checkpoint_path=AUTOENCODER_CHECKPOINT)
    return _anomaly


# --------------------------------------------------------------------------
# Request helpers
# --------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def read_upload():
    """
    Decode the uploaded image straight from memory.

    The original API wrote every upload into an uploads/ directory and never
    removed it, so the disk grew without bound and two users uploading the same
    filename overwrote each other. Decoding in memory removes the attack
    surface, the cleanup problem and the collision at once.

    Returns:
        (image, error_response, status). image is None when error_response is set.
    """
    if "file" not in request.files:
        return None, {"success": False, "error": "No file provided"}, 400

    upload = request.files["file"]
    if upload.filename == "":
        return None, {"success": False, "error": "Empty filename"}, 400
    if not allowed_file(upload.filename):
        return None, {
            "success": False,
            "error": f"Unsupported type. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        }, 400

    buffer = np.frombuffer(upload.read(), np.uint8)
    if buffer.size == 0:
        return None, {"success": False, "error": "Empty upload"}, 400

    # imdecode validates the actual bytes, so a .jpg full of something else is
    # rejected here rather than trusted because of its extension.
    #
    # It can also RAISE rather than return None — cv2.error on a corrupt header,
    # or an allocation failure on a decompression-bomb image whose pixel
    # dimensions dwarf its compressed size. An uncaught exception here escapes
    # as a 500 with an empty body, so a bad upload looks like a server fault.
    try:
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except cv2.error as e:
        return None, {"success": False,
                      "error": f"Could not decode image: {e}"}, 400
    if image is None:
        return None, {"success": False, "error": "Not a decodable image"}, 400

    return image, None, None


def encode_jpeg(image: np.ndarray) -> str:
    """BGR image -> base64 JPEG, so an annotated frame can ride in the JSON."""
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return ""
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def marker_size_from_request(default: float = 50.0) -> float:
    try:
        return float(request.form.get("marker_size_mm", default))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health_check():
    """Liveness plus what is actually loadable — no unconditional 'ready'."""
    return jsonify({
        "status": "online",
        "service": "ZENITH INSPECT-AR",
        "models": {
            "aruco": "ready",
            "defect_detector": "available" if DETECTOR_CHECKPOINT.is_file() else "checkpoint missing",
            "anomaly_autoencoder": "available" if AUTOENCODER_CHECKPOINT.is_file() else "checkpoint missing",
        },
        "loaded": {
            "pipeline": _pipeline is not None,
            "anomaly": _anomaly is not None,
            "text_reader": _text_reader is not None,
        },
    })


@app.route("/api/models", methods=["GET"])
def model_status():
    """Which checkpoints exist on disk, and how large."""
    def describe(path: Path):
        if not path.is_file():
            return {"present": False}
        return {"present": True, "size_bytes": path.stat().st_size, "path": path.name}

    return jsonify({
        "defect_detector": describe(DETECTOR_CHECKPOINT),
        "anomaly_autoencoder": describe(AUTOENCODER_CHECKPOINT),
        "note": "Checkpoints are gitignored and are not distributed with the repo.",
    })


@app.route("/api/calibrate", methods=["POST"])
def calibrate_camera():
    """Recover the pixel-to-millimetre ratio from an ArUco marker in the image."""
    image, error, status = read_upload()
    if error:
        return jsonify(error), status

    result = _calibrator.calibrate(image, marker_size_mm=marker_size_from_request())

    if result["success"]:
        return jsonify({
            "success": True,
            "calibration_data": result["calibration"],
            "marker_count": result["marker_count"],
            "reference_size_mm": result["reference_size_mm"],
        })

    # A failed calibration is a 200 with success=false, not a 400: the request
    # was valid, the image simply had no marker in it.
    return jsonify({
        "success": False,
        "error": result.get("error", "No ArUco marker detected"),
        "marker_count": 0,
    })


@app.route("/api/analyze", methods=["POST"])
def analyze_object():
    """
    Supervised inspection: find damage, and measure it when a marker is present.

    Optional form fields:
        marker_size_mm    printed marker size, default 50.0
        annotate          "1" to include a base64 annotated JPEG
    """
    image, error, status = read_upload()
    if error:
        return jsonify(error), status

    try:
        started = time.time()
        pipeline = get_pipeline()

        # Calibration is per-request and explicitly optional. The old version
        # passed the ratio through unconditionally; that ratio used to default
        # to 1.0 when no marker was found, which silently relabelled pixel
        # counts as millimetres. It is None on failure now, and dimensions are
        # simply omitted rather than invented.
        calibrated = pipeline.calibrate(image, marker_size_mm=marker_size_from_request())

        results = pipeline.process_frame(image)

        detections = []
        for det in results.get("detections", []):
            obj_id = det["id"]
            entry = {
                "id": obj_id,
                "class": det.get("class"),
                "confidence": round(float(det.get("confidence", 0.0)), 4),
                "bbox": [round(float(v), 1) for v in det.get("bbox", [])],
                "untrained_class": det.get("untrained_class", False),
            }
            # .get() throughout: a detection can legitimately have no dimension
            # entry (uncalibrated, or the contour pass found nothing). The old
            # code indexed directly and raised KeyError into a 500.
            dims = results.get("dimensions", {}).get(obj_id)
            if dims:
                entry["dimensions_mm"] = {
                    "width": round(dims.get("width_mm", 0.0), 2),
                    "height": round(dims.get("height_mm", 0.0), 2),
                    "area": round(dims.get("area_mm2", 0.0), 2),
                }
            detections.append(entry)

        response = {
            "success": True,
            "calibrated": bool(calibrated),
            "pixel_to_mm_ratio": pipeline.pixel_to_mm_ratio,
            "objects_detected": len(detections),
            "detections": detections,
            # Every trained class is a damage class; there is no healthy class
            # in the training data. Any detection at all therefore means damage.
            "overall_status": "FAIL" if detections else "PASS",
            "processing_time_s": round(time.time() - started, 3),
        }

        if request.form.get("annotate") == "1":
            response["annotated_jpeg_base64"] = encode_jpeg(
                pipeline.draw_results(image, results))

        return jsonify(response)

    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except Exception as e:
        return jsonify({"success": False, "error": f"{type(e).__name__}: {e}"}), 500


@app.route("/api/anomaly", methods=["POST"])
def detect_anomaly():
    """
    Unsupervised inspection via autoencoder reconstruction error.

    Finds anything unlike defect-free aircraft skin, including damage types the
    supervised detector was never labelled for.

    Optional form fields:
        threshold   0-1 cut on the normalised map, default 0.6
        heatmap     "1" to include a base64 heatmap overlay JPEG
    """
    image, error, status = read_upload()
    if error:
        return jsonify(error), status

    try:
        threshold = float(request.form.get("threshold", 0.6))
    except (TypeError, ValueError):
        threshold = 0.6

    try:
        started = time.time()
        detector = get_anomaly_detector()
        result = detector.detect(image, threshold=threshold)

        response = {
            "success": True,
            "regions_found": len(result["regions"]),
            "regions": [
                {
                    "id": r["id"],
                    "bbox": [round(v, 1) for v in r["bbox"]],
                    "area_px": round(r["area"], 1),
                    "peak_score": round(r["peak_score"], 6),
                    "mean_score": round(r["mean_score"], 6),
                }
                for r in result["regions"]
            ],
            "anomaly_ratio": round(result["anomaly_ratio"], 4),
            "relative_error": (round(result["relative_error"], 3)
                               if result["relative_error"] is not None else None),
            "threshold": threshold,
            "processing_time_s": round(time.time() - started, 3),
        }

        if request.form.get("heatmap") == "1":
            response["heatmap_jpeg_base64"] = encode_jpeg(result["heatmap"])

        return jsonify(response)

    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except Exception as e:
        return jsonify({"success": False, "error": f"{type(e).__name__}: {e}"}), 500


@app.route("/api/text", methods=["POST"])
def read_text():
    """
    Read serial numbers, part numbers and labels stamped on a component.

    A defect report is only actionable if it names the part it was found on.
    This reads that identifier straight off the airframe.

    Optional form fields:
        serials_only  "1" to return only strings that look like part numbers
        annotate      "1" to include a base64 JPEG with the reads boxed
    """
    image, error, status = read_upload()
    if error:
        return jsonify(error), status

    try:
        started = time.time()
        reader = get_text_reader()
        results = reader.read(image)

        if request.form.get("serials_only") == "1":
            results = [r for r in results if r["looks_like_serial"]]

        response = {
            "success": True,
            "count": len(results),
            "results": [
                {
                    "text": r["text"],
                    "confidence": r["confidence"],
                    "bbox": [round(v, 1) for v in r["bbox"]],
                    "looks_like_serial": r["looks_like_serial"],
                }
                for r in results
            ],
            "serials": [r["text"] for r in results if r["looks_like_serial"]],
            "processing_time_s": round(time.time() - started, 3),
        }

        if request.form.get("annotate") == "1":
            response["annotated_jpeg_base64"] = encode_jpeg(reader.draw(image, results))

        return jsonify(response)

    except ImportError as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except Exception as e:
        return jsonify({"success": False, "error": f"{type(e).__name__}: {e}"}), 500


@app.errorhandler(413)
def too_large(_):
    return jsonify({"success": False, "error": "File exceeds the 50 MB limit"}), 413


# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Bind to loopback by default. The original bound 0.0.0.0 with debug=True,
    # which exposes the Werkzeug debugger to the whole network — and that
    # debugger executes arbitrary Python from the browser. Opt in deliberately:
    #     set INSPECT_AR_HOST=0.0.0.0
    host = os.environ.get("INSPECT_AR_HOST", "127.0.0.1")
    port = int(os.environ.get("INSPECT_AR_PORT", "5000"))
    debug = os.environ.get("INSPECT_AR_DEBUG") == "1"

    print("=" * 60)
    print("ZENITH INSPECT-AR — HTTP API")
    print("=" * 60)
    print(f"  detector checkpoint    : {'found' if DETECTOR_CHECKPOINT.is_file() else 'MISSING'}")
    print(f"  autoencoder checkpoint : {'found' if AUTOENCODER_CHECKPOINT.is_file() else 'MISSING'}")
    print(f"  listening on           : http://{host}:{port}")
    if debug and host != "127.0.0.1":
        print("  WARNING: debug mode on a non-loopback host allows remote code execution")
    print("=" * 60)

    app.run(debug=debug, host=host, port=port)
