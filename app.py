import sys
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import subprocess
import tempfile
import urllib.request
import json
import datetime
import time
import mimetypes

# ── Firebase ──────────────────────────────────────────────────────────────────
try:
    import google.auth
    import google.auth.transport.requests
    from google.oauth2 import service_account
    import requests as _requests
    _firebase_ok = True
except ImportError:
    _firebase_ok = False

# Load .env from the project directory
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_env_path = os.path.join(_PROJECT_DIR, ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

_SERVICE_ACCOUNT_PATH = os.path.join(
    _PROJECT_DIR,
    os.environ.get("FIREBASE_SERVICE_ACCOUNT", "serviceAccountKey.json"),
)
_FIRESTORE_SCOPES = [
    "https://www.googleapis.com/auth/datastore",
    "https://www.googleapis.com/auth/firebase",
    "https://www.googleapis.com/auth/devstorage.full_control",
]
_firebase_project_id = None
_firebase_storage_bucket = None
_firebase_credentials = None


def _load_firebase():
    """Load service account credentials and project metadata."""
    global _firebase_ok, _firebase_project_id, _firebase_storage_bucket, _firebase_credentials
    if not _firebase_ok:
        print("[firebase] pip install google-auth google-auth-httplib2 requests")
        return False
    if not os.path.exists(_SERVICE_ACCOUNT_PATH):
        print(f"[firebase] serviceAccountKey.json not found at {_SERVICE_ACCOUNT_PATH}")
        _firebase_ok = False
        return False
    try:
        _firebase_credentials = service_account.Credentials.from_service_account_file(
            _SERVICE_ACCOUNT_PATH, scopes=_FIRESTORE_SCOPES
        )
        with open(_SERVICE_ACCOUNT_PATH) as f:
            sa = json.load(f)
        _firebase_project_id = sa["project_id"]
        # Storage bucket: default is <project_id>.firebasestorage.app
        _firebase_storage_bucket = sa.get(
            "storage_bucket", f"{_firebase_project_id}.firebasestorage.app"
        )
        return True
    except Exception as e:
        print(f"[firebase] Failed to load credentials: {e}")
        _firebase_ok = False
        return False


def _get_access_token():
    """Return a fresh OAuth2 bearer token."""
    _firebase_credentials.refresh(google.auth.transport.requests.Request())
    return _firebase_credentials.token


def _classify_from_confidence(avg_conf_pct: float) -> str:
    """Map average confidence % → classification label."""
    if avg_conf_pct >= 75:
        return "Good"
    elif avg_conf_pct >= 50:
        return "Fair"
    elif avg_conf_pct >= 25:
        return "Poor"
    else:
        return "Bad"


def _upload_image_to_storage(image_path: str, image_name: str) -> tuple[str, str]:
    """
    Upload a local image file to Firebase Storage.
    Returns (public_download_url, storage_path).
    """
    token = _get_access_token()
    ext = os.path.splitext(image_path)[1] or ".jpg"
    timestamp_ms = int(time.time() * 1000)
    name_no_ext = os.path.splitext(image_name)[0]
    storage_path = f"Images/{name_no_ext}_{timestamp_ms}{ext}"
    encoded_path = urllib.request.quote(storage_path, safe="")
    upload_url = (
        f"https://firebasestorage.googleapis.com/v0/b/"
        f"{_firebase_storage_bucket}/o?uploadType=media&name={encoded_path}"
    )
    content_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    resp = _requests.post(
        upload_url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": content_type,
        },
        data=image_bytes,
        timeout=30,
    )
    resp.raise_for_status()
    # Build public download URL with token
    upload_token = resp.json().get("downloadTokens", "")
    download_url = (
        f"https://firebasestorage.googleapis.com/v0/b/{_firebase_storage_bucket}"
        f"/o/{encoded_path}?alt=media&token={upload_token}"
    )
    return download_url, storage_path


def _post_crack_record(
    label: str,
    classification: str,
    location: str,
    datetime_str: str,
    length_cm: str,
    width_cm: str,
    depth_cm: str,
    image_url: str,
    image_path: str,
    image_name: str,
    description: str = "",
) -> str:
    """
    POST a crack_record document to Firestore REST API.
    Returns the new document ID.
    """
    token = _get_access_token()
    url = (
        f"https://firestore.googleapis.com/v1/projects/{_firebase_project_id}"
        f"/databases/(default)/documents/crack_records"
    )
    now_iso = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    body = {
        "fields": {
            "label":          {"stringValue": label},
            "description":    {"stringValue": description},
            "classification": {"stringValue": classification},
            "location":       {"stringValue": location},
            "datetime":       {"stringValue": datetime_str},
            "length":         {"stringValue": length_cm},
            "width":          {"stringValue": width_cm},
            "depth":          {"stringValue": depth_cm},
            "imageName":      {"stringValue": image_name},
            "imageUrl":       {"stringValue": image_url},
            "imagePath":      {"stringValue": image_path},
            "createdAt":      {"timestampValue": now_iso},
        }
    }
    resp = _requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=15,
    )
    resp.raise_for_status()
    # Document name is like projects/.../documents/crack_records/<id>
    doc_name = resp.json().get("name", "")
    return doc_name.split("/")[-1]


_firebase_ready = _load_firebase()
# ─────────────────────────────────────────────────────────────────────────────

# ── Flask API ─────────────────────────────────────────────────────────────────
# Serves the latest annotated frame at http://<your-mac-ip>:8000/api/frame
# The display ESP32 polls this endpoint to mirror what the app sees.
try:
    from flask import Flask, Response

    _flask_ok = True
except ImportError:
    _flask_ok = False

_frame_lock = threading.Lock()
_frame_jpg = None  # latest annotated frame as JPEG bytes


def _set_latest_frame(bgr):
    global _frame_jpg
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if ok:
        with _frame_lock:
            _frame_jpg = bytes(buf)


_app_instance = None  # set after CrackDetectorApp is created


def _run_flask():
    if not _flask_ok:
        print("[display] pip install flask  →  enables ESP32 display mirror")
        return
    app = Flask(__name__)

    @app.route("/api/frame")
    def frame():
        with _frame_lock:
            data = _frame_jpg
        if data is None:
            return Response(status=204)
        return Response(
            data,
            mimetype="image/jpeg",
            headers={"Content-Length": str(len(data)), "Cache-Control": "no-cache"},
        )

    @app.route("/health")
    def health():
        return {"status": "ok"}

    @app.route("/api/snap", methods=["POST"])
    def api_snap():
        # Triggers capture + inference in whatever mode the app is currently in
        if _app_instance is not None:
            _app_instance.root.after(0, _app_instance._upload_image)
        return {"status": "ok"}

    import logging

    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=8000, threaded=True)


threading.Thread(target=_run_flask, daemon=True).start()


def _get_local_ip():
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ─────────────────────────────────────────────────────────────────────────────


def resource_path(relative):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative)


MODEL_PATH = resource_path("best.pt")

# ── Design Tokens ─────────────────────────────────────────────────────────────
BG = "#F9FAFB"
CARD = "#FFFFFF"
BORDER = "#E5E7EB"
PRIMARY = "#2563EB"
PRIMARY_HVR = "#1D4ED8"
PRIMARY_TINT = "#EFF6FF"
PRIMARY_MID = "#BFDBFE"
TEXT_HEAD = "#1F2937"
TEXT_BODY = "#374151"
TEXT_MUTED = "#6B7280"
SUCCESS = "#059669"
DANGER = "#DC2626"
WARNING = "#D97706"
SIDEBAR_W = 300

MASK_COLORS = [
    (37, 99, 235),
    (5, 150, 105),
    (220, 38, 38),
    (217, 119, 6),
    (124, 58, 237),
    (6, 182, 212),
    (236, 72, 153),
    (16, 185, 129),
]


class CrackDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CrackScan — Concrete Analysis")
        self.root.geometry("1240x800")
        self.root.minsize(1040, 680)
        self.root.configure(bg=BG)

        self.model = None
        self.current_image_path = None
        self.photo = None
        self.camera_running = False
        self.ip_cam_running = False
        self.cap = None
        self.mode = "camera"
        self.confidence = tk.DoubleVar(value=0.5)
        self.ip_url_var = tk.StringVar(value="http://192.168.1.100")
        self.cam_distance_var = tk.StringVar(value="60")  # cm from surface
        self.focal_length_var = tk.StringVar(value="3.6")  # mm — ESP32-CAM default
        self._last_result = None          # stores latest YOLO result
        self._last_annotated = None       # stores latest annotated BGR frame
        self._firebase_location = "Centro"  # default barangay

        global _app_instance
        _app_instance = self
        self._setup_styles()
        self._build_ui()
        self._load_model()
        # Start in Live Cam mode by default
        self.root.after(100, lambda: self._switch_mode("camera"))

    # ── Styles ────────────────────────────────────────────────────────────────
    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Blue.Horizontal.TScale",
            background=CARD,
            troughcolor="#DBEAFE",
            sliderlength=20,
            sliderrelief="flat",
            borderwidth=0,
        )

    # ── UI ───────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_main()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    def _build_sidebar(self):
        self.sidebar = tk.Frame(
            self.root,
            bg=CARD,
            width=SIDEBAR_W,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.pack_propagate(False)

        # ── Scrollable inner frame ────────────────────────────────────────────
        _sb = tk.Scrollbar(self.sidebar, orient="vertical", width=6)
        _sb.pack(side="right", fill="y")
        _scroll_canvas = tk.Canvas(
            self.sidebar, bg=BG, bd=0, highlightthickness=0,
            yscrollcommand=_sb.set
        )
        _scroll_canvas.pack(side="left", fill="both", expand=True)
        _sb.config(command=_scroll_canvas.yview)
        self.sidebar = tk.Frame(_scroll_canvas, bg=BG)
        _cw = _scroll_canvas.create_window((0, 0), window=self.sidebar, anchor="nw")
        self.sidebar.bind("<Configure>", lambda e: _scroll_canvas.configure(
            scrollregion=_scroll_canvas.bbox("all")
        ))
        _scroll_canvas.bind("<Configure>", lambda e: _scroll_canvas.itemconfig(
            _cw, width=e.width
        ))
        _scroll_canvas.bind_all("<MouseWheel>", lambda e: _scroll_canvas.yview_scroll(
            int(-1 * (e.delta / 120)), "units"
        ))
        # ─────────────────────────────────────────────────────────────────────

        # Brand
        header = tk.Frame(self.sidebar, bg=PRIMARY, height=64)
        header.pack(fill="x")
        header.pack_propagate(False)
        brand = tk.Frame(header, bg=PRIMARY)
        brand.pack(side="left", padx=18, fill="y")
        icon_bg = tk.Frame(brand, bg=PRIMARY_HVR, width=32, height=32)
        icon_bg.pack(side="left", pady=16)
        icon_bg.pack_propagate(False)
        tk.Label(
            icon_bg, text="⬡", font=("Helvetica", 13, "bold"), fg=CARD, bg=PRIMARY_HVR
        ).place(relx=0.5, rely=0.5, anchor="center")
        title_col = tk.Frame(brand, bg=PRIMARY)
        title_col.pack(side="left", padx=(10, 0), pady=14)
        tk.Label(
            title_col,
            text="CrackScan",
            font=("Helvetica", 14, "bold"),
            fg=CARD,
            bg=PRIMARY,
        ).pack(anchor="w")
        tk.Label(
            title_col,
            text="Concrete Analysis",
            font=("Helvetica", 10),
            fg=PRIMARY_MID,
            bg=PRIMARY,
        ).pack(anchor="w")

        # API Server IP bar
        _local_ip = _get_local_ip()
        ip_bar = tk.Frame(self.sidebar, bg=PRIMARY_TINT, highlightbackground=PRIMARY_MID, highlightthickness=1)
        ip_bar.pack(fill="x", padx=16, pady=(10, 0))
        tk.Label(
            ip_bar,
            text="API SERVER",
            font=("Helvetica", 8, "bold"),
            fg=TEXT_MUTED,
            bg=PRIMARY_TINT,
        ).pack(side="left", padx=(10, 6), pady=6)
        tk.Label(
            ip_bar,
            text=f"{_local_ip}:8000",
            font=("Helvetica", 11, "bold"),
            fg=PRIMARY,
            bg=PRIMARY_TINT,
        ).pack(side="left", pady=6)

        # Mode tabs (3)
        self._sidebar_label("MODE")
        toggle_wrap = tk.Frame(
            self.sidebar, bg="#F3F4F6", highlightbackground=BORDER, highlightthickness=1
        )
        toggle_wrap.pack(fill="x", padx=16, pady=(6, 0))
        inner = tk.Frame(toggle_wrap, bg="#F3F4F6", padx=3, pady=3)
        inner.pack(fill="x")

        self.cam_tab = tk.Button(
            inner,
            text="Live Cam",
            font=("Helvetica", 11, "bold"),
            fg=PRIMARY,
            bg=CARD,
            relief="flat",
            cursor="hand2",
            pady=7,
            bd=0,
            highlightthickness=0,
            activeforeground=PRIMARY,
            activebackground=PRIMARY_TINT,
            command=lambda: self._switch_mode("camera"),
        )
        self.cam_tab.pack(side="left", fill="x", expand=True, padx=(0, 1))

        self.ip_tab = tk.Button(
            inner,
            text="IP Cam",
            font=("Helvetica", 11),
            fg=TEXT_MUTED,
            bg="#F3F4F6",
            relief="flat",
            cursor="hand2",
            pady=7,
            bd=0,
            highlightthickness=0,
            activeforeground=PRIMARY,
            activebackground=PRIMARY_TINT,
            command=lambda: self._switch_mode("ipcam"),
        )
        self.ip_tab.pack(side="left", fill="x", expand=True, padx=(1, 0))

        # Live camera controls
        self.camera_controls = tk.Frame(self.sidebar, bg=CARD)
        self._sidebar_label("CAMERA", parent=self.camera_controls)
        cam_card = self._card(self.camera_controls)
        self.cam_btn = self._list_btn(
            cam_card, "▶  Start Camera", SUCCESS, self._toggle_camera, divider=False
        )

        # IP camera controls
        self.ipcam_controls = tk.Frame(self.sidebar, bg=CARD)
        self._sidebar_label("ESP32-CAM", parent=self.ipcam_controls)

        url_card = tk.Frame(
            self.ipcam_controls,
            bg=CARD,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        url_card.pack(fill="x", padx=16, pady=(6, 0))
        tk.Label(
            url_card,
            text="Camera IP Address",
            font=("Helvetica", 10, "bold"),
            fg=TEXT_MUTED,
            bg=CARD,
        ).pack(anchor="w", padx=12, pady=(10, 4))
        url_entry = tk.Entry(
            url_card,
            textvariable=self.ip_url_var,
            font=("Helvetica", 12),
            fg=TEXT_HEAD,
            bg="#F9FAFB",
            relief="flat",
            bd=0,
            highlightbackground=BORDER,
            highlightthickness=1,
            insertbackground=PRIMARY,
        )
        url_entry.pack(fill="x", padx=12, pady=(0, 4), ipady=7)
        tk.Label(
            url_card,
            text="Stream: /stream  •  Snapshot: /capture",
            font=("Helvetica", 9),
            fg=TEXT_MUTED,
            bg=CARD,
        ).pack(anchor="w", padx=12, pady=(0, 10))

        ip_btn_card = self._card(self.ipcam_controls)
        self.ip_connect_btn = self._list_btn(
            ip_btn_card,
            "▶  Connect Stream",
            SUCCESS,
            self._toggle_ip_stream,
            divider=True,
        )
        # Large prominent Take Picture button
        snap_frame = tk.Frame(self.ipcam_controls, bg=CARD)
        snap_frame.pack(fill="x", padx=16, pady=(10, 0))
        self.ip_snap_btn = tk.Button(
            snap_frame,
            text="📷  Take Picture",
            font=("Helvetica", 14, "bold"),
            fg=CARD,
            bg=PRIMARY,
            relief="flat",
            cursor="hand2",
            pady=14,
            bd=0,
            highlightthickness=0,
            activeforeground=CARD,
            activebackground=PRIMARY_HVR,
            command=self._ip_snapshot,
        )
        self.ip_snap_btn.pack(fill="x")

        # Confidence
        tk.Frame(self.sidebar, bg=BORDER, height=1).pack(
            fill="x", padx=16, pady=(16, 0)
        )
        self._sidebar_label("DETECTION THRESHOLD")
        conf_card = self._card(self.sidebar)
        conf_row = tk.Frame(conf_card, bg=CARD)
        conf_row.pack(fill="x", padx=14, pady=(12, 4))
        tk.Label(
            conf_row, text="Confidence", font=("Helvetica", 12), fg=TEXT_BODY, bg=CARD
        ).pack(side="left")
        self.conf_label = tk.Label(
            conf_row, text="50%", font=("Helvetica", 12, "bold"), fg=PRIMARY, bg=CARD
        )
        self.conf_label.pack(side="right")
        self.slider = ttk.Scale(
            conf_card,
            from_=0.1,
            to=1.0,
            orient="horizontal",
            variable=self.confidence,
            style="Blue.Horizontal.TScale",
            command=self._on_slider,
        )
        self.slider.pack(fill="x", padx=14, pady=(0, 4))
        ticks = tk.Frame(conf_card, bg=CARD)
        ticks.pack(fill="x", padx=14, pady=(0, 10))
        tk.Label(
            ticks, text="10%", font=("Helvetica", 10), fg=TEXT_MUTED, bg=CARD
        ).pack(side="left")
        tk.Label(
            ticks, text="100%", font=("Helvetica", 10), fg=TEXT_MUTED, bg=CARD
        ).pack(side="right")

        # Camera distance + focal length
        tk.Frame(self.sidebar, bg=BORDER, height=1).pack(
            fill="x", padx=16, pady=(16, 0)
        )
        self._sidebar_label("CAMERA SETUP")
        cam_setup_card = tk.Frame(
            self.sidebar, bg=CARD, highlightbackground=BORDER, highlightthickness=1
        )
        cam_setup_card.pack(fill="x", padx=16, pady=(6, 0))

        # Distance row
        dist_row = tk.Frame(cam_setup_card, bg=CARD)
        dist_row.pack(fill="x", padx=12, pady=(10, 4))
        tk.Label(
            dist_row,
            text="Distance to surface",
            font=("Helvetica", 10, "bold"),
            fg=TEXT_MUTED,
            bg=CARD,
        ).pack(side="left")
        tk.Label(
            dist_row, text="cm", font=("Helvetica", 10), fg=TEXT_MUTED, bg=CARD
        ).pack(side="right")
        dist_entry = tk.Entry(
            cam_setup_card,
            textvariable=self.cam_distance_var,
            font=("Helvetica", 12),
            fg=TEXT_HEAD,
            bg="#F9FAFB",
            relief="flat",
            bd=0,
            highlightbackground=BORDER,
            highlightthickness=1,
            insertbackground=PRIMARY,
            width=8,
        )
        dist_entry.pack(fill="x", padx=12, pady=(0, 4), ipady=6)

        # Focal length row
        focal_row = tk.Frame(cam_setup_card, bg=CARD)
        focal_row.pack(fill="x", padx=12, pady=(6, 4))
        tk.Label(
            focal_row,
            text="Focal length",
            font=("Helvetica", 10, "bold"),
            fg=TEXT_MUTED,
            bg=CARD,
        ).pack(side="left")
        tk.Label(
            focal_row, text="mm", font=("Helvetica", 10), fg=TEXT_MUTED, bg=CARD
        ).pack(side="right")
        focal_entry = tk.Entry(
            cam_setup_card,
            textvariable=self.focal_length_var,
            font=("Helvetica", 12),
            fg=TEXT_HEAD,
            bg="#F9FAFB",
            relief="flat",
            bd=0,
            highlightbackground=BORDER,
            highlightthickness=1,
            insertbackground=PRIMARY,
            width=8,
        )
        focal_entry.pack(fill="x", padx=12, pady=(0, 4), ipady=6)
        tk.Label(
            cam_setup_card,
            text="ESP32-CAM: 3.6mm  •  Webcam: 3.7mm",
            font=("Helvetica", 9),
            fg=TEXT_MUTED,
            bg=CARD,
        ).pack(anchor="w", padx=12, pady=(0, 10))

        # Stats
        tk.Frame(self.sidebar, bg=BORDER, height=1).pack(
            fill="x", padx=16, pady=(16, 0)
        )
        self._sidebar_label("RESULTS")
        stats_card = self._card(self.sidebar)
        self.stat_count = self._stat_row(stats_card, "Cracks Found", "—", False)
        self.stat_conf = self._stat_row(stats_card, "Avg Confidence", "—", False)
        self.stat_highest = self._stat_row(stats_card, "Highest", "—", False)
        self.stat_lowest = self._stat_row(stats_card, "Lowest", "—", True)

        # Instances
        tk.Frame(self.sidebar, bg=BORDER, height=1).pack(
            fill="x", padx=16, pady=(16, 0)
        )
        self._sidebar_label("INSTANCES")
        self.detection_frame = tk.Frame(self.sidebar, bg=BG)
        self.detection_frame.pack(fill="both", expand=True, padx=16, pady=(6, 0))

        # ── Save to Firebase ──────────────────────────────────────────────
        tk.Frame(self.sidebar, bg=BORDER, height=1).pack(
            fill="x", padx=16, pady=(16, 0)
        )
        self._sidebar_label("FIREBASE")
        firebase_card = tk.Frame(
            self.sidebar, bg=CARD, highlightbackground=BORDER, highlightthickness=1
        )
        firebase_card.pack(fill="x", padx=16, pady=(6, 0))

        # Location row inside the card
        loc_row = tk.Frame(firebase_card, bg=CARD)
        loc_row.pack(fill="x", padx=12, pady=(10, 4))
        tk.Label(
            loc_row,
            text="Location (Barangay)",
            font=("Helvetica", 10, "bold"),
            fg=TEXT_MUTED,
            bg=CARD,
        ).pack(side="left")
        self._fb_location_var = tk.StringVar(value=self._firebase_location)
        loc_entry = tk.Entry(
            firebase_card,
            textvariable=self._fb_location_var,
            font=("Helvetica", 12),
            fg=TEXT_HEAD,
            bg="#F9FAFB",
            relief="flat",
            bd=0,
            highlightbackground=BORDER,
            highlightthickness=1,
            insertbackground=PRIMARY,
        )
        loc_entry.pack(fill="x", padx=12, pady=(0, 10), ipady=6)

        # Save button
        self.save_fb_btn = tk.Button(
            self.sidebar,
            text="☁  Save to Firebase",
            font=("Helvetica", 13, "bold"),
            fg=CARD,
            bg=PRIMARY,
            relief="flat",
            cursor="hand2",
            pady=13,
            bd=0,
            highlightthickness=0,
            activeforeground=CARD,
            activebackground=PRIMARY_HVR,
            state="disabled",
            command=self._save_to_firebase,
        )
        self.save_fb_btn.pack(fill="x", padx=16, pady=(8, 0))

        self.fb_status_var = tk.StringVar(value="")
        self.fb_status_lbl = tk.Label(
            self.sidebar,
            textvariable=self.fb_status_var,
            font=("Helvetica", 10),
            fg=TEXT_MUTED,
            bg=BG,
            wraplength=SIDEBAR_W - 32,
            justify="left",
        )
        self.fb_status_lbl.pack(anchor="w", padx=16, pady=(4, 0))
        # ────────────────────────────────────────────────────────────────────

        # Status bar
        status_bar = tk.Frame(
            self.sidebar,
            bg="#F3F4F6",
            highlightbackground=BORDER,
            highlightthickness=1,
            height=40,
        )
        status_bar.pack(fill="x", side="bottom")
        status_bar.pack_propagate(False)
        self.status_dot = tk.Label(
            status_bar, text="●", font=("Helvetica", 10), fg=TEXT_MUTED, bg="#F3F4F6"
        )
        self.status_dot.pack(side="left", padx=(12, 4))
        self.status_var = tk.StringVar(value="Loading model…")
        self.status_lbl = tk.Label(
            status_bar,
            textvariable=self.status_var,
            font=("Helvetica", 11),
            fg=TEXT_MUTED,
            bg="#F3F4F6",
            anchor="w",
        )
        self.status_lbl.pack(side="left", fill="x", expand=True)

    # ── Main Panel ────────────────────────────────────────────────────────────
    def _build_main(self):
        self.main = tk.Frame(self.root, bg=BG)
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.grid_rowconfigure(1, weight=1)
        self.main.grid_columnconfigure(0, weight=1)

        topbar = tk.Frame(
            self.main,
            bg=CARD,
            height=56,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        topbar.grid(row=0, column=0, sticky="ew")
        topbar.grid_propagate(False)
        tk.Label(
            topbar,
            text="Detection View",
            font=("Helvetica", 15, "bold"),
            fg=TEXT_HEAD,
            bg=CARD,
        ).pack(side="left", padx=24, pady=16)
        self.mode_pill = tk.Label(
            topbar,
            text="IMAGE MODE",
            font=("Helvetica", 10, "bold"),
            fg=CARD,
            bg=PRIMARY,
            padx=10,
            pady=3,
        )
        self.mode_pill.pack(side="right", padx=24, pady=16)
        self.ip_live_dot = tk.Label(
            topbar,
            text="● LIVE",
            font=("Helvetica", 10, "bold"),
            fg=CARD,
            bg=SUCCESS,
            padx=10,
            pady=3,
        )

        content = tk.Frame(self.main, bg=BG)
        content.grid(row=1, column=0, sticky="nsew", padx=16, pady=16)
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=1)

        self.empty_state = tk.Frame(
            content, bg=CARD, highlightbackground=BORDER, highlightthickness=1
        )
        self.empty_state.grid(row=0, column=0, sticky="nsew")
        empty_inner = tk.Frame(self.empty_state, bg=CARD)
        empty_inner.place(relx=0.5, rely=0.5, anchor="center")
        icon_wrap = tk.Frame(
            empty_inner,
            bg=PRIMARY_TINT,
            highlightbackground=PRIMARY_MID,
            highlightthickness=1,
            width=72,
            height=72,
        )
        icon_wrap.pack(pady=(0, 16))
        icon_wrap.pack_propagate(False)
        tk.Label(icon_wrap, text="📷", font=("Helvetica", 28), bg=PRIMARY_TINT).place(
            relx=0.5, rely=0.5, anchor="center"
        )
        tk.Label(
            empty_inner,
            text="No image loaded",
            font=("Helvetica", 16, "bold"),
            fg=TEXT_HEAD,
            bg=CARD,
        ).pack()
        tk.Label(
            empty_inner,
            text="Take a photo, start live camera,\nor connect an ESP32-CAM to begin",
            font=("Helvetica", 12),
            fg=TEXT_MUTED,
            bg=CARD,
            justify="center",
        ).pack(pady=(6, 0))

        self.canvas_wrap = tk.Frame(
            content, bg="#F1F5F9", highlightbackground=BORDER, highlightthickness=1
        )
        self.canvas_wrap.grid(row=0, column=0, sticky="nsew")
        self.canvas_wrap.grid_remove()
        self.canvas_wrap.grid_rowconfigure(0, weight=1)
        self.canvas_wrap.grid_columnconfigure(0, weight=1)
        self.canvas = tk.Label(self.canvas_wrap, bg="#F1F5F9")
        self.canvas.grid(row=0, column=0, sticky="nsew")

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _sidebar_label(self, text, parent=None):
        p = parent or self.sidebar
        bg = CARD if parent else BG
        tk.Label(
            p, text=text, font=("Helvetica", 10, "bold"), fg=TEXT_MUTED, bg=bg
        ).pack(anchor="w", padx=16, pady=(14, 0))

    def _card(self, parent):
        f = tk.Frame(parent, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        f.pack(fill="x", padx=16, pady=(6, 0))
        return f

    def _list_btn(self, parent, text, color, cmd, divider=True, state="normal"):
        btn = tk.Button(
            parent,
            text=text,
            font=("Helvetica", 13),
            fg=color,
            bg=CARD,
            relief="flat",
            cursor="hand2",
            pady=12,
            bd=0,
            highlightthickness=0,
            anchor="w",
            padx=14,
            state=state,
            command=cmd,
            activeforeground=color,
            activebackground=PRIMARY_TINT,
        )
        btn.pack(fill="x")
        if divider:
            tk.Frame(parent, bg=BORDER, height=1).pack(fill="x")
        return btn

    def _stat_row(self, parent, label, value, is_last):
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x")
        tk.Label(row, text=label, font=("Helvetica", 12), fg=TEXT_BODY, bg=CARD).pack(
            side="left", padx=14, pady=11
        )
        val = tk.Label(
            row, text=value, font=("Helvetica", 12, "bold"), fg=TEXT_MUTED, bg=CARD
        )
        val.pack(side="right", padx=14)
        if not is_last:
            tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=14)
        return val

    def _set_status(self, msg, color=None):
        self.status_var.set(msg)
        c = color or TEXT_MUTED
        self.status_lbl.config(fg=c)
        self.status_dot.config(fg=c)

    def _show_empty(self):
        self.canvas_wrap.grid_remove()
        self.empty_state.grid()

    def _show_canvas(self):
        self.empty_state.grid_remove()
        self.canvas_wrap.grid()

    def _on_slider(self, val):
        self.conf_label.config(text=f"{int(float(val)*100)}%")

    # ── Mode Switch ───────────────────────────────────────────────────────────
    def _switch_mode(self, mode):
        if self.camera_running:
            self._stop_camera()
        if self.ip_cam_running:
            self._stop_ip_stream()

        self.mode = mode
        self.photo = None
        self._show_empty()
        self._clear_stats()

        for tab in [self.cam_tab, self.ip_tab]:
            tab.config(bg="#F3F4F6", fg=TEXT_MUTED, font=("Helvetica", 11))
        self.camera_controls.pack_forget()
        self.ipcam_controls.pack_forget()
        self.ip_live_dot.pack_forget()

        if mode == "camera":
            self.cam_tab.config(bg=CARD, fg=PRIMARY, font=("Helvetica", 11, "bold"))
            self.camera_controls.pack(fill="x")
            self.mode_pill.config(text="LIVE CAM", bg=SUCCESS)
        elif mode == "ipcam":
            self.ip_tab.config(bg=CARD, fg=PRIMARY, font=("Helvetica", 11, "bold"))
            self.ipcam_controls.pack(fill="x")
            self.mode_pill.config(text="IP CAM", bg=WARNING)

    # ── Model ─────────────────────────────────────────────────────────────────
    def _load_model(self):
        self._set_status("Loading model…")

        def load():
            try:
                self.model = YOLO(MODEL_PATH)
                self.root.after(0, lambda: self._set_status("Model ready", SUCCESS))
            except Exception as e:
                self.root.after(0, lambda: self._set_status(f"Error: {e}", DANGER))

        threading.Thread(target=load, daemon=True).start()

    # ── Image Capture ─────────────────────────────────────────────────────────
    def _upload_image(self):
        self._set_status("Taking photo…", PRIMARY)
        self.upload_btn.config(state="disabled", fg=TEXT_MUTED)
        self.root.update()
        if sys.platform == "win32":
            threading.Thread(target=self._capture_windows, daemon=True).start()
        else:
            self._capture_mac()

    def _capture_windows(self):
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                self.root.after(
                    0, lambda: self._set_status("✗ No camera found", DANGER)
                )
                self.root.after(
                    0, lambda: self.upload_btn.config(state="normal", fg=PRIMARY)
                )
                return
            for _ in range(5):
                cap.read()
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                self.root.after(
                    0, lambda: self._set_status("✗ Failed to capture", DANGER)
                )
                self.root.after(
                    0, lambda: self.upload_btn.config(state="normal", fg=PRIMARY)
                )
                return
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.close()
            cv2.imwrite(tmp.name, frame)
            self.current_image_path = tmp.name
            self.root.after(0, lambda: self._display_image(tmp.name))
            self.root.after(
                0, lambda: self.analyze_btn.config(state="normal", fg=PRIMARY)
            )
            self.root.after(
                0, lambda: self._set_status("Photo ready — click Analyze", TEXT_MUTED)
            )
            self.root.after(0, self._clear_stats)
        except Exception as e:
            self.root.after(0, lambda: self._set_status(f"✗ {e}", DANGER))
        finally:
            self.root.after(
                0, lambda: self.upload_btn.config(state="normal", fg=PRIMARY)
            )

    def _capture_mac(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()
        result = subprocess.run(["imagesnap", "-w", "1", tmp.name], capture_output=True)
        self.upload_btn.config(state="normal", fg=PRIMARY)
        if result.returncode != 0:
            self._set_status("✗ Run: brew install imagesnap", DANGER)
            return
        self.current_image_path = tmp.name
        self._display_image(tmp.name)
        self.analyze_btn.config(state="normal", fg=PRIMARY)
        self._set_status("Photo ready — click Analyze", TEXT_MUTED)
        self._clear_stats()

    # ── Inference ─────────────────────────────────────────────────────────────
    def _run_inference(self):
        if not self.model or not self.current_image_path:
            return
        self.analyze_btn.config(state="disabled", fg=TEXT_MUTED)
        self._set_status("Analyzing image…", PRIMARY)
        conf = self.confidence.get()

        def infer():
            try:
                preds = self.model.predict(
                    self.current_image_path, conf=conf, verbose=False
                )
                result = preds[0]
                frame = cv2.imread(self.current_image_path)
                annotated = self._draw_masks(frame, result)
                self.root.after(0, lambda: self._show_result(annotated, result))
            except Exception as e:
                self.root.after(0, lambda: self._set_status(f"Error: {e}", DANGER))
            finally:
                self.root.after(
                    0, lambda: self.analyze_btn.config(state="normal", fg=PRIMARY)
                )

        threading.Thread(target=infer, daemon=True).start()

    # ── Live Camera ───────────────────────────────────────────────────────────
    def _toggle_camera(self):
        if self.camera_running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self._set_status("Camera not found", DANGER)
            return
        self.camera_running = True
        self.cam_btn.config(text="■  Stop Camera", fg=DANGER)
        self._set_status("● Live camera running", SUCCESS)
        threading.Thread(target=self._camera_loop, daemon=True).start()

    def _stop_camera(self):
        self.camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.cam_btn.config(text="▶  Start Camera", fg=SUCCESS)
        self._set_status("Camera stopped", TEXT_MUTED)
        self._show_empty()
        self.photo = None
        self._clear_stats()

    def _camera_loop(self):
        while self.camera_running:
            # Drain buffer — grab twice, use only the latest frame
            self.cap.grab()
            ret, frame = self.cap.retrieve()
            if not ret:
                break
            conf = self.confidence.get()
            if self.model:
                preds = self.model.predict(frame, conf=conf, verbose=False)
                result = preds[0]
                annotated = self._draw_masks(frame, result)
                self.root.after(
                    0, lambda r=result, a=annotated: self._show_result(a, r)
                )
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.root.after(0, lambda f=rgb: self._display_image_array(f))

    # ── IP Camera ─────────────────────────────────────────────────────────────
    def _base_url(self):
        return self.ip_url_var.get().rstrip("/")

    def _toggle_ip_stream(self):
        if self.ip_cam_running:
            self._stop_ip_stream()
        else:
            self._start_ip_stream()

    def _start_ip_stream(self):
        base = self._base_url()
        stream_url = f"{base}/stream"
        cap = cv2.VideoCapture(stream_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            self.cap = cap
            self.ip_cam_running = True
            self.ip_connect_btn.config(text="■  Disconnect", fg=DANGER)
            self._set_status(f"● Streaming {base}", SUCCESS)
            self.ip_live_dot.pack(side="right", padx=(0, 8), pady=16)
            threading.Thread(target=self._ip_stream_loop, daemon=True).start()
        else:
            cap.release()
            self._set_status("MJPEG failed — using snapshot polling", WARNING)
            self.ip_cam_running = True
            self.ip_connect_btn.config(text="■  Disconnect", fg=DANGER)
            self.ip_live_dot.pack(side="right", padx=(0, 8), pady=16)
            threading.Thread(target=self._ip_snapshot_loop, daemon=True).start()

    def _stop_ip_stream(self):
        self.ip_cam_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.ip_connect_btn.config(text="▶  Connect Stream", fg=SUCCESS)
        self.ip_live_dot.pack_forget()
        self._set_status("IP camera disconnected", TEXT_MUTED)
        self._show_empty()
        self.photo = None
        self._clear_stats()

    def _ip_stream_loop(self):
        while self.ip_cam_running and self.cap:
            # Drain buffer — always use the freshest frame
            self.cap.grab()
            ret, frame = self.cap.retrieve()
            if not ret:
                self.root.after(
                    0,
                    lambda: self._set_status(
                        "Stream lost — retrying snapshot…", WARNING
                    ),
                )
                self.cap.release()
                self.cap = None
                threading.Thread(target=self._ip_snapshot_loop, daemon=True).start()
                break
            conf = self.confidence.get()
            if self.model:
                preds = self.model.predict(frame, conf=conf, verbose=False)
                result = preds[0]
                annotated = self._draw_masks(frame, result)
                self.root.after(
                    0, lambda r=result, a=annotated: self._show_result(a, r)
                )
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.root.after(0, lambda f=rgb: self._display_image_array(f))

    def _ip_snapshot_loop(self):
        import time

        snap_url = f"{self._base_url()}/capture"
        while self.ip_cam_running:
            frame = self._fetch_snapshot(snap_url)
            if frame is not None:
                conf = self.confidence.get()
                if self.model:
                    preds = self.model.predict(frame, conf=conf, verbose=False)
                    result = preds[0]
                    annotated = self._draw_masks(frame, result)
                    self.root.after(
                        0, lambda r=result, a=annotated: self._show_result(a, r)
                    )
                else:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.root.after(0, lambda f=rgb: self._display_image_array(f))
            else:
                self.root.after(
                    0, lambda: self._set_status("✗ Cannot reach camera", DANGER)
                )
                time.sleep(2)

    def _fetch_snapshot(self, url):
        try:
            req = urllib.request.urlopen(url, timeout=3)
            img_array = np.frombuffer(req.read(), dtype=np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def _ip_snapshot(self):
        snap_url = f"{self._base_url()}/capture"
        self._set_status("Grabbing snapshot…", PRIMARY)
        self.ip_snap_btn.config(state="disabled", fg=TEXT_MUTED)

        def grab():
            frame = self._fetch_snapshot(snap_url)
            if frame is None:
                self.root.after(
                    0,
                    lambda: self._set_status(
                        "✗ Cannot reach camera — check IP", DANGER
                    ),
                )
                self.root.after(
                    0, lambda: self.ip_snap_btn.config(state="normal", fg=PRIMARY)
                )
                return
            conf = self.confidence.get()
            if self.model:
                preds = self.model.predict(frame, conf=conf, verbose=False)
                result = preds[0]
                annotated = self._draw_masks(frame, result)
                self.root.after(
                    0, lambda r=result, a=annotated: self._show_result(a, r)
                )
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.root.after(0, lambda f=rgb: self._display_image_array(f))
            self.root.after(
                0, lambda: self.ip_snap_btn.config(state="normal", fg=PRIMARY)
            )

        threading.Thread(target=grab, daemon=True).start()

    # ── Drawing ───────────────────────────────────────────────────────────────
    def _px_to_cm(self, px, frame_h):
        """Convert pixel length to cm using pinhole camera model.
        cm_per_px = (sensor_h_mm * distance_cm) / (focal_mm * frame_h_px)
        Sensor height assumed 3.6mm (1/4" sensor, common in ESP32-CAM & webcams).
        """
        try:
            distance_cm = float(self.cam_distance_var.get())
            focal_mm = float(self.focal_length_var.get())
            sensor_h_mm = 3.6  # 1/4" sensor height in mm
            cm_per_px = (sensor_h_mm * distance_cm) / (focal_mm * frame_h)
            return px * cm_per_px
        except (ValueError, ZeroDivisionError):
            return None

    def _draw_masks(self, img, result):
        overlay = img.copy()
        self._crack_dims = []  # list of (w_px, h_px) per detection
        self._last_frame_h = img.shape[0]  # store for sidebar cm conversion
        ih, iw = img.shape[:2]

        if result.masks is not None:
            label_boxes = []  # track placed label rects to avoid overlap

            for i, (mask, box) in enumerate(zip(result.masks.data, result.boxes)):
                color = MASK_COLORS[i % len(MASK_COLORS)]
                m = mask.cpu().numpy()
                m = cv2.resize(m, (iw, ih)) > 0.5
                overlay[m] = (overlay[m] * 0.35 + np.array(color) * 0.65).astype(
                    np.uint8
                )
                mask_uint = (m * 255).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask_uint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(overlay, contours, -1, color, 2)

                # Bounding box
                x1, y1, x2, y2 = (
                    int(box.xyxy[0][0]),
                    int(box.xyxy[0][1]),
                    int(box.xyxy[0][2]),
                    int(box.xyxy[0][3]),
                )
                crack_w = x2 - x1
                crack_h = y2 - y1
                self._crack_dims.append((crack_w, crack_h))

                # Centroid of segmented mask
                M = cv2.moments(mask_uint)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Build label lines
                conf = float(box.conf)
                frame_h = ih
                w_cm = self._px_to_cm(crack_w, frame_h)
                h_cm = self._px_to_cm(crack_h, frame_h)
                if w_cm is not None:
                    line1 = f"Confidence: {conf*100:.0f}%"
                    line2 = f"L {h_cm:.1f}cm x W {w_cm:.1f}cm"
                else:
                    line1 = f"Confidence: {conf*100:.0f}%"
                    line2 = f"L {crack_h}px x W {crack_w}px"

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.52
                thickness = 1
                pad = 8

                (w1, th1), _ = cv2.getTextSize(line1, font, font_scale, thickness)
                (w2, th2), _ = cv2.getTextSize(line2, font, font_scale, thickness)
                box_w = max(w1, w2) + pad * 2
                box_h = th1 + th2 + pad * 3  # top pad + between + bottom pad

                # Diagonal offset from centroid — try 4 quadrants to avoid overlap
                offset = 60
                candidates = [
                    (cx + offset, cy - offset),  # top-right
                    (cx - offset - box_w, cy - offset),  # top-left
                    (cx + offset, cy + offset),  # bottom-right
                    (cx - offset - box_w, cy + offset),  # bottom-left
                ]

                def overlaps(rx1, ry1, rx2, ry2):
                    for ex1, ey1, ex2, ey2 in label_boxes:
                        if rx1 < ex2 and rx2 > ex1 and ry1 < ey2 and ry2 > ey1:
                            return True
                    return False

                lx, ly = candidates[0]  # default
                for ox, oy in candidates:
                    # Clamp to image bounds
                    ox = max(2, min(ox, iw - box_w - 2))
                    oy = max(2, min(oy, ih - box_h - 2))
                    if not overlaps(ox, oy, ox + box_w, oy + box_h):
                        lx, ly = ox, oy
                        break

                lx = max(2, min(lx, iw - box_w - 2))
                ly = max(2, min(ly, ih - box_h - 2))
                label_boxes.append((lx, ly, lx + box_w, ly + box_h))

                # Pointer dot on centroid
                cv2.circle(overlay, (cx, cy), 5, color, -1, cv2.LINE_AA)
                cv2.circle(overlay, (cx, cy), 5, (255, 255, 255), 1, cv2.LINE_AA)

                # Pointer line: centroid → nearest edge of label box
                label_cx = lx + box_w // 2
                label_cy = ly + box_h // 2
                dx = cx - label_cx
                dy = cy - label_cy
                norm = max(1, (dx**2 + dy**2) ** 0.5)
                edge_x = int(label_cx + (dx / norm) * (box_w // 2))
                edge_y = int(label_cy + (dy / norm) * (box_h // 2))
                cv2.line(
                    overlay, (cx, cy), (edge_x, edge_y), (255, 255, 255), 2, cv2.LINE_AA
                )
                cv2.line(overlay, (cx, cy), (edge_x, edge_y), color, 1, cv2.LINE_AA)

                # Label box — filled + white border
                cv2.rectangle(overlay, (lx, ly), (lx + box_w, ly + box_h), color, -1)
                cv2.rectangle(
                    overlay, (lx, ly), (lx + box_w, ly + box_h), (255, 255, 255), 1
                )

                # Crack number pill above label box
                tag = f" #{i+1} "
                (tw, tth), _ = cv2.getTextSize(tag, font, 0.40, 1)
                pill_x2 = lx + tw + 6
                pill_y1 = ly - tth - 6
                cv2.rectangle(overlay, (lx, pill_y1), (pill_x2, ly), color, -1)
                cv2.rectangle(overlay, (lx, pill_y1), (pill_x2, ly), (255, 255, 255), 1)
                cv2.putText(
                    overlay,
                    tag,
                    (lx + 3, ly - 3),
                    font,
                    0.40,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                # Text lines
                cv2.putText(
                    overlay,
                    line1,
                    (lx + pad, ly + pad + th1),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    overlay,
                    line2,
                    (lx + pad, ly + pad * 2 + th1 + th2),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

        return overlay

    # ── Display ───────────────────────────────────────────────────────────────
    def _show_result(self, annotated, result):
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        self._display_image_array(rgb)
        self._update_stats(result)
        if self.mode == "ipcam":
            self._update_detections(result)
        # ── Store latest result for Firebase save ────────────────────────
        self._last_result = result
        self._last_annotated = annotated.copy()
        if _firebase_ready and len(result.boxes) > 0:
            self.save_fb_btn.config(state="normal")
        else:
            self.save_fb_btn.config(state="disabled")
        # ── Send annotated frame to display ESP32 via /api/frame ──────────────
        threading.Thread(
            target=_set_latest_frame, args=(annotated,), daemon=True
        ).start()

    def _display_image(self, path):
        self._render_pil(Image.open(path))

    def _display_image_array(self, arr):
        self._render_pil(Image.fromarray(arr))

    def _render_pil(self, img):
        self._show_canvas()
        w = max(self.canvas_wrap.winfo_width(), 600)
        h = max(self.canvas_wrap.winfo_height(), 520)
        img.thumbnail((w - 2, h - 2), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.config(image=self.photo)

    # ── Stats ─────────────────────────────────────────────────────────────────
    def _update_stats(self, result):
        boxes = result.boxes
        if len(boxes) == 0:
            for w in [
                self.stat_count,
                self.stat_conf,
                self.stat_highest,
                self.stat_lowest,
            ]:
                w.config(text="—", fg=TEXT_MUTED)
            return
        confs = [float(b.conf) * 100 for b in boxes]
        self.stat_count.config(text=str(len(boxes)), fg=DANGER)
        self.stat_conf.config(text=f"{np.mean(confs):.1f}%", fg=PRIMARY)
        self.stat_highest.config(text=f"{max(confs):.1f}%", fg=SUCCESS)
        self.stat_lowest.config(text=f"{min(confs):.1f}%", fg=PRIMARY)

    def _clear_stats(self):
        for w in [self.stat_count, self.stat_conf, self.stat_highest, self.stat_lowest]:
            w.config(text="—", fg=TEXT_MUTED)
        for w in self.detection_frame.winfo_children():
            w.destroy()
        self._last_result = None
        self._last_annotated = None
        self.save_fb_btn.config(state="disabled")
        self.fb_status_var.set("")

    def _update_detections(self, result):
        for w in self.detection_frame.winfo_children():
            w.destroy()
        dims = getattr(self, "_crack_dims", [])
        for i, box in enumerate(result.boxes):
            rgb = MASK_COLORS[i % len(MASK_COLORS)]
            color_hex = "#%02x%02x%02x" % rgb
            conf = float(box.conf) * 100
            cw, ch = dims[i] if i < len(dims) else (0, 0)

            item = tk.Frame(
                self.detection_frame,
                bg=CARD,
                highlightbackground=BORDER,
                highlightthickness=1,
            )
            item.pack(fill="x", pady=(0, 5))

            # Top row — color swatch + name + confidence badge
            top = tk.Frame(item, bg=CARD)
            top.pack(fill="x", padx=12, pady=(10, 4))
            swatch = tk.Frame(top, bg=color_hex, width=10, height=10)
            swatch.pack(side="left", padx=(0, 8))
            swatch.pack_propagate(False)
            tk.Label(
                top,
                text=f"Crack #{i+1}",
                font=("Helvetica", 12, "bold"),
                fg=TEXT_HEAD,
                bg=CARD,
            ).pack(side="left")
            badge_bg = tk.Frame(
                top,
                bg=PRIMARY_TINT,
                highlightbackground=PRIMARY_MID,
                highlightthickness=1,
            )
            badge_bg.pack(side="right")
            tk.Label(
                badge_bg,
                text=f"{conf:.0f}%",
                font=("Helvetica", 11, "bold"),
                fg=PRIMARY,
                bg=PRIMARY_TINT,
                padx=8,
                pady=3,
            ).pack()

            # Bottom row — width / height dimensions
            # Convert to cm if possible
            frame_h = getattr(self, "_last_frame_h", 320)
            w_cm = self._px_to_cm(cw, frame_h)
            if w_cm is not None:
                h_cm = self._px_to_cm(ch, frame_h)
                dim_text = f"W: {w_cm:.1f} cm   H: {h_cm:.1f} cm"
            else:
                dim_text = f"W: {cw}px   H: {ch}px"
            dim_row = tk.Frame(item, bg=CARD)
            dim_row.pack(fill="x", padx=12, pady=(0, 10))
            tk.Label(
                dim_row,
                text=dim_text,
                font=("Helvetica", 10),
                fg=TEXT_MUTED,
                bg=CARD,
            ).pack(side="left")

            tk.Frame(item, bg=BORDER, height=1).pack(fill="x", padx=12)


    # ── Firebase Save ──────────────────────────────────────────────────
    def _save_to_firebase(self):
        """Collect inference data and POST to Firestore + Storage in a thread."""
        result = self._last_result
        annotated = self._last_annotated
        if result is None or annotated is None:
            self.fb_status_var.set("⚠ No inference result to save.")
            self.fb_status_lbl.config(fg=WARNING)
            return

        self.save_fb_btn.config(state="disabled")
        self.fb_status_var.set("⏳ Uploading…")
        self.fb_status_lbl.config(fg=PRIMARY)

        def _worker():
            try:
                # ── Compute dimensions from detected cracks ─────────────────────
                dims = getattr(self, "_crack_dims", [])
                frame_h = getattr(self, "_last_frame_h", 320)
                lengths, widths = [], []
                for i, box in enumerate(result.boxes):
                    cw, ch = dims[i] if i < len(dims) else (0, 0)
                    w_cm = self._px_to_cm(cw, frame_h)
                    h_cm = self._px_to_cm(ch, frame_h)
                    if w_cm is not None:
                        widths.append(w_cm)
                        lengths.append(h_cm)
                    else:
                        widths.append(float(cw))
                        lengths.append(float(ch))

                avg_length = sum(lengths) / len(lengths) if lengths else 0.0
                avg_width  = sum(widths)  / len(widths)  if widths  else 0.0
                # depth: not directly measured — stored as "0"
                depth_str  = "0"

                # ── Confidence & classification ──────────────────────────────
                confs = [float(b.conf) * 100 for b in result.boxes]
                avg_conf = sum(confs) / len(confs)
                classification = _classify_from_confidence(avg_conf)

                # ── Auto-generate label & datetime ────────────────────────────
                now = datetime.datetime.now()
                label = f"Crack-{now.strftime('%Y-%m-%d-%H%M%S')}"
                datetime_str = now.strftime("%Y-%m-%dT%H:%M")
                location = self._fb_location_var.get().strip() or "Centro"

                # ── Save annotated frame to a temp file for upload ───────────────
                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                tmp.close()
                cv2.imwrite(tmp.name, annotated)
                image_name = f"{label}.jpg"

                # ── Upload image to Firebase Storage ─────────────────────────
                self.root.after(
                    0, lambda: self.fb_status_var.set("⏳ Uploading image…")
                )
                image_url, storage_path = _upload_image_to_storage(
                    tmp.name, image_name
                )
                os.unlink(tmp.name)

                # ── POST document to Firestore ───────────────────────────────
                self.root.after(
                    0, lambda: self.fb_status_var.set("⏳ Saving record…")
                )
                doc_id = _post_crack_record(
                    label=label,
                    classification=classification,
                    location=location,
                    datetime_str=datetime_str,
                    length_cm=f"{avg_length:.2f}",
                    width_cm=f"{avg_width:.2f}",
                    depth_cm=depth_str,
                    image_url=image_url,
                    image_path=storage_path,
                    image_name=image_name,
                    description=(
                        f"{len(result.boxes)} crack(s) detected. "
                        f"Avg confidence: {avg_conf:.1f}%."
                    ),
                )
                _msg = f"✓ Saved! [{classification}] ID: {doc_id[:8]}…"
                def _on_success(msg=_msg):
                    self.fb_status_var.set(msg)
                    self.fb_status_lbl.config(fg=SUCCESS)
                    self.save_fb_btn.config(state="normal")
                self.root.after(0, _on_success)
            except Exception as e:
                err = str(e)
                def _on_error(msg=err):
                    self.fb_status_var.set(f"✗ Error: {msg}")
                    self.fb_status_lbl.config(fg=DANGER)
                    self.save_fb_btn.config(state="normal")
                self.root.after(0, _on_error)

        threading.Thread(target=_worker, daemon=True).start()
    # ────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    root = tk.Tk()
    app = CrackDetectorApp(root)
    root.mainloop()
