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


def resource_path(relative):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative)


MODEL_PATH = resource_path("best.pt")

# ── Design Tokens: Blue & White SaaS ──────────────────────────────────────────
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
        self.cap = None
        self.mode = "image"
        self.confidence = tk.DoubleVar(value=0.5)

        self._setup_styles()
        self._build_ui()
        self._load_model()

    # ── Styles ─────────────────────────────────────────────────────────────────
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

    # ── UI ─────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_main()

    # ── Sidebar ────────────────────────────────────────────────────────────────
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

        # Brand header — deep blue bar
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

        # Mode toggle
        self._sidebar_label("MODE")
        toggle_wrap = tk.Frame(
            self.sidebar, bg="#F3F4F6", highlightbackground=BORDER, highlightthickness=1
        )
        toggle_wrap.pack(fill="x", padx=16, pady=(6, 0))
        inner = tk.Frame(toggle_wrap, bg="#F3F4F6", padx=3, pady=3)
        inner.pack(fill="x")

        self.img_tab = tk.Button(
            inner,
            text="Image",
            font=("Helvetica", 12, "bold"),
            fg=PRIMARY,
            bg=CARD,
            relief="flat",
            cursor="hand2",
            pady=8,
            bd=0,
            highlightthickness=0,
            activeforeground=PRIMARY,
            activebackground=PRIMARY_TINT,
            command=lambda: self._switch_mode("image"),
        )
        self.img_tab.pack(side="left", fill="x", expand=True, padx=(0, 2))

        self.cam_tab = tk.Button(
            inner,
            text="Live Cam",
            font=("Helvetica", 12),
            fg=TEXT_MUTED,
            bg="#F3F4F6",
            relief="flat",
            cursor="hand2",
            pady=8,
            bd=0,
            highlightthickness=0,
            activeforeground=PRIMARY,
            activebackground=PRIMARY_TINT,
            command=lambda: self._switch_mode("camera"),
        )
        self.cam_tab.pack(side="left", fill="x", expand=True, padx=(2, 0))

        # Image controls
        self.image_controls = tk.Frame(self.sidebar, bg=CARD)
        self.image_controls.pack(fill="x")
        self._sidebar_label("CAPTURE", parent=self.image_controls)
        img_card = self._card(self.image_controls)
        self.upload_btn = self._list_btn(
            img_card, "📸  Take Photo", PRIMARY, self._upload_image, divider=True
        )
        self.analyze_btn = self._list_btn(
            img_card,
            "⬡  Analyze Image",
            TEXT_MUTED,
            self._run_inference,
            divider=False,
            state="disabled",
        )

        # Camera controls
        self.camera_controls = tk.Frame(self.sidebar, bg=CARD)
        self._sidebar_label("CAMERA", parent=self.camera_controls)
        cam_card = self._card(self.camera_controls)
        self.cam_btn = self._list_btn(
            cam_card, "▶  Start Camera", SUCCESS, self._toggle_camera, divider=False
        )

        # Confidence slider
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

        # Instances list
        tk.Frame(self.sidebar, bg=BORDER, height=1).pack(
            fill="x", padx=16, pady=(16, 0)
        )
        self._sidebar_label("INSTANCES")
        self.detection_frame = tk.Frame(self.sidebar, bg=BG)
        self.detection_frame.pack(fill="both", expand=True, padx=16, pady=(6, 0))

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

    # ── Main panel ─────────────────────────────────────────────────────────────
    def _build_main(self):
        self.main = tk.Frame(self.root, bg=BG)
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.grid_rowconfigure(1, weight=1)
        self.main.grid_columnconfigure(0, weight=1)

        # Top bar
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

        # Content area
        content = tk.Frame(self.main, bg=BG)
        content.grid(row=1, column=0, sticky="nsew", padx=16, pady=16)
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=1)

        # Empty state card
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
        tk.Label(
            icon_wrap, text="📷", font=("Helvetica", 28), bg=PRIMARY_TINT, fg=PRIMARY
        ).place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(
            empty_inner,
            text="No image loaded",
            font=("Helvetica", 16, "bold"),
            fg=TEXT_HEAD,
            bg=CARD,
        ).pack()
        tk.Label(
            empty_inner,
            text="Take a photo or start live camera\nto begin crack detection",
            font=("Helvetica", 12),
            fg=TEXT_MUTED,
            bg=CARD,
            justify="center",
        ).pack(pady=(6, 0))

        # Image canvas
        self.canvas_wrap = tk.Frame(
            content, bg="#F1F5F9", highlightbackground=BORDER, highlightthickness=1
        )
        self.canvas_wrap.grid(row=0, column=0, sticky="nsew")
        self.canvas_wrap.grid_remove()
        self.canvas_wrap.grid_rowconfigure(0, weight=1)
        self.canvas_wrap.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Label(self.canvas_wrap, bg="#F1F5F9")
        self.canvas.grid(row=0, column=0, sticky="nsew")

    # ── Helpers ────────────────────────────────────────────────────────────────
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

    def _set_status(self, msg, color):
        self.status_var.set(msg)
        self.status_lbl.config(fg=color)
        self.status_dot.config(fg=color)

    # ── Slider ─────────────────────────────────────────────────────────────────
    def _on_slider(self, val):
        self.conf_label.config(text=f"{int(float(val)*100)}%")

    # ── Mode ───────────────────────────────────────────────────────────────────
    def _switch_mode(self, mode):
        if self.camera_running:
            self._stop_camera()
        self.mode = mode
        if mode == "image":
            self.img_tab.config(bg=CARD, fg=PRIMARY, font=("Helvetica", 12, "bold"))
            self.cam_tab.config(bg="#F3F4F6", fg=TEXT_MUTED, font=("Helvetica", 12))
            self.camera_controls.pack_forget()
            self.image_controls.pack(fill="x")
            self.mode_pill.config(text="IMAGE MODE", bg=PRIMARY)
        else:
            self.cam_tab.config(bg=CARD, fg=PRIMARY, font=("Helvetica", 12, "bold"))
            self.img_tab.config(bg="#F3F4F6", fg=TEXT_MUTED, font=("Helvetica", 12))
            self.image_controls.pack_forget()
            self.camera_controls.pack(fill="x")
            self.mode_pill.config(text="LIVE CAM", bg=SUCCESS)
        self.photo = None
        self._show_empty()
        self._clear_stats()

    def _show_empty(self):
        self.canvas_wrap.grid_remove()
        self.empty_state.grid()

    def _show_canvas(self):
        self.empty_state.grid_remove()
        self.canvas_wrap.grid()

    # ── Model ──────────────────────────────────────────────────────────────────
    def _load_model(self):
        self._set_status("Loading model…", TEXT_MUTED)

        def load():
            try:
                self.model = YOLO(MODEL_PATH)
                self.root.after(0, lambda: self._set_status("Model ready", SUCCESS))
            except Exception as e:
                self.root.after(0, lambda: self._set_status(f"Error: {e}", DANGER))

        threading.Thread(target=load, daemon=True).start()

    # ── Capture ────────────────────────────────────────────────────────────────
    def _upload_image(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()
        self._set_status("Taking photo…", PRIMARY)
        self.upload_btn.config(state="disabled", fg=TEXT_MUTED)
        self.root.update()
        result = subprocess.run(["imagesnap", "-w", "1", tmp.name], capture_output=True)
        self.upload_btn.config(state="normal", fg=PRIMARY)
        if result.returncode != 0:
            self._set_status("Run: brew install imagesnap", DANGER)
            return
        self.current_image_path = tmp.name
        self._display_image(tmp.name)
        self.analyze_btn.config(state="normal", fg=PRIMARY)
        self._set_status("Photo ready — click Analyze", TEXT_MUTED)
        self._clear_stats()

    # ── Inference ──────────────────────────────────────────────────────────────
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

    # ── Camera ─────────────────────────────────────────────────────────────────
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
        self._set_status("Live camera running", SUCCESS)
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
            ret, frame = self.cap.read()
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

    # ── Drawing ────────────────────────────────────────────────────────────────
    def _draw_masks(self, img, result):
        overlay = img.copy()
        if result.masks is not None:
            for i, (mask, box) in enumerate(zip(result.masks.data, result.boxes)):
                color = MASK_COLORS[i % len(MASK_COLORS)]
                m = mask.cpu().numpy()
                m = cv2.resize(m, (img.shape[1], img.shape[0])) > 0.5
                overlay[m] = (overlay[m] * 0.35 + np.array(color) * 0.65).astype(
                    np.uint8
                )
                # Crisp contour edge
                mask_uint = (m * 255).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask_uint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(overlay, contours, -1, color, 2)
                # Confidence badge
                x1, y1 = int(box.xyxy[0][0]), int(box.xyxy[0][1])
                conf = float(box.conf)
                label = f"#{i+1}  {conf*100:.0f}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                px, py = 8, 5
                bx1, by1 = x1, max(y1 - th - py * 2 - 2, 2)
                bx2, by2 = x1 + tw + px * 2, max(y1 - 2, th + py * 2 + 2)
                cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, -1)
                cv2.putText(
                    overlay,
                    label,
                    (bx1 + px, by2 - py),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        return overlay

    # ── Display ────────────────────────────────────────────────────────────────
    def _show_result(self, annotated, result):
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        self._display_image_array(rgb)
        self._update_stats(result)
        if self.mode == "image":
            self._update_detections(result)
            n = len(result.boxes)
            self._set_status(
                f"{n} crack{'s' if n != 1 else ''} detected",
                DANGER if n > 0 else SUCCESS,
            )

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

    # ── Stats ──────────────────────────────────────────────────────────────────
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

    def _update_detections(self, result):
        for w in self.detection_frame.winfo_children():
            w.destroy()
        for i, box in enumerate(result.boxes):
            rgb = MASK_COLORS[i % len(MASK_COLORS)]
            color_hex = "#%02x%02x%02x" % rgb
            conf = float(box.conf) * 100

            item = tk.Frame(
                self.detection_frame,
                bg=CARD,
                highlightbackground=BORDER,
                highlightthickness=1,
            )
            item.pack(fill="x", pady=(0, 5))

            left = tk.Frame(item, bg=CARD)
            left.pack(side="left", padx=12, pady=10, fill="x", expand=True)

            swatch = tk.Frame(left, bg=color_hex, width=10, height=10)
            swatch.pack(side="left", padx=(0, 8))
            swatch.pack_propagate(False)

            tk.Label(
                left,
                text=f"Crack #{i+1}",
                font=("Helvetica", 12, "bold"),
                fg=TEXT_HEAD,
                bg=CARD,
            ).pack(side="left")

            badge_bg = tk.Frame(
                item,
                bg=PRIMARY_TINT,
                highlightbackground=PRIMARY_MID,
                highlightthickness=1,
            )
            badge_bg.pack(side="right", padx=12, pady=10)
            tk.Label(
                badge_bg,
                text=f"{conf:.0f}%",
                font=("Helvetica", 11, "bold"),
                fg=PRIMARY,
                bg=PRIMARY_TINT,
                padx=8,
                pady=3,
            ).pack()


if __name__ == "__main__":
    root = tk.Tk()
    app = CrackDetectorApp(root)
    root.mainloop()
