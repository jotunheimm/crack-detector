import sys
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import subprocess
import tempfile


def resource_path(relative):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative)


MODEL_PATH = resource_path("best.pt")

COLORS = [
    (255, 80, 80),
    (80, 255, 120),
    (80, 160, 255),
    (255, 220, 40),
    (255, 80, 220),
    (40, 230, 220),
    (255, 160, 40),
    (160, 80, 255),
]


class CrackDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CrackScan")
        self.root.geometry("1100x720")
        self.root.minsize(900, 600)
        self.root.configure(bg="#0d0d0d")

        self.model = None
        self.current_image_path = None
        self.photo = None
        self.camera_running = False
        self.cap = None
        self.mode = "image"

        self._build_ui()
        self._load_model()

    def _build_ui(self):
        self.sidebar = tk.Frame(self.root, bg="#111111", width=260)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        tk.Label(
            self.sidebar,
            text="⬡ CRACKSCAN",
            font=("Courier New", 13, "bold"),
            fg="#ff4444",
            bg="#111111",
            pady=20,
        ).pack(fill="x", padx=20)

        tk.Frame(self.sidebar, bg="#2a2a2a", height=1).pack(fill="x", padx=16)

        tk.Label(
            self.sidebar,
            text="MODE",
            font=("Courier New", 9),
            fg="#555555",
            bg="#111111",
        ).pack(anchor="w", padx=20, pady=(18, 6))

        mode_frame = tk.Frame(self.sidebar, bg="#111111")
        mode_frame.pack(fill="x", padx=16)

        self.img_mode_btn = tk.Button(
            mode_frame,
            text="IMAGE",
            font=("Courier New", 9, "bold"),
            fg="#0d0d0d",
            bg="#ff4444",
            relief="flat",
            cursor="hand2",
            pady=8,
            command=lambda: self._switch_mode("image"),
        )
        self.img_mode_btn.pack(side="left", fill="x", expand=True, padx=(0, 2))

        self.cam_mode_btn = tk.Button(
            mode_frame,
            text="LIVE CAM",
            font=("Courier New", 9, "bold"),
            fg="#ffffff",
            bg="#1a1a1a",
            relief="flat",
            cursor="hand2",
            pady=8,
            command=lambda: self._switch_mode("camera"),
        )
        self.cam_mode_btn.pack(side="left", fill="x", expand=True, padx=(2, 0))

        tk.Frame(self.sidebar, bg="#2a2a2a", height=1).pack(
            fill="x", padx=16, pady=(16, 0)
        )

        self.image_controls = tk.Frame(self.sidebar, bg="#111111")
        self.image_controls.pack(fill="x", padx=16, pady=(12, 0))

        tk.Label(
            self.image_controls,
            text="INPUT",
            font=("Courier New", 9),
            fg="#555555",
            bg="#111111",
        ).pack(anchor="w", pady=(0, 6))

        self.upload_btn = tk.Button(
            self.image_controls,
            text="📸  TAKE PHOTO",
            font=("Courier New", 10, "bold"),
            fg="#ffffff",
            bg="#ff4444",
            activebackground="#cc2222",
            activeforeground="#ffffff",
            relief="flat",
            cursor="hand2",
            pady=10,
            command=self._upload_image,
        )
        self.upload_btn.pack(fill="x", pady=(0, 6))

        self.analyze_btn = tk.Button(
            self.image_controls,
            text="⬡  ANALYZE",
            font=("Courier New", 10, "bold"),
            fg="#0d0d0d",
            bg="#aaaaaa",
            relief="flat",
            cursor="hand2",
            pady=10,
            state="disabled",
            command=self._run_inference,
        )
        self.analyze_btn.pack(fill="x")

        self.camera_controls = tk.Frame(self.sidebar, bg="#111111")

        self.cam_btn = tk.Button(
            self.camera_controls,
            text="▶  START CAMERA",
            font=("Courier New", 10, "bold"),
            fg="#ffffff",
            bg="#ff4444",
            activebackground="#cc2222",
            relief="flat",
            cursor="hand2",
            pady=10,
            command=self._toggle_camera,
        )
        self.cam_btn.pack(fill="x", padx=0, pady=(12, 0))

        tk.Frame(self.sidebar, bg="#2a2a2a", height=1).pack(
            fill="x", padx=16, pady=(16, 0)
        )

        tk.Label(
            self.sidebar,
            text="DETECTION STATS",
            font=("Courier New", 9),
            fg="#555555",
            bg="#111111",
        ).pack(anchor="w", padx=20, pady=(16, 8))

        self.stat_count = self._stat_row("CRACKS FOUND", "—")
        self.stat_conf = self._stat_row("AVG CONFIDENCE", "—")
        self.stat_highest = self._stat_row("HIGHEST CONF", "—")
        self.stat_lowest = self._stat_row("LOWEST CONF", "—")

        tk.Frame(self.sidebar, bg="#2a2a2a", height=1).pack(
            fill="x", padx=16, pady=(16, 0)
        )

        tk.Label(
            self.sidebar,
            text="INSTANCES",
            font=("Courier New", 9),
            fg="#555555",
            bg="#111111",
        ).pack(anchor="w", padx=20, pady=(16, 6))

        self.detection_frame = tk.Frame(self.sidebar, bg="#111111")
        self.detection_frame.pack(fill="both", expand=True, padx=16)

        self.status_var = tk.StringVar(value="Loading model...")
        tk.Label(
            self.sidebar,
            textvariable=self.status_var,
            font=("Courier New", 8),
            fg="#444444",
            bg="#111111",
            wraplength=220,
            justify="left",
        ).pack(anchor="w", padx=20, pady=(8, 16))

        self.canvas_frame = tk.Frame(self.root, bg="#0d0d0d")
        self.canvas_frame.pack(side="right", fill="both", expand=True)

        self.canvas = tk.Label(
            self.canvas_frame,
            bg="#0d0d0d",
            text="Take a photo or start live camera",
            font=("Courier New", 14),
            fg="#2a2a2a",
        )
        self.canvas.pack(fill="both", expand=True, padx=20, pady=20)

    def _stat_row(self, label, value):
        frame = tk.Frame(self.sidebar, bg="#1a1a1a", pady=8)
        frame.pack(fill="x", pady=2)
        tk.Label(
            frame, text=label, font=("Courier New", 7), fg="#555555", bg="#1a1a1a"
        ).pack(anchor="w", padx=10)
        val = tk.Label(
            frame,
            text=value,
            font=("Courier New", 12, "bold"),
            fg="#ff4444",
            bg="#1a1a1a",
        )
        val.pack(anchor="w", padx=10)
        return val

    def _switch_mode(self, mode):
        if self.camera_running:
            self._stop_camera()
        self.mode = mode
        if mode == "image":
            self.img_mode_btn.config(bg="#ff4444", fg="#0d0d0d")
            self.cam_mode_btn.config(bg="#1a1a1a", fg="#ffffff")
            self.camera_controls.pack_forget()
            self.image_controls.pack(fill="x", padx=16, pady=(12, 0))
            self.canvas.config(text="Take a photo to begin", image="")
            self.photo = None
        else:
            self.cam_mode_btn.config(bg="#ff4444", fg="#0d0d0d")
            self.img_mode_btn.config(bg="#1a1a1a", fg="#ffffff")
            self.image_controls.pack_forget()
            self.camera_controls.pack(fill="x", padx=16, pady=(12, 0))
            self.canvas.config(text="Press START CAMERA to begin", image="")
            self.photo = None
        self._clear_stats()

    def _load_model(self):
        self.status_var.set("Loading model...")

        def load():
            try:
                self.model = YOLO(MODEL_PATH)
                self.status_var.set("✓ Model ready")
            except Exception as e:
                self.status_var.set(f"✗ Error: {e}")

        threading.Thread(target=load, daemon=True).start()

    def _upload_image(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()

        self.status_var.set("📸 Taking photo...")
        self.upload_btn.config(state="disabled", bg="#aaaaaa")
        self.root.update()

        result = subprocess.run(["imagesnap", "-w", "1", tmp.name], capture_output=True)

        self.upload_btn.config(state="normal", bg="#ff4444")

        if result.returncode != 0:
            self.status_var.set("✗ Could not capture. Run: brew install imagesnap")
            return

        self.current_image_path = tmp.name
        self._display_image(tmp.name)
        self.analyze_btn.config(state="normal", bg="#ff4444", fg="#ffffff")
        self.status_var.set("📸 Photo taken! Click ANALYZE.")
        self._clear_stats()

    def _run_inference(self):
        if not self.model or not self.current_image_path:
            return
        self.analyze_btn.config(state="disabled", bg="#aaaaaa", fg="#0d0d0d")
        self.status_var.set("Analyzing...")

        def infer():
            try:
                preds = self.model.predict(
                    self.current_image_path, conf=0.5, verbose=False
                )
                result = preds[0]
                frame = cv2.imread(self.current_image_path)
                annotated = self._draw_masks(frame, result)
                self.root.after(0, lambda: self._show_result(annotated, result))
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"✗ Error: {e}"))
            finally:
                self.root.after(
                    0,
                    lambda: self.analyze_btn.config(
                        state="normal", bg="#ff4444", fg="#ffffff"
                    ),
                )

        threading.Thread(target=infer, daemon=True).start()

    def _toggle_camera(self):
        if self.camera_running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_var.set("✗ Camera not found")
            return
        self.camera_running = True
        self.cam_btn.config(text="■  STOP CAMERA", bg="#cc2222")
        self.status_var.set("✓ Live camera running...")
        threading.Thread(target=self._camera_loop, daemon=True).start()

    def _stop_camera(self):
        self.camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.cam_btn.config(text="▶  START CAMERA", bg="#ff4444")
        self.status_var.set("Camera stopped.")
        self.canvas.config(text="Press START CAMERA to begin", image="")
        self.photo = None
        self._clear_stats()

    def _camera_loop(self):
        while self.camera_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            if self.model:
                preds = self.model.predict(frame, conf=0.5, verbose=False)
                result = preds[0]
                annotated = self._draw_masks(frame, result)
                self.root.after(
                    0, lambda r=result, a=annotated: self._show_result(a, r)
                )
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.root.after(0, lambda f=rgb: self._display_image_array(f))

    def _draw_masks(self, img, result):
        overlay = img.copy()
        if result.masks is not None:
            for i, (mask, box) in enumerate(zip(result.masks.data, result.boxes)):
                color = COLORS[i % len(COLORS)]
                conf = float(box.conf)
                m = mask.cpu().numpy()
                m = cv2.resize(m, (img.shape[1], img.shape[0])) > 0.5
                overlay[m] = (overlay[m] * 0.35 + np.array(color) * 0.65).astype(
                    np.uint8
                )
                x1, y1 = int(box.xyxy[0][0]), int(box.xyxy[0][1])
                label = f"#{i+1} {conf*100:.1f}%"
                cv2.putText(
                    overlay,
                    label,
                    (x1 + 4, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA,
                )
        return overlay

    def _show_result(self, annotated, result):
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        self._display_image_array(rgb)
        self._update_stats(result)
        if self.mode == "image":
            self._update_detections(result)
            n = len(result.boxes)
            self.status_var.set(f"✓ Done — {n} crack(s) found")

    def _display_image(self, path):
        img = Image.open(path)
        self._render_pil(img)

    def _display_image_array(self, arr):
        self._render_pil(Image.fromarray(arr))

    def _render_pil(self, img):
        w = self.canvas_frame.winfo_width() or 800
        h = self.canvas_frame.winfo_height() or 680
        img.thumbnail((w - 40, h - 40), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.config(image=self.photo, text="")

    def _update_stats(self, result):
        boxes = result.boxes
        if len(boxes) == 0:
            self.stat_count.config(text="0")
            self.stat_conf.config(text="—")
            self.stat_highest.config(text="—")
            self.stat_lowest.config(text="—")
            return
        confs = [float(b.conf) * 100 for b in boxes]
        self.stat_count.config(text=str(len(boxes)))
        self.stat_conf.config(text=f"{np.mean(confs):.1f}%")
        self.stat_highest.config(text=f"{max(confs):.1f}%")
        self.stat_lowest.config(text=f"{min(confs):.1f}%")

    def _clear_stats(self):
        for w in [self.stat_count, self.stat_conf, self.stat_highest, self.stat_lowest]:
            w.config(text="—")
        for w in self.detection_frame.winfo_children():
            w.destroy()

    def _update_detections(self, result):
        for w in self.detection_frame.winfo_children():
            w.destroy()
        for i, box in enumerate(result.boxes):
            color = "#%02x%02x%02x" % COLORS[i % len(COLORS)]
            conf = float(box.conf) * 100
            row = tk.Frame(self.detection_frame, bg="#1a1a1a", pady=6)
            row.pack(fill="x", pady=2)
            tk.Label(
                row, text="  ●", font=("Courier New", 10), fg=color, bg="#1a1a1a"
            ).pack(side="left")
            tk.Label(
                row,
                text=f"Crack #{i+1}",
                font=("Courier New", 9, "bold"),
                fg="#cccccc",
                bg="#1a1a1a",
            ).pack(side="left", padx=4)
            tk.Label(
                row,
                text=f"{conf:.1f}%",
                font=("Courier New", 9, "bold"),
                fg=color,
                bg="#1a1a1a",
            ).pack(side="right", padx=10)


if __name__ == "__main__":
    root = tk.Tk()
    app = CrackDetectorApp(root)
    root.mainloop()
