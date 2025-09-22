import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import threading
import time
import os
from datetime import datetime
from ultralytics import YOLO

# ---------- Config ----------
DEFAULT_MODEL = "yolov8n.pt"       # liviano (CPU-friendly)
AVAILABLE_MODELS = ["yolov8n.pt", "yolov8s.pt"]
DEFAULT_CAMERA = 0                 # índice de cámara
DEFAULT_CONF = 0.5
DEFAULT_IMGSZ = 640                # bajá a 416/384 si va lento
TARGET_MAX_WIDTH = 960             # ancho máx. de render en label (auto-resize)
SAVE_DIR = "runs/detect"

# ---------- Util ----------
def now_ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# ---------- App ----------
class DetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector local (YOLOv8 + Tkinter)")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Estado
        self.running = False
        self.paused = False
        self.cap = None
        self.model = None
        self.frame = None
        self.video_thread = None
        self.video_writer = None
        self.save_video = tk.BooleanVar(value=False)
        self.flip_h = tk.BooleanVar(value=True)   # espejo por defecto (webcam)
        self.dark_mode = tk.BooleanVar(value=True)

        self._last_infer_t = 0.0
        self._fps_ma = 0.0
        self._last_frame_t = time.time()
        self._display_w = TARGET_MAX_WIDTH
        self._display_h = None
        self._current_size = (0, 0)
        self._lock = threading.Lock()

        # ---------- Estilos ----------
        self._build_style(dark=True)

        # ---------- Layout ----------
        container = ttk.Frame(root, padding=10, style="App.TFrame")
        container.grid(row=0, column=0, sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # Header
        header = ttk.Frame(container, style="App.TFrame")
        header.grid(row=0, column=0, sticky="we")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="YOLOv8 Detector", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Button(header, text="Modo Claro/Oscuro", style="Tool.TButton",
                   command=self.toggle_theme).grid(row=0, column=1, sticky="e", padx=(8,0))

        ttk.Separator(container).grid(row=1, column=0, sticky="we", pady=(8,8))

        # Video area
        video_area = ttk.Frame(container, style="App.TFrame")
        video_area.grid(row=2, column=0, sticky="nsew")
        container.rowconfigure(2, weight=1)
        container.columnconfigure(0, weight=1)

        self.video_label = ttk.Label(video_area, style="Video.TLabel")
        self.video_label.grid(row=0, column=0, sticky="nsew")
        video_area.rowconfigure(0, weight=1)
        video_area.columnconfigure(0, weight=1)

        ttk.Separator(container).grid(row=3, column=0, sticky="we", pady=(8,8))

        # Controls
        controls = ttk.Frame(container, style="App.TFrame")
        controls.grid(row=4, column=0, sticky="we", pady=(0,6))
        for c in range(12):
            controls.columnconfigure(c, weight=1)

        # Línea 1
        ttk.Label(controls, text="Cámara:", style="Label.TLabel").grid(row=0, column=0, sticky="w")
        self.cam_idx = tk.IntVar(value=DEFAULT_CAMERA)
        self.cam_spin = ttk.Spinbox(controls, from_=0, to=8, textvariable=self.cam_idx, width=5)
        self.cam_spin.grid(row=0, column=1, sticky="w", padx=(4,12))

        ttk.Label(controls, text="Modelo:", style="Label.TLabel").grid(row=0, column=2, sticky="e")
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.model_cb = ttk.Combobox(controls, textvariable=self.model_var, values=AVAILABLE_MODELS,
                                     width=16, state="readonly")
        self.model_cb.grid(row=0, column=3, sticky="w", padx=(4,12))

        ttk.Label(controls, text="Confianza:", style="Label.TLabel").grid(row=0, column=4, sticky="w")
        self.conf_var = tk.DoubleVar(value=DEFAULT_CONF)
        self.conf_scale = ttk.Scale(controls, from_=0.1, to=0.9, orient="horizontal", variable=self.conf_var)
        self.conf_scale.grid(row=0, column=5, sticky="we", padx=(4,6))
        self.conf_label = ttk.Label(controls, text=f"{DEFAULT_CONF:.2f}", style="Mono.TLabel")
        self.conf_label.grid(row=0, column=6, sticky="w")
        self.conf_var.trace_add("write", lambda *_: self.conf_label.config(text=f"{self.conf_var.get():.2f}"))

        ttk.Label(controls, text="ImgSz:", style="Label.TLabel").grid(row=0, column=7, sticky="e")
        self.imgsz_var = tk.IntVar(value=DEFAULT_IMGSZ)
        self.imgsz_spin = ttk.Spinbox(controls, from_=320, to=960, increment=32,
                                      textvariable=self.imgsz_var, width=6)
        self.imgsz_spin.grid(row=0, column=8, sticky="w", padx=(4,12))

        self.flip_chk = ttk.Checkbutton(controls, text="Espejo", variable=self.flip_h)
        self.flip_chk.grid(row=0, column=9, sticky="w", padx=(4,12))

        # Línea 2 (toolbar)
        toolbar = ttk.Frame(container, style="App.TFrame")
        toolbar.grid(row=5, column=0, sticky="we", pady=(0,6))
        for c in range(10):
            toolbar.columnconfigure(c, weight=1)

        self.save_chk = ttk.Checkbutton(toolbar, text="Guardar video (MP4)", variable=self.save_video)
        self.save_chk.grid(row=0, column=0, sticky="w", padx=(0,12))

        self.start_btn = ttk.Button(toolbar, text="▶ Start (Barra espaciadora)", style="Primary.TButton",
                                    command=self.start)
        self.start_btn.grid(row=0, column=1, sticky="we")
        self.stop_btn = ttk.Button(toolbar, text="■ Stop", command=self.stop, state="disabled")
        self.stop_btn.grid(row=0, column=2, sticky="we", padx=(8,0))
        self.snap_btn = ttk.Button(toolbar, text="📸 Screenshot (P)", command=self.screenshot, state="disabled")
        self.snap_btn.grid(row=0, column=3, sticky="we", padx=(8,0))
        self.quit_btn = ttk.Button(toolbar, text="Salir", command=self.on_close)
        self.quit_btn.grid(row=0, column=9, sticky="e")

        # Status bar
        status = ttk.Frame(container, style="Status.TFrame", padding=(6,4))
        status.grid(row=6, column=0, sticky="we")
        for c in range(6):
            status.columnconfigure(c, weight=1)
        self.stat_left = ttk.Label(status, text="Listo.", style="Status.TLabel")
        self.stat_left.grid(row=0, column=0, sticky="w")
        self.stat_mid = ttk.Label(status, text="", style="Status.TLabel")
        self.stat_mid.grid(row=0, column=1, sticky="w")
        self.stat_right = ttk.Label(status, text="", style="Status.TLabel")
        self.stat_right.grid(row=0, column=5, sticky="e")

        # Hotkeys
        root.bind("<space>", lambda e: self._toggle_run())
        root.bind("<p>", lambda e: self.screenshot())
        root.bind("<P>", lambda e: self.screenshot())
        root.bind("g", lambda e: self._toggle_save())
        root.bind("+", lambda e: self._bump_conf(0.05))
        root.bind("-", lambda e: self._bump_conf(-0.05))

        # Inicial
        ensure_dir(SAVE_DIR)
        self._update_status()

    # ---------- Estilos / Tema ----------
    def _build_style(self, dark=True):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        if dark:
            bg = "#0B1220"      # fondo
            fg = "#E5E7EB"      # texto
            sub = "#9CA3AF"     # secundario
            acc = "#2563EB"     # primario
            card = "#111827"
            status_bg = "#0F172A"
        else:
            bg = "#F8FAFC"
            fg = "#111827"
            sub = "#334155"
            acc = "#2563EB"
            card = "#FFFFFF"
            status_bg = "#EEF2FF"

        self.root.configure(bg=bg)
        style.configure("App.TFrame", background=bg)
        style.configure("Title.TLabel", background=bg, foreground=fg, font=("Segoe UI", 16, "bold"))
        style.configure("Label.TLabel", background=bg, foreground=sub, font=("Segoe UI", 10))
        style.configure("Mono.TLabel", background=bg, foreground=fg, font=("Consolas", 10))
        style.configure("TLabel", background=bg, foreground=fg)
        style.configure("Video.TLabel", background=card, foreground=fg, anchor="center")
        style.configure("Tool.TButton", padding=6)
        style.configure("Primary.TButton", padding=8, relief="flat")
        style.map("Primary.TButton", background=[("active", acc)], foreground=[("active", "#ffffff")])

        # Widgets base
        style.configure("TButton", background=card, foreground=fg, padding=6)
        style.configure("TCheckbutton", background=bg, foreground=fg)
        style.configure("TScale", background=bg)
        style.configure("TSpinbox", fieldbackground="#1F2937" if dark else "#FFFFFF",
                        foreground=fg, arrowsize=14)
        style.configure("TCombobox", fieldbackground="#1F2937" if dark else "#FFFFFF",
                        foreground=fg)
        style.configure("Status.TFrame", background=status_bg)
        style.configure("Status.TLabel", background=status_bg, foreground=fg, font=("Segoe UI", 9))

    def toggle_theme(self):
        self.dark_mode.set(not self.dark_mode.get())
        self._build_style(dark=self.dark_mode.get())

    # ---------- Lógica ----------
    def load_model(self, path):
        try:
            t0 = time.time()
            self.model = YOLO(path)
            dt = (time.time() - t0) * 1000
            self._set_status(f"Modelo cargado: {os.path.basename(path)} ({dt:.0f} ms)")
        except Exception as e:
            messagebox.showerror("Error cargando modelo", str(e))
            self.model = None

    def start(self):
        if self.running:
            return
        if self.model is None:
            self.load_model(self.model_var.get())
            if self.model is None:
                return

        # Abrir cámara
        idx = int(self.cam_idx.get())
        # CAP_DSHOW en Windows mejora apertura
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not self.cap or not self.cap.isOpened():
            if self.cap:
                self.cap.release()
            self.cap = None
            messagebox.showerror("Cámara", f"No se pudo abrir la cámara {idx}")
            return

        # Ajustes cámara (opcionales)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        try:
            cv2.setNumThreads(0)  # ceder CPU a Tk
        except Exception:
            pass

        # VideoWriter si corresponde
        self._open_writer_if_needed()

        self.running = True
        self.paused = False
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.snap_btn.config(state="normal")

        # Hilo de procesamiento
        self.video_thread = threading.Thread(target=self.loop, daemon=True)
        self.video_thread.start()

    def _open_writer_if_needed(self):
        if self.save_video.get():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ensure_dir(SAVE_DIR)
            out_path = os.path.join(SAVE_DIR, f"output_{now_ts()}.mp4")
            self.video_writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            self._set_status(f"Grabando en {out_path}")
        else:
            self.video_writer = None

    def stop(self):
        self.running = False
        self.paused = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.snap_btn.config(state="disabled")

    def loop(self):
        infer_every_n = 1
        frame_count = 0
        imgsz = int(self.imgsz_var.get())

        # FPS baseline
        self._fps_ma = 0.0
        self._last_frame_t = time.time()

        while self.running and self.cap and self.cap.isOpened():
            if self.paused:
                time.sleep(0.05)
                continue

            ret, frame = self.cap.read()
            if not ret:
                break

            if self.flip_h.get():
                frame = cv2.flip(frame, 1)

            t1 = time.time()
            annotated = frame

            do_infer = (frame_count % infer_every_n == 0)
            if do_infer and self.model is not None:
                try:
                    results = self.model.predict(
                        source=frame,
                        conf=float(self.conf_var.get()),
                        imgsz=imgsz,
                        verbose=False
                    )
                    annotated = results[0].plot()
                except Exception:
                    annotated = frame

            # Guardar si corresponde
            if self.video_writer is not None:
                self.video_writer.write(annotated)

            # Mostrar en Tk (redimensionado suave según ancho disponible)
            self.show_on_label(annotated)

            # Métricas
            t2 = time.time()
            infer_ms = (t2 - t1) * 1000.0
            self._last_infer_t = infer_ms

            dt = t2 - self._last_frame_t
            self._last_frame_t = t2
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            # media móvil simple
            self._fps_ma = 0.9 * self._fps_ma + 0.1 * inst_fps if self._fps_ma > 0 else inst_fps

            # Status
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._update_status_runtime(w, h)

            frame_count += 1
            time.sleep(0.001)  # ceder CPU

        # Cleanup
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        self._set_status("Stopped.")

    def show_on_label(self, bgr_frame):
        # Calcula tamaño target una sola vez o si cambió la ventana
        h, w = bgr_frame.shape[:2]
        if self._display_h is None or (w, h) != self._current_size:
            # Ancho preferido: min(label width, TARGET_MAX_WIDTH)
            try:
                label_w = self.video_label.winfo_width()
                label_w = label_w if label_w > 50 else TARGET_MAX_WIDTH
            except Exception:
                label_w = TARGET_MAX_WIDTH

            target_w = min(label_w, TARGET_MAX_WIDTH)
            scale = target_w / float(w)
            target_h = int(h * scale)
            self._display_w, self._display_h = target_w, target_h
            self._current_size = (w, h)

        # Resize para display (sin tocar archivo original)
        if self._display_w and self._display_h:
            disp = cv2.resize(bgr_frame, (self._display_w, self._display_h), interpolation=cv2.INTER_LINEAR)
        else:
            disp = bgr_frame

        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # Evitar GC
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def screenshot(self):
        if not self.running or self.cap is None:
            self._set_status("No hay video para capturar.")
            return
        # Hacemos una última lectura para capturar contenido actual
        ret, frame = self.cap.read()
        if not ret:
            self._set_status("No se pudo capturar frame.")
            return
        if self.flip_h.get():
            frame = cv2.flip(frame, 1)

        # Inferencia de una pasada para guardar anotado
        try:
            results = self.model.predict(source=frame, conf=float(self.conf_var.get()),
                                         imgsz=int(self.imgsz_var.get()), verbose=False)
            annotated = results[0].plot()
        except Exception:
            annotated = frame

        out_dir = os.path.join(SAVE_DIR, "frames")
        ensure_dir(out_dir)
        path = os.path.join(out_dir, f"frame_{now_ts()}.png")
        cv2.imwrite(path, annotated)
        self._set_status(f"Screenshot guardado: {path}")

    def model_cksum(self):
        # “checksum” simple: string del modelo seleccionado
        return self.model_var.get()

    def on_close(self):
        self.stop()
        self.root.after(150, self.root.destroy)

    # ---------- UI helpers ----------
    def _toggle_run(self):
        if self.running:
            # Pausa / reanuda
            self.paused = not self.paused
            self._set_status("Pausado." if self.paused else "Reanudado.")
        else:
            self.start()

    def _toggle_save(self):
        self.save_video.set(not self.save_video.get())
        if self.running:
            # Reabrir writer en caliente
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self._open_writer_if_needed()

    def _bump_conf(self, delta):
        v = float(self.conf_var.get())
        nv = max(0.1, min(0.9, v + delta))
        self.conf_var.set(round(nv, 2))

    def _set_status(self, text):
        self.stat_left.config(text=text)

    def _update_status(self):
        self.stat_mid.config(
            text=f"Modelo: {os.path.basename(self.model_var.get())}  |  Cam: {self.cam_idx.get()}"
        )
        self.stat_right.config(text="Listo")
        # refresco periódico del bloque estático por si cambian selects
        self.root.after(1000, self._update_status)

    def _update_status_runtime(self, w, h):
        fps = self._fps_ma
        inf = self._last_infer_t
        self.stat_right.config(text=f"{fps:.1f} FPS  ·  {w}x{h}  ·  {inf:.0f} ms/inf")

# ---------- Main ----------
def main():
    root = tk.Tk()
    # Escalado y tema base
    try:
        root.call("tk", "scaling", 1.25)
    except Exception:
        pass

    app = DetectorApp(root)
    root.minsize(900, 600)
    root.mainloop()

if __name__ == "__main__":
    main()
