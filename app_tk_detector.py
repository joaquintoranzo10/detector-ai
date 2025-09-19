import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import threading
import time
from ultralytics import YOLO

# ---------- Config ----------
DEFAULT_MODEL = "yolov8n.pt"   # liviano (recomendado CPU)
AVAILABLE_MODELS = ["yolov8n.pt", "yolov8s.pt"]  # podés agregar más si querés
DEFAULT_CAMERA = 0             # índice de cámara
DEFAULT_CONF = 0.5
DEFAULT_IMGSZ = 640            # bajá a 416/384 si va lento

# ---------- App ----------
class DetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector local (YOLOv8 + Tkinter)")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Estado
        self.running = False
        self.cap = None
        self.model = None
        self.frame = None
        self.video_thread = None
        self.last_infer_time = 0.0
        self.video_writer = None
        self.save_video = tk.BooleanVar(value=False)

        # ----- UI -----
        container = ttk.Frame(root, padding=8)
        container.grid(row=0, column=0, sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # Canvas para video
        self.video_label = ttk.Label(container)
        self.video_label.grid(row=0, column=0, columnspan=4, sticky="nsew", pady=(0,8))
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        # Controles
        ttk.Label(container, text="Cámara:").grid(row=1, column=0, sticky="w")
        self.cam_idx = tk.IntVar(value=DEFAULT_CAMERA)
        self.cam_spin = ttk.Spinbox(container, from_=0, to=5, textvariable=self.cam_idx, width=5)
        self.cam_spin.grid(row=1, column=1, sticky="w", padx=(4,12))

        ttk.Label(container, text="Modelo:").grid(row=1, column=2, sticky="e")
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.model_cb = ttk.Combobox(container, textvariable=self.model_var, values=AVAILABLE_MODELS, width=14, state="readonly")
        self.model_cb.grid(row=1, column=3, sticky="w")

        ttk.Label(container, text="Confianza:").grid(row=2, column=0, sticky="w", pady=(8,0))
        self.conf_var = tk.DoubleVar(value=DEFAULT_CONF)
        self.conf_scale = ttk.Scale(container, from_=0.1, to=0.9, orient="horizontal", variable=self.conf_var)
        self.conf_scale.grid(row=2, column=1, sticky="we", pady=(8,0))
        self.conf_label = ttk.Label(container, text=f"{DEFAULT_CONF:.2f}")
        self.conf_label.grid(row=2, column=2, sticky="w", pady=(8,0))
        self.conf_var.trace_add("write", lambda *_: self.conf_label.config(text=f"{self.conf_var.get():.2f}"))

        ttk.Label(container, text="ImgSz:").grid(row=2, column=3, sticky="e", pady=(8,0))
        self.imgsz_var = tk.IntVar(value=DEFAULT_IMGSZ)
        self.imgsz_spin = ttk.Spinbox(container, from_=320, to=960, increment=32, textvariable=self.imgsz_var, width=6)
        self.imgsz_spin.grid(row=2, column=3, sticky="w", padx=(48,0), pady=(8,0))

        self.save_chk = ttk.Checkbutton(container, text="Guardar video anotado (runs/detect/output.mp4)", variable=self.save_video)
        self.save_chk.grid(row=3, column=0, columnspan=3, sticky="w", pady=(8,0))

        self.start_btn = ttk.Button(container, text="Start", command=self.start)
        self.start_btn.grid(row=4, column=0, sticky="we", pady=12)
        self.stop_btn = ttk.Button(container, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.grid(row=4, column=1, sticky="we", pady=12)
        self.quit_btn = ttk.Button(container, text="Salir", command=self.on_close)
        self.quit_btn.grid(row=4, column=3, sticky="e", pady=12)

        # Estirar columnas para que no se rompa el layout
        for c in range(4):
            container.columnconfigure(c, weight=1)

    # --------- Lógica ----------
    def load_model(self, path):
        try:
            self.model = YOLO(path)
        except Exception as e:
            messagebox.showerror("Error cargando modelo", str(e))
            self.model = None

    def start(self):
        if self.running:
            return
        # Cargar modelo si no está o cambió
        if self.model is None or self.model_cksum() != self.model_var.get():
            self.load_model(self.model_var.get())
            if self.model is None:
                return

        # Abrir cámara
        idx = int(self.cam_idx.get())
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # CAP_DSHOW ayuda en Windows
        if not self.cap.isOpened():
            messagebox.showerror("Cámara", f"No se pudo abrir la cámara {idx}")
            self.cap.release()
            self.cap = None
            return

        # VideoWriter si corresponde
        if self.save_video.get():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            import os
            os.makedirs("runs/detect", exist_ok=True)
            out_path = "runs/detect/output.mp4"
            self.video_writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        else:
            self.video_writer = None

        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        # Procesamiento en hilo para no congelar la UI
        self.video_thread = threading.Thread(target=self.loop, daemon=True)
        self.video_thread.start()

    def stop(self):
        self.running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def loop(self):
        # Para aliviar CPU, podés “saltar” frames: p.ej., inferir cada 2
        infer_every_n = 1
        frame_count = 0
        imgsz = int(self.imgsz_var.get())

        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Inferencia (cada N frames)
            do_infer = (frame_count % infer_every_n == 0)
            annotated = frame
            if do_infer and self.model is not None:
                try:
                    results = self.model.predict(source=frame, conf=float(self.conf_var.get()),
                                                 imgsz=imgsz, verbose=False)
                    annotated = results[0].plot()  # cajas y labels pintadas
                except Exception as e:
                    # Si falla una inferencia, mostramos el frame crudo y seguimos
                    annotated = frame

            # Guardar si corresponde
            if self.video_writer is not None:
                self.video_writer.write(annotated)

            # Mostrar en Tk
            self.show_on_label(annotated)

            frame_count += 1
            # Pequeño sleep para no saturar CPU (ajustá si querés más FPS)
            time.sleep(0.001)

        # Cleanup
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def show_on_label(self, bgr_frame):
        # Convertir BGR->RGB y a ImageTk
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        # redimensionar suave al ancho del label (opcional)
        # img = img.resize((800, int(800 * img.height / img.width)))
        imgtk = ImageTk.PhotoImage(image=img)

        # Evitar que el GC lo borre
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def model_cksum(self):
        # “checksum” simple: la ruta del modelo actual
        return self.model_var.get()

    def on_close(self):
        self.stop()
        self.root.after(100, self.root.destroy)


def main():
    root = tk.Tk()
    # Tema por defecto decente
    try:
        from tkinter import ttk
        root.call("tk", "scaling", 1.25)
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass

    app = DetectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
