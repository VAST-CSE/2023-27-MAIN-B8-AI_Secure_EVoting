
import cv2
import numpy as np
import threading
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from PIL import Image, ImageTk
from insightface.app import FaceAnalysis

BASE_DIR = Path(__file__).resolve().parent
KNOWN_FACES_DIR = BASE_DIR / "known_faces"
MATCH_THRESHOLD = 0.50
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

BG = "#0B1220"
PANEL = "#151F32"
PANEL_LIGHT = "#1D2A40"
TEXT = "#F1F5F9"
MUTED = "#94A3B8"
GREEN = "#22C55E"
RED = "#F87171"
BLUE = "#38BDF8"
YELLOW = "#FBBF24"


class EVotingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI-Powered Secure E-Voting System")
        self.root.geometry("1200x760")
        self.root.minsize(950, 650)
        self.root.configure(bg=BG)

        self.running = False
        self.camera = None
        self.worker = None
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_status = {
            "name": "Camera stopped",
            "detail": "Start the camera to begin recognition",
            "faces": 0,
            "color": MUTED
        }

        self.known_people = []
        self.model = None

        self.build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close_app)

        self.set_status(
            "Model loading",
            "Loading AI model and registered reference photos...",
            0,
            YELLOW
        )
        self.root.after(100, self.initialize_model)

    def build_ui(self):
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=24, pady=(20, 12))

        tk.Label(
            header,
            text="AI-POWERED SECURE E-VOTING SYSTEM",
            bg=BG,
            fg=TEXT,
            font=("Segoe UI", 21, "bold")
        ).pack(anchor="w")

        tk.Label(
            header,
            text="BIOMETRIC FACE RECOGNITION  •  PHASE 1 PROTOTYPE",
            bg=BG,
            fg=BLUE,
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", pady=(5, 0))

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=24, pady=8)

        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        camera_panel = tk.Frame(body, bg=PANEL)
        camera_panel.grid(
            row=0, column=0, sticky="nsew", padx=(0, 10)
        )

        tk.Label(
            camera_panel,
            text="LIVE CAMERA FEED",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI", 12, "bold")
        ).pack(anchor="w", padx=18, pady=(16, 10))

        self.video_label = tk.Label(
            camera_panel,
            text="Camera is not running",
            bg="#050A12",
            fg=MUTED,
            font=("Segoe UI", 14),
            anchor="center"
        )
        self.video_label.pack(
            fill="both", expand=True, padx=16, pady=(0, 16)
        )

        side = tk.Frame(body, bg=BG)
        side.grid(row=0, column=1, sticky="nsew")
        side.grid_columnconfigure(0, weight=1)

        self.status_card = tk.Frame(side, bg=PANEL)
        self.status_card.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        tk.Label(
            self.status_card,
            text="RECOGNITION STATUS",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", padx=18, pady=(16, 8))

        self.status_badge = tk.Label(
            self.status_card,
            text="INITIALIZING",
            bg=PANEL_LIGHT,
            fg=YELLOW,
            font=("Segoe UI", 11, "bold"),
            padx=12,
            pady=8
        )
        self.status_badge.pack(anchor="w", padx=18)

        self.name_label = tk.Label(
            self.status_card,
            text="Please wait...",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI", 22, "bold"),
            wraplength=310,
            justify="left"
        )
        self.name_label.pack(anchor="w", padx=18, pady=(14, 4))

        self.detail_label = tk.Label(
            self.status_card,
            text="Preparing face recognition",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 10),
            wraplength=310,
            justify="left"
        )
        self.detail_label.pack(
            anchor="w", padx=18, pady=(0, 18)
        )

        info_card = tk.Frame(side, bg=PANEL)
        info_card.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        tk.Label(
            info_card,
            text="SESSION INFORMATION",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", padx=18, pady=(16, 10))

        self.camera_value = self.info_row(
            info_card, "Camera", "Stopped"
        )
        self.people_value = self.info_row(
            info_card, "Registered photos", "0"
        )
        self.faces_value = self.info_row(
            info_card, "Faces detected", "0"
        )

        controls = tk.Frame(side, bg=PANEL)
        controls.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        tk.Label(
            controls,
            text="CAMERA CONTROLS",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", padx=18, pady=(16, 10))

        self.start_button = tk.Button(
            controls,
            text="Start Camera",
            command=self.start_camera,
            bg=GREEN,
            fg="#052E16",
            activebackground="#16A34A",
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            cursor="hand2",
            padx=12,
            pady=10
        )
        self.start_button.pack(
            fill="x", padx=18, pady=(0, 8)
        )

        self.stop_button = tk.Button(
            controls,
            text="Stop Camera",
            command=self.stop_camera,
            bg=PANEL_LIGHT,
            fg=TEXT,
            activebackground="#334155",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            cursor="hand2",
            padx=12,
            pady=9,
            state="disabled"
        )
        self.stop_button.pack(
            fill="x", padx=18, pady=(0, 16)
        )

        tk.Label(
            side,
            text="DEMO ONLY — Face recognition is not proof of voter eligibility.",
            bg=BG,
            fg=YELLOW,
            font=("Segoe UI", 9),
            wraplength=340,
            justify="left"
        ).grid(row=3, column=0, sticky="w", pady=4)

        footer = tk.Label(
            self.root,
            text="Academic prototype  |  Local webcam processing",
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 9)
        )
        footer.pack(fill="x", padx=24, pady=(4, 14))

    def info_row(self, parent, title, value):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", padx=18, pady=5)

        tk.Label(
            row,
            text=title,
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 10)
        ).pack(side="left")

        value_label = tk.Label(
            row,
            text=value,
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI", 10, "bold")
        )
        value_label.pack(side="right")
        return value_label

    def set_status(self, name, detail, faces, color):
        self.latest_status = {
            "name": name,
            "detail": detail,
            "faces": faces,
            "color": color
        }

    def initialize_model(self):
        try:
            self.model = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"]
            )
            self.model.prepare(ctx_id=-1, det_size=(640, 640))

            if not KNOWN_FACES_DIR.is_dir():
                raise FileNotFoundError(
                    f"Folder not found:\n{KNOWN_FACES_DIR}"
                )

            for photo_path in KNOWN_FACES_DIR.iterdir():
                if not photo_path.is_file():
                    continue

                if photo_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                image = cv2.imread(str(photo_path))
                if image is None:
                    print(f"Could not read {photo_path.name}")
                    continue

                faces = self.model.get(image)

                if len(faces) != 1:
                    print(
                        f"Skipped {photo_path.name}: "
                        "expected exactly one face."
                    )
                    continue

                self.known_people.append({
                    "name": photo_path.stem,
                    "embedding": faces[0].normed_embedding.copy()
                })

                print(f"Registered: {photo_path.stem}")

            if not self.known_people:
                raise RuntimeError(
                    "No valid reference faces found.\n"
                    "Add clear photos to the known_faces folder."
                )

            self.people_value.config(
                text=str(len(self.known_people))
            )
            self.start_button.config(state="normal")
            self.set_status(
                "Ready",
                "AI model loaded. Start the camera to identify faces.",
                0,
                GREEN
            )
            self.refresh_ui()

        except Exception as error:
            self.start_button.config(state="disabled")
            self.set_status(
                "Setup error",
                str(error),
                0,
                RED
            )
            self.refresh_ui()
            messagebox.showerror(
                "Initialization Error",
                str(error)
            )

    def start_camera(self):
        if self.running:
            return

        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():
            self.camera.release()
            self.camera = None
            messagebox.showerror(
                "Camera Error",
                "Could not open the webcam. Check camera permissions."
            )
            return

        self.running = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.camera_value.config(text="Running")

        self.set_status(
            "Scanning",
            "Looking for a face...",
            0,
            BLUE
        )

        self.worker = threading.Thread(
            target=self.recognition_loop,
            daemon=True
        )
        self.worker.start()
        self.refresh_ui()

    def recognition_loop(self):
        while self.running:
            success, frame = self.camera.read()

            if not success:
                self.set_status(
                    "Camera error",
                    "Could not read the webcam feed.",
                    0,
                    RED
                )
                break

            faces = self.model.get(frame)

            detected_names = []

            for face in faces:
                embedding = face.normed_embedding
                best_name = "Unknown Person"
                best_score = -1.0

                for person in self.known_people:
                    score = float(
                        np.dot(person["embedding"], embedding)
                    )

                    if score > best_score:
                        best_score = score
                        best_name = person["name"]

                if best_score >= MATCH_THRESHOLD:
                    label = best_name
                    color = (50, 205, 50)
                    detected_names.append(best_name)
                else:
                    label = "Unknown Person"
                    color = (70, 70, 245)
                    detected_names.append("Unknown Person")

                x1, y1, x2, y2 = map(int, face.bbox)

                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), color, 2
                )

                cv2.putText(
                    frame,
                    f"{label} ({best_score:.2f})",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2
                )

            if not faces:
                status = {
                    "name": "No face detected",
                    "detail": "Position your face in front of the camera.",
                    "faces": 0,
                    "color": MUTED
                }
            elif len(detected_names) == 1:
                if detected_names[0] == "Unknown Person":
                    status = {
                        "name": "Unknown Person",
                        "detail": "No registered face matched the threshold.",
                        "faces": 1,
                        "color": RED
                    }
                else:
                    status = {
                        "name": detected_names[0],
                        "detail": "Registered face matched. Eligibility has not been checked.",
                        "faces": 1,
                        "color": GREEN
                    }
            else:
                status = {
                    "name": f"{len(detected_names)} faces detected",
                    "detail": "Multiple faces are visible. Review each label in the camera feed.",
                    "faces": len(detected_names),
                    "color": BLUE
                }

            with self.lock:
                self.latest_frame = frame.copy()
                self.latest_status = status

        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def refresh_ui(self):
        with self.lock:
            frame = (
                self.latest_frame.copy()
                if self.latest_frame is not None
                else None
            )
            status = self.latest_status.copy()

        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            width = max(self.video_label.winfo_width(), 320)
            height = max(self.video_label.winfo_height(), 240)

            image = Image.fromarray(frame)
            image.thumbnail((width, height), Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(image=image)
            self.video_label.config(image=photo, text="")
            self.video_label.image = photo

        self.status_badge.config(
            text=status["name"].upper(),
            fg=status["color"]
        )
        self.name_label.config(text=status["name"])
        self.detail_label.config(text=status["detail"])
        self.faces_value.config(text=str(status["faces"]))

        if self.running:
            self.root.after(30, self.refresh_ui)

    def stop_camera(self):
        self.running = False

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.camera_value.config(text="Stopped")

        with self.lock:
            self.latest_frame = None
            self.latest_status = {
                "name": "Camera stopped",
                "detail": "Start the camera to begin recognition.",
                "faces": 0,
                "color": MUTED
            }

        self.video_label.config(
            image="",
            text="Camera is not running"
        )
        self.video_label.image = None

        self.refresh_ui()

    def close_app(self):
        self.running = False

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = EVotingApp(root)
    root.mainloop()
