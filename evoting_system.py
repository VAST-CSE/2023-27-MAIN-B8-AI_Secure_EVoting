
import os
import cv2
import time
import uuid
import queue
import sqlite3
import threading
import hashlib
import hmac
import tkinter as tk

from pathlib import Path
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
from insightface.app import FaceAnalysis


# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).resolve().parent
KNOWN_FACES_DIR = BASE_DIR / "known_faces"
DATABASE_FILE = BASE_DIR / "evoting.db"

CANDIDATES = [
    "Candidate A",
    "Candidate B",
    "Candidate C",
]

ADMIN_PASSWORD = "Admin@123"

# Experimental starting value only. Calibrate using your own test photos.
MATCH_THRESHOLD = 0.42

CAMERA_INDEX = 0
RECOGNITION_INTERVAL = 6


# ==================== DATABASE ====================

def connect_db():
    connection = sqlite3.connect(DATABASE_FILE, timeout=10)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection

def initialise_database():
    with connect_db() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS voters (
                voter_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                has_voted INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS candidates (
                candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE
            );

            CREATE TABLE IF NOT EXISTS ballots (
                ballot_id TEXT PRIMARY KEY,
                candidate_id INTEGER NOT NULL,
                cast_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (candidate_id)
                    REFERENCES candidates(candidate_id)
            );

            CREATE TABLE IF NOT EXISTS election_settings (
                setting_key TEXT PRIMARY KEY,
                setting_value TEXT NOT NULL
            );
        """)

        for candidate in CANDIDATES:
            db.execute(
                "INSERT OR IGNORE INTO candidates(name) VALUES(?)",
                (candidate,)
            )

        db.execute("""
            INSERT OR IGNORE INTO election_settings
            (setting_key, setting_value)
            VALUES ('ended', '0')
        """)


def sync_registered_voters():
    if not KNOWN_FACES_DIR.exists():
        KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)

    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    with connect_db() as db:
        for image_path in sorted(KNOWN_FACES_DIR.iterdir()):
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in supported_extensions:
                continue

            name = image_path.stem.strip()

            if name:
                db.execute(
                    "INSERT OR IGNORE INTO voters(name) VALUES(?)",
                    (name,)
                )


def check_voters():
    with connect_db() as db:
        rows = db.execute(
            "SELECT voter_id, name, has_voted FROM voters ORDER BY name"
        ).fetchall()

        print("\n--- REGISTERED VOTERS IN DATABASE ---")

        for row in rows:
            print(row)

        print(f"\nTotal database records: {len(rows)}")


def fix_renamed_voter():
    with connect_db() as db:
        old_name = "Vishnu"
        new_name = "TS Vishnu"

        old_voter = db.execute(
            "SELECT has_voted FROM voters WHERE name = ?",
            (old_name,)
        ).fetchone()

        new_voter = db.execute(
            "SELECT has_voted FROM voters WHERE name = ?",
            (new_name,)
        ).fetchone()

        if old_voter and new_voter:
            if old_voter[0] == 0 and new_voter[0] == 0:
                db.execute(
                    "DELETE FROM voters WHERE name = ?",
                    (old_name,)
                )
                print("Removed obsolete voter record: Vishnu")
            else:
                print("No changes made. Check voting status first.")
        else:
            print("Both voter records were not found.")


def get_voter_status(name):
    with connect_db() as db:
        return db.execute(
            "SELECT has_voted FROM voters WHERE name = ?",
            (name,)
        ).fetchone()


def election_has_ended():
    with connect_db() as db:
        row = db.execute(
            "SELECT setting_value FROM election_settings "
            "WHERE setting_key = 'ended'"
        ).fetchone()
        return bool(row and row[0] == "1")


def set_election_ended(ended):
    with connect_db() as db:
        db.execute(
            "UPDATE election_settings SET setting_value = ? "
            "WHERE setting_key = 'ended'",
            ("1" if ended else "0",)
        )


def cast_secret_ballot(voter_name, candidate_name):
    """
    Voter status and ballot are updated in one transaction.
    The ballots table contains no voter name or voter ID.
    """
    with connect_db() as db:
        db.execute("BEGIN IMMEDIATE")

        ended = db.execute(
            "SELECT setting_value FROM election_settings "
            "WHERE setting_key = 'ended'"
        ).fetchone()

        if ended and ended[0] == "1":
            raise ValueError("The election has ended.")

        voter = db.execute(
            "SELECT voter_id, has_voted FROM voters WHERE name = ?",
            (voter_name,)
        ).fetchone()

        if voter is None:
            raise ValueError("This person is not registered.")

        voter_id, has_voted = voter

        if has_voted:
            raise ValueError("This voter has already voted.")

        candidate = db.execute(
            "SELECT candidate_id FROM candidates WHERE name = ?",
            (candidate_name,)
        ).fetchone()

        if candidate is None:
            raise ValueError("Invalid candidate.")

        updated = db.execute(
            "UPDATE voters SET has_voted = 1 "
            "WHERE voter_id = ? AND has_voted = 0",
            (voter_id,)
        )

        if updated.rowcount != 1:
            raise ValueError("This voter has already voted.")

        # Deliberately do not insert voter_id or voter_name here.
        db.execute(
            "INSERT INTO ballots(ballot_id, candidate_id) VALUES(?, ?)",
            (str(uuid.uuid4()), candidate[0])
        )

        db.commit()


def get_statistics():
    with connect_db() as db:
        total_votes = db.execute(
            "SELECT COUNT(*) FROM ballots"
        ).fetchone()[0]

        rows = db.execute("""
            SELECT c.name, COUNT(b.ballot_id) AS vote_count
            FROM candidates c
            LEFT JOIN ballots b ON c.candidate_id = b.candidate_id
            GROUP BY c.candidate_id, c.name
            ORDER BY vote_count DESC, c.name ASC
        """).fetchall()

    return total_votes, rows


def reset_database():
    with connect_db() as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("DELETE FROM ballots")
        db.execute("UPDATE voters SET has_voted = 0")
        db.execute(
            "UPDATE election_settings SET setting_value = '0' "
            "WHERE setting_key = 'ended'"
        )
        db.commit()


# ==================== FACE RECOGNITION ====================

def load_reference_faces():
    print("Loading InsightFace model...")
    face_app = FaceAnalysis(name="buffalo_l")
    face_app.prepare(ctx_id=-1, det_size=(640, 640))

    known_embeddings = {}
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for image_path in sorted(KNOWN_FACES_DIR.iterdir()):
        if image_path.suffix.lower() not in supported_extensions:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not read: {image_path.name}")
            continue

        faces = face_app.get(image)

        if not faces:
            print(f"No face found in: {image_path.name}")
            continue

        largest_face = max(
            faces,
            key=lambda face: (
                (face.bbox[2] - face.bbox[0]) *
                (face.bbox[3] - face.bbox[1])
            )
        )

        embedding = largest_face.normed_embedding
        known_embeddings[image_path.stem] = embedding

        print(f"Registered reference loaded: {image_path.stem}")

    return face_app, known_embeddings


def recognise_face(face, known_embeddings):
    if not known_embeddings:
        return "Unknown Person", 0.0

    embedding = face.normed_embedding
    best_name = "Unknown Person"
    best_score = -1.0

    for name, known_embedding in known_embeddings.items():
        score = float(embedding @ known_embedding)

        if score > best_score:
            best_score = score
            best_name = name

    if best_score < MATCH_THRESHOLD:
        return "Unknown Person", best_score

    return best_name, best_score


# ==================== APPLICATION ====================

class EVotingApp:
    
    def __init__(self, root, face_app, known_embeddings):
        self.root = root
        self.face_app = face_app
        self.known_embeddings = known_embeddings

        self.camera = None
        self.camera_thread = None
        self.camera_running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.recognition_queue = queue.Queue()

        self.current_voter = None
        self.current_score = 0.0
        self.last_recognition_time = 0
        self.frame_count = 0
        self.photo = None

        self.ballot_open = False
        self.recognition_pending = False

        self.root.title("AI-Powered Secure E-Voting System")
        self.root.geometry("1150x760")
        self.root.minsize(950, 650)
        self.root.configure(bg="#101827")
        self.root.protocol("WM_DELETE_WINDOW", self.close_app)

        self.build_interface()
        self.update_summary()
        self.poll_queues()

        if election_has_ended():
            self.status_var.set("ELECTION ENDED")
            self.detail_var.set(
                "An administrator must reset the election to reopen voting."
            )


    def build_interface(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Treeview",
            background="#1e293b",
            foreground="white",
            fieldbackground="#1e293b",
            rowheight=28
        )
        style.configure(
            "Treeview.Heading",
            background="#334155",
            foreground="white"
        )

        header = tk.Frame(self.root, bg="#101827")
        header.pack(fill="x", padx=24, pady=(20, 12))

        tk.Label(
            header,
            text="AI-POWERED SECURE E-VOTING SYSTEM",
            font=("Segoe UI", 21, "bold"),
            fg="white",
            bg="#101827"
        ).pack(anchor="w")

        tk.Label(
            header,
            text="BIOMETRIC FACE RECOGNITION",  # PHASE 1 PROTOTYPE
            font=("Segoe UI", 10),
            fg="#38bdf8",
            bg="#101827"
        ).pack(anchor="w", pady=(5, 0))

        body = tk.Frame(self.root, bg="#101827")
        body.pack(fill="both", expand=True, padx=24, pady=8)

        left = tk.Frame(body, bg="#1e293b")
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))

        right = tk.Frame(body, bg="#1e293b", width=320)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        tk.Label(
            left,
            text="LIVE CAMERA",
            font=("Segoe UI", 12, "bold"),
            fg="white",
            bg="#1e293b"
        ).pack(anchor="w", padx=14, pady=(12, 6))

        self.video_label = tk.Label(
            left,
            text="Camera is stopped",
            bg="#0b1220",
            fg="#94a3b8",
            font=("Segoe UI", 14),
            width=70,
            height=22
        )
        self.video_label.pack(fill="both", expand=True, padx=14, pady=(0, 12))

        controls = tk.Frame(left, bg="#1e293b")
        controls.pack(fill="x", padx=14, pady=(0, 14))

        tk.Button(
            controls, text="Start Camera", command=self.start_camera,
            bg="#0284c7", fg="white", activebackground="#0369a1",
            relief="flat", padx=14, pady=8
        ).pack(side="left", padx=(0, 8))

        tk.Button(
            controls, text="Stop Camera", command=self.stop_camera,
            bg="#475569", fg="white", relief="flat",
            padx=14, pady=8
        ).pack(side="left")

        tk.Label(
            right, text="VOTER VERIFICATION",
            font=("Segoe UI", 12, "bold"),
            fg="white", bg="#1e293b"
        ).pack(anchor="w", padx=16, pady=(16, 10))

        self.status_var = tk.StringVar(value="Waiting for voter")
        tk.Label(
            right, textvariable=self.status_var,
            font=("Segoe UI", 17, "bold"),
            fg="#38bdf8", bg="#1e293b",
            wraplength=280, justify="left"
        ).pack(anchor="w", padx=16, pady=(0, 6))

        self.detail_var = tk.StringVar(
            value="Start the camera and look directly at it."
        )
        tk.Label(
            right, textvariable=self.detail_var,
            font=("Segoe UI", 10),
            fg="#cbd5e1", bg="#1e293b",
            wraplength=280, justify="left"
        ).pack(anchor="w", padx=16, pady=(0, 16))

        self.summary_var = tk.StringVar()
        tk.Label(
            right, textvariable=self.summary_var,
            font=("Segoe UI", 10),
            fg="#e2e8f0", bg="#0f172a",
            justify="left", padx=12, pady=12
        ).pack(fill="x", padx=12, pady=6)

        tk.Button(
            right, text="Cast Vote",
            command=self.open_ballot,
            bg="#16a34a", fg="white",
            activebackground="#15803d",
            relief="flat", font=("Segoe UI", 11, "bold"),
            padx=12, pady=10
        ).pack(fill="x", padx=16, pady=(12, 6))

        tk.Button(
            right, text="View Election Results",
            command=self.show_results,
            bg="#7c3aed", fg="white",
            relief="flat", padx=12, pady=9
        ).pack(fill="x", padx=16, pady=6)

        tk.Button(
            right, text="End Election",
            command=self.end_election,
            bg="#b45309", fg="white",
            relief="flat", padx=12, pady=9
        ).pack(fill="x", padx=16, pady=6)

        tk.Button(
            right, text="Reset Election (Admin)",
            command=self.admin_reset,
            bg="#b91c1c", fg="white",
            relief="flat", padx=12, pady=9
        ).pack(fill="x", padx=16, pady=6)

        tk.Label(
            self.root,
            text=(
                "BIOMETRIC VERIFICATION • Face recognition may produce "
                "incorrect matches. Please ensure voter eligibility."
            ),
            fg="#94a3b8", bg="#101827",
            font=("Segoe UI", 9)
        ).pack(pady=(4, 14))

    # ---------------- Camera ----------------

    def start_camera(self):
        if self.camera_running:
            return

        if election_has_ended():
            messagebox.showwarning(
                "Election ended",
                "Voting is closed. Contact the election administrator to reopen voting."
            )
            return

        self.camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

        if not self.camera.isOpened():
            self.camera.release()
            self.camera = cv2.VideoCapture(CAMERA_INDEX)

        if not self.camera.isOpened():
            self.camera = None
            messagebox.showerror(
                "Camera error",
                "Could not open the camera. Check permissions and CAMERA_INDEX."
            )
            return

        self.camera_running = True
        self.camera_thread = threading.Thread(
            target=self.camera_loop, daemon=True
        )
        self.camera_thread.start()

        self.status_var.set("Scanning for a registered voter")
        self.detail_var.set("Keep your face visible and look toward the camera.")

    
    
    def stop_camera(self):
        self.camera_running = False

        camera = self.camera
        self.camera = None

        if camera is not None:
            camera.release()

        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        self.video_label.configure(
            image="",
            text="Camera is stopped",
            bg="#0b1220",
            fg="#94a3b8"
        )
        self.photo = None


    
    
    def camera_loop(self):
        last_faces = []

        while self.camera_running:
            camera = self.camera

            if camera is None:
                break

            ok, frame = camera.read()

            if not self.camera_running:
                break

            if not ok:
                time.sleep(0.05)
                continue

            self.frame_count += 1

            if self.frame_count % RECOGNITION_INTERVAL == 0:
                try:
                    last_faces = self.face_app.get(frame)

                    if last_faces:
                        largest = max(
                            last_faces,
                            key=lambda f: (
                                (f.bbox[2] - f.bbox[0]) *
                                (f.bbox[3] - f.bbox[1])
                            )
                        )

                        name, score = recognise_face(
                            largest, self.known_embeddings
                        )

                        if self.recognition_queue.empty():
                            self.recognition_queue.put((name, score))
                    else:
                        last_faces = []

                except Exception as exc:
                    print("Recognition error:", exc)
                    last_faces = []

            for face in last_faces:
                x1, y1, x2, y2 = map(int, face.bbox)

                name, score = recognise_face(
                    face, self.known_embeddings
                )

                voter = (
                    get_voter_status(name)
                    if name != "Unknown Person"
                    else None
                )

                if voter is not None:
                    colour = (0, 220, 80)
                    label = f"{name} ({score:.2f})"
                else:
                    colour = (0, 0, 255)
                    label = "PERSON NOT REGISTERED"

                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), colour, 3
                )

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2

                (text_width, text_height), baseline = (
                    cv2.getTextSize(
                        label, font, font_scale, thickness
                    )
                )

                label_y = max(y1 - 10, text_height + 10)

                cv2.rectangle(
                    frame,
                    (x1, label_y - text_height - 8),
                    (x1 + text_width + 8, label_y + baseline),
                    colour,
                    -1
                )

                cv2.putText(
                    frame,
                    label,
                    (x1 + 4, label_y - 3),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA
                )

            if not self.camera_running:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image.thumbnail((760, 520))

            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            if self.camera_running:
                self.frame_queue.put(image)


    
    def poll_queues(self):
        if not self.root.winfo_exists():
            return

        try:
            latest_image = None

            while True:
                latest_image = self.frame_queue.get_nowait()

        except queue.Empty:
            pass

        if latest_image is not None and self.camera_running:
            self.photo = ImageTk.PhotoImage(image=latest_image)
            self.video_label.configure(image=self.photo, text="")

        try:
            name, score = self.recognition_queue.get_nowait()
            self.handle_recognition(name, score)

        except queue.Empty:
            pass

        self.root.after(40, self.poll_queues)


    # ---------------- Voter workflow ----------------

    
    
    
    def handle_recognition(self, name, score):
        if not self.camera_running:
            return

        if self.ballot_open or self.recognition_pending:
            return

        if election_has_ended():
            self.stop_camera()
            self.current_voter = None
            self.status_var.set("ELECTION ENDED")
            self.detail_var.set("Voting is closed.")
            self.update_summary()
            return

        if name == "Unknown Person":
            self.status_var.set("Unknown Person")
            self.detail_var.set(
                f"Person not registered or face not matched. "
                f"Similarity: {score:.2f}"
            )
            return

        voter = get_voter_status(name)

        if voter is None:
            self.recognition_pending = True
            self.current_voter = None
            self.status_var.set("Not Registered")
            self.detail_var.set("Person is not registered.")

            self.root.after(1500, self.stop_unregistered_camera)
            return

        self.current_voter = name
        self.current_score = score
        self.recognition_pending = True

        self.status_var.set(f"Recognised: {name}")

        if voter[0] == 1:
            self.detail_var.set(
                "Already voted. Camera stopping..."
            )
            self.root.after(1500, self.finish_repeat_voter)
            return

        self.detail_var.set(
            "Face verified. Camera stopping..."
        )
        self.root.after(1500, self.finish_new_voter)

    def stop_unregistered_camera(self):
        self.stop_camera()
        self.recognition_pending = False
        self.status_var.set("Not Registered")
        self.detail_var.set(
            "Person is not registered. Camera is stopped."
        )

    def finish_new_voter(self):
        if self.current_voter is None or self.ballot_open:
            self.recognition_pending = False
            return

        if not self.camera_running:
            self.recognition_pending = False
            return

        self.stop_camera()
        self.open_ballot()

    
    def finish_repeat_voter(self):
        name = self.current_voter

        self.stop_camera()
        self.current_voter = None
        self.current_score = 0.0
        self.recognition_pending = False

        self.status_var.set("Waiting for voter")
        self.detail_var.set(
            "Click Start Camera when the next voter is ready."
        )

        self.update_summary()

        messagebox.showwarning(
            "Already Voted",
            f"{name} has already cast a vote.\n\n"
            "A second vote is not permitted.",
            parent=self.root
        )


    
    def open_ballot(self):
        if self.ballot_open:
            return

        if election_has_ended():
            self.current_voter = None
            self.recognition_pending = False
            messagebox.showwarning(
                "Election ended",
                "Voting is closed."
            )
            return

        if self.current_voter is None:
            self.recognition_pending = False
            return

        name = self.current_voter
        status = get_voter_status(name)

        if status is None:
            self.current_voter = None
            self.recognition_pending = False
            self.status_var.set("Not Registered")
            self.detail_var.set("This person is not registered.")
            return

        if status[0] == 1:
            self.current_voter = None
            self.recognition_pending = False
            self.status_var.set("Already Voted")
            self.detail_var.set(
                f"{name} has already voted. No second ballot is allowed."
            )
            return

        self.ballot_open = True

        ballot = tk.Toplevel(self.root)
        ballot.title("Secret Ballot")
        ballot.geometry("430x370")
        ballot.configure(bg="#101827")
        ballot.transient(self.root)
        ballot.grab_set()

        def close_ballot():
            self.ballot_open = False
            self.recognition_pending = False
            self.current_voter = None
            self.current_score = 0.0

            try:
                ballot.grab_release()
            except tk.TclError:
                pass

            if ballot.winfo_exists():
                ballot.destroy()

            self.status_var.set("Voting cancelled")
            self.detail_var.set(
                "Click Start Camera when the next voter is ready."
            )

        ballot.protocol("WM_DELETE_WINDOW", close_ballot)

        tk.Label(
            ballot,
            text="BALLOT",
            font=("Segoe UI", 20, "bold"),
            fg="white",
            bg="#101827"
        ).pack(pady=(20, 6))

        tk.Label(
            ballot,
            text=f"Verified voter: {name}",
            font=("Segoe UI", 11),
            fg="#38bdf8",
            bg="#101827"
        ).pack(pady=(0, 8))

        tk.Label(
            ballot,
            text="Choose one candidate. Your choice is not stored with your name.",
            font=("Segoe UI", 9),
            fg="#cbd5e1",
            bg="#101827",
            wraplength=370
        ).pack(pady=(0, 14))

        choice = tk.StringVar(value="")

        for candidate in CANDIDATES:
            tk.Radiobutton(
                ballot,
                text=candidate,
                variable=choice,
                value=candidate,
                font=("Segoe UI", 12),
                fg="white",
                bg="#101827",
                selectcolor="#334155",
                activebackground="#101827",
                activeforeground="white"
            ).pack(anchor="w", padx=55, pady=5)

        def submit_vote():
            selected = choice.get()

            if not selected:
                messagebox.showwarning(
                    "Select candidate",
                    "Choose a candidate first.",
                    parent=ballot
                )
                return

            if not messagebox.askyesno(
                "Confirm ballot",
                f"Submit your vote for {selected}?\n"
                "You cannot change it after submission.",
                parent=ballot
            ):
                return

            try:
                cast_secret_ballot(name, selected)

            except (ValueError, sqlite3.Error) as exc:
                messagebox.showerror(
                    "Vote not submitted",
                    str(exc),
                    parent=ballot
                )
                return

            self.ballot_open = False
            self.recognition_pending = False

            try:
                ballot.grab_release()
            except tk.TclError:
                pass

            ballot.destroy()

            self.stop_camera()
            self.current_voter = None
            self.current_score = 0.0

            self.status_var.set("Vote Recorded")
            self.detail_var.set(
                "Vote saved. Click Start Camera for the next voter."
            )

            self.update_summary()

            messagebox.showinfo(
                "Vote Recorded",
                "Your vote was recorded successfully.\n\n"
                "The camera is off. Start it for the next voter."
            )

        tk.Button(
            ballot,
            text="Confirm & Cast Vote",
            command=submit_vote,
            bg="#16a34a",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            padx=14,
            pady=9
        ).pack(pady=18)

    # ---------------- Results and controls ----------------

    
    def update_summary(self):
        with connect_db() as db:
            registered = db.execute(
                "SELECT COUNT(*) FROM voters"
            ).fetchone()[0]

            voted = db.execute(
                "SELECT COUNT(*) FROM voters WHERE has_voted = 1"
            ).fetchone()[0]

        state = "ENDED" if election_has_ended() else "OPEN"
        turnout = (voted / registered * 100) if registered else 0

        self.summary_var.set(
            f"ELECTION: {state}\n\n"
            f"Registered voters: {registered}\n"
            f"People voted: {voted}\n"
            f"Voter turnout: {turnout:.1f}%"
        )



    
    def show_results(self):
        entered = simpledialog.askstring(
            "Admin Authentication",
            "Enter admin password to view election results:",
            show="*",
            parent=self.root
        )

        if entered is None:
            return

        if not hmac.compare_digest(entered, ADMIN_PASSWORD):
            messagebox.showerror(
                "Access Denied",
                "Incorrect admin password.",
                parent=self.root
            )
            return

        total, rows = get_statistics()

        window = tk.Toplevel(self.root)
        window.title("Election Results - Admin")
        window.geometry("720x540")
        window.configure(bg="#101827")
        window.transient(self.root)
        window.grab_set()

        tk.Label(
            window, text="ELECTION RESULTS - ADMIN ONLY",
            font=("Segoe UI", 20, "bold"),
            fg="white", bg="#101827"
        ).pack(pady=(18, 6))

        tk.Label(
            window, text=f"Total valid ballots: {total}",
            font=("Segoe UI", 11),
            fg="#38bdf8", bg="#101827"
        ).pack(pady=(0, 12))

        columns = ("rank", "candidate", "votes", "percentage", "difference")
        table = ttk.Treeview(
            window, columns=columns, show="headings", height=8
        )

        headings = {
            "rank": "Rank",
            "candidate": "Candidate",
            "votes": "Votes",
            "percentage": "Percentage",
            "difference": "Votes behind leader"
        }

        widths = {
            "rank": 65,
            "candidate": 190,
            "votes": 90,
            "percentage": 110,
            "difference": 150
        }

        for column in columns:
            table.heading(column, text=headings[column])
            table.column(column, width=widths[column], anchor="center")

        table.pack(fill="x", padx=18, pady=8)

        max_votes = rows[0][1] if rows else 0
        second_votes = rows[1][1] if len(rows) > 1 else 0

        tied_winners = [
            name for name, votes in rows if votes == max_votes
        ] if max_votes > 0 else []

        for rank, (name, votes) in enumerate(rows, start=1):
            percentage = votes / total * 100 if total else 0
            difference = max_votes - votes

            table.insert(
                "", "end",
                values=(
                    rank, name, votes,
                    f"{percentage:.2f}%", difference
                )
            )

        if total == 0:
            outcome = "No votes have been cast yet."
        elif len(tied_winners) > 1:
            outcome = "TIE: " + ", ".join(tied_winners)
        else:
            outcome = f"WINNER: {tied_winners[0]}"

        margin = max_votes - second_votes if total else 0
        margin_text = (
            f"Winning margin over second place: {margin} vote(s)."
            if total else "Winning margin: not available yet."
        )

        tk.Label(
            window, text=outcome,
            font=("Segoe UI", 15, "bold"),
            fg="#4ade80" if total and len(tied_winners) == 1 else "#fbbf24",
            bg="#101827", wraplength=650
        ).pack(pady=(14, 4))

        tk.Label(
            window, text=margin_text,
            font=("Segoe UI", 11),
            fg="#e2e8f0", bg="#101827"
        ).pack(pady=4)

        tk.Label(
            window,
            text="Votes behind leader = leading vote count minus candidate vote count.",
            font=("Segoe UI", 9),
            fg="#94a3b8", bg="#101827"
        ).pack(pady=(6, 12))


    
    def end_election(self):
        if election_has_ended():
            messagebox.showinfo("Election", "The election has already ended.")
            self.update_summary()
            return

        entered = simpledialog.askstring(
            "Admin Authentication",
            "Enter admin password to end the election:",
            show="*",
            parent=self.root
        )

        if entered is None:
            return

        if not hmac.compare_digest(entered, ADMIN_PASSWORD):
            messagebox.showerror("Access Denied", "Incorrect admin password.")
            return

        if not messagebox.askyesno(
            "Confirm End Election",
            "Are you sure you want to end the election?"
        ):
            return

        set_election_ended(True)
        self.current_voter = None
        self.stop_camera()
        self.status_var.set("ELECTION ENDED")
        self.detail_var.set("Voting is closed. Only an administrator can reopen the election.")
        self.update_summary()

        messagebox.showinfo(
            "Election Ended",
            "Election ended successfully. Existing votes are preserved."
        )


    def admin_reset(self):
        entered = simpledialog.askstring(
            "Admin authentication",
            "Enter the admin password:",
            show="*",
            parent=self.root
        )

        if entered is None:
            return

        if not hmac.compare_digest(entered, ADMIN_PASSWORD):
            messagebox.showerror("Access denied", "Incorrect admin password.")
            return

        if not messagebox.askyesno(
            "Confirm Election Reset",
            "This will permanently delete all election ballots and reset "
            "every voter's voting status. This action cannot be undone.\n\n"
            "Are you sure you want to continue?"
        ):
            return

        reset_database()
        self.current_voter = None
        self.status_var.set("Election reset")
        self.detail_var.set("All votes have been cleared. The election is now open.")
        self.update_summary()

        messagebox.showinfo(
            "Reset Complete",
            "Election reset successfully. Voter registration and reference "
            "photos have been preserved."
        )

    def close_app(self):
        self.stop_camera()
        self.root.destroy()


# ==================== START PROGRAM ====================

def main():
    if not KNOWN_FACES_DIR.exists():
        KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)

    initialise_database()
    sync_registered_voters()
    
    # fix_renamed_voter()  # One-time cleanup for the old Vishnu record
    # check_voters()       # Enable when debugging voter records


    print("Registered voter names are taken from image filenames.")
    print("Loading face-recognition model; first launch may take a while.")

    face_app, known_embeddings = load_reference_faces()

    if not known_embeddings:
        print("WARNING: No usable reference faces were loaded.")
        print("Add clear face photos to the known_faces folder and restart.")

    root = tk.Tk()
    EVotingApp(root, face_app, known_embeddings)
    root.mainloop()


if __name__ == "__main__":
    main()
