"""
Real-Time AI Voice Call Detection
==================================
Premium, classy dark UI with refined aesthetics.

Now with REAL VoIP call detection and audio recording!
"""

import customtkinter as ctk
import shutil
import sys
import os
from datetime import datetime
from scipy.io.wavfile import write as write_wav
import numpy as np

# Add parent dir to path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.audio_recorder import AudioRecorder
from modules.voip_monitor import VoIPMonitor

# Notifications
try:
    from plyer import notification
    HAS_NOTIFY = True
except ImportError:
    HAS_NOTIFY = False

def notify(title: str, msg: str):
    if HAS_NOTIFY:
        try:
            notification.notify(title=title, message=msg, app_name="Voice AI", timeout=5)
        except:
            pass


# ============================================
# PREMIUM COLOR PALETTES (DARK & LIGHT)
# ============================================
ctk.set_appearance_mode("dark")

DARK_THEME = {
    # Backgrounds - deep, rich blacks
    "bg": "#08080c",
    "surface": "#0f0f14",
    "elevated": "#16161d",
    "border": "#1f1f2a",
    
    # Text hierarchy
    "text": "#fafafa",
    "text_dim": "#a1a1aa",
    "text_muted": "#52525b",
    
    # Accent - elegant teal
    "accent": "#14b8a6",
    "accent_soft": "#0d9488",
    
    # Status - muted, sophisticated
    "safe": "#34d399",
    "warn": "#fbbf24",
    "alert": "#f87171",
    
    # Button text
    "btn_text": "#000",
}

LIGHT_THEME = {
    # Backgrounds - clean, warm whites
    "bg": "#f8fafc",
    "surface": "#ffffff",
    "elevated": "#f1f5f9",
    "border": "#cbd5e1",
    
    # Text hierarchy - darker for better visibility
    "text": "#0f172a",
    "text_dim": "#1e293b",
    "text_muted": "#334155",
    
    # Accent - elegant teal (slightly deeper for light mode)
    "accent": "#0d9488",
    "accent_soft": "#14b8a6",
    
    # Status - muted, sophisticated
    "safe": "#059669",
    "warn": "#d97706",
    "alert": "#dc2626",
    
    # Button text
    "btn_text": "#fff",
}

# Active color palette (starts with dark)
C = DARK_THEME.copy()


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Voice AI Detection")
        self.geometry("880x740")
        self.minsize(800, 680)
        self.configure(fg_color=C["bg"])
        
        self._running = False
        self._last = None
        self._job = None
        self._is_dark_mode = True  # Track current theme
        
        # Real VoIP monitoring components
        self._voip_monitor: VoIPMonitor | None = None
        self._audio_recorder: AudioRecorder | None = None
        self._call_active = False
        self._frames_processed = 0
        
        # Temp folder for audio clips
        self._temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
        self._setup_temp_folder()
        
        self._build_ui()
        
        # Cleanup on close
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        
    def _on_close(self):
        """Clean up resources before closing."""
        self._stop()
        self._cleanup_temp_folder()
        self.destroy()
    
    def _setup_temp_folder(self):
        """Create temp folder for audio clips."""
        if os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir)
        os.makedirs(self._temp_dir, exist_ok=True)
        
    def _cleanup_temp_folder(self):
        """Remove temp folder and all audio clips."""
        if os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir)
            print(f"[Cleanup] Removed temp folder with {self._frames_processed} audio clips")
        
    def _build_ui(self):
        # Main padding
        pad = ctk.CTkFrame(self, fg_color="transparent")
        pad.pack(fill="both", expand=True, padx=32, pady=28)
        
        # ─────────────────────────────────────
        # HEADER
        # ─────────────────────────────────────
        header = ctk.CTkFrame(pad, fg_color="transparent")
        header.pack(fill="x", pady=(0, 24))
        
        # Title section with centered title and right-aligned button
        title_section = ctk.CTkFrame(header, fg_color="transparent", height=50)
        title_section.pack(fill="x")
        title_section.pack_propagate(False)
        
        # Theme toggle button (right side, using place for absolute positioning)
        self.theme_btn = ctk.CTkButton(
            title_section, text="☀",
            font=ctk.CTkFont(size=18),
            fg_color=C["surface"], hover_color=C["elevated"],
            text_color=C["text_dim"], width=40, height=40,
            corner_radius=20, border_width=1, border_color=C["border"],
            command=self._toggle_theme
        )
        self.theme_btn.place(relx=1.0, rely=0.5, anchor="e")
        
        # Title centered in the section
        self.title_lbl = ctk.CTkLabel(
            title_section, text="Voice Detection",
            font=ctk.CTkFont(family="Segoe UI Light", size=36),
            text_color=C["text"]
        )
        self.title_lbl.place(relx=0.5, rely=0.5, anchor="center")
        
        self.sub_lbl = ctk.CTkLabel(
            header, text="Neural Analysis System",
            font=ctk.CTkFont(size=14, weight="normal"),
            text_color=C["text_muted"]
        )
        self.sub_lbl.pack(pady=(4, 0))
        
        # ─────────────────────────────────────
        # WAVEFORM
        # ─────────────────────────────────────
        self.wave = ctk.CTkCanvas(
            pad, bg=C["bg"], highlightthickness=0, height=36
        )
        self.wave.pack(fill="x", pady=(0, 20))
        self._bars = [6] * 50
        self._wave_active = False
        
        # ─────────────────────────────────────
        # CALL DETECTION BANNER
        # ─────────────────────────────────────
        self.call_frame = ctk.CTkFrame(
            pad, fg_color=C["surface"],
            corner_radius=10, border_width=1, border_color=C["accent"]
        )
        self.call_label = ctk.CTkLabel(
            self.call_frame, 
            text="📞  VoIP Call Detected — Recording Active",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=C["accent"]
        )
        self.call_label.pack(pady=14, padx=24)
        
        # ─────────────────────────────────────
        # MAIN STATUS CARD
        # ─────────────────────────────────────
        self.status_card = ctk.CTkFrame(
            pad, fg_color=C["surface"],
            corner_radius=16, border_width=1, border_color=C["border"]
        )
        self.status_card.pack(fill="x", pady=(0, 20))
        
        inner = ctk.CTkFrame(self.status_card, fg_color="transparent")
        inner.pack(pady=36, padx=48)
        
        self.status_lbl = ctk.CTkLabel(
            inner, text="READY",
            font=ctk.CTkFont(family="Segoe UI", size=44, weight="bold"),
            text_color=C["text_muted"]
        )
        self.status_lbl.pack()
        
        self.frames_lbl = ctk.CTkLabel(
            inner, text="0 frames recorded",
            font=ctk.CTkFont(size=16),
            text_color=C["text_dim"]
        )
        self.frames_lbl.pack(pady=(8, 0))
        
        # Thin divider
        div = ctk.CTkFrame(inner, height=1, fg_color=C["border"], width=200)
        div.pack(pady=20)
        
        self.desc_lbl = ctk.CTkLabel(
            inner, text="Press Start to begin monitoring",
            font=ctk.CTkFont(size=14),
            text_color=C["text_muted"]
        )
        self.desc_lbl.pack()
        
        # ─────────────────────────────────────
        # CONTROLS CARD
        # ─────────────────────────────────────
        self.ctrl_card = ctk.CTkFrame(
            pad, fg_color=C["surface"],
            corner_radius=14, border_width=1, border_color=C["border"]
        )
        self.ctrl_card.pack(fill="x", pady=(0, 20))
        
        ctrl_inner = ctk.CTkFrame(self.ctrl_card, fg_color="transparent")
        ctrl_inner.pack(fill="both", expand=True, padx=24, pady=20)
        
        btn_row = ctk.CTkFrame(ctrl_inner, fg_color="transparent")
        btn_row.pack(fill="x")
        btn_row.grid_columnconfigure(0, weight=1)
        btn_row.grid_columnconfigure(1, weight=1)
        
        self.start_btn = ctk.CTkButton(
            btn_row, text="Start Monitoring",
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=C["accent"], hover_color=C["accent_soft"],
            text_color=C["btn_text"], height=44, corner_radius=10,
            command=self._start
        )
        self.start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        
        self.stop_btn = ctk.CTkButton(
            btn_row, text="Stop",
            font=ctk.CTkFont(size=15),
            fg_color="transparent", hover_color=C["elevated"],
            border_width=1, border_color=C["border"],
            text_color=C["text_dim"], height=44, corner_radius=10,
            command=self._stop, state="disabled"
        )
        self.stop_btn.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        
        self.state_lbl = ctk.CTkLabel(
            ctrl_inner, text="● Idle",
            font=ctk.CTkFont(size=13),
            text_color=C["text_muted"]
        )
        self.state_lbl.pack(pady=(12, 0))
        
    # ─────────────────────────────────────
    # WAVEFORM ANIMATION
    # ─────────────────────────────────────
    def _draw_wave(self):
        self.wave.delete("all")
        w = self.wave.winfo_width() or 600
        h = self.wave.winfo_height() or 36
        bw = (w / len(self._bars)) - 1
        
        for i, bh in enumerate(self._bars):
            x = i * (bw + 1)
            y = (h - bh) / 2
            color = C["accent"] if self._wave_active else C["border"]
            self.wave.create_rectangle(x, y, x+bw, y+bh, fill=color, outline="")
            
    def _animate_wave(self):
        if not self._wave_active:
            return
        for i in range(len(self._bars)):
            t = random.randint(6, 28)
            self._bars[i] += (t - self._bars[i]) * 0.25
        self._draw_wave()
        self.after(50, self._animate_wave)
        
    def _start_wave(self):
        self._wave_active = True
        self._animate_wave()
        
    def _stop_wave(self):
        self._wave_active = False
        self._bars = [6] * 50
        self._draw_wave()
        
    # ─────────────────────────────────────
    # VOIP CALLBACKS
    # ─────────────────────────────────────
    def _on_call_start(self):
        """Called when a VoIP call is detected."""
        self._call_active = True
        self.after(0, self._update_call_ui, True)
        
        # Start recording
        if self._audio_recorder:
            self._audio_recorder.start()
            
        notify("📞 Call Detected", "VoIP call started — recording audio")
    
    def _on_call_end(self):
        """Called when VoIP call ends."""
        self._call_active = False
        self.after(0, self._update_call_ui, False)
        
        # Stop recording
        if self._audio_recorder:
            self._audio_recorder.stop()
            
        notify("📴 Call Ended", f"Processed {self._frames_processed} frames")
    
    def _update_call_ui(self, call_active: bool):
        """Update UI based on call state (runs on main thread)."""
        if call_active:
            self.call_frame.pack(fill="x", pady=(0, 16), before=self.status_card)
            self.desc_lbl.configure(text="Recording call audio...")
            self._start_wave()
        else:
            self.call_frame.pack_forget()
            self.desc_lbl.configure(text="Waiting for VoIP call...")
            self._stop_wave()
        
    # ─────────────────────────────────────
    # CONTROLS
    # ─────────────────────────────────────
    def _start(self):
        self._running = True
        self._last = None
        self._frames_processed = 0
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.state_lbl.configure(text="● Monitoring", text_color=C["accent"])
        self.frames_lbl.configure(text="0 frames recorded")
        
        # Initialize real components
        self._audio_recorder = AudioRecorder(frame_duration=3.0, samplerate=48000)
        self._voip_monitor = VoIPMonitor(
            poll_interval=1.0,
            on_call_start=self._on_call_start,
            on_call_end=self._on_call_end
        )
        self._voip_monitor.start()
        
        self.status_lbl.configure(text="MONITORING", text_color=C["accent"])
        self.desc_lbl.configure(text="Waiting for VoIP call...")
        
        # Start processing loop
        self._process_frames()
        
    def _stop(self):
        self._running = False
        
        # Stop VoIP monitor
        if self._voip_monitor:
            self._voip_monitor.stop()
            self._voip_monitor = None
            
        # Stop audio recorder
        if self._audio_recorder:
            self._audio_recorder.stop()
            self._audio_recorder = None
        
        if self._job:
            self.after_cancel(self._job)
            self._job = None
            
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.state_lbl.configure(text="● Idle", text_color=C["text_muted"])
        self._stop_wave()
        self._reset_status()
        self.call_frame.pack_forget()
        
    def _reset_status(self):
        self.status_lbl.configure(text="READY", text_color=C["text_muted"])
        self.frames_lbl.configure(text="0 frames recorded")
        self.desc_lbl.configure(text="Press Start to begin monitoring")
        
    # ─────────────────────────────────────
    # FRAME PROCESSING LOOP
    # ─────────────────────────────────────
    def _process_frames(self):
        """Check for new audio frames and process them."""
        if not self._running:
            return
            
        if self._audio_recorder and self._call_active:
            frame = self._audio_recorder.get_frame(timeout=0.05)
            
            if frame is not None:
                self._frames_processed += 1
                
                # Save audio frame to temp folder
                self._save_audio_frame(frame)
                
                # Update frame counter in UI
                word = "frame" if self._frames_processed == 1 else "frames"
                self.frames_lbl.configure(text=f"{self._frames_processed} {word} recorded")
                
        self._job = self.after(100, self._process_frames)
    
    def _save_audio_frame(self, frame):
        """Save audio frame as WAV file to temp folder."""
        try:
            timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]  # HH:MM:SS_mmm
            filename = f"frame_{self._frames_processed:04d}_{timestamp}.wav"
            filepath = os.path.join(self._temp_dir, filename)
            
            # Convert to int16 for WAV
            audio_data = (frame * 32767).astype(np.int16)
            write_wav(filepath, 48000, audio_data)
            
            print(f"[Recording] Saved: {filename}")
        except Exception as e:
            print(f"[Recording] Error saving frame: {e}")
        
    
    # ─────────────────────────────────────
    # THEME TOGGLE
    # ─────────────────────────────────────
    def _toggle_theme(self):
        global C
        self._is_dark_mode = not self._is_dark_mode
        
        if self._is_dark_mode:
            C = DARK_THEME.copy()
            ctk.set_appearance_mode("dark")
            self.theme_btn.configure(text="☀")
        else:
            C = LIGHT_THEME.copy()
            ctk.set_appearance_mode("light")
            self.theme_btn.configure(text="🌙")
        
        self._apply_theme()
    
    def _apply_theme(self):
        # Main window
        self.configure(fg_color=C["bg"])
        
        # Waveform canvas
        self.wave.configure(bg=C["bg"])
        self._draw_wave()
        
        # Theme toggle button
        self.theme_btn.configure(
            fg_color=C["surface"],
            hover_color=C["elevated"],
            text_color=C["text_dim"],
            border_color=C["border"]
        )
        
        # Header
        self.title_lbl.configure(text_color=C["text"])
        self.sub_lbl.configure(text_color=C["text_muted"])
        
        # Call frame
        self.call_frame.configure(fg_color=C["surface"], border_color=C["accent"])
        self.call_label.configure(text_color=C["accent"])
        
        # Status card
        self.status_card.configure(fg_color=C["surface"], border_color=C["border"])
        
        # Status labels (only update color if not running)
        if not self._running:
            self.status_lbl.configure(text_color=C["text_muted"])
        self.frames_lbl.configure(text_color=C["text_dim"])
        self.desc_lbl.configure(text_color=C["text_muted"])
        
        # Controls card
        self.ctrl_card.configure(fg_color=C["surface"], border_color=C["border"])
        
        # Control buttons
        self.start_btn.configure(
            fg_color=C["accent"],
            hover_color=C["accent_soft"],
            text_color=C["btn_text"]
        )
        self.stop_btn.configure(
            hover_color=C["elevated"],
            border_color=C["border"],
            text_color=C["text_dim"]
        )
        
        # State label
        if not self._running:
            self.state_lbl.configure(text_color=C["text_muted"])


if __name__ == "__main__":
    app = App()
    app.mainloop()
