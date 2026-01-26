"""
Real-Time AI Voice Call Detection
==================================
Premium, classy dark UI with refined aesthetics.

⚠️ DEMO MODE — Mocked behavior only.
"""

import customtkinter as ctk
import random
from datetime import datetime

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
        
        self._build_ui()
        
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
        # ALERT BANNER (hidden initially)
        # ─────────────────────────────────────
        self.alert_frame = ctk.CTkFrame(
            pad, fg_color=C["surface"],
            corner_radius=10, border_width=1, border_color=C["alert"]
        )
        self.alert_label = ctk.CTkLabel(
            self.alert_frame, 
            text="⚠  AI Voice Detected — Review Required",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=C["alert"]
        )
        self.alert_label.pack(pady=14, padx=24)
        
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
        
        self.conf_lbl = ctk.CTkLabel(
            inner, text="—",
            font=ctk.CTkFont(size=16),
            text_color=C["text_dim"]
        )
        self.conf_lbl.pack(pady=(8, 0))
        
        # Thin divider
        div = ctk.CTkFrame(inner, height=1, fg_color=C["border"], width=200)
        div.pack(pady=20)
        
        self.desc_lbl = ctk.CTkLabel(
            inner, text="Press Start to begin analysis",
            font=ctk.CTkFont(size=14),
            text_color=C["text_muted"]
        )
        self.desc_lbl.pack()
        
        # ─────────────────────────────────────
        # METER + CONTROLS ROW
        # ─────────────────────────────────────
        row = ctk.CTkFrame(pad, fg_color="transparent")
        row.pack(fill="x", pady=(0, 20))
        row.grid_columnconfigure(0, weight=3)
        row.grid_columnconfigure(1, weight=2)
        
        # Meter card
        self.meter_card = ctk.CTkFrame(
            row, fg_color=C["surface"],
            corner_radius=14, border_width=1, border_color=C["border"]
        )
        self.meter_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        meter_inner = ctk.CTkFrame(self.meter_card, fg_color="transparent")
        meter_inner.pack(fill="x", padx=24, pady=20)
        
        meter_head = ctk.CTkFrame(meter_inner, fg_color="transparent")
        meter_head.pack(fill="x")
        
        self.prob_lbl = ctk.CTkLabel(
            meter_head, text="Probability",
            font=ctk.CTkFont(size=14),
            text_color=C["text_dim"]
        )
        self.prob_lbl.pack(side="left")
        
        self.pct_lbl = ctk.CTkLabel(
            meter_head, text="0%",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=C["accent"]
        )
        self.pct_lbl.pack(side="right")
        
        self.prog = ctk.CTkProgressBar(
            meter_inner, height=6, corner_radius=3,
            fg_color=C["elevated"], progress_color=C["safe"]
        )
        self.prog.pack(fill="x", pady=(14, 0))
        self.prog.set(0)
        
        # Controls card
        self.ctrl_card = ctk.CTkFrame(
            row, fg_color=C["surface"],
            corner_radius=14, border_width=1, border_color=C["border"]
        )
        self.ctrl_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        
        ctrl_inner = ctk.CTkFrame(self.ctrl_card, fg_color="transparent")
        ctrl_inner.pack(fill="both", expand=True, padx=20, pady=20)
        
        self.start_btn = ctk.CTkButton(
            ctrl_inner, text="Start",
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=C["accent"], hover_color=C["accent_soft"],
            text_color=C["btn_text"], height=44, corner_radius=10,
            command=self._start
        )
        self.start_btn.pack(fill="x", pady=(0, 8))
        
        self.stop_btn = ctk.CTkButton(
            ctrl_inner, text="Stop",
            font=ctk.CTkFont(size=15),
            fg_color="transparent", hover_color=C["elevated"],
            border_width=1, border_color=C["border"],
            text_color=C["text_dim"], height=44, corner_radius=10,
            command=self._stop, state="disabled"
        )
        self.stop_btn.pack(fill="x")
        
        self.state_lbl = ctk.CTkLabel(
            ctrl_inner, text="● Idle",
            font=ctk.CTkFont(size=13),
            text_color=C["text_muted"]
        )
        self.state_lbl.pack(pady=(12, 0))
        
        # ─────────────────────────────────────
        # LOG SECTION
        # ─────────────────────────────────────
        self.log_card = ctk.CTkFrame(
            pad, fg_color=C["surface"],
            corner_radius=14, border_width=1, border_color=C["border"]
        )
        self.log_card.pack(fill="both", expand=True)
        
        log_head = ctk.CTkFrame(self.log_card, fg_color="transparent")
        log_head.pack(fill="x", padx=24, pady=(18, 12))
        
        self.history_lbl = ctk.CTkLabel(
            log_head, text="History",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=C["text_dim"]
        )
        self.history_lbl.pack(side="left")
        
        self.cnt_lbl = ctk.CTkLabel(
            log_head, text="0",
            font=ctk.CTkFont(size=14),
            text_color=C["text_muted"]
        )
        self.cnt_lbl.pack(side="right")
        
        self.log_scroll = ctk.CTkScrollableFrame(
            self.log_card, fg_color=C["elevated"],
            corner_radius=10
        )
        self.log_scroll.pack(fill="both", expand=True, padx=20, pady=(0, 18))
        
        # Log header row
        hdr = ctk.CTkFrame(self.log_scroll, fg_color="transparent")
        hdr.pack(fill="x", pady=(8, 6), padx=8)
        ctk.CTkLabel(hdr, text="TIME", width=80, font=ctk.CTkFont(size=11), text_color=C["text_muted"]).pack(side="left")
        ctk.CTkLabel(hdr, text="RESULT", width=110, font=ctk.CTkFont(size=11), text_color=C["text_muted"]).pack(side="left", padx=(20,0))
        ctk.CTkLabel(hdr, text="CONF", font=ctk.CTkFont(size=11), text_color=C["text_muted"]).pack(side="left", padx=(20,0))
        
        self._log_entries = []
        
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
    # CONTROLS
    # ─────────────────────────────────────
    def _start(self):
        self._running = True
        self._last = None
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.state_lbl.configure(text="● Analyzing", text_color=C["accent"])
        self._clear_log()
        self._start_wave()
        self._tick()
        
    def _stop(self):
        self._running = False
        if self._job:
            self.after_cancel(self._job)
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.state_lbl.configure(text="● Idle", text_color=C["text_muted"])
        self._stop_wave()
        self._reset_status()
        self.alert_frame.pack_forget()
        
    def _reset_status(self):
        self.status_lbl.configure(text="READY", text_color=C["text_muted"])
        self.conf_lbl.configure(text="—")
        self.desc_lbl.configure(text="Press Start to begin analysis")
        self.prog.set(0)
        self.pct_lbl.configure(text="0%", text_color=C["accent"])
        
    # ─────────────────────────────────────
    # MOCK UPDATE LOOP
    # ─────────────────────────────────────
    def _tick(self):
        if not self._running:
            return
            
        # Generate weighted random confidence
        r = random.choices([0, 1, 2], weights=[0.5, 0.28, 0.22])[0]
        if r == 0:
            conf = random.randint(8, 36)
        elif r == 1:
            conf = random.randint(44, 66)
        else:
            conf = random.randint(74, 94)
            
        # Determine status
        if conf < 40:
            status, color = "HUMAN", C["safe"]
            desc = "Natural speech patterns confirmed"
            self.alert_frame.pack_forget()
        elif conf < 70:
            status, color = "SUSPICIOUS", C["warn"]
            desc = "Anomalies detected in voice pattern"
            self.alert_frame.pack_forget()
            if self._last != "SUSPICIOUS":
                notify("⚡ Suspicious", f"{conf}% confidence — review needed")
        else:
            status, color = "AI DETECTED", C["alert"]
            desc = "High probability of synthetic voice"
            self.alert_frame.pack(fill="x", pady=(0, 16), before=self.status_card)
            if self._last != "AI DETECTED":
                notify("🚨 AI Detected", f"{conf}% confidence — potential fraud")
                
        self._last = status
        
        # Update UI
        self.status_lbl.configure(text=status, text_color=color)
        self.conf_lbl.configure(text=f"{conf}% confidence")
        self.desc_lbl.configure(text=desc)
        
        self.prog.set(conf / 100)
        self.prog.configure(progress_color=color)
        self.pct_lbl.configure(text=f"{conf}%", text_color=color)
        
        self._add_log(status, conf, color)
        
        self._job = self.after(random.randint(2200, 3200), self._tick)
        
    # ─────────────────────────────────────
    # LOG
    # ─────────────────────────────────────
    def _add_log(self, status: str, conf: int, color: str):
        row = ctk.CTkFrame(self.log_scroll, fg_color="transparent")
        row.pack(fill="x", pady=3, padx=8)
        
        t = datetime.now().strftime("%H:%M:%S")
        ctk.CTkLabel(row, text=t, width=80, font=ctk.CTkFont(family="Consolas", size=13), text_color=C["text_dim"]).pack(side="left")
        ctk.CTkLabel(row, text=status, width=110, font=ctk.CTkFont(size=13, weight="bold"), text_color=color).pack(side="left", padx=(20,0))
        ctk.CTkLabel(row, text=f"{conf}%", font=ctk.CTkFont(size=13), text_color=C["text_dim"]).pack(side="left", padx=(20,0))
        
        self._log_entries.append(row)
        self.cnt_lbl.configure(text=str(len(self._log_entries)))
        
    def _clear_log(self):
        for e in self._log_entries:
            e.destroy()
        self._log_entries = []
        self.cnt_lbl.configure(text="0")
    
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
        
        # Alert frame
        self.alert_frame.configure(fg_color=C["surface"], border_color=C["alert"])
        self.alert_label.configure(text_color=C["alert"])
        
        # Status card
        self.status_card.configure(fg_color=C["surface"], border_color=C["border"])
        
        # Status labels (only update color if not running)
        if not self._running:
            self.status_lbl.configure(text_color=C["text_muted"])
        self.conf_lbl.configure(text_color=C["text_dim"])
        self.desc_lbl.configure(text_color=C["text_muted"])
        
        # Meter card
        self.meter_card.configure(fg_color=C["surface"], border_color=C["border"])
        self.prob_lbl.configure(text_color=C["text_dim"])
        
        # Progress bar
        self.prog.configure(fg_color=C["elevated"])
        if not self._running:
            self.pct_lbl.configure(text_color=C["accent"])
        
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
        
        # Log card
        self.log_card.configure(fg_color=C["surface"], border_color=C["border"])
        self.history_lbl.configure(text_color=C["text_dim"])
        
        # Log scroll
        self.log_scroll.configure(fg_color=C["elevated"])
        
        # Count label
        self.cnt_lbl.configure(text_color=C["text_muted"])


if __name__ == "__main__":
    app = App()
    app.mainloop()
