"""
VoIP Call Detection Module
==========================
Detects active VoIP calls by checking for VoIP apps playing audio.
"""

import psutil
import warnings
from pycaw.pycaw import AudioUtilities, IAudioSessionControl2

# Suppress pycaw COM warnings for devices with missing properties
warnings.filterwarnings("ignore", message="COMError attempting to get property")


VOIP_KEYWORDS = [
    "whatsapp",
    "zoom",
    "teams",
    "discord",
    "telegram",
    "signal",
    "skype",
    "viber",
    "slack",
    "webex",
]


def detect_voip_sources():
    """
    Detect VoIP apps that are ACTIVELY playing audio.
    
    Returns list of detected VoIP sources with PID, name, and process chain.
    Only matches apps in State=1 (AudioSessionStateActive), meaning they
    are currently outputting audio, not just have an open session.
    """
    sessions = AudioUtilities.GetAllSessions()
    detected = []

    for s in sessions:
        try:
            # Only match ACTIVE sessions (actually playing audio NOW)
            # State 0 = Inactive, State 1 = Active, State 2 = Expired
            if s.State != 1:
                continue
                
            ctl = s._ctl.QueryInterface(IAudioSessionControl2)
            pid = ctl.GetProcessId()

            if pid == 0:
                continue

            proc = psutil.Process(pid)
            name = proc.name().lower()
            chain = [p.name().lower() for p in proc.parents()] + [name]

            # Detect using keywords in process name or parent chain
            if any(key in name for key in VOIP_KEYWORDS) or \
               any(key in p for p in chain for key in VOIP_KEYWORDS):
                detected.append({
                    "pid": pid,
                    "name": name,
                    "chain": chain
                })

        except Exception:
            continue

    return detected


def is_call_active():
    """
    Returns True if a VoIP app is actively playing audio.
    
    This indicates an active call since VoIP apps typically only play
    audio during calls (ringtones, voice, etc).
    """
    return len(detect_voip_sources()) > 0


if __name__ == "__main__":
    sources = detect_voip_sources()
    
    if sources:
        print("\n🔊 Active VoIP Audio Sources Detected:")
        for s in sources:
            print(f"  - PID {s['pid']} → {s['name']}")
            print(f"    Parent chain: {s['chain']}")
        print(f"\n📞 Call Active: True")
    else:
        print("\n🔇 No VoIP audio sources detected.")
        print(f"\n📞 Call Active: False")
