import psutil
from pycaw.pycaw import AudioUtilities, IAudioSessionControl2
from comtypes import CLSCTX_ALL, cast


VOIP_KEYWORDS = [
    "whatsapp",
    "zoom",
    "teams",
    "skype",
    "discord",
    "telegram",
    "signal",
    "viber",
    "chrome",
    "msedge",
    "brave"
]


def get_audio_sessions_fixed():
    """Return list of dicts with: pid, process_name, is_active_audio"""
    sessions = AudioUtilities.GetAllSessions()
    results = []

    for s in sessions:
        try:
            ctl = s._ctl.QueryInterface(IAudioSessionControl2)
            pid = ctl.GetProcessId()

            # PID may be 0 or invalid for system streams
            if pid == 0:
                continue

            proc = psutil.Process(pid)
            name = proc.name().lower()

            # Active means it's outputting audio or "in use"
            is_active = (s.State == 1 or s.State == 0)

            results.append({
                "pid": pid,
                "name": name,
                "active": is_active
            })

        except Exception:
            continue

    return results


def detect_voip_sources():
    """Detect VoIP apps by scanning audio sessions + parent chains."""
    sessions = get_audio_sessions_fixed()
    detected = []

    for ses in sessions:
        if not ses["active"]:
            continue

        pid = ses["pid"]
        name = ses["name"]

        try:
            proc = psutil.Process(pid)
            chain = [p.name().lower() for p in proc.parents()] + [name]

            # Detect using keywords
            if any(key in name for key in VOIP_KEYWORDS) or \
               any(key in p for p in chain for key in VOIP_KEYWORDS):
                detected.append({
                    "pid": pid,
                    "name": name,
                    "chain": chain
                })

        except psutil.NoSuchProcess:
            continue

    return detected


def is_call_active():
    return len(detect_voip_sources()) > 0


if __name__ == "__main__":
    sources = detect_voip_sources()

    if sources:
        print("\nActive VoIP Audio Sources Detected:")
        for s in sources:
            print(f"- PID {s['pid']} → {s['name']}")
            print(f"  Parent chain: {s['chain']}")
    else:
        print("\nNo VoIP sources detected.")
