import psutil
import warnings
from pycaw.pycaw import AudioUtilities, IAudioSessionControl2
from comtypes import CLSCTX_ALL, cast
from ctypes import POINTER
from pycaw.pycaw import IAudioMeterInformation

# Suppress pycaw COM warnings for devices with missing properties
warnings.filterwarnings("ignore", message="COMError attempting to get property")


VOIP_KEYWORDS = [
    "whatsapp",
    "zoom",
    "teams",
    "discord",
    "telegram",
    "signal"
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


def is_microphone_active():
    """Check if any application is actively using the microphone."""
    try:
        from pycaw.pycaw import AudioUtilities as AU
        from comtypes import CLSCTX_ALL
        
        # Get the default microphone device
        devices = AU.GetAllDevices()
        for device in devices:
            try:
                # Check if it's a capture device (microphone) and active
                if device.FriendlyName and "microphone" in device.FriendlyName.lower():
                    # Try to get audio meter to check if it's actively receiving audio
                    return True
            except Exception:
                continue
        
        # Alternative: Check capture sessions via MMDevice API
        import comtypes
        from pycaw.pycaw import IMMDeviceEnumerator, IMMDevice, CLSID_MMDeviceEnumerator
        
        enumerator = comtypes.CoCreateInstance(
            CLSID_MMDeviceEnumerator,
            IMMDeviceEnumerator,
            comtypes.CLSCTX_ALL
        )
        # eCapture = 1 means input devices (microphones)
        device = enumerator.GetDefaultAudioEndpoint(1, 0)  # 1 = eCapture, 0 = eConsole
        
        if device:
            # Get sessions for the capture device
            mgr = device.Activate(
                AudioUtilities.IID_IAudioSessionManager2,
                CLSCTX_ALL,
                None
            )
            session_enum = mgr.GetSessionEnumerator()
            
            for i in range(session_enum.GetCount()):
                session = session_enum.GetSession(i)
                if session:
                    ctl2 = session.QueryInterface(IAudioSessionControl2)
                    state = ctl2.GetState()
                    if state == 1:  # AudioSessionStateActive
                        return True
                        
    except Exception as e:
        # Fallback: check if any VoIP process has open handles to audio capture
        pass
    
    return False


def is_call_active():
    """Returns True only if a VoIP app is playing audio AND microphone is in use."""
    voip_detected = len(detect_voip_sources()) > 0
    mic_active = is_microphone_active()
    return voip_detected and mic_active


if __name__ == "__main__":
    sources = detect_voip_sources()
    mic_active = is_microphone_active()

    print(f"\n🎤 Microphone Active: {mic_active}")
    
    if sources:
        print("\n🔊 Active VoIP Audio Sources Detected:")
        for s in sources:
            print(f"  - PID {s['pid']} → {s['name']}")
            print(f"    Parent chain: {s['chain']}")
    else:
        print("\n🔇 No VoIP audio sources detected.")
    
    print(f"\n📞 Call Active: {is_call_active()}")

