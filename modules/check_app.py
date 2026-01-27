"""
VoIP Call Detection Module
==========================
Detects active VoIP calls by checking for VoIP apps playing audio AND using microphone.
Uses both audio output and microphone input detection to distinguish calls from voice messages.
"""

import psutil
import warnings
from pycaw.pycaw import AudioUtilities, IAudioSessionControl2
from pycaw.pycaw import EDataFlow, DEVICE_STATE
from ctypes import POINTER, cast
import comtypes

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


def _get_voip_pids_with_audio_output():
    """Get PIDs of VoIP apps that are actively playing audio."""
    sessions = AudioUtilities.GetAllSessions()
    voip_pids = set()

    for s in sessions:
        try:
            # Only match ACTIVE sessions (actually playing audio NOW)
            if s.State != 1:
                continue
                
            ctl = s._ctl.QueryInterface(IAudioSessionControl2)
            pid = ctl.GetProcessId()

            if pid == 0:
                continue

            proc = psutil.Process(pid)
            name = proc.name().lower()
            chain = [p.name().lower() for p in proc.parents()] + [name]

            # Check if it's a VoIP app
            if any(key in name for key in VOIP_KEYWORDS) or \
               any(key in p for p in chain for key in VOIP_KEYWORDS):
                voip_pids.add(pid)

        except Exception:
            continue

    return voip_pids


def _get_voip_pids_with_mic_input():
    """
    Get PIDs of VoIP apps that are actively using the microphone.
    Uses pycaw to enumerate capture device sessions.
    """
    voip_mic_pids = set()
    
    try:
        # Get all audio devices including capture (microphone) devices
        from pycaw.pycaw import AudioUtilities as AU
        
        # Get the device enumerator
        devices = AU.GetAllDevices()
        
        for device in devices:
            try:
                # Check if it's a capture (input) device
                # We need to get sessions from capture endpoints
                device_state = device.GetState()
                if device_state != DEVICE_STATE.ACTIVE.value:
                    continue
                    
                # Try to get sessions from this device
                # For capture devices, we check which processes have active sessions
                session_manager = device.Activate(
                    IAudioSessionControl2._iid_, 
                    comtypes.CLSCTX_ALL, 
                    None
                )
            except Exception:
                continue
    except Exception:
        pass
    
    # Alternative approach: Get capture sessions from default microphone
    try:
        from pycaw.pycaw import AudioUtilities as AU
        from pycaw.pycaw import IMMDeviceEnumerator, IMMDevice
        
        # Get the default microphone device
        device_enumerator = comtypes.CoCreateInstance(
            AudioUtilities.CLSID_MMDeviceEnumerator,
            IMMDeviceEnumerator,
            comtypes.CLSCTX_ALL
        )
        
        # EDataFlow.eCapture = 1 for capture (microphone) devices
        try:
            default_mic = device_enumerator.GetDefaultAudioEndpoint(
                EDataFlow.eCapture.value,  # Capture device (microphone)
                0  # eConsole role
            )
            
            if default_mic:
                # Get the session manager for capture device
                from pycaw.pycaw import IAudioSessionManager2
                mgr = default_mic.Activate(
                    IAudioSessionManager2._iid_,
                    comtypes.CLSCTX_ALL,
                    None
                )
                
                if mgr:
                    from pycaw.pycaw import IAudioSessionEnumerator
                    enum = mgr.GetSessionEnumerator()
                    
                    if enum:
                        count = enum.GetCount()
                        for i in range(count):
                            try:
                                session = enum.GetSession(i)
                                ctl2 = session.QueryInterface(IAudioSessionControl2)
                                state = ctl2.GetState()
                                
                                # State 1 = Active (currently using mic)
                                if state == 1:
                                    pid = ctl2.GetProcessId()
                                    if pid and pid != 0:
                                        # Check if it's a VoIP app
                                        try:
                                            proc = psutil.Process(pid)
                                            name = proc.name().lower()
                                            if any(key in name for key in VOIP_KEYWORDS):
                                                voip_mic_pids.add(pid)
                                        except:
                                            pass
                            except:
                                continue
        except Exception:
            pass
    except Exception:
        pass
    
    return voip_mic_pids


def detect_any_mic_activity(threshold: float = 0.0001, sample_duration: float = 0.15):
    """
    Check if the microphone has any signal (indicating it's being used for a call).
    
    Uses soundcard to sample mic and detect any activity above noise floor.
    During a call, even silence has some mic activity from background noise/processing.
    
    Returns True if mic has signal above threshold, False otherwise.
    """
    try:
        import soundcard as sc
        import numpy as np
        import warnings
        
        # Suppress soundcard warnings
        warnings.filterwarnings("ignore", message="data discontinuity")
        
        # Get default microphone
        default_mic = sc.default_microphone()
        if default_mic is None:
            return False
        
        # Sample mic
        samplerate = 16000
        num_frames = int(samplerate * sample_duration)
        
        with default_mic.recorder(samplerate=samplerate) as recorder:
            data = recorder.record(numframes=num_frames)
        
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        # Calculate RMS (audio energy level)
        rms = np.sqrt(np.mean(data ** 2))
        
        # Very low threshold - catches any mic activity
        return rms > threshold
        
    except Exception:
        return False


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


def is_mic_in_use_by_voip():
    """
    Check if any VoIP app is using the microphone.
    Uses _get_voip_pids_with_mic_input to get VoIP processes with active mic.
    """
    voip_mic_pids = _get_voip_pids_with_mic_input()
    return len(voip_mic_pids) > 0


def is_call_active(require_mic: bool = False):
    """
    Returns True if a VoIP app is actively playing audio.
    
    Args:
        require_mic: If True, also requires microphone to be in use.
                     This helps distinguish actual calls from voice messages.
    
    This indicates an active call since VoIP apps typically only play
    audio during calls (ringtones, voice, etc).
    """
    has_voip_audio = len(detect_voip_sources()) > 0
    
    if not require_mic:
        return has_voip_audio
    
    # Strict mode: require both VoIP audio AND microphone in use
    if has_voip_audio:
        mic_active = detect_any_mic_activity()
        return mic_active
    
    return False


def is_call_active_strict():
    """
    Returns True only if VoIP audio is playing AND microphone is in use.
    This is the stricter check that distinguishes calls from voice messages.
    
    Voice messages typically only play audio (no mic usage).
    Actual calls require both audio output AND mic input.
    """
    return is_call_active(require_mic=True)


if __name__ == "__main__":
    print("=" * 50)
    print("VoIP Call Detection Module Test")
    print("=" * 50)
    
    # Check for VoIP audio sources
    sources = detect_voip_sources()
    
    print("\n📊 Detection Results:")
    print("-" * 30)
    
    if sources:
        print("\n🔊 Active VoIP Audio Sources Detected:")
        for s in sources:
            print(f"  - PID {s['pid']} → {s['name']}")
            print(f"    Parent chain: {s['chain']}")
    else:
        print("\n🔇 No VoIP audio sources detected.")
    
    # Check microphone activity
    mic_active = detect_any_mic_activity()
    mic_status = "🎤 Microphone IN USE" if mic_active else "🔇 Microphone not active"
    print(f"\n{mic_status}")
    
    # Show call detection results
    print("\n" + "-" * 30)
    call_loose = is_call_active(require_mic=False)
    call_strict = is_call_active(require_mic=True)
    
    print(f"📞 Call Active (Audio Only):   {'✅ YES' if call_loose else '❌ NO'}")
    print(f"📞 Call Active (Audio + Mic):  {'✅ YES' if call_strict else '❌ NO'}")
    
    if call_loose and not call_strict:
        print("\n💡 Tip: VoIP audio detected but no mic - likely a voice message playback!")
    elif call_strict:
        print("\n📱 Active voice call confirmed (both audio and mic in use)")
