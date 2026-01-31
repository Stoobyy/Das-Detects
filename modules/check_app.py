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


def detect_any_mic_activity():
    """
    Check if ANY process is actively using the microphone.
    Returns True if microphone is in use, False otherwise.
    
    Uses Windows Registry CapabilityAccessManager as the primary method,
    which reliably tracks mic usage across all apps.
    """
    try:
        import winreg
        
        # Registry path where Windows tracks microphone access
        base_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone"
        
        # Get list of running process names for verification
        running_processes = set()
        for proc in psutil.process_iter(['name']):
            try:
                running_processes.add(proc.info['name'].lower())
            except:
                pass
        
        def check_key_for_active_mic(key_path):
            """Check a registry key for active mic usage"""
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                    try:
                        start_time, _ = winreg.QueryValueEx(key, "LastUsedTimeStart")
                        stop_time, _ = winreg.QueryValueEx(key, "LastUsedTimeStop")
                        
                        # Mic is IN USE when start > stop (started but not yet stopped)
                        if start_time > stop_time:
                            # For NonPackaged apps, verify process is running
                            key_name = key_path.split("\\")[-1]
                            if "#" in key_name:
                                exe_name = key_name.split("#")[-1].lower()
                                if exe_name in running_processes:
                                    return True
                            else:
                                # Packaged app (UWP) - always valid
                                return True
                    except FileNotFoundError:
                        pass
                    
                    # Check subkeys
                    try:
                        i = 0
                        while True:
                            subkey_name = winreg.EnumKey(key, i)
                            subkey_path = f"{key_path}\\{subkey_name}"
                            if check_key_for_active_mic(subkey_path):
                                return True
                            i += 1
                    except OSError:
                        pass
            except (FileNotFoundError, PermissionError):
                pass
            return False
        
        return check_key_for_active_mic(base_path)
        
    except Exception:
        return False


def detect_voip_sources():
    """
    Detect VoIP apps that have an active audio session OR are running with mic active.
    
    Returns list of detected VoIP sources with PID, name, and process chain.
    Uses multiple detection methods:
    1. Audio session detection - only State=1 (actually playing audio)
    2. Running process + mic active detection (fallback for UWP apps)
    """
    sessions = AudioUtilities.GetAllSessions()
    detected = []
    detected_pids = set()

    # Method 1: Check audio sessions - only ACTIVE sessions (State=1)
    for s in sessions:
        try:
            # Only match ACTIVE sessions (actually playing audio NOW)
            # This prevents detecting apps that just have a session open
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
                detected_pids.add(pid)

        except Exception:
            continue

    # Method 2: Fallback - check running VoIP processes with mic active
    # This catches UWP apps like WhatsApp that don't create standard audio sessions
    if detect_any_mic_activity():
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                name = proc.info['name'].lower()
                pid = proc.info['pid']
                
                if pid in detected_pids:
                    continue  # Already detected via audio session
                
                if any(key in name for key in VOIP_KEYWORDS):
                    chain = [p.name().lower() for p in psutil.Process(pid).parents()] + [name]
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
