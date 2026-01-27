"""
VoIP Monitor Module
===================
Background service that monitors for active VoIP calls and manages audio recording.
Polls call detection and triggers callbacks on state changes.
"""

import threading
import time
from typing import Callable

# Import the VoIP detection functions
from modules.check_app import is_call_active, detect_voip_sources


class VoIPMonitor:
    """
    Background monitor that detects VoIP calls and manages recording lifecycle.
    Runs in a separate thread and invokes callbacks on call state changes.
    """
    
    def __init__(
        self,
        poll_interval: float = 1.0,
        on_call_start: Callable[[], None] | None = None,
        on_call_end: Callable[[], None] | None = None,
        on_status_update: Callable[[bool, list], None] | None = None,
    ):
        """
        Args:
            poll_interval: How often to check for calls (seconds)
            on_call_start: Callback when a call is detected
            on_call_end: Callback when call ends
            on_status_update: Callback(is_active, sources) on each poll
        """
        self.poll_interval = poll_interval
        self.on_call_start = on_call_start
        self.on_call_end = on_call_end
        self.on_status_update = on_status_update
        
        self._running = False
        self._thread: threading.Thread | None = None
        self._call_active = False
        self._lock = threading.Lock()
    
    def start(self) -> bool:
        """Start monitoring in background thread. Returns True if started."""
        with self._lock:
            if self._running:
                return False
            
            self._running = True
            self._call_active = False
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()
            return True
    
    def stop(self):
        """Stop monitoring."""
        with self._lock:
            self._running = False
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
    
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running
    
    def is_call_detected(self) -> bool:
        """Check if a call is currently detected."""
        return self._call_active
    
    def _monitor_loop(self):
        """Internal monitoring loop - runs in background thread."""
        import comtypes
        comtypes.CoInitialize()  # Required for pycaw in background thread
        
        try:
            self._monitor_loop_inner()
        finally:
            comtypes.CoUninitialize()
    
    def _monitor_loop_inner(self):
        """Actual monitoring logic."""
        while self._running:
            try:
                # Check current call state - VoIP app actively playing audio
                sources = detect_voip_sources()
                call_now_active = len(sources) > 0
                
                # Detect state transitions
                was_active = self._call_active
                self._call_active = call_now_active
                
                # Fire callbacks on state change
                if call_now_active and not was_active:
                    # Call just started
                    if self.on_call_start:
                        try:
                            self.on_call_start()
                        except Exception as e:
                            print(f"[VoIPMonitor] on_call_start error: {e}")
                
                elif not call_now_active and was_active:
                    # Call just ended
                    if self.on_call_end:
                        try:
                            self.on_call_end()
                        except Exception as e:
                            print(f"[VoIPMonitor] on_call_end error: {e}")
                
                # Always fire status update
                if self.on_status_update:
                    try:
                        self.on_status_update(call_now_active, sources)
                    except Exception as e:
                        print(f"[VoIPMonitor] on_status_update error: {e}")
                
            except Exception as e:
                print(f"[VoIPMonitor] Poll error: {e}")
            
            time.sleep(self.poll_interval)


# Quick test
if __name__ == "__main__":
    print("Testing VoIPMonitor...")
    
    def on_start():
        print("📞 CALL STARTED!")
    
    def on_end():
        print("📴 CALL ENDED!")
    
    def on_update(active, sources):
        status = "🟢 In Call" if active else "⚪ No Call"
        print(f"  Status: {status} | Sources: {len(sources)}")
    
    monitor = VoIPMonitor(
        poll_interval=2.0,
        on_call_start=on_start,
        on_call_end=on_end,
        on_status_update=on_update
    )
    
    monitor.start()
    print("Monitoring for 30 seconds... Make/end a call to test.\n")
    
    try:
        time.sleep(30)
    except KeyboardInterrupt:
        pass
    
    monitor.stop()
    print("\nMonitor stopped.")
