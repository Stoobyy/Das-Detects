"""
Debug script for VoIP detection logic.
Run this to see what the system detects as active VoIP apps and microphone usage.
"""
import sys
import os
import time
import psutil

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.check_app import detect_voip_sources, detect_any_mic_activity

def loop():
    print("Monitoring VoIP sources and Mic activity... (Press Ctrl+C to stop)")
    print("-" * 60)
    print(f"{'TIMESTAMP':<10} | {'MIC':<5} | {'VOIP SOURCES'}")
    print("-" * 60)
    
    while True:
        timestamp = time.strftime("%H:%M:%S")
        
        try:
            mic_active = detect_any_mic_activity()
            sources = detect_voip_sources()
            
            source_names = [s['name'] for s in sources]
            
            mic_str = "ON" if mic_active else "OFF"
            sources_str = ", ".join(source_names) if source_names else "None"
            
            # Highlight if active
            if mic_active or source_names:
                print(f"{timestamp:<10} | {mic_str:<5} | {sources_str}")
            else:
                # Print dot for heartbeat if nothing interesting
                sys.stdout.write(".")
                sys.stdout.flush()
                
        except Exception as e:
            print(f"\nError: {e}")
            
        time.sleep(1.0)

if __name__ == "__main__":
    try:
        loop()
    except KeyboardInterrupt:
        print("\nStopped.")
