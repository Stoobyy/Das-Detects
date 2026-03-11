import uiautomation as auto

def debug_whatsapp_ui():
    print("Searching for Voice call window...")
    # Look for WhatsApp Voice call window
    call_win = auto.WindowControl(Name="Voice call")
    if not call_win.Exists(0, 2):
        call_win = auto.WindowControl(Name="Video call")
        if not call_win.Exists(0, 1):
            print("Call window not found!")
            return
            
    print(f"Call window found: '{call_win.Name}' [{call_win.ClassName}]. Printing elements...")
    
    # Increase depth and look at everything, including text properties
    for item, depth in auto.WalkControl(call_win, maxDepth=15):
        if hasattr(item, 'Name') and item.Name and len(item.Name.strip()) > 0:
            print(f"Depth {depth} [{item.ControlType}]: {repr(item.Name)}")

if __name__ == "__main__":
    debug_whatsapp_ui()
