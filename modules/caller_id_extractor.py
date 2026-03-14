import uiautomation as auto
import re
import logging

class CallerIDExtractor:
    def __init__(self):
        pass

    def extract_caller_id(self) -> str | None:
        """
        Extracts the caller ID from the active WhatsApp call window.
        Returns the valid phone number, or None if not an unsaved number or not found.
        """
        try:
            # Look for Voice call or Video call window
            call_win = auto.WindowControl(Name="Voice call")
            if not call_win.Exists(0, 0):
                call_win = auto.WindowControl(Name="Video call")
                if not call_win.Exists(0, 0):
                    return None
            
            # Walk through controls in the Call window up to depth 15
            for item, depth in auto.WalkControl(call_win, maxDepth=15):
                if hasattr(item, 'Name') and item.Name:
                    valid_num = self.validate_number(item.Name)
                    if valid_num:
                        return valid_num
                            
            return None
        except Exception as e:
            logging.error(f"Error extracting WhatsApp Caller ID: {e}")
            return None

    @staticmethod
    def validate_number(text: str) -> str | None:
        """
        Checks if the extracted text is an unsaved number (starts with + and contains mostly numbers).
        Returns the clean number if valid, or None if it's a saved contact name (contains letters).
        """
        text = text.strip()
        if not text.startswith('+'):
            return None
            
        # Check if it contains any letters. If it does, it's a saved contact.
        if re.search(r'[a-zA-Z]', text):
            return None
            
        # Extract digits
        digits = re.sub(r'\D', '', text)
        if len(digits) >= 7:
            return '+' + digits
        return None
