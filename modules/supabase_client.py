import os
import logging
from datetime import datetime, timezone
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

class SupabaseClientDB:
    def __init__(self):
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        
        self.client: Client | None = None
        if url and key:
            try:
                self.client = create_client(url, key)
            except Exception as e:
                logging.error(f"Failed to initialize Supabase client: {e}")
        else:
            logging.warning("SUPABASE_URL or SUPABASE_KEY not set. Supabase integration disabled.")

    def check_flagged_number(self, phone_hash: str) -> bool:
        """
        Queries the flagged_numbers table.
        Returns True if strike_count > 0 for this phone_hash.
        """
        if not self.client:
            return False
            
        try:
            response = self.client.table("flagged_numbers").select("strike_count").eq("phone_hash", phone_hash).execute()
            if response.data:
                return response.data[0].get("strike_count", 0) > 0
            return False
        except Exception as e:
            logging.error(f"Error checking flagged number: {e}")
            return False

    def flag_number(self, phone_hash: str) -> bool:
        """
        Upserts the phone_hash, incrementing strike_count and updating last_flagged_at.
        """
        if not self.client:
            return False
            
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            
            response = self.client.table("flagged_numbers").select("strike_count").eq("phone_hash", phone_hash).execute()
            
            if response.data:
                new_strikes = response.data[0].get("strike_count", 0) + 1
                self.client.table("flagged_numbers").update({
                    "strike_count": new_strikes,
                    "last_flagged_at": now_iso
                }).eq("phone_hash", phone_hash).execute()
            else:
                self.client.table("flagged_numbers").insert({
                    "phone_hash": phone_hash,
                    "strike_count": 1,
                    "last_flagged_at": now_iso
                }).execute()
                
            return True
        except Exception as e:
            logging.error(f"Error flagging number: {e}")
            return False
