import os
import json
import logging
import asyncio
from datetime import datetime, timedelta, date, time as dtime, timezone
from typing import Dict, Any, List, Union, Optional, AsyncGenerator
import redis
from google.oauth2 import service_account
from googleapiclient.discovery import build
from dotenv import load_dotenv
import time

# --- Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("cache_updater")

# --- Configuration ---
GOOGLE_CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID", "primary")
GOOGLE_CRED_FILE   = "credentials.json"
# MODIFICATION: Use REDIS_URL for deployment flexibility
REDIS_URL = "redis://default:ea4589af328841a28283b872173cc899@fly-test-little-voice-4200-redis.upstash.io:6379"
DAYS_TO_CACHE = 7 # How many days ahead to pre-populate the cache
CACHE_EXPIRATION_SECONDS = 60 * 60 * 4 # 4 hours

# --- Helper Functions ---
def to_rfc3339(dt: datetime) -> str:
    """Converts a datetime object to an RFC3339 string with Z timezone."""
    return dt.isoformat().replace('+00:00', 'Z')

def format_time_for_response(dt):
    """Helper function to format time in a cross-platform way."""
    return dt.strftime("%I:%M %p").lstrip("0")

def get_availability_for_day(calendar_service, target_date: date) -> List[str]:
    """
    Connects to Google Calendar and fetches the list of available 1-hour slots for a given day.
    """
    logger.info(f"Fetching availability from Google API for date: {target_date}")

    start_of_day = datetime.combine(target_date, dtime.min, tzinfo=timezone.utc)
    end_of_day = datetime.combine(target_date, dtime.max, tzinfo=timezone.utc)
    
    business_start = start_of_day.replace(hour=9)
    business_end = start_of_day.replace(hour=17)

    try:
        events_result = calendar_service.events().list(
            calendarId=GOOGLE_CALENDAR_ID,
            timeMin=to_rfc3339(start_of_day),
            timeMax=to_rfc3339(end_of_day),
            singleEvents=True,
            orderBy='startTime'
        ).execute()
    except Exception as e:
        logger.error(f"Could not fetch events for {target_date}. Error: {e}")
        return []

    events = events_result.get('items', [])
    busy_times = []
    for event in events:
        start_str = event['start'].get('dateTime')
        end_str = event['end'].get('dateTime')
        if start_str and end_str:
            # Using dateutil.parser would be more robust, but this handles the common 'Z' format
            if 'Z' in start_str: start_str = start_str.replace('Z', '+00:00')
            if 'Z' in end_str: end_str = end_str.replace('Z', '+00:00')
            busy_times.append(
                (datetime.fromisoformat(start_str), datetime.fromisoformat(end_str))
            )

    available_slots = []
    current_time = business_start
    while current_time < business_end:
        is_busy = any(start < current_time + timedelta(hours=1) and end > current_time for start, end in busy_times)
        if not is_busy:
            formatted_time = format_time_for_response(current_time)
            available_slots.append(formatted_time)
        current_time += timedelta(hours=1)
    
    return available_slots

def update_availability_cache():
    """
    Main function to pre-populate the Redis cache with Google Calendar availability.
    """
    logger.info("Starting cache update process...")
    try:
        # MODIFICATION: Connect using REDIS_URL
        if not REDIS_URL:
            raise redis.exceptions.ConnectionError("REDIS_URL environment variable not set.")
        
        redis_client = redis.from_url(REDIS_URL)
        redis_client.ping()
        logger.info("Successfully connected to Redis.")
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Could not connect to Redis. Aborting. Error: {e}")
        return

    try:
        credentials = service_account.Credentials.from_service_account_file(
            GOOGLE_CRED_FILE, scopes=["https://www.googleapis.com/auth/calendar.readonly"]
        )
        calendar_service = build("calendar", "v3", credentials=credentials)
        logger.info("Successfully authenticated with Google Calendar API.")
    except Exception as e:
        logger.error(f"Could not connect to Google Calendar API. Aborting. Error: {e}")
        return

    today = date.today()
    for i in range(DAYS_TO_CACHE):
        target_date = today + timedelta(days=i)
        
        cache_key = f"availability:{target_date.isoformat()}"
        
        available_slots = get_availability_for_day(calendar_service, target_date)
        
        slots_json = json.dumps(available_slots)
        redis_client.set(cache_key, slots_json, ex=CACHE_EXPIRATION_SECONDS)
        logger.info(f"Updated cache for {target_date}: {len(available_slots)} slots available.")
    
    logger.info("Cache update process finished successfully.")


if __name__ == "__main__":
    update_availability_cache()