import os
import json
import logging
import re
import asyncio
from datetime import datetime, timedelta, date, time as dtime, timezone
from typing import Dict, Any, List, Union, Optional, AsyncGenerator
import functools
# --- Environment Setup ---

from dotenv import load_dotenv
load_dotenv()

# --- Main Imports ---
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
import redis
import dateparser
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
# --- LangChain / OpenAI ---
from langchain_openai import ChatOpenAI
# from langchain_community.vectorstores import Pinecone
from langchain_pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessageChunk
from langchain.memory import ConversationBufferMemory
# This block should go right after app = FastAPI(...)
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.prompts import PromptTemplate

# Define the custom prompt for concise answers

# --- Google Calendar ---
from google.oauth2 import service_account
from googleapiclient.discovery import build

# ─── Logging & Env Loading ──────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("receptionist_agent")

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV       = os.getenv("PINECONE_ENV")
PINECONE_INDEX     = os.getenv("PINECONE_INDEX")
GOOGLE_CALENDAR_ID = "primary"
TIMEZONE = "UTC"
GOOGLE_CRED_FILE   = "credentials.json" # This is okay to be hardcoded if the file is always in the same directory

REDIS_URL = os.getenv("REDIS_URL")

if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX]):
    raise RuntimeError("A required API key or configuration is missing. Check your .env file.")

# ─── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(title="Streaming AI Receptionist with Caching and Parallel Tools", version="2.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# This line tells FastAPI to serve all files from the 'static' directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# ─── Service Clients (LLM, DB, Cache) ───────────────────────────────────────────
llm = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=OPENAI_API_KEY, streaming=True)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

try:
    if not REDIS_URL:
        raise redis.exceptions.ConnectionError("REDIS_URL environment variable not set.")
    
    # Connect using the full URL, which includes host, port, and password
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
    logger.info("Successfully connected to Redis via URL.")
    
except redis.exceptions.ConnectionError as e:
    logger.warning(f"Could not connect to Redis. Caching will be disabled. Error: {e}")
    redis_client = None
    
    
prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Keep the answer as concise as possible, ideally in one to two sentences.

Context: {context}

Question: {question}

Helpful Answer:"""

QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

try:
    vectorstore = Pinecone.from_existing_index(index_name=PINECONE_INDEX, embedding=embeddings)
    qa_chain = RetrievalQA.from_chain_type(llm=llm,retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),chain_type_kwargs={"prompt": QA_PROMPT}  )
    logger.info("Successfully connected to Pinecone index.")
except Exception as e:
    logger.error(f"Failed to initialize Pinecone. FAQs will not be available. Error: {e}")
    qa_chain = None

try:
    if vectorstore:
        vectorstore.similarity_search("initial prewarm query", k=1)
        logger.info("Pinecone prewarm successful.")
except Exception as e:
    logger.warning(f"Pinecone prewarm failed: {e}")

try:
    credentials = service_account.Credentials.from_service_account_file(
        GOOGLE_CRED_FILE, scopes=["https://www.googleapis.com/auth/calendar"]
    )
    calendar_service = build("calendar", "v3", credentials=credentials)
    logger.info("Successfully authenticated with Google Calendar API.")
except FileNotFoundError:
    logger.error(f"Google credentials file not found at '{GOOGLE_CRED_FILE}'. Calendar functions will fail.")
    calendar_service = None
except Exception as e:
    logger.error(f"Failed to initialize Google Calendar service. Error: {e}")
    calendar_service = None

# ─── Agent Tools ────────────────────────────────────────────────────────────────
def format_time_for_response(dt):
    return dt.strftime("%#I:%M %p" if os.name == 'nt' else "%-I:%M %p")

def to_rfc3339(dt: datetime) -> str:
    return dt.isoformat()

@tool
async def answer_frequently_asked_question(question: str) -> str:
    """
    Use this tool to answer questions about Efficient AI, its services, pricing, or what it does.
    It is the primary tool for any general inquiry or question about the company.
    """
    if not qa_chain:
        return "I'm sorry, my knowledge base is currently unavailable."
    logger.info(f"Answering FAQ for question: '{question}'")
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: qa_chain.invoke(question))
    return result.get('result', "I couldn't find an answer to that question.")

class CheckAvailabilityInput(BaseModel):
    day: str = Field(description="The specific day to check for availability, such as 'tomorrow', 'next Tuesday', or 'July 25th'.")

@tool(args_schema=CheckAvailabilityInput)
async def list_available_slots_for_day(day: str) -> str:
    """
    Use this tool when a user asks about availability or wants to know what times are free on a specific day.
    """
    if not calendar_service:
        return "I'm sorry, I can't check the calendar right now."

    parsed_date = dateparser.parse(day, settings={'TIMEZONE': 'UTC', 'RETURN_AS_TIMEZONE_AWARE': True})
    if not parsed_date:
        return f"I'm sorry, I didn't understand the day '{day}'. Could you be more specific?"

    target_date = parsed_date.date()
    cache_key = f"availability:{target_date.isoformat()}"
    loop = asyncio.get_running_loop()

    if redis_client:
        try:
            # MODIFICATION: Replaced asyncio.to_thread
            cached_slots_json = await loop.run_in_executor(None, redis_client.get, cache_key)
            if cached_slots_json:
                logger.info(f"Cache HIT for date: {target_date}")
                available_slots = json.loads(cached_slots_json)
                if not available_slots:
                    return f"Sorry, there are no available slots on {target_date.strftime('%A, %b %d')}."
                return f"On {target_date.strftime('%A, %b %d')}, the following times are available: {', '.join(available_slots)}."
        except Exception as e:
            logger.error(f"Redis cache read error: {e}. Falling back to API.")

    logger.info(f"Cache MISS for date: {target_date}. Fetching from Google API.")

    start_of_day = datetime.combine(target_date, dtime.min, tzinfo=timezone.utc)
    end_of_day = datetime.combine(target_date, dtime.max, tzinfo=timezone.utc)
    # ... business hours logic ...
    business_start = start_of_day.replace(hour=9)
    business_end = start_of_day.replace(hour=17)
    
    # ... Google Calendar fetching logic ...
    try:
        events_result = await loop.run_in_executor(None, lambda: calendar_service.events().list(
            calendarId=GOOGLE_CALENDAR_ID, timeMin=to_rfc3339(start_of_day), timeMax=to_rfc3339(end_of_day),
            singleEvents=True, orderBy='startTime'
        ).execute())
    except Exception as e:
        logger.error(f"Error fetching Google Calendar events: {e}")
        return "I encountered an error while trying to check the calendar."

    # ... slot calculation logic ...
    events = events_result.get('items', [])
    busy_times = []
    for event in events:
        start_str = event['start'].get('dateTime')
        end_str = event['end'].get('dateTime')
        if start_str and end_str:
            busy_times.append((dateparser.parse(start_str), dateparser.parse(end_str)))

    available_slots = []
    now_utc = datetime.now(timezone.utc)
    current_time = business_start

    if target_date == now_utc.date():
        # If checking for today, start from the current time.
        current_time = max(business_start, now_utc)
        # Round up to the next full hour to ensure we only suggest future slots.
        if current_time.minute > 0 or current_time.second > 0:
            current_time = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    while current_time < business_end:
        is_busy = any(start < current_time + timedelta(minutes=59) and end > current_time for start, end in busy_times)
        if not is_busy:
            available_slots.append(format_time_for_response(current_time))
        current_time += timedelta(hours=1)
    latest_slots = available_slots[:2]
    if redis_client:
        try:
            slots_json = json.dumps(available_slots)
            # MODIFICATION: Replaced asyncio.to_thread
            # await loop.run_in_executor(None,lambda: redis_client.set(cache_key, slots_json, ex=3600))
            set_with_expiry = functools.partial(redis_client.set, cache_key, slots_json, ex=3600)
            await loop.run_in_executor(None, set_with_expiry)
            logger.info(f"Set cache for {target_date}")
        except Exception as e:
            logger.error(f"Redis cache write error: {e}")

    if not latest_slots:
        return f"Sorry, there are no available slots on {target_date.strftime('%A, %b %d')}."

    # MODIFICATION: Update the return string to be more specific.
    return f"On {target_date.strftime('%A, %b %d')}, the two latest available times are: {', '.join(latest_slots)}."

class BookSlotInput(BaseModel):
    full_name: str = Field(description="The user's full name for the meeting invitation.")
    requested_time: str = Field(description="The specific date and time for the booking, e.g., 'tomorrow at 4pm'.")

# MODIFICATION: Removed phone_number from the function signature and body
@tool(args_schema=BookSlotInput)
async def book_demo_slot(full_name: str, requested_time: str) -> str:
    """
    Use this tool to book a new demo or meeting. Only use it when you have ALL required information.
    """
    if not calendar_service:
        return "I'm sorry, I can't book meetings right now due to a calendar connection issue."

    parsed_time = dateparser.parse(requested_time, settings={'TO_TIMEZONE': 'UTC', 'RETURN_AS_TIMEZONE_AWARE': True})
    if not parsed_time or parsed_time.time() == dtime(0, 0):
        return f"I'm sorry, I couldn't understand the time '{requested_time}'. Please provide a specific time, like '2 PM' or '14:00'."

    start_time = parsed_time
    end_time = start_time + timedelta(hours=1)
    logger.info(f"Attempting to book slot for {full_name} at {start_time}")
    loop = asyncio.get_running_loop()

    try:
        events_result = await loop.run_in_executor(None, lambda: calendar_service.events().list(
            calendarId=GOOGLE_CALENDAR_ID, timeMin=to_rfc3339(start_time), timeMax=to_rfc3339(end_time), singleEvents=True
        ).execute())
        if events_result.get('items', []):
            return f"I'm so sorry, but it looks like the {format_time_for_response(start_time)} slot on {start_time.strftime('%A, %b %d')} was just taken. Could you suggest another time?"
    except Exception as e:
        logger.error(f"Error checking for slot availability before booking: {e}")
        return "I had trouble confirming if that slot is free. Please try again."

    event_summary = f"Demo: ({full_name})"
    event_body = {
        "summary": event_summary,
        "description": f"Booked via AI Assistant for {full_name}.", # Removed phone number from description
        "start": {"dateTime": to_rfc3339(start_time), "timeZone": "UTC"},
        "end": {"dateTime": to_rfc3339(end_time), "timeZone": "UTC"},
    }
    try:
        created_event = await loop.run_in_executor(None, lambda: calendar_service.events().insert(calendarId=GOOGLE_CALENDAR_ID, body=event_body).execute())
        logger.info(f"Successfully booked event ID: {created_event['id']}")
        if redis_client:
            cache_key = f"availability:{start_time.date().isoformat()}"
            await loop.run_in_executor(None, redis_client.delete, cache_key)
            logger.info(f"Cache invalidated for {start_time.date()}")
        first_name = full_name.split()[0]
        return f"All set, {first_name}! I have booked your demo for {start_time.strftime('%A, %b %d at')} {format_time_for_response(start_time)}. Is there anything else I can help with?"
    except Exception as e:
        logger.error(f"Error creating Google Calendar event: {e}")
        return "I'm sorry, I encountered an unexpected error while trying to book the meeting."



class RescheduleSlotInput(BaseModel):
    full_name: str = Field(description="The user's full name, which was used for the original booking.")
    original_time: str = Field(description="The original date and time of the meeting they want to reschedule, e.g., 'tomorrow at 2pm' or 'my meeting on Friday'.")
    new_time: str = Field(description="The new desired date and time for the meeting, e.g., 'next Monday at 10am'.")

@tool(args_schema=RescheduleSlotInput)
async def reschedule_demo_slot(full_name: str, original_time: str, new_time: str) -> str:
    """
    Use this tool to reschedule an existing demo or meeting.
    You must collect the user's full name, the original meeting time, and the new desired time before calling this tool.
    """
    if not calendar_service:
        return "I'm sorry, I can't modify the calendar right now."

    parsed_original = dateparser.parse(original_time, settings={'TO_TIMEZONE': 'UTC', 'RETURN_AS_TIMEZONE_AWARE': True})
    if not parsed_original:
        return f"I'm sorry, I didn't understand the original meeting time: '{original_time}'."

    parsed_new = dateparser.parse(new_time, settings={'TO_TIMEZONE': 'UTC', 'RETURN_AS_TIMEZONE_AWARE': True})
    if not parsed_new:
        return f"I'm sorry, I didn't understand the new time you requested: '{new_time}'."

    logger.info(f"Attempting to reschedule meeting for {full_name} from {parsed_original} to {parsed_new}")
    loop = asyncio.get_running_loop()

    search_start = datetime.combine(parsed_original.date(), dtime.min, tzinfo=timezone.utc)
    search_end = datetime.combine(parsed_original.date(), dtime.max, tzinfo=timezone.utc)
    
    try:
        events_result = await loop.run_in_executor(None, lambda: calendar_service.events().list(
            calendarId=GOOGLE_CALENDAR_ID, timeMin=to_rfc3339(search_start), timeMax=to_rfc3339(search_end), singleEvents=True
        ).execute())
    except Exception as e:
        logger.error(f"Error searching for original event: {e}")
        return "I had trouble searching for your original meeting. Please try again."

    original_event = None
    for event in events_result.get('items', []):
        start_str = event['start'].get('dateTime')
        event_start_time = dateparser.parse(start_str)
        if full_name.lower() in event.get('summary', '').lower() and abs(event_start_time - parsed_original) < timedelta(minutes=15):
            original_event = event
            break
    
    if not original_event:
        return f"I'm sorry, I couldn't find a meeting for '{full_name}' around {format_time_for_response(parsed_original)} on {parsed_original.strftime('%A, %b %d')}."

    new_start_time = parsed_new
    new_end_time = new_start_time + timedelta(hours=1)
    
    try:
        events_result = await loop.run_in_executor(None, lambda: calendar_service.events().list(
            calendarId=GOOGLE_CALENDAR_ID, timeMin=to_rfc3339(new_start_time), timeMax=to_rfc3339(new_end_time), singleEvents=True
        ).execute())
        for event in events_result.get('items', []):
            if event['id'] != original_event['id']:
                return f"I'm so sorry, but the new time you requested is already booked."
    except Exception as e:
        logger.error(f"Error checking new slot availability: {e}")
        return "I had trouble confirming if the new slot is free."

    original_event['start']['dateTime'] = to_rfc3339(new_start_time)
    original_event['end']['dateTime'] = to_rfc3339(new_end_time)

    try:
        updated_event = await loop.run_in_executor(None, lambda: calendar_service.events().update(
            calendarId=GOOGLE_CALENDAR_ID, eventId=original_event['id'], body=original_event
        ).execute())
        logger.info(f"Successfully rescheduled event ID: {updated_event['id']}")

        if redis_client:
            old_cache_key = f"availability:{parsed_original.date().isoformat()}"
            new_cache_key = f"availability:{new_start_time.date().isoformat()}"
            # MODIFICATION: Replaced asyncio.to_thread
            await loop.run_in_executor(None, redis_client.delete, old_cache_key, new_cache_key)
            logger.info(f"Cache invalidated for {parsed_original.date()} and {new_start_time.date()}")

        return f"I've successfully rescheduled your meeting to {new_start_time.strftime('%A, %b %d at')} {format_time_for_response(new_start_time)}."
    except Exception as e:
        logger.error(f"Error updating event: {e}")
        return "I'm sorry, I encountered an error while rescheduling your meeting."
    
    
# ─── Agent Definition ───────────────────────────────────────────────────────────
AGENT_PROMPT = """
You are Anna, a friendly and highly efficient AI virtual assistant for a company called "Efficient AI".
Your goal is to be helpful, polite, and conversational.
You are responsible for:
1. Answering questions about "Efficient AI".
2. Checking calendar availability for demos.
3. Booking new demos for potential clients.
4. Rescheduling existing demos for clients.

Today's date is {today}.

IMPORTANT INSTRUCTIONS:
- **No Initial Greeting**: The user has already been greeted. Do not say "Hello," "Hi," or introduce yourself. Jump directly into answering the user's question or fulfilling their request.
- **Check Conversation History First**: If a user asks about a time they have just booked in the current conversation, check the chat history. If you've already confirmed their booking, remind them of it directly instead of using a tool. For example: "Yes, that slot is for you! I have you down for Tuesday at 2 PM."
- **Slot-First Booking Process**: When a user asks to book a specific time (e.g., "Book 3 PM tomorrow"), you MUST follow this sequence:
    1. First, use the `list_available_slots_for_day` tool to confirm that specific time is actually available.
    2. If the slot is available, THEN you should ask the user for their full name.
    3. Once you have all details (time, name), you can finally call the `book_demo_slot` tool.
- When a user wants to reschedule a meeting, you must have their full name, the original meeting time, and the new desired time before calling the `reschedule_demo_slot` tool.
- **IMPORTANT: Before asking for these details, you MUST FIRST check the conversation history for a recently booked appointment. If you find one for the user, confirm those details with them (e.g., "I see you have a demo booked for tomorrow at 4 PM, is that the one you'd like to change?"). Only ask for information you are missing.**
- DO NOT make up information. If you don't have the information, you MUST ask the user for it.
- Be conversational. For example, say "Let me check the calendar for you" before using a tool.
- If a user's request is ambiguous, ask for clarification.
- If you book or reschedule a meeting successfully, always ask if there is anything else I can help with.
- **Be Concise**: Keep all responses as short and to-the-point as possible. Do not use filler phrases or unnecessary pleasantries.
- **Gather Information Incrementally**: If you have some details for a booking (like the time) but are missing others (like the name), only ask for the specific information you are missing. Do not re-ask for details the user has already provided.
"""

tools = [
    answer_frequently_asked_question,
    list_available_slots_for_day,
    book_demo_slot,
    reschedule_demo_slot,
]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", AGENT_PROMPT.format(today=date.today().strftime("%A, %B %d, %Y"))),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_openai_tools_agent(llm, tools, prompt)

_sessions: Dict[str, AgentExecutor] = {}
_memories: Dict[str, ConversationBufferMemory] = {}

class MessageIn(BaseModel):
    session_id: str
    text: str

# async def stream_agent_response(session_id: str, user_input: str) -> AsyncGenerator[str, None]:
#     """
#     Handles the agent interaction with token-level streaming and correct session initialization.
#     """
#     if session_id not in _sessions:
#         logger.info(f"Creating new session: {session_id}")
#         _memories[session_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#         _sessions[session_id] = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=_memories[session_id])

#     agent_executor = _sessions[session_id]

#     try:
#         async for chunk in agent_executor.astream_log({"input": user_input}):
#             for op in chunk.ops:
#                 path = op.get("path", "")
#                 if path == "/streamed_output/-":
#                     value = op.get("value")
#                     if isinstance(value, AIMessageChunk):
#                         yield value.content

#     except Exception as e:
#         logger.error(f"Error during agent streaming for session {session_id}: {e}", exc_info=True)
#         yield "I'm sorry, a technical error occurred. Please try again in a moment."


async def stream_agent_response(session_id: str, user_input: str) -> AsyncGenerator[str, None]:
    """
    Handles the agent interaction with token-level streaming and correct session initialization
    using Redis for persistent, multi-worker-safe conversation history.
    """
    if not redis_client:
        yield "I'm sorry, my memory system is currently unavailable. I cannot access our past conversation."
        return

    # 1. Initialize history from Redis for the current session
    message_history = RedisChatMessageHistory(session_id=session_id, url=str(REDIS_URL))

    # 2. Create a memory object with the Redis-backed history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=message_history,
        return_messages=True,
        output_key="output"
    )

    # 3. Create the agent executor for this specific request
    executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, memory=memory
    )

    try:
        # astream() yields dictionaries containing the output. We need to parse them.
        async for chunk in executor.astream({"input": user_input}):
            # The agent's final answer is in the 'output' key.
            if "output" in chunk:
                yield chunk["output"]
    except Exception as e:
        logger.error(f"Error during agent streaming for session {session_id}: {e}", exc_info=True)
        yield "I'm sorry, a technical error occurred. Please try again in a moment."


@app.post("/stream_message")
async def stream_message_endpoint(msg: MessageIn):
    return StreamingResponse(
        stream_agent_response(msg.session_id, msg.text),
        media_type="text/event-stream",  # or "text/plain"
    )

class SessionIdIn(BaseModel):
    session_id: str

@app.post("/reset_session")
async def reset_session(payload: SessionIdIn):
    """Endpoint to reset a user's session and conversation history."""
    session_id = payload.session_id
    if session_id in _sessions:
        del _sessions[session_id]
        del _memories[session_id]
        logger.info(f"Session {session_id} has been reset.")
        return {"status": "success", "message": f"Session {session_id} reset."}
    return {"status": "error", "message": "Session not found."}


@app.get("/")
async def read_root():
    return FileResponse('static/test.html')


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("V7parallel:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)