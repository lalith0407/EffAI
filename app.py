import asyncio
import base64
import json
import sys
import os
from aiohttp import web, WSMsgType, ClientSession, TCPConnector
import websockets
import audioop # Imported but not directly used in this logic
from dotenv import load_dotenv
import uuid
import re
import aiohttp

AGENT_SERVER_URL = "https://test-little-voice-4200.fly.dev/stream_message"
# Load secrets from .env if available
load_dotenv()
sys.stdout.reconfigure(line_buffering=True)

# --- Configuration ---
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "b47673813f12f3245e3805eb4f547d1e4e78e69e") # Replace with your actual key or load from .env
# ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "sk_bb292bc19795a159078e963a997835ab063de7b597b8966a")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "sk_46f694dd001701f3f2a8930bab1a931f8d15bc5a17d12577")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "cgSgspJ2msm6clMCkdW9")

if not DEEPGRAM_API_KEY:
    print("CRITICAL ERROR: DEEPGRAM_API_KEY not found in environment variables or .env file.")
    sys.exit(1)
if not ELEVENLABS_API_KEY:
    print("CRITICAL ERROR: ELEVENLABS_API_KEY not found in environment variables or .env file.")
    sys.exit(1)

DEEPGRAM_URL = (
    "wss://api.deepgram.com/v1/listen?"
    "encoding=mulaw&sample_rate=8000&channels=1&language=en-US"
    "&interim_results=true&punctuate=false&smart_format=false"
    "&utterance_end_ms=1200" # CORRECTED: was utterance_end
    "&endpointing=4000"
    "&model=nova-2-phonecall"
)

# --- ElevenLabs TTS Class ---
class ElevenLabsTTS:
    def __init__(self, sw_websocket, stream_sid):
        self.sw_websocket = sw_websocket
        self.stream_sid = stream_sid
        self.el_ws = None
        self.context_id = f"ctx_{uuid.uuid4().hex[:8]}"
        self.active = True
        self.last_text = ""

    async def connect(self):
        url = (
            f"wss://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/"
            f"multi-stream-input?output_format=ulaw_8000"
            f"&optimize_streaming_latency=4"
            f"&auto_mode=true"
            f"&inactivity_timeout=600"
            f"&apply_text_normalization=on"
        )
        try:
            self.el_ws = await websockets.connect(
                url,
                extra_headers={"xi-api-key": ELEVENLABS_API_KEY},
                ping_interval=30,
                ping_timeout=60
            )
            await self.el_ws.send(json.dumps({
                "text": " ", # Initial handshake with minimal silence
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.8,"speaking_rate": 0.8},
                "context_id": self.context_id
            }))
            print("🎧 ElevenLabs (ultra-low-latency) connection established")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to connect to ElevenLabs: {e}", flush=True)
            self.active = False

    async def send_text(self, text):
        if not self.el_ws or not self.active:
            print("ElevenLabs WebSocket not active, cannot send text.")
            return
        if not text.strip() or text == self.last_text:
            return

        try:
            await self.el_ws.send(json.dumps({
                "text": text,
                "context_id": self.context_id,
                "flush": True
            }))
            self.last_text = text
            print(f"🔊 Sent text to ElevenLabs: \"{text[:40]}{'...' if len(text) > 40 else ''}\"", flush=True)
        except websockets.ConnectionClosed:
            print("ElevenLabs WebSocket connection closed, cannot send text.", flush=True)
            self.active = False
        except Exception as e:
            print(f"⚠️ Error sending text to ElevenLabs: {e}", flush=True)

    async def audio_forwarder(self):
        while self.active:
            try:
                response = await asyncio.wait_for(self.el_ws.recv(), timeout=10.0)
                data = json.loads(response)
                if "audio" in data and data["audio"]:
                    await self.sw_websocket.send_str(json.dumps({
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {"payload": data["audio"]}
                    }))
                elif "is_final" in data and data["is_final"]:
                    print("ElevenLabs finished synthesizing current text segment.", flush=True)
                elif "error" in data:
                    print(f"ElevenLabs error: {data['error']}", flush=True)

            except asyncio.TimeoutError:
                pass
            except websockets.ConnectionClosed:
                print("ElevenLabs audio forwarder detected connection closed.", flush=True)
                self.active = False
                break
            except Exception as e:
                print(f"⚠️ Error in ElevenLabs audio forwarder: {e}", flush=True)
                self.active = False
                break

    async def run(self):
        await self.connect()
        if self.active:
            asyncio.create_task(self.audio_forwarder())

    async def close(self):
        self.active = False
        if self.el_ws:
            try:
                await self.el_ws.close()
                print("ElevenLabs connection closed.", flush=True)
            except Exception as e:
                print(f"Error closing ElevenLabs connection: {e}", flush=True)
        self.el_ws = None
        
async def prewarm_agent(agent_session_id: str):
    """Sends a dummy request to the agent to wake it up and load models."""
    print("🚀 Pre-warming agent server...", flush=True)
    payload = {"session_id": agent_session_id, "text": "Health check"}
    try:
        async with aiohttp.ClientSession() as session:
            # Set a long timeout to account for cold starts
            async with session.post(AGENT_SERVER_URL, json=payload, timeout=30) as resp:
                if resp.status == 200:
                    await resp.read() # Consume the response to free up the connection
                    print("✅ Agent pre-warmed successfully.", flush=True)
                else:
                    error_text = await resp.text()
                    print(f"⚠️ Agent pre-warm failed with status {resp.status}: {error_text}", flush=True)
    except asyncio.TimeoutError:
        print("⚠️ Agent pre-warm timed out. The server might be slow to start.", flush=True)
    except Exception as e:
        print(f"⚠️ An exception occurred during agent pre-warm: {e}", flush=True)


# --- Deepgram Management Class ---
class DeepgramASR:
    def __init__(self, api_key: str, dg_url: str, audio_queue: asyncio.Queue, tts_client: ElevenLabsTTS, call_shutdown_event: asyncio.Event,stream_sid: str, agent_session_id: str, caller_id: str):
        self.api_key = api_key
        self.dg_url = dg_url
        self.audio_queue = audio_queue
        self.tts_client = tts_client
        self.call_shutdown_event = call_shutdown_event
        self.dg_ws = None
        self.active = False
        self.connector = TCPConnector(force_close=True, enable_cleanup_closed=True, ttl_dns_cache=300)
        self.session = ClientSession(connector=self.connector)
        self.stream_sid = stream_sid
        self.http_session = aiohttp.ClientSession()
        self.agent_session_id = agent_session_id 
        self.transcript_buffer = "" # ADD THIS LINE
        self.caller_id = caller_id
        
    async def stream_agent_to_tts(self, transcript: str):
        """
        Connects to the agent server, gets the streaming response, and sends it
        to the TTS client sentence by sentence.
        """
        payload = {"session_id": self.agent_session_id, "text": transcript, "caller_id": self.caller_id}
        print(payload)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(AGENT_SERVER_URL, json=payload) as resp:
                    if resp.status == 200:
                        buffer = ""
                        # Read the streaming response chunk by chunk
                        async for chunk in resp.content.iter_any():
                            buffer += chunk.decode('utf-8')
                            
                            # Use regex to find sentence boundaries (. ! ?)
                            # This provides a more natural cadence for the TTS
                            sentence_match = re.search(r'(?<=[.!?])\s*', buffer)
                            if sentence_match:
                                sentence = buffer[:sentence_match.end()]
                                buffer = buffer[sentence_match.end():]
                                if sentence.strip():
                                    await self.tts_client.send_text(sentence)
                        
                        # Send any remaining text in the buffer after the stream ends
                        if buffer.strip():
                            await self.tts_client.send_text(buffer)
                    else:
                        error_text = await resp.text()
                        print(f"Error from agent server: {resp.status} - {error_text}")
                        await self.tts_client.send_text("I'm having a little trouble thinking right now.")
        except Exception as e:
            print(f"Error connecting to agent server: {e}")
            await self.tts_client.send_text("I'm unable to connect to my brain at the moment.")
            
            
    async def connect(self):
        try:
            self.dg_ws = await self.session.ws_connect(
                self.dg_url,
                headers={"Authorization": f"Token {self.api_key}"}
            )
            self.active = True
            print("✅ Deepgram ASR connection established.", flush=True)
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Deepgram: {e}", flush=True)
            self.active = False
            return False

    async def send_audio_to_deepgram(self):
        while self.active and not self.call_shutdown_event.is_set():
            try:
                audio_chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=1.0)
                if audio_chunk is None: # Signal to stop
                    break
                if not self.dg_ws.closed:
                    await self.dg_ws.send_bytes(audio_chunk)
                else:
                    print("Deepgram WS closed while sending audio, dropping chunk.", flush=True)
            except asyncio.TimeoutError:
                continue # No audio, just keep waiting/checking shutdown
            except Exception as e:
                print(f"⚠️ Error sending audio to Deepgram: {e}", flush=True)
                break # Break on send error, will trigger reconnect

    async def receive_transcripts_from_deepgram(self):
        while self.active and not self.call_shutdown_event.is_set():
            try:
                msg = await asyncio.wait_for(self.dg_ws.receive(), timeout=5.0)
                if msg.type == WSMsgType.TEXT:
                    dg_data = json.loads(msg.data)
                    
                    # ## MODIFIED LOGIC STARTS HERE ##
                    
                    is_utterance_end = dg_data.get("type") == "UtteranceEnd"
                    is_final = False
                    
                    # First, try to get a transcript ONLY if it's not an UtteranceEnd message.
                    if not is_utterance_end:
                        transcript = dg_data.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")
                        is_final = dg_data.get("is_final", False)
                        if transcript:
                            self.transcript_buffer = transcript

                    # Now, check the trigger conditions with the potentially updated buffer.
                    if self.transcript_buffer and (is_utterance_end or is_final):
                        trigger_reason = "final transcript" if is_final else "utterance end"
                        print(f"Triggering agent on {trigger_reason}: '{self.transcript_buffer}'")

                        asyncio.create_task(self.stream_agent_to_tts(self.transcript_buffer))
                        
                        # Reset the buffer to prevent sending the same text again.
                        self.transcript_buffer = ""

                elif msg.type == WSMsgType.CLOSE:
                    print("Deepgram WebSocket closed by server during transcript reception.", flush=True)
                    self.active = False
                    break
                elif msg.type == WSMsgType.ERROR:
                    print(f"Deepgram WebSocket error: {msg.data}. Signalling reconnect.", flush=True)
                    self.active = False
                    break
            except asyncio.TimeoutError:
                pass # No transcript, keep listening
            except Exception as e:
                print(f"⚠️ Error receiving transcript from Deepgram: {e}", flush=True)
                self.active = False
                break
            

    async def run(self):
        while not self.call_shutdown_event.is_set():
            if not await self.connect():
                print("Deepgram connection failed. Retrying in 1 second...", flush=True)
                await asyncio.sleep(1)
                continue

            # We successfully connected, start the send/receive tasks for this connection
            tasks_for_dg_session = [
                self.send_audio_to_deepgram(),
                self.receive_transcripts_from_deepgram(),
                # deepgram_keepalive is implicitly handled by DG's endpointing and auto-disconnect.
                # If DG doesn't disconnect, it's fine. If it does, we reconnect.
            ]
            
            # Create tasks explicitly to avoid DeprecationWarning
            dg_session_tasks = [asyncio.create_task(t) for t in tasks_for_dg_session]
            
            # Wait for any of these to complete (i.e., connection closes, error, or shutdown)
            done, pending = await asyncio.wait(dg_session_tasks, return_when=asyncio.FIRST_COMPLETED)
            
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True) # Ensure cleanup

            if self.call_shutdown_event.is_set():
                print("Deepgram ASR run loop exiting due to call shutdown.", flush=True)
                break
            else:
                print("Deepgram ASR connection ended. Attempting to re-establish for next turn.", flush=True)
                # Small delay before reconnecting if not a full call shutdown
                await asyncio.sleep(0.5)

    async def close(self):
        self.active = False
        if self.dg_ws:
            try:
                await self.dg_ws.close()
                print("Deepgram ASR connection closed gracefully.", flush=True)
            except Exception as e:
                print(f"Error closing Deepgram WS: {e}", flush=True)
        if self.session:
            await self.session.close()
            print("Deepgram ASR client session closed.", flush=True)


# --- SignalWire Audio Handler ---
async def handle_audio(ws):
    print("🎧 SignalWire client connected!", flush=True)
    caller_id: str = None
    # Queue for audio data from SignalWire to Deepgram
    audio_data_queue = asyncio.Queue()
    
    tts_client: ElevenLabsTTS = None
    stream_sid: str = None

    # Event to signal the overall call termination
    call_shutdown_event = asyncio.Event()
    agent_session_id = f"call_{uuid.uuid4().hex}"
    print(f"Generated new Agent Session ID for this call: {agent_session_id}")
    dg_asr: DeepgramASR = None # Placeholder for Deepgram ASR instance

    async def signalwire_audio_reader():
        nonlocal tts_client, stream_sid, caller_id # To update outer scope variables
        try:
            async for msg in ws: # Listen for messages from SignalWire
                if call_shutdown_event.is_set():
                    break # Stop if overall call is shutting down

                if msg.type == WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    event_type = data.get("event")

                    if event_type == "start":
                        print(f"SignalWire Start Event Data: {json.dumps(data, indent=2)}")
                        asyncio.create_task(prewarm_agent(agent_session_id))
                        stream_sid = data["start"]["streamSid"]
                        caller_id = data.get("start", {}).get("from")
                        if not caller_id:
                            caller_id = data.get("start", {}).get("customParameters", {}).get("caller_number")

                        print(f"Call received from number: {caller_id}", flush=True)
                        print(f"SignalWire stream started with SID: {stream_sid}", flush=True)
                        if not tts_client: # Initialize TTS client only once per call
                            tts_client = ElevenLabsTTS(ws, stream_sid)
                            await tts_client.run()
                            # Send initial greeting here
                            await tts_client.send_text("Hi, this is Anna with efficient Ai, I am here to answer any questions, help you book a demo, or share more about our services. How can I help you today?")
                            #### Hi there this is Anna from efficient AI. How can I help you today? ####
                        
                        # Initialize Deepgram ASR after SignalWire stream starts and TTS is ready
                        nonlocal dg_asr
                        dg_asr = DeepgramASR(DEEPGRAM_API_KEY, DEEPGRAM_URL, audio_data_queue, tts_client, call_shutdown_event,stream_sid=stream_sid,agent_session_id=agent_session_id, caller_id=caller_id)
                        # Start Deepgram ASR in the background. It will handle its own reconnects.
                        asyncio.create_task(dg_asr.run())


                    elif event_type == "media":
                        payload = base64.b64decode(data["media"]["payload"])
                        await audio_data_queue.put(payload) # Put audio into queue for Deepgram to pick up

                    elif event_type == "stop":
                        print("SignalWire 'stop' event received. Signaling call shutdown.", flush=True)
                        call_shutdown_event.set() # Signal overall call end
                        break # Exit this loop

                elif msg.type == WSMsgType.CLOSE:
                    print("SignalWire WebSocket closed by client. Signaling call shutdown.", flush=True)
                    call_shutdown_event.set() # Signal overall call end
                    break # Exit loop on client close

                elif msg.type == WSMsgType.ERROR:
                    print(f"SignalWire WebSocket error: {msg.data}. Signaling call shutdown.", flush=True)
                    call_shutdown_event.set()
                    break

        except Exception as e:
            print(f"⚠️ Critical error in signalwire_audio_reader: {e}", flush=True)
            call_shutdown_event.set() # Ensure full shutdown on critical error
        finally:
            print("SignalWire audio reader task finished.", flush=True)
            # Signal the Deepgram sender to stop by putting None (poison pill)
            await audio_data_queue.put(None) 


    # Run the SignalWire audio reader task in the background
    audio_reader_task = asyncio.create_task(signalwire_audio_reader())
    
    # Wait for the entire call to shut down
    await call_shutdown_event.wait()

    print("Call shutdown event triggered. Starting final cleanup...", flush=True)

    # Final cleanup after the entire call session ends
    try:
        # Cancel the audio reader task if it's still running
        audio_reader_task.cancel()
        try:
            await audio_reader_task # Wait for it to finish canceling/cleanup
        except asyncio.CancelledError:
            pass # Expected if cancelled

        # Close Deepgram ASR client if it was initialized
        if dg_asr:
            await dg_asr.close()
        
        # Close ElevenLabs client if it was initialized
        if tts_client:
            await tts_client.close()
        
        # Log final transcript (this would ideally be gathered from an LLM interaction, not just Deepgram's output)
        # For simplicity, we removed the `final_transcripts` list as it was mostly for debugging Deepgram's internal state.
        # In a multi-turn conversation, you'd collect LLM responses here.
        print("✅ Call session ended.", flush=True)
        
        # Ensure SignalWire WebSocket is closed
        if not ws.closed:
            try:
                await ws.close()
                print("🔌 SignalWire WebSocket connection ended cleanly.", flush=True)
            except Exception as e:
                print(f"Error closing SignalWire WS during cleanup: {e}", flush=True)

    except Exception as e:
        print(f"🔴 Fatal error during final cleanup: {e}", flush=True)


# --- Aiohttp Application Setup ---

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    print(f"New SignalWire connection received from: {request.remote}", flush=True)
    await handle_audio(ws)
    print("🔌 SignalWire WebSocket handler finished.", flush=True)
    return ws

async def root_handler(request):
    return web.Response(text="✅ EfficientAI phone assistant is running!")

app = web.Application()
app.router.add_get("/", root_handler)
app.router.add_get("/ws", websocket_handler)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    print(f"🚀 App starting on ws://0.0.0.0:{port}/ws", flush=True)
    web.run_app(app, port=port)

