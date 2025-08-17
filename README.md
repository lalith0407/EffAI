AI Receptionist Platform

Efficient AI is a production-grade, low-latency voice AI stack designed to act as a 24/7 virtual receptionist. The system integrates real-time speech pipelines with LangChain-driven orchestration to deliver fully automated phone conversations that can book, reschedule, and manage customer interactions with human-like fluency.

🔧 Architecture Overview

Speech In/Out

Deepgram ASR for low-latency speech-to-text on streaming audio.

ElevenLabs TTS (ulaw_8000) for sub-200ms speech synthesis with natural prosody.

AI Orchestration Layer

LangChain Agent with custom tools for FAQs, scheduling, and rescheduling.

OpenAI GPT-4o as the LLM core, configured for streaming token-level output.

Pinecone Vector DB for retrieval-augmented generation (RAG) on company FAQs.

RedisChatMessageHistory for persistent, multi-worker conversation state.

Scheduling/Integration

Google Calendar API for real-time slot management and bookings.

Redis caching layer for availability lookups + invalidation on booking events.

Service Mesh

FastAPI for agent endpoints + LangChain integration.

Aiohttp WebSocket server for SignalWire call streaming.

Uvicorn as ASGI runtime, with coroutine-driven concurrency.

📡 Data Flow

Caller audio is streamed via SignalWire → queued to Deepgram ASR.

ASR transcripts are buffered + segmented → dispatched to LangChain Agent.

LangChain Agent invokes tools (Pinecone RAG, Calendar API, Redis cache).

Streaming agent output is sentence-segmented → piped into ElevenLabs TTS.

Audio frames are streamed back in real-time to the caller.

⚙️ Key Features

True end-to-end streaming conversation pipeline (WS → ASR → LLM → TTS).

Agentic orchestration with LangChain + tool-calling for structured actions.

Cache-aware calendar availability lookups (Redis + Google Calendar).

Fault-tolerant design: reconnection loops for Deepgram and ElevenLabs, poison-pill queues for graceful shutdown.

Scalable state management with Redis-backed message history for distributed workers.
