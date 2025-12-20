import time
import os
import base64
import json
import requests
import logging
import asyncio
import threading
import math
import struct
import re
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
from google import genai
import websockets

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Configuration
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/voice-agent', methods=['POST'])
def voice_agent():
    start_time = time.time()
    logger.info("Received request for /voice-agent")
    try:
        data = request.json
        if not data or 'audio_base64' not in data:
            logger.warning("Missing audio_base64 in request body")
            return jsonify({"error": "Missing audio_base64 in request body"}), 400

        audio_base64 = data['audio_base64']
        audio_binary = base64.b64decode(audio_base64)

        # 1. Sarvam Speech to Text
        stt_response = sarvam_stt(audio_binary)
        if not stt_response or 'transcript' not in stt_response:
            logger.error(f"STT Failed or no transcript: {stt_response}")
            return jsonify({"error": "Failed to transcribe audio", "details": stt_response}), 500
        
        transcript = stt_response['transcript']
        logger.info(f"Transcript received: {transcript}")

        # 2. Gemini Call Agent
        print("Calling Gemini Agent...")
        gemini_response = gemini_agent(transcript)
        
        # 3. Parse Gemini Response
        agent_data = parse_gemini_response(gemini_response)
        print(f"Gemini response parsed: {agent_data}")
        
        # 4. Sarvam Text to Speech
        print("Calling Sarvam TTS...")
        tts_response = sarvam_tts(agent_data['response_text'])
        
        # Final Response
        audio_data = tts_response.get('audios', [None])[0] if tts_response else None
        
        end_time = time.time()
        total_delay = end_time - start_time
        
        # Consolidated Log for Debugging
        print("--- Voice Agent Response Summary ---")
        print(f"1. STT Transcript: {transcript}")
        print(f"2. Agent Response Text: {agent_data['response_text']}")
        print(f"3. TTS Audio: {'RECEIVED' if audio_data else 'MISSING'} (Length: {len(audio_data) if audio_data else 0})")
        print(f"4. TOTAL DELAY: {total_delay:.2f} seconds")
        print("-------------------------------------")

        return jsonify({
            "audio": audio_data,
            "transcript": agent_data['response_text'],
            "intent": agent_data.get('intent'),
            "action": agent_data.get('action'),
            "input_transcript": transcript,
            "latency_seconds": total_delay
        })

    except Exception as e:
        print(f"Unhandled exception in /voice-agent: {str(e)}")
        return jsonify({"error": str(e)}), 500

def sarvam_stt(audio_binary):
    url = "https://api.sarvam.ai/speech-to-text"
    headers = {"api-subscription-key": SARVAM_API_KEY}
    
    # Strictly matching the curl -F fields in order
    data = [
        ('model', 'saarika:v2.5'),
        ('language_code', 'en-IN')
    ]
    
    files = {
        'file': ('file.wav', audio_binary, 'audio/wav')
    }

    print(f"STT Request URL: {url}")
    print(f"STT Payload Data: {data}")
    
    try:
        # requests will automatically set Content-Type to multipart/form-data with boundary
        response = requests.post(url, headers=headers, files=files, data=data)
        
        print(f"STT Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("STT Transcription successful")
            return result
        else:
            print(f"STT API Error ({response.status_code}): {response.text}")
            return {"error": response.text, "status_code": response.status_code}
            
    except Exception as e:
        print(f"STT Request failed with exception: {e}")
        return {"error": str(e), "status_code": 500}

def gemini_agent(transcript):
    model_name = 'gemini-2.5-flash-lite'
    print(f"Generating content with Gemini model: {model_name}")
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=transcript,
            config={
                'system_instruction': "You are a Senior Sales Executive for TrackerBI, an AI Audio Analytics platform powered by VocalIQ. Your goal is to convert Call Center Managers by highlighting how we provide 100% call coverage and detect revenue leakage. Lead with discovery questions about their current quality monitoring. Never invent pricing; say it is customized. Respond in JSON format with: response_text (2 sentences max, consultative and ROI-focused), intent, action, and language. Plain text only."
            }
        )
        return response.text
    except Exception as e:
        print(f"Error in gemini_agent with model {model_name}: {str(e)}")
        raise e

def parse_gemini_response(text):
    # Remove markdown code blocks if present
    cleaned_text = text.replace('```json', '').replace('```', '').strip()
    try:
        return json.loads(cleaned_text)
    except Exception as e:
        return {
            "response_text": "I apologize, there was a technical issue. Please repeat your query.",
            "intent": "error",
            "action": "none",
            "error": str(e),
            "raw_response": text
        }

def sarvam_tts(text):
    url = "https://api.sarvam.ai/text-to-speech"
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "target_language_code": "en-IN",
        "speaker": "manisha",
        "pitch": 0,
        "pace": 1,
        "loudness": 1,
        "speech_sample_rate": 16000,
        "enable_preprocessing": True,
        "model": "bulbul:v2",
        "inputs": [text],
        "output_audio_format": "wav"
    }
    
    print(f"Making TTS request to {url}")
    try:
        response = requests.post(url, headers=headers, json=payload)
        print(f"TTS Response Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"TTS Response Keys: {list(result.keys())}")
            if 'audios' in result and result['audios']:
                audio_str = result['audios'][0]
                print(f"TTS Audio received, length: {len(audio_str)}")
                print(f"Audio prefix: {audio_str[:50]}")
            return result
        else:
            print(f"TTS Error Response: {response.text}")
            return None
    except Exception as e:
        print(f"Exception during TTS request: {str(e)}")
        return None

@app.route('/voice-agent', methods=['POST'])
def voice_agent_legacy():
    # Keeping old REST endpoint for compatibility if needed
    return voice_agent() # Calls the legacy REST function

# --- Real-time WebSocket Logic ---
from sarvamai import AsyncSarvamAI
from google.genai import types as genai_types

# Initialize Gemini Client (Sync client but we use its .aio property)
genai_client = genai.Client(api_key=GEMINI_API_KEY)
gemini_async_client = genai_client.aio

class SarvamSession:
    def __init__(self, sid):
        self.sid = sid
        self.sarvam_client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
        self.stt_ctx = None
        self.stt_stream = None
        self.tts_ctx = None
        self.tts_stream = None
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        self.is_ai_speaking = False
        self.is_active = True
        self.stt_connected = False
        self.tts_connected = False
        self.speech_ended_flag = False # Track END_SPEECH for finalizing transcripts
        self.last_processed_transcript = "" # Prevent double trigger
        self.history = [] # Conversation history for Gemini
        self.transcript_buffer = [] # Buffer for concatenating split speech
        self.llm_task = None # Track current LLM task for cancellation
        
        # Latency & Control tracking
        self.speech_end_time = 0
        self.ai_response_start_time = 0 # To prevent instant barge-in
        self.llm_start_time = 0
        self.llm_first_chunk_time = 0
        self.tts_first_audio_time = 0
        
    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def connect_stt(self):
        if self.stt_connected:
            print(f"--- [STT] Handshake already active for {self.sid} ---")
            return
        print(f"--- [STT] Connecting with SDK for {self.sid} ---")
        try:
            self.stt_ctx = self.sarvam_client.speech_to_text_streaming.connect(
                language_code="en-IN",
                model="saarika:v2.5",
                input_audio_codec="pcm_s16le", 
                sample_rate="16000",
                high_vad_sensitivity=False, # Reduced sensitivity to prevent noise-based barge-in
                vad_signals=True,
                flush_signal=True
            )
            self.stt_stream = await self.stt_ctx.__aenter__()
            self.stt_connected = True
            print(f"--- [STT] SDK Connected for session {self.sid} ---")
            asyncio.run_coroutine_threadsafe(self.receive_stt(), self.loop)
            asyncio.run_coroutine_threadsafe(self.ping_loop("stt"), self.loop)
        except Exception as e:
            print(f"--- [STT] SDK Connection Failed: {e} ---")

    async def connect_tts(self):
        if self.tts_connected:
            print(f"--- [TTS] Handshake already active for {self.sid} ---")
            return
        print(f"--- [TTS] Connecting with SDK for {self.sid} ---")
        try:
            self.tts_ctx = self.sarvam_client.text_to_speech_streaming.connect(model="bulbul:v2")
            self.tts_stream = await self.tts_ctx.__aenter__()
            self.tts_connected = True
            
            # Initial configuration
            await self.tts_stream.configure(
                target_language_code="en-IN",
                speaker="manisha",
                speech_sample_rate=16000,
                enable_preprocessing=True,
                output_audio_codec="wav"
            )
            
            print(f"--- [TTS] SDK Connected and Configured for session {self.sid} ---")
            asyncio.run_coroutine_threadsafe(self.receive_tts(), self.loop)
            asyncio.run_coroutine_threadsafe(self.ping_loop("tts"), self.loop)
            
            # Notify frontend that streams are ready
            socketio.emit('session-ready', {"status": "connected"}, room=self.sid)
        except Exception as e:
            print(f"--- [TTS] SDK Connection Failed: {e} ---")

    async def ping_loop(self, stream_type):
        """Keep-alive heartbeat for Sarvam WebSockets."""
        while self.is_active:
            try:
                await asyncio.sleep(20)
                if stream_type == "tts" and self.tts_stream:
                    await self.tts_stream.ping()
            except Exception:
                break

    async def send_audio(self, audio_chunk):
        if self.stt_stream:
            # Volume Check (First chunk only)
            if not hasattr(self, '_audio_logged'):
                count = len(audio_chunk) // 2
                if count > 0:
                    samples = struct.unpack(f"<{count}h", audio_chunk)
                    rms = math.sqrt(sum(s*s for s in samples) / count)
                    print(f"--- [STT-MIC] First chunk received ({len(audio_chunk)} bytes). RMS Volume: {rms:.2f} ---")
                self._audio_logged = True
            
            audio_base64 = base64.b64encode(audio_chunk).decode("utf-8")
            await self.stt_stream.transcribe(audio=audio_base64)

    async def receive_stt(self):
        print(f"--- [STT-LOOP] Started for {self.sid} ---")
        try:
            async for message in self.stt_stream:
                msg_dict = message.dict()
                m_type = msg_dict.get("type")
                data = msg_dict.get("data", {})
                
                print(f"--- [STT-MSG] {m_type} ---") # Print message type at least

                if m_type == "events":
                    signal = data.get("signal_type")
                    print(f"--- [STT-EVENT] {signal} ---")
                    if signal == "START_SPEECH":
                        self.speech_ended_flag = False # New thought starting
                        
                        # Only Barge-in if AI has been speaking for at least 0.5s to avoid echo/noise false positives
                        time_since_response_start = time.time() - self.ai_response_start_time if self.ai_response_start_time > 0 else 0
                        
                        if self.is_ai_speaking and time_since_response_start > 0.5:
                            print(f"--- [BARGE-IN] Interrupting AI due to {signal} ({time_since_response_start:.2f}s into response) ---")
                            socketio.emit('interrupt', {}, room=self.sid)
                            self.is_ai_speaking = False
                        elif self.is_ai_speaking:
                            print(f"--- [BARGE-IN-IGNORE] Speech detected too early ({time_since_response_start:.2f}s) - likely echo/noise ---")
                        
                        socketio.emit('stt-event', {"type": "speech_start"}, room=self.sid)
                    elif signal == "END_SPEECH":
                        print("--- [STT-DETECTION] Voice segment ended (Flagging for finalization) ---")
                        self.speech_ended_flag = True
                        self.speech_end_time = time.time() # Start measuring latency
                        socketio.emit('stt-event', {"type": "speech_end"}, room=self.sid)

                elif m_type == "data":
                    transcript = data.get("transcript")
                    is_final = data.get("is_final") if data.get("is_final") is not None else msg_dict.get("is_final", False)
                    
                    # Force finalization if we just had an END_SPEECH signal
                    effective_final = is_final or self.speech_ended_flag
                    
                    if transcript:
                        print(f"--- [STT-DATA] '{transcript}' (final={is_final}, force_final={self.speech_ended_flag}) ---")
                        socketio.emit('stt-transcript', {"transcript": transcript, "is_final": effective_final}, room=self.sid)
                    
                    if effective_final:
                        if transcript and transcript.strip():
                            # Buffer the transcript to prevent split sentences
                            self.transcript_buffer.append(transcript.strip())
                            
                            # Debounce LLM call: wait 600ms to see if more speech follows
                            if hasattr(self, 'llm_trigger_timer'):
                                self.llm_trigger_timer.cancel()
                            
                            self.llm_trigger_timer = self.loop.call_later(
                                0.8, # Increased to 800ms for better sentence aggregation
                                lambda: asyncio.run_coroutine_threadsafe(self.trigger_llm_if_buffered(), self.loop)
                            )
                        elif is_final:
                            print("--- [STT-EMPTY] Final flag received but transcript empty ---")
                        
                        # Reset flag after processing the 'final' chunk of the segment
                        self.speech_ended_flag = False
                
                elif m_type == "error":
                    print(f"--- [STT-ERROR] {data} ---")
                elif m_type != "events":
                    print(f"--- [STT-OTHER] {m_type}: {data} ---")
        except Exception as e:
            print(f"--- [STT-RECEIVE-EXCEPTION] {e} ---")

    async def trigger_llm_if_buffered(self):
        """Processes the buffered transcripts and triggers Gemini."""
        if not self.transcript_buffer:
            return
            
        full_transcript = " ".join(self.transcript_buffer)
        self.transcript_buffer = [] # Clear buffer
        
        if full_transcript == self.last_processed_transcript:
            return
            
        stt_latency = time.time() - self.speech_end_time if self.speech_end_time > 0 else 0
        print(f"--- [STT-AGGREGATED] Handing to LLM: '{full_transcript}' (Latency: {stt_latency:.2f}s) ---")
        self.last_processed_transcript = full_transcript
        
        # Add to history
        self.history.append({"role": "user", "parts": [{"text": full_transcript}]})
        
        # Cancel any previous active LLM task (Barge-in / Overlap)
        if self.llm_task and not self.llm_task.done():
            print("--- [LLM-CANCEL] Cancelling previous active generation ---")
            self.llm_task.cancel()
            
        self.llm_task = asyncio.create_task(self.process_gemini_streaming())

    async def process_gemini_streaming(self):
        """Streams Gemini response asynchronously and feeds Sarvam TTS."""
        try:
            self.llm_start_time = time.time()
            self.ai_response_start_time = time.time() # Mark start of response process
            print("--- [LLM-PROCESS] Starting generation with history context ---")
            full_response = ""
            
            # Using gemini-2.5-flash-lite as explicitly requested
            model_id = 'gemini-2.5-flash-lite'
            fallback_model = 'gemini-2.5-flash'
            
            # Start Gemini Stream with fallback for 429
            try:
                print(f"--- [LLM-FETCH] Requesting async stream from {model_id}... (History length: {len(self.history)}) ---")
                response_iter = await gemini_async_client.models.generate_content_stream(
                    model=model_id,
                    contents=self.history,
                    config=genai_types.GenerateContentConfig(
                        system_instruction="You are a Senior Sales Executive for TrackerBI, an AI Audio Analytics platform powered by Gemini 2.0. Your goal is to convert Call Center Managers by highlighting how we provide 100% call coverage and detect revenue leakage. Lead with discovery questions about their current quality monitoring. Use key phrases like 'Actionable Intelligence' and 'Data-driven coaching'. Never invent pricing; say it is customized. Respond in 2 sentences max. Plain text only, no emojis or formatting."
                    )
                )
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"--- [LLM-QUOTA] {model_id} exhausted. Falling back to {fallback_model}... ---")
                    response_iter = await gemini_async_client.models.generate_content_stream(
                        model=fallback_model,
                        contents=self.history,
                        system_instruction="You are a Senior Sales Executive for TrackerBI, an AI Audio Analytics platform powered by Gemini 2.0. Your goal is to convert Call Center Managers by highlighting how we provide 100% call coverage and detect revenue leakage. Lead with discovery questions about their current quality monitoring. Use key phrases like 'Actionable Intelligence' and 'Data-driven coaching'. Never invent pricing; say it is customized. Respond in 2 sentences max. Plain text only, no emojis or formatting."
                        )
                    
                else:
                    raise e

            if not self.tts_stream:
                print("--- [LLM-WARN] TTS stream is not initialized. Audio will not be generated. ---")

            self.is_ai_speaking = True
            async for chunk in response_iter:
                if not self.is_ai_speaking: 
                    print("--- [LLM-STOP] Barge-in detected, stopping Gemini stream. ---")
                    break
                
                chunk_text = chunk.text
                if chunk_text:
                    if not self.llm_first_chunk_time:
                        self.llm_first_chunk_time = time.time()
                        ttft = self.llm_first_chunk_time - self.llm_start_time
                        print(f"--- [LLM-TTFT] Time to first chunk: {ttft:.2f}s ---")

                    print(f"--- [AI-CHUNK] {chunk_text} ---")
                    full_response += chunk_text
                    
                    if self.tts_stream:
                        # Clean chunk and ensure it contains meaningful text for TTS
                        clean_chunk = chunk_text.replace("*", "").replace("#", "").strip()
                        if clean_chunk and re.search(r'[a-zA-Z0-9]', clean_chunk):
                            await self.tts_stream.convert(text=clean_chunk)
                        else:
                            print(f"--- [TTS-SKIP] Skipping punctuation-only or empty chunk: '{chunk_text}' ---")
                    
                    socketio.emit('ai-response-partial', {"text": chunk_text}, room=self.sid)

            if self.is_ai_speaking:
                if self.tts_stream:
                    print("--- [TTS-FINAL] Flushing audio stream... ---")
                    await self.tts_stream.flush()
                
                llm_duration = time.time() - self.llm_start_time
                ttft = self.llm_first_chunk_time - self.llm_start_time if self.llm_first_chunk_time else 0
                stt_latency = self.llm_start_time - self.speech_end_time if self.speech_end_time > 0 else 0
                
                print(f"--- [AI-FULL] Response: {full_response} (Total LLM Time: {llm_duration:.2f}s) ---")
                
                # Update history with AI response in correct GenAI SDK format
                self.history.append({"role": "model", "parts": [{"text": full_response}]})
                
                socketio.emit('ai-response', {
                    "response_text": full_response,
                    "metrics": {
                        "stt_latency": round(stt_latency, 2),
                        "ttft": round(ttft, 2),
                        "llm_duration": round(llm_duration, 2)
                    }
                }, room=self.sid)
                # Keep is_ai_speaking True for a few seconds to allow barge-in during audio playback
                async def _delayed_ai_stop():
                    await asyncio.sleep(5.0) # Assume audio takes up to 5s to finish playing
                    if self.is_ai_speaking:
                        print("--- [AI-STOP] Turn finally complete ---")
                        self.is_ai_speaking = False
                
                asyncio.run_coroutine_threadsafe(_delayed_ai_stop(), self.loop)
                
                # Reset for next turn
                self.ai_response_start_time = 0
                self.llm_first_chunk_time = 0
                self.tts_first_audio_time = 0

        except Exception as e:
            print(f"--- [LLM-EXCEPTION] {e} ---")
            socketio.emit('ai-response', {"response_text": "I'm having trouble thinking right now."}, room=self.sid)
            self.is_ai_speaking = False

    async def receive_tts(self):
        print(f"--- [TTS-LOOP] Started for {self.sid} ---")
        try:
            async for message in self.tts_stream:
                msg_dict = message.dict()
                m_type = msg_dict.get("type")
                
                if m_type == "audio":
                    audio_base64 = msg_dict.get("data", {}).get("audio")
                    if audio_base64:
                        if not self.tts_first_audio_time:
                            self.tts_first_audio_time = time.time()
                            # End-to-end latency: from when user stopped speaking to first audio chunk
                            e2e_latency = self.tts_first_audio_time - self.speech_end_time if self.speech_end_time > 0 else 0
                            print(f"--- [TTS-AUDIO] First Chunk received. E2E Latency: {e2e_latency:.2f}s ---")
                        else:
                            print(f"--- [TTS-AUDIO] Chunk received ({len(audio_base64)} chars) ---")
                        
                        audio_bytes = base64.b64decode(audio_base64)
                        socketio.emit('audio-chunk', audio_bytes, room=self.sid)
                elif m_type == "error":
                    print(f"--- [TTS-ERROR] {msg_dict} ---")
                else:
                    # Catch config acks or pings
                    print(f"--- [TTS-EVENT] {m_type}: {msg_dict.get('data')} ---")
        except Exception as e:
            print(f"--- [TTS-RECEIVE-EXCEPTION] {e} ---")

    async def summarize_and_save(self):
        """Generates a summary of the conversation and saves it to a file with retries and fallback."""
        if not self.history:
            print(f"--- [SUMMARY] No history to summarize for {self.sid} ---")
            return

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                print(f"--- [SUMMARY] Attempt {attempt + 1}: Generating summary for {self.sid}... ---")
                summary_request = self.history + [{"role": "user", "parts": [{"text": "Please provide a extremely concise summary of our entire conversation above."}]}]
                
                # Use gemini-2.5-flash-lite for summary
                response = await gemini_async_client.models.generate_content(
                    model='gemini-2.5-flash-lite',
                    contents=summary_request,
                    config=genai_types.GenerateContentConfig(
                        system_instruction="Summarize the conversation accurately in 1-2 short sentences."
                    )
                )
                
                summary_text = response.text
                break # Success!
                
            except Exception as e:
                print(f"--- [SUMMARY-ATTEMPT-FAILED] Attempt {attempt + 1} failed: {e} ---")
                if attempt < max_retries:
                    wait_time = (attempt + 1) * 2
                    print(f"--- [SUMMARY-RETRY] Waiting {wait_time}s before retry... ---")
                    await asyncio.sleep(wait_time)
                else:
                    # Fallback to raw history save
                    print("--- [SUMMARY-FALLBACK] Max retries reached. Saving raw history instead. ---")
                    summary_text = "ERROR: Summarization service overloaded. Below is the raw conversation log:\n\n"
                    for turn in self.history:
                        role = "User" if turn["role"] == "user" else "AI"
                        text = turn["parts"][0]["text"]
                        summary_text += f"{role}: {text}\n"

        # Save to file
        try:
            os.makedirs("summaries", exist_ok=True)
            filename = f"summaries/summary_{self.sid}_{int(time.time())}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"Session ID: {self.sid}\n")
                f.write(f"Time: {time.ctime()}\n")
                f.write("-" * 30 + "\n")
                f.write(summary_text)
            
            print(f"--- [SUMMARY] Saved to {filename} ---")
            socketio.emit('summary-saved', {"filename": filename, "summary": summary_text}, room=self.sid)
        except Exception as file_err:
            print(f"--- [SUMMARY-FILE-ERROR] {file_err} ---")

    def close(self):
        self.is_active = False
        self.is_ai_speaking = False
        async def _close():
            # Generate summary before closing if there's history
            await self.summarize_and_save()
            
            if self.stt_ctx: await self.stt_ctx.__aexit__(None, None, None)
            if self.tts_ctx: await self.tts_ctx.__aexit__(None, None, None)
            self.loop.stop()
            print(f"--- Session {self.sid} SDK CLOSED ---")
        asyncio.run_coroutine_threadsafe(_close(), self.loop)

sessions = {}

@socketio.on('connect')
def handle_connect():
    sid = request.sid
    print(f"--- Client {sid} CONNECTED ---")
    sessions[sid] = SarvamSession(sid)

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    if sid in sessions:
        sessions[sid].close()
        del sessions[sid]
    print(f"--- Client {sid} DISCONNECTED ---")

@socketio.on('start-streaming')
def handle_start_streaming():
    sid = request.sid
    session = sessions.get(sid)
    if session:
        print(f"--- [START] Initializing SDK for {sid} ---")
        asyncio.run_coroutine_threadsafe(session.connect_stt(), session.loop)
        asyncio.run_coroutine_threadsafe(session.connect_tts(), session.loop)

@socketio.on('audio-data')
def handle_audio_data(data):
    sid = request.sid
    session = sessions.get(sid)
    if session:
        # Avoid flood but confirm arrival
        if not hasattr(session, '_audio_arrival_logged'):
            print(f"--- [WS-AUDIO] First buffer arrived from {sid} ---")
            session._audio_arrival_logged = True
        asyncio.run_coroutine_threadsafe(session.send_audio(data), session.loop)

@socketio.on('stop-session')
def handle_stop_session():
    sid = request.sid
    if sid in sessions:
        print(f"--- [STOP] Explicit stop requested for {sid} ---")
        sessions[sid].close()
        # Optional: we don't necessarily delete it yet if we want to confirm summary delivery
        # but usually close() is the end of the line.

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)


