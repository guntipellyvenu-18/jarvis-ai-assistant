"""
Integrated voice pipeline with JARVIS autonomous agent.
Features: Ollama LLM integration, tool dispatch from voice, session persistence.
"""

import json
import logging
import queue
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, List, Tuple, Any, Dict

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import torch
except ImportError:
    torch = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    from ollama import chat
except ImportError:
    chat = None

logger = logging.getLogger("jarvis.voice_agent")

# ----------------------------
# Configuration
# ----------------------------

MIC_SAMPLE_RATE = 16000
VAD_FRAME_SIZE = 512
VAD_SPEECH_THRESHOLD = 0.5
SILENCE_TRIGGER_FRAMES = 25
PRE_ROLL_CHUNKS = 10

PIPER_MODEL_PATH = "en_US-lessac-medium.onnx"
PIPER_SAMPLE_RATE = 22050
WHISPER_MODEL_SIZE = "base.en"
OLLAMA_MODEL = "llama3.1"

DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if torch is not None and torch.cuda.is_available() else "int8"

# ----------------------------
# Data Models
# ----------------------------

@dataclass
class VoiceTurn:
    timestamp: str
    user_text: str
    transcription_confidence: float
    agent_response: str
    tools_used: List[str]
    
    def to_dict(self):
        return asdict(self)

@dataclass
class AgentToolCall:
    name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]
    timestamp: str

# ----------------------------
# TTS Engine
# ----------------------------

class PiperStreamingPlayer:
    """Piper TTS with streaming output."""
    
    def __init__(self, model_path: str, sample_rate: int = 22050):
        self.model_path = model_path
        self.sample_rate = sample_rate

    def speak(self, text: str) -> bool:
        if not text or not text.strip():
            return False

        if sd is None:
            logger.error("sounddevice is not installed")
            return False

        try:
            cmd = [
                sys.executable,
                "-m",
                "piper",
                "--model",
                self.model_path,
                "--output-raw",
            ]

            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            if proc.stdin is None or proc.stdout is None:
                logger.error("Piper process pipes unavailable")
                return False

            proc.stdin.write(text.encode("utf-8"))
            proc.stdin.close()

            with sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
            ) as stream:
                chunk_size = 2048
                while True:
                    data = proc.stdout.read(chunk_size)
                    if not data:
                        break
                    chunk = np.frombuffer(data, dtype=np.int16)
                    stream.write(chunk)

            stderr = proc.stderr.read().decode("utf-8", errors="replace")
            rc = proc.wait(timeout=30)
            if rc != 0:
                logger.error("Piper error: %s", stderr)
                return False

            return True

        except subprocess.TimeoutExpired:
            logger.error("Piper timeout")
            return False
        except Exception:
            logger.exception("TTS failed")
            return False

# ----------------------------
# VAD Engine
# ----------------------------

class SileroVADEngine:
    """Silero Voice Activity Detection."""
    
    def __init__(self):
        if torch is None:
            raise RuntimeError("torch required for VAD")
        try:
            logger.info("Loading Silero-VAD...")
            self.model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
            self.model.eval()
            logger.info("Silero-VAD loaded")
        except Exception:
            logger.exception("Failed to load Silero-VAD")
            raise

    def speech_probability(self, frame: np.ndarray, sample_rate: int) -> float:
        try:
            tensor = torch.from_numpy(frame)
            with torch.no_grad():
                prob = self.model(tensor, sample_rate).item()
            return float(prob)
        except Exception:
            logger.exception("VAD inference failed")
            return 0.0

# ----------------------------
# Transcription Engine
# ----------------------------

class WhisperTranscriber:
    """Faster-Whisper for speech-to-text."""
    
    def __init__(self, model_size: str, device: str, compute_type: str):
        if WhisperModel is None:
            raise RuntimeError("faster-whisper required")
        logger.info("Loading Whisper (%s) on %s", model_size, device)
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray, language: str = "en") -> Tuple[str, float]:
        try:
            segments, _ = self.model.transcribe(
                audio,
                beam_size=1,
                language=language,
            )

            texts = []
            confidences = []
            for seg in segments:
                text = seg.text.strip()
                if text:
                    texts.append(text)
                    confidences.append(getattr(seg, "avg_logprob", 0.0))

            full_text = " ".join(texts).strip()
            confidence = float(np.mean(confidences)) if confidences else 0.0
            return full_text, confidence
        except Exception:
            logger.exception("Transcription failed")
            return "", 0.0

# ----------------------------
# Agent Integration
# ----------------------------

class OllamaAgentBridge:
    """Bridge to JARVIS autonomous agent via Ollama."""
    
    def __init__(self, model: str = OLLAMA_MODEL, tools_schema: Optional[List[Dict[str, Any]]] = None):
        self.model = model
        self.tools_schema = tools_schema or []
        self.tool_registry: Dict[str, Callable] = {}
        self.conversation_history: List[Dict[str, str]] = []

    def register_tool(self, name: str, func: Callable) -> None:
        """Register a tool that the agent can call."""
        self.tool_registry[name] = func
        logger.info("Registered tool: %s", name)

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a registered tool."""
        if name not in self.tool_registry:
            return {"status": "error", "message": f"Tool '{name}' not found"}
        
        try:
            result = self.tool_registry[name](**args)
            return {"status": "success", "result": result}
        except Exception as exc:
            logger.exception("Tool execution failed: %s", name)
            return {"status": "error", "message": str(exc)}

    def reasoning_loop(self, user_input: str, max_iterations: int = 3) -> Tuple[str, List[AgentToolCall]]:
        """Execute reasoning loop with optional tool dispatch.
        
        Returns:
            (final_response, list_of_tool_calls)
        """
        if chat is None:
            logger.warning("Ollama chat not available; using mock response")
            return f"Mock response to: {user_input}", []

        self.conversation_history.append({"role": "user", "content": user_input})
        tool_calls_made: List[AgentToolCall] = []

        for iteration in range(max_iterations):
            logger.info("Reasoning iteration %d/%d", iteration + 1, max_iterations)

            try:
                response = chat(
                    model=self.model,
                    messages=self.conversation_history,
                    tools=self.tools_schema if self.tools_schema else None,
                )
            except Exception as exc:
                logger.exception("Ollama request failed")
                return f"Error: {exc}", tool_calls_made

            assistant_content = response.message.content or ""
            tool_calls = getattr(response.message, "tool_calls", None) or []

            # No tools called: agent has answered
            if not tool_calls:
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_content
                })
                logger.info("Agent concluded reasoning")
                return assistant_content, tool_calls_made

            # Process tool calls
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": [
                    {
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                tool_name = tc.function.name
                tool_args = self._normalize_args(tc.function.arguments)
                logger.info("Executing tool: %s", tool_name)

                result = self._execute_tool(tool_name, tool_args)
                
                tool_call_record = AgentToolCall(
                    name=tool_name,
                    arguments=tool_args,
                    result=result,
                    timestamp=datetime.utcnow().isoformat(),
                )
                tool_calls_made.append(tool_call_record)

                self.conversation_history.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": json.dumps(result),
                })

        logger.warning("Max reasoning iterations reached")
        final_msg = self.conversation_history[-1].get("content", "")
        return final_msg, tool_calls_made

    def _normalize_args(self, raw_args: Any) -> Dict[str, Any]:
        """Normalize tool arguments."""
        if raw_args is None:
            return {}
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, str):
            try:
                return json.loads(raw_args)
            except json.JSONDecodeError:
                logger.warning("Failed to parse args as JSON: %s", raw_args)
                return {}
        return {}

    def reset_history(self) -> None:
        """Reset conversation history."""
        self.conversation_history = []

# ----------------------------
# Integrated Voice-Agent Pipeline
# ----------------------------

class VoiceAgentPipeline:
    """Voice I/O + JARVIS agent reasoning loop."""
    
    def __init__(
        self,
        agent: OllamaAgentBridge,
        session_dir: Optional[str] = "./voice_sessions",
    ):
        if sd is None:
            raise RuntimeError("sounddevice required")

        self.agent = agent
        self.audio_queue: queue.Queue = queue.Queue()
        self.is_speaking = False
        self.turns: List[VoiceTurn] = []

        self.vad = SileroVADEngine()
        self.whisper = WhisperTranscriber(
            WHISPER_MODEL_SIZE,
            DEVICE,
            COMPUTE_TYPE,
        )
        self.tts = PiperStreamingPlayer(PIPER_MODEL_PATH, PIPER_SAMPLE_RATE)

        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning("Audio status: %s", status)
        self.audio_queue.put(indata.flatten().copy())

    def save_session(self, session_id: str):
        session_file = self.session_dir / f"{session_id}.json"
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump([turn.to_dict() for turn in self.turns], f, indent=2)
        logger.info("Session saved: %s", session_file)

    def handle_turn(self, audio_data: np.ndarray):
        """Process one speech turn with agent reasoning."""
        # 1. Transcribe
        user_text, confidence = self.whisper.transcribe(audio_data)
        if not user_text.strip():
            logger.warning("No speech recognized")
            return

        print(f"[User] ({confidence:.2f}): {user_text}")

        # 2. Agent reasoning with tool dispatch
        bot_reply, tool_calls = self.agent.reasoning_loop(user_text, max_iterations=3)
        print(f"[Agent]: {bot_reply}")

        if tool_calls:
            print(f"[Tools]: Executed {len(tool_calls)} tool(s)")
            for tc in tool_calls:
                print(f"  - {tc.name}: {tc.result.get('status', 'unknown')}")

        # 3. Speak response
        self.is_speaking = True
        try:
            self.tts.speak(bot_reply)
        finally:
            self.is_speaking = False

        # 4. Log turn
        self.turns.append(
            VoiceTurn(
                timestamp=datetime.utcnow().isoformat(),
                user_text=user_text,
                transcription_confidence=confidence,
                agent_response=bot_reply,
                tools_used=[tc.name for tc in tool_calls],
            )
        )

    def run(self):
        """Main voice loop."""
        session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        print("\n[✓] Voice-Agent pipeline active. Speak into the microphone...\n")

        pre_roll_buffer = []
        speech_buffer = []
        is_recording = False
        silence_counter = 0

        try:
            with sd.InputStream(
                samplerate=MIC_SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=VAD_FRAME_SIZE,
                callback=self.audio_callback,
                latency="low",
            ):
                while True:
                    # Prevent echo while speaking
                    if self.is_speaking:
                        while not self.audio_queue.empty():
                            try:
                                self.audio_queue.get_nowait()
                            except queue.Empty:
                                break
                        continue

                    try:
                        frame = self.audio_queue.get(timeout=1.0)
                    except queue.Empty:
                        continue

                    speech_prob = self.vad.speech_probability(frame, MIC_SAMPLE_RATE)

                    if speech_prob >= VAD_SPEECH_THRESHOLD:
                        if not is_recording:
                            print("\n[🎙] Voice detected. Recording...", end="", flush=True)
                            is_recording = True
                            speech_buffer = pre_roll_buffer.copy()

                        speech_buffer.append(frame)
                        silence_counter = 0

                    else:
                        if is_recording:
                            speech_buffer.append(frame)
                            silence_counter += 1

                            if silence_counter >= SILENCE_TRIGGER_FRAMES:
                                print(" [Done]")
                                is_recording = False
                                silence_counter = 0

                                complete_audio = np.concatenate(speech_buffer)
                                speech_buffer = []
                                pre_roll_buffer = []

                                self.handle_turn(complete_audio)
                        else:
                            pre_roll_buffer.append(frame)
                            if len(pre_roll_buffer) > PRE_ROLL_CHUNKS:
                                pre_roll_buffer.pop(0)

        except KeyboardInterrupt:
            print("\n[*] Shutting down gracefully...")
            self.save_session(session_id)
            sys.exit(0)
        except Exception:
            logger.exception("Voice pipeline crashed")
            self.save_session(session_id)
            raise

# ----------------------------
# Example tools for agent
# ----------------------------

def get_system_status() -> Dict[str, Any]:
    """Tool: Get system status."""
    try:
        import psutil
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
        }
    except Exception as exc:
        return {"error": str(exc)}

def set_alarm(duration_minutes: int, message: str = "Alarm") -> Dict[str, Any]:
    """Tool: Set a simple alarm."""
    import time
    return {
        "status": "alarm_set",
        "duration_minutes": duration_minutes,
        "message": message,
        "set_time": datetime.utcnow().isoformat(),
    }

# ----------------------------
# Entry point
# ----------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    try:
        # Initialize agent
        agent = OllamaAgentBridge(model=OLLAMA_MODEL)
        
        # Register tools
        agent.register_tool("get_system_status", get_system_status)
        agent.register_tool("set_alarm", set_alarm)

        # Create voice pipeline
        pipeline = VoiceAgentPipeline(agent=agent)

        # Run
        pipeline.run()

    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[*] Terminated.")
        sys.exit(0)
