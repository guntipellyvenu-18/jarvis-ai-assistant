"""
Production-grade voice I/O pipeline for JARVIS autonomous agent.
Features: VAD-driven recording, streaming TTS, session persistence, error recovery.
"""

import json
import logging
import sys
import threading
import queue
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, List, Tuple
from enum import Enum

import numpy as np

try:
    import sounddevice as sd
    import torch
    from faster_whisper import WhisperModel
except ImportError:
    sd = None  # type: ignore
    torch = None  # type: ignore
    WhisperModel = None  # type: ignore

try:
    import subprocess
except ImportError:
    subprocess = None  # type: ignore

logger = logging.getLogger("jarvis.voice")

# ============= Configuration =============

@dataclass
class VoiceConfig:
    """Voice pipeline configuration."""
    mic_sample_rate: int = 16000          # Whisper + Silero-VAD requirement
    vad_frame_size: int = 512             # 32ms at 16kHz
    vad_threshold: float = 0.5            # Speech probability threshold
    silence_trigger_frames: int = 25      # ~800ms silence to end utterance
    pre_roll_chunks: int = 10             # ~320ms pre-speech buffer
    piper_model_path: str = "en_US-lessac-medium.onnx"
    piper_sample_rate: int = 22050
    whisper_model: str = "base.en"        # tiny.en, base.en, small.en
    device: str = "cuda" if torch and torch.cuda.is_available() else "cpu"
    compute_type: str = "float16" if torch and torch.cuda.is_available() else "int8"
    mic_timeout: float = 5.0              # Mic read timeout (seconds)
    max_utterance_length: int = 300       # Max audio duration (seconds)
    enable_session_logging: bool = True
    session_log_dir: Path = Path("./voice_sessions")

class VoiceState(Enum):
    """Pipeline execution state."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"

@dataclass
class VoiceSegment:
    """Audio segment metadata."""
    timestamp: str
    duration: float
    text: str
    confidence: float
    audio_frames: int

# ============= TTS Engine =============

class PiperStreamingPlayer:
    """Low-latency streaming TTS via Piper."""

    def __init__(self, model_path: str, sample_rate: int = 22050):
        if not subprocess:
            raise RuntimeError("subprocess module required")
        self.model_path = model_path
        self.sample_rate = sample_rate
        logger.info("Piper TTS initialized: %s @ %dHz", model_path, sample_rate)

    def speak(self, text: str) -> bool:
        """Stream TTS output directly to speaker.
        
        Returns True on success, False on error.
        """
        if not text or not text.strip():
            return False

        try:
            cmd = [
                sys.executable, "-m", "piper",
                "--model", self.model_path,
                "--output-raw"
            ]

            logger.debug("Launching Piper: %s", " ".join(cmd))
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Write text and close stdin
            proc.stdin.write(text.encode("utf-8"))
            proc.stdin.close()

            # Stream PCM to speaker
            if not sd:
                logger.error("sounddevice not available")
                proc.terminate()
                return False

            with sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16"
            ) as stream:
                chunk_size = 2048
                while True:
                    data = proc.stdout.read(chunk_size)
                    if not data:
                        break
                    audio_chunk = np.frombuffer(data, dtype=np.int16)
                    stream.write(audio_chunk)

            returncode = proc.wait(timeout=30)
            if returncode != 0:
                stderr = proc.stderr.read().decode("utf-8", errors="replace")
                logger.error("Piper failed: %s", stderr)
                return False

            logger.info("TTS output streamed: %d characters", len(text))
            return True

        except subprocess.TimeoutExpired:
            logger.error("Piper process timeout")
            proc.kill()
            return False
        except Exception as exc:
            logger.exception("TTS streaming failed")
            return False

# ============= VAD Engine =============

class SileroVADEngine:
    """Silero Voice Activity Detection wrapper."""

    def __init__(self):
        if not torch:
            raise RuntimeError("torch required for VAD")
        
        logger.info("Loading Silero-VAD v5...")
        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True
        )
        self.model.eval()
        logger.info("Silero-VAD ready")

    def get_speech_probability(self, frame: np.ndarray, sample_rate: int) -> float:
        """Get speech probability for a frame.
        
        Args:
            frame: Audio frame as numpy array (float32)
            sample_rate: Sample rate in Hz
            
        Returns:
            Speech probability [0.0, 1.0]
        """
        try:
            tensor_frame = torch.from_numpy(frame)
            with torch.no_grad():
                prob = self.model(tensor_frame, sample_rate).item()
            return prob
        except Exception as exc:
            logger.exception("VAD inference failed")
            return 0.0

# ============= Transcription Engine =============

class WhisperTranscriber:
    """Faster-Whisper transcription wrapper."""

    def __init__(self, model_size: str, device: str, compute_type: str):
        if not WhisperModel:
            raise RuntimeError("faster-whisper required")
        
        logger.info("Loading Whisper (%s) on %s...", model_size, device)
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
        logger.info("Whisper ready")

    def transcribe(self, audio: np.ndarray, language: str = "en") -> Tuple[str, float]:
        """Transcribe audio to text.
        
        Returns:
            (transcribed_text, avg_confidence)
        """
        try:
            segments, _ = self.model.transcribe(
                audio,
                beam_size=1,
                language=language
            )
            texts = [seg.text.strip() for seg in segments]
            confidences = [getattr(seg, "confidence", 0.8) for seg in segments]
            
            full_text = " ".join(texts)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            logger.info("Transcribed %d segments, avg confidence: %.2f", len(segments), avg_confidence)
            return full_text, avg_confidence

        except Exception as exc:
            logger.exception("Transcription failed")
            return "", 0.0

# ============= Voice Agent Pipeline =============

class VoiceAgentPipeline:
    """End-to-end voice I/O pipeline with VAD, transcription, and TTS."""

    def __init__(
        self,
        response_generator: Callable[[str], str],
        config: Optional[VoiceConfig] = None,
    ):
        """Initialize voice pipeline.
        
        Args:
            response_generator: Callable(user_text) -> bot_response
            config: VoiceConfig instance
        """
        if not sd:
            raise RuntimeError("sounddevice required")

        self.config = config or VoiceConfig()
        self.response_generator = response_generator
        self.state = VoiceState.IDLE

        # Initialize components
        logger.info("Initializing voice pipeline...")
        self.whisper = WhisperTranscriber(
            self.config.whisper_model,
            self.config.device,
            self.config.compute_type
        )
        self.vad = SileroVADEngine()
        self.tts = PiperStreamingPlayer(
            self.config.piper_model_path,
            self.config.piper_sample_rate
        )

        self.audio_queue: queue.Queue = queue.Queue()
        self.is_speaking = False
        self.session_log: List[VoiceSegment] = []

        # Create session directory
        if self.config.enable_session_logging:
            self.config.session_log_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Voice pipeline ready")

    def audio_callback(self, indata, frames, time_info, status):
        """sounddevice callback: enqueue incoming audio."""
        if status and status.messages:
            logger.warning("Audio input warning: %s", status.messages)
        self.audio_queue.put(indata.flatten().copy())

    def _save_session(self, session_id: str) -> None:
        """Save session log to disk."""
        if not self.config.enable_session_logging or not self.session_log:
            return

        log_file = self.config.session_log_dir / f"{session_id}.json"
        try:
            with open(log_file, "w") as f:
                json.dump([asdict(seg) for seg in self.session_log], f, indent=2)
            logger.info("Session saved: %s", log_file)
        except Exception as exc:
            logger.exception("Failed to save session")

    def run(self, session_id: Optional[str] = None) -> None:
        """Run voice listening loop."""
        if not session_id:
            session_id = datetime.utcnow().isoformat()

        self.session_log = []
        logger.info("Starting voice loop (session: %s)", session_id)
        print(f"\n[✓] Ready. Speak into microphone (session: {session_id})...\n")

        try:
            with sd.InputStream(
                samplerate=self.config.mic_sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.config.vad_frame_size,
                callback=self.audio_callback,
                latency="low"
            ):
                self._listen_loop()
        except Exception as exc:
            logger.exception("Voice loop error")
            self.state = VoiceState.ERROR
        finally:
            self._save_session(session_id)
            logger.info("Voice loop terminated")

    def _listen_loop(self) -> None:
        """Main listening loop with VAD-driven recording."""
        pre_roll_buffer = []
        speech_buffer = []
        is_recording = False
        silent_frames = 0
        total_frames = 0

        while True:
            # Drain queue if agent is speaking
            if self.is_speaking:
                while not self.audio_queue.empty():
                    try:
                        self.audio_queue.get_nowait()
                    except queue.Empty:
                        break
                continue

            try:
                frame = self.audio_queue.get(timeout=self.config.mic_timeout)
            except queue.Empty:
                if is_recording:
                    logger.warning("Mic timeout during recording; ending utterance")
                    is_recording = False
                continue

            total_frames += 1

            # Get speech probability
            speech_prob = self.vad.get_speech_probability(
                frame,
                self.config.mic_sample_rate
            )

            if speech_prob >= self.config.vad_threshold:
                if not is_recording:
                    self.state = VoiceState.LISTENING
                    logger.info("Speech detected (prob=%.3f)", speech_prob)
                    print("\n[🎙] Recording...", end="", flush=True)
                    is_recording = True
                    speech_buffer = pre_roll_buffer.copy()

                speech_buffer.append(frame)
                silent_frames = 0

            else:  # No speech
                if is_recording:
                    speech_buffer.append(frame)
                    silent_frames += 1

                    if silent_frames >= self.config.silence_trigger_frames:
                        print(" [Done]")
                        is_recording = False

                        # Process the complete utterance
                        audio_data = np.concatenate(speech_buffer)
                        speech_buffer = []
                        pre_roll_buffer = []
                        silent_frames = 0

                        self.state = VoiceState.PROCESSING
                        self._handle_turn(audio_data)
                        self.state = VoiceState.IDLE

                else:
                    # Maintain pre-roll buffer
                    pre_roll_buffer.append(frame)
                    if len(pre_roll_buffer) > self.config.pre_roll_chunks:
                        pre_roll_buffer.pop(0)

    def _handle_turn(self, audio_data: np.ndarray) -> None:
        """Process one speech turn: transcribe → generate → speak."""
        try:
            # 1. Transcription
            user_text, confidence = self.whisper.transcribe(audio_data)

            if not user_text or confidence < 0.3:
                logger.warning("Low confidence transcription (%.2f), skipping", confidence)
                return

            print(f"[User] ({confidence:.2f}): {user_text}")

            # 2. Generate response
            bot_reply = self.response_generator(user_text)
            print(f"[Agent]: {bot_reply}")

            # 3. Speak response
            self.state = VoiceState.SPEAKING
            self.is_speaking = True
            try:
                success = self.tts.speak(bot_reply)
                if not success:
                    logger.error("TTS failed")
            finally:
                self.is_speaking = False

            # 4. Log segment
            segment = VoiceSegment(
                timestamp=datetime.utcnow().isoformat(),
                duration=len(audio_data) / self.config.mic_sample_rate,
                text=f"User: {user_text}\nAgent: {bot_reply}",
                confidence=confidence,
                audio_frames=len(audio_data)
            )
            self.session_log.append(segment)

        except Exception as exc:
            logger.exception("Turn processing error")
            self.state = VoiceState.ERROR

# ============= Demo & Integration =============

def demo_response_generator(user_text: str) -> str:
    """Simple demo response generator.
    
    In production, integrate with JARVIS agent:
        agent = AutonomousAgent(...)
        return agent.execute_flow(user_text)
    """
    text_lower = user_text.lower()

    responses = {
        "hello": "Greetings. Voice pipeline online and operational.",
        "time": "All internal clocks synchronized and running.",
        "status": "All subsystems nominal. Ready for commands.",
        "health": "System diagnostics: CPU and memory within acceptable range.",
    }

    for keyword, response in responses.items():
        if keyword in text_lower:
            return response

    return f"Command received: {user_text}. Standing by for further instructions."

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    try:
        config = VoiceConfig()
        pipeline = VoiceAgentPipeline(
            response_generator=demo_response_generator,
            config=config
        )
        pipeline.run()

    except KeyboardInterrupt:
        print("\n[*] Voice pipeline terminated by user")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Fatal error")
        sys.exit(1)
