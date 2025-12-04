import json
import requests
import os
import sys
import time

# ==== CONFIGURATION ====
"""
Simple runner that calls the local `fast.py` FastAPI `/tts` endpoint for testing.

This script reads a local text file and a local reference audio file (speaker),
POSTs the audio bytes to the `/tts` endpoint with an `X-Payload` header containing
the TTS request JSON, and writes the returned `audio/wav` response to disk.

By default this uses `http://127.0.0.1:8000/tts`. Set `LOCAL_TTS_ENDPOINT`,
`TEXT_FILE_LOCAL` and `REFERENCE_VOICE_LOCAL` via environment variables if needed.
"""

# Local FastAPI TTS endpoint (change port if you started uvicorn on a different port)
LOCAL_TTS_ENDPOINT = os.environ.get("LOCAL_TTS_ENDPOINT", "http://127.0.0.1:58003/tts")

TEXT_FILE_LOCAL = os.environ.get("TEXT_FILE_LOCAL", os.path.expanduser("~/voice_cloner/voice.txt"))
REFERENCE_VOICE_LOCAL = os.environ.get("REFERENCE_VOICE_LOCAL", os.path.expanduser("~/voice_cloner/patrick_voice.wav"))

USE_LOCAL = True


def call_local_tts(text_file_path: str, ref_voice_path: str, out_path: str = "output_local_tts.wav") -> bool:
    if not os.path.exists(text_file_path):
        print(f"❌ Text file not found: {text_file_path}")
        return False
    if not os.path.exists(ref_voice_path):
        print(f"❌ Reference voice not found: {ref_voice_path}")
        return False

    # Read text
    with open(text_file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # Build payload for /tts endpoint (X-Payload header)
    payload = {
        "text": text,
        # When sending raw audio bytes in the request body, set speaker_ref_path to None
        "speaker_ref_path": None,
        "guidance": 3.0,
        "top_p": 0.95,
    }

    headers = {
        "X-Payload": json.dumps(payload),
        "Accept": "audio/wav",
        "Content-Type": "application/octet-stream",
    }

    # Read audio bytes to send as request body
    with open(ref_voice_path, "rb") as af:
        audio_bytes = af.read()

    print(f"🔄 Sending local TTS request to {LOCAL_TTS_ENDPOINT} (this may trigger model initialization)")

    try:
        resp = requests.post(LOCAL_TTS_ENDPOINT, data=audio_bytes, headers=headers, timeout=900)
    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

    print(f"Response: {resp.status_code} {resp.reason}")
    if resp.status_code == 200:
        # Save returned audio
        with open(out_path, "wb") as out_f:
            out_f.write(resp.content)
        print(f"✅ Saved synthesized audio to: {out_path}")
        return True
    else:
        print(f"❌ TTS failed: {resp.status_code} - {resp.text}")
        return False


def main():
    if USE_LOCAL:
        print("Using local TTS endpoint flow")
        success = call_local_tts(TEXT_FILE_LOCAL, REFERENCE_VOICE_LOCAL)
        if not success:
            print("Local TTS call failed. If the server isn't running, start it with uvicorn and try again.")
            sys.exit(1)
    else:
        print("USE_LOCAL is False — Azure/Salad flow is not implemented in this runner yet.")


if __name__ == "__main__":
    main()

