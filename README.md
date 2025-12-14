# MetaVoice Speech Synthesis

This project provides a FastAPI server for MetaVoice speech synthesis and a client script for voice cloning.

## Project Structure

- `src/python/api/inference/` - FastAPI server for speech synthesis
- `voice_cloner/` - Client script for testing voice cloning

## Server Setup (Ubuntu/Linux)

### 1. Clone and Setup Environment

```bash
git clone <repository-url>
cd metavoice-on-salad

# Create virtual environment
python3 -m venv metavoice_env
source metavoice_env/bin/activate

# Install requirements
pip install -r src/python/api/inference/requirements.txt
```

### 2. Fix PyTorch Compatibility Issues

If you encounter PyTorch version conflicts, install compatible versions:

```bash
pip install "transformers>=4.35.0,<4.40.0"
```

### 3. Start the Server

```bash
cd src/python/api/inference

# Set Python path and start server
PYTHONPATH=/path/to/metavoice-on-salad/src/python/api/inference uvicorn fast:app --host 0.0.0.0 --port 43304
```

### 4. Verify Server is Running

```bash
curl http://localhost:43304/health
# Should return: {"status":"ok"}
```

## Client Setup (Local Machine)

### 1. Setup Environment

```bash
cd metavoice-on-salad

# Create virtual environment
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install client requirements
pip install requests
```

### 2. Prepare Voice Cloning Files

Create the required directories and files:

```bash
mkdir -p ~/voice_cloner

# Create text file with content to synthesize
echo "Your text content here" > ~/voice_cloner/voice.txt

# Add your reference voice file (WAV format recommended)
cp /path/to/your/voice.wav ~/voice_cloner/patrick_voice.wav
```

### 3. Update Server Endpoint

Edit `voice_cloner/voice_clone_runner.py` and update the endpoint:

```python
LOCAL_TTS_ENDPOINT = "http://YOUR_SERVER_IP:43304/tts"
```

### 4. Run Voice Cloning

```bash
python3 voice_cloner/voice_clone_runner.py
```

The synthesized audio will be saved as `output_local_tts.wav`.

## Configuration

### Environment Variables

- `LOCAL_TTS_ENDPOINT` - TTS server endpoint (default: http://127.0.0.1:43304/tts)
- `TEXT_FILE_LOCAL` - Path to text file (default: ~/voice_cloner/voice.txt)
- `REFERENCE_VOICE_LOCAL` - Path to reference voice file (default: ~/voice_cloner/patrick_voice.wav)

### Server Configuration

The server can be configured by modifying the `ServingConfig` class in `fast.py`:

- `port` - Server port (default: 58003)
- `seed` - Random seed for sampling (default: 1337)
- `temperature` - Sampling temperature (default: 1.0)

## Troubleshooting

### CUDA Out of Memory

If you encounter GPU memory issues:

```bash
# Kill existing processes
pkill -f python
pkill -f uvicorn

# Reset GPU
nvidia-smi --gpu-reset

# Set memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0
```

### PyTorch Compatibility

If you see `fx_graph_cache` errors, the code includes automatic version detection and fallbacks.

### NLTK Tokenizer Issues

The code includes fallback sentence splitting if NLTK fails to download required data.

## API Endpoints

- `GET /health` - Health check
- `POST /tts` - Text-to-speech synthesis
- `POST /process_short_text` - Azure blob processing (short text)
- `POST /process_long_text` - Azure blob processing (long text, background task)

## Features

- **Long Text Support**: Automatically splits long text into sentences for better quality
- **Voice Cloning**: Uses reference audio to clone voice characteristics
- **GPU Acceleration**: Optimized for NVIDIA GPUs
- **Flexible Input**: Supports various audio formats via FFmpeg conversion
- **Error Handling**: Robust error handling with fallbacks

## Requirements

### Server Requirements
- Python 3.10+
- NVIDIA GPU with CUDA support
- FFmpeg
- 8GB+ GPU memory recommended

### Client Requirements
- Python 3.7+
- Internet connection to server