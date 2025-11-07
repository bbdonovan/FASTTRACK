#!/bin/bash
set -e

#MODEL_NAME="llama3.2:3b-instruct"
#DATA_DIR="/root/.ollama/models"  # default Ollama storage directory

echo "================================================="
echo " Launching Ollama server for model ${MODEL_NAME}"
echo "================================================="

# If you want to force pull the model:
#echo "Pulling model ${MODEL_NAME} if not present..."
#ollama pull ${MODEL_NAME}

# Start the Ollama server with GPU support
echo "Starting Ollama server on port 11434..."
export OLLAMA_HOST=0.0.0.0
ollama serve &

# Wait for Ollama to start (it needs a few seconds)
sleep 5

# Register your local model with Ollama
if ! ollama list | grep -q "llama3.2-local"; then
    echo "Creating local Ollama model from /models/Meta-Llama-3.2-3B-Instruct ..."
    ollama create llama3.2-local -f /app/Modelfile
else
    echo "Model llama3.2-local already registered."
fi

# Keep container running
tail -f /dev/null