#!/bin/bash

models=(
  "../models/beehive-llama-3.2-1b-instruct-fp16.gguf"
  "../models/Phi-3-mini-4k-instruct-fp16.gguf"
  "../models/Qwen3-1.7B-f16.gguf"
  "../models/Mistral-7B-Instruct-v0.3.fp16.gguf"
  "../models/Qwen3-8B-f16.gguf"
)

for model in "${models[@]}"; do
  name=$(basename "$model" .gguf)
   # file size (human readable, GB/MB)
  if command -v numfmt &> /dev/null; then
    size=$(stat -c%s "$model" 2>/dev/null | numfmt --to=iec --suffix=B)   # Linux
    [ -z "$size" ] && size=$(stat -f%z "$model" 2>/dev/null | numfmt --to=iec --suffix=B)  # macOS
  else
    size=$(stat -c%s "$model" 2>/dev/null || stat -f%z "$model" 2>/dev/null)
    size="${size} bytes"
  fi

  # colors
  CYAN="\033[1;36m"
  YELLOW="\033[1;33m"
  RESET="\033[0m"

  width=$(tput cols)   # get terminal width
  line=$(printf '━%.0s' $(seq 1 $width))

  echo -e "\n${CYAN}${line}${RESET}"
    echo -e "   🚀 Running Model: ${YELLOW}$name${RESET} (size: ${YELLOW}$size${RESET}) 🚀"
#  echo -e "   🚀 Running Model: ${YELLOW}$name${RESET} 🚀"
  echo -e "${CYAN}${line}${RESET} \n"

  cmd=(
    java @argfile
    -cp /home/devoxx2025-demo/java-ai-demos/jitllm/target/jitllm-1.0.0-jdk21.jar
    org.beehive.jitllm.JitllmApp
    --model "$model"
    --stream true
    --echo false
    -p "Who are you?"
    --instruct
  )

  # Pretty print the command (one-liner)
  echo -e "java @argfile -cp /home/devoxx2025-demo/java-ai-demos/jitllm/target/jitllm-1.0.0-jdk21.jar org.beehive.jitllm.JitllmApp --model \"$model\" --stream true --echo false -p \"Who are you?\" --instruct \n"

  # Execute it
  "${cmd[@]}"

  #java @argfile -cp /home/devoxx2025-demo/java-ai-demos/jitllm/target/jitllm-1.0.0-jdk21.jar org.beehive.jitllm.JitllmApp --model "$model"  --stream true --echo false -p "Who are you?" --instruct

   #./jitllm --gpu --model "$model" --prompt "Who are you?"
done

