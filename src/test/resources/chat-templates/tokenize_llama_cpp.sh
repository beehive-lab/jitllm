#!/usr/bin/env bash
#
# Records llama.cpp's tokenization of the rendered Llama prompts, which LlamaTemplateTokenizationAccelTest
# compares the engine's token ids against. Only the scenarios the engine must reproduce exactly are
# recorded: Llama 3.2's tool conversations and Llama 3.1's plain chat.
#
#   LLAMA_TOKENIZE=~/llama.cpp/build/bin/llama-tokenize ./tokenize_llama_cpp.sh <model.gguf>
#
# Any Llama 3.x GGUF will do: 3.1 and 3.2 share one vocabulary. --no-escape keeps the JSON's
# backslashes as written, --no-bos because the rendered text already starts with <|begin_of_text|>.
set -euo pipefail
cd "$(dirname "$0")"
MODEL=$1
TOKENIZE=${LLAMA_TOKENIZE:-llama-tokenize}
for f in llama-3.2/system_and_tool.txt llama-3.2/no_system_two_tools.txt llama-3.2/call_and_result.txt \
    llama-3.1/chat_user_only.txt llama-3.1/chat_system_and_user.txt llama-3.1/chat_multi_turn.txt; do
  "$TOKENIZE" -m "$MODEL" -f "$f" --ids --no-bos --no-escape --log-disable 2>/dev/null \
    | tail -1 > "${f%.txt}.llama-cpp-ids.txt"
  echo "$f $(tr -cd ',' < "${f%.txt}.llama-cpp-ids.txt" | wc -c)"
done
