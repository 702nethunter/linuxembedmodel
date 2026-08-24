#!/usr/bin/env bash
# Publish the trained encoder to the Hugging Face Hub.
#
# Run this ON THE GPU BOX, where the weights already live -- uploading from
# there avoids pulling 162 MB down and pushing it back up.
#
#   huggingface-cli login          # once, paste a WRITE token
#   HF_REPO=<user>/linuxembed bash scripts/publish_hf.sh
#
# The model card at modelcard/README.md is uploaded as the repo README.
set -euo pipefail

cd "$(dirname "$0")/.."

: "${HF_REPO:?set HF_REPO=<your-hf-username>/linuxembed}"
MODEL_DIR="${MODEL_DIR:-artifacts/embed-stage1-infonce}"
HF="${HF:-huggingface-cli}"

[ -f "$MODEL_DIR/model.safetensors" ] || {
  echo "no model at $MODEL_DIR" >&2; exit 1; }

$HF whoami >/dev/null 2>&1 || {
  echo "not logged in — run: $HF login  (needs a WRITE token)" >&2; exit 1; }

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# Ship only what SentenceTransformer needs to load. checkpoints/ holds optimizer
# state that is several times the model size and is useless to a consumer.
for f in model.safetensors config.json config_sentence_transformers.json \
         modules.json sentence_bert_config.json special_tokens_map.json \
         tokenizer.json tokenizer_config.json; do
  cp "$MODEL_DIR/$f" "$STAGE/"
done
mkdir -p "$STAGE/1_Pooling" "$STAGE/2_Normalize"
cp "$MODEL_DIR/1_Pooling/config.json" "$STAGE/1_Pooling/"
[ -d "$MODEL_DIR/2_Normalize" ] && cp -r "$MODEL_DIR/2_Normalize/." "$STAGE/2_Normalize/" || true

# The model card replaces the auto-generated README, with the repo id filled in.
sed "s|REPLACE_WITH_HF_REPO_ID|$HF_REPO|g" modelcard/README.md > "$STAGE/README.md"

echo "  staged $(du -sh "$STAGE" | cut -f1) for $HF_REPO"
ls -la "$STAGE"

$HF repo create "$HF_REPO" --type model -y 2>/dev/null || \
  echo "  (repo already exists, uploading into it)"

$HF upload "$HF_REPO" "$STAGE" . --repo-type model \
  --commit-message "from-scratch kernel C embedding model, 0.905 recall@1 over 914k chunks"

echo
echo "  published: https://huggingface.co/$HF_REPO"
echo "  load with: SentenceTransformer(\"$HF_REPO\")"
