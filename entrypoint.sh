#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
#  Entrypoint for HF Spaces — loads secrets from ENV variable
# ─────────────────────────────────────────────────────────────────────
#  HF Spaces injects secrets as environment variables automatically.
#
#  Option 1 (recommended): Add each secret individually in Settings:
#    HF_TOKEN=hf_...
#    MODEL_NAME=Qwen/Qwen2.5-3B-Instruct
#    USE_LOCAL_MODEL=1
#    ADAPTER_MODEL_ID=Hk4crprasad/email-triage-grpo
#
#  Option 2: Paste ALL env vars as a single secret named "ENV":
#    HF_TOKEN=hf_xxx
#    MODEL_NAME=Qwen/Qwen2.5-3B-Instruct
#    USE_LOCAL_MODEL=1
#    ADAPTER_MODEL_ID=Hk4crprasad/email-triage-grpo
#
#  This script detects Option 2 and exports each line.
# ─────────────────────────────────────────────────────────────────────

# If a secret named ENV exists, parse and export each KEY=VALUE line
if [ -n "$ENV" ]; then
    echo "📦 Loading environment from ENV secret..."
    while IFS= read -r line; do
        # Skip empty lines and comments
        line=$(echo "$line" | xargs)  # trim whitespace
        [[ -z "$line" || "$line" == \#* ]] && continue
        # Export KEY=VALUE
        export "$line"
        # Print key name (not value) for debugging
        key=$(echo "$line" | cut -d= -f1)
        echo "  ✓ $key"
    done <<< "$ENV"
    echo "✅ Environment loaded"
fi

# Start the server
exec uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers 1
