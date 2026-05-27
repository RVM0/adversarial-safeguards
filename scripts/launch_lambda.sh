#!/usr/bin/env bash
# Launch a Lambda Labs A100 instance and run the sweep.
#
# Prerequisites:
#   - Lambda Labs API key in $LAMBDA_API_KEY
#   - SSH key uploaded to Lambda dashboard, name in $LAMBDA_SSH_KEY_NAME
#   - HuggingFace token in $HF_TOKEN
#
# Cost estimate: A100 80GB at $1.29/hr × 60 hrs (2.5 days) ≈ $77.

set -euo pipefail

: "${LAMBDA_API_KEY?must be set}"
: "${LAMBDA_SSH_KEY_NAME?must be set}"
: "${HF_TOKEN?must be set}"

INSTANCE_TYPE="${INSTANCE_TYPE:-gpu_1x_a100_sxm4}"
REGION="${REGION:-us-west-1}"

echo "[lambda] Launching $INSTANCE_TYPE in $REGION..."

LAUNCH_RESPONSE=$(curl -fsSL \
    -X POST \
    -H "Content-Type: application/json" \
    -u "$LAMBDA_API_KEY:" \
    "https://cloud.lambdalabs.com/api/v1/instance-operations/launch" \
    -d "{
        \"region_name\": \"$REGION\",
        \"instance_type_name\": \"$INSTANCE_TYPE\",
        \"ssh_key_names\": [\"$LAMBDA_SSH_KEY_NAME\"],
        \"name\": \"advsafe-sweep\"
    }")

INSTANCE_ID=$(echo "$LAUNCH_RESPONSE" | python -c "import json,sys; print(json.load(sys.stdin)['data']['instance_ids'][0])")
echo "[lambda] Launched: $INSTANCE_ID"
echo "[lambda] Waiting for active status..."

while true; do
    STATUS=$(curl -fsSL -u "$LAMBDA_API_KEY:" \
        "https://cloud.lambdalabs.com/api/v1/instances/$INSTANCE_ID" \
        | python -c "import json,sys; print(json.load(sys.stdin)['data']['status'])")
    if [ "$STATUS" = "active" ]; then break; fi
    echo "[lambda] status=$STATUS, sleeping 20s..."
    sleep 20
done

IP=$(curl -fsSL -u "$LAMBDA_API_KEY:" \
    "https://cloud.lambdalabs.com/api/v1/instances/$INSTANCE_ID" \
    | python -c "import json,sys; print(json.load(sys.stdin)['data']['ip'])")

echo "[lambda] Instance active at $IP"
echo ""
echo "[lambda] To start the sweep:"
echo "    ssh ubuntu@$IP"
echo "    git clone <your-repo-url> advsafe"
echo "    cd advsafe"
echo "    export HF_TOKEN=$HF_TOKEN"
echo "    bash scripts/setup_env.sh"
echo "    bash scripts/download_models.sh"
echo "    bash scripts/download_datasets.sh"
echo "    nohup advsafe-sweep --config configs/experiments/sweep.yaml > sweep.log 2>&1 &"
echo ""
echo "[lambda] To tear down (IMPORTANT — billing continues until then):"
echo "    curl -X POST -u $LAMBDA_API_KEY: \\"
echo "         -H 'Content-Type: application/json' \\"
echo "         -d '{\"instance_ids\":[\"$INSTANCE_ID\"]}' \\"
echo "         https://cloud.lambdalabs.com/api/v1/instance-operations/terminate"
