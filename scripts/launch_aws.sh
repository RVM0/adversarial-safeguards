#!/usr/bin/env bash
# Launch an AWS p4de.24xlarge (8× A100 80GB) instance for the sweep.
#
# IMPORTANT: AWS A100 instances are ~8× more expensive than Lambda Labs per GPU,
# because AWS only sells A100s in 8-GPU bundles. We recommend Lambda Labs for
# cost; this script is provided because the user asked for an AWS path.
#
# Cost estimate: p4de.24xlarge at ~$40.97/hr × 60 hrs (2.5 days) ≈ $2,458.
# At spot pricing (~30% off): ~$1,720.
#
# A cheaper AWS option: g5.48xlarge (8× A10G 24GB) at ~$16/hr × 60 = ~$960.
# A10G can run all 4 models with QLoRA (24GB is tight for 32B but feasible).

set -euo pipefail

: "${AWS_PROFILE:=default}"
: "${AWS_REGION:=us-east-1}"
: "${KEY_PAIR_NAME?must be set}"
: "${SECURITY_GROUP_ID?must be set}"
: "${SUBNET_ID?must be set}"
: "${HF_TOKEN?must be set}"

INSTANCE_TYPE="${INSTANCE_TYPE:-g5.48xlarge}"  # 8x A10G is the cost-rational AWS pick
AMI_ID="${AMI_ID:-}"  # If empty, look up Deep Learning Base AMI

if [ -z "$AMI_ID" ]; then
    echo "[aws] Looking up latest Deep Learning Base AMI..."
    AMI_ID=$(aws ec2 describe-images \
        --region "$AWS_REGION" \
        --owners amazon \
        --filters "Name=name,Values=Deep Learning Base GPU AMI (Ubuntu 22.04)*" \
                  "Name=state,Values=available" \
        --query "sort_by(Images, &CreationDate)[-1].ImageId" \
        --output text)
    echo "[aws] AMI: $AMI_ID"
fi

USER_DATA=$(cat <<'EOF' | base64
#!/bin/bash
set -e
cd /home/ubuntu
apt-get update
sudo -u ubuntu git clone <PUT_YOUR_REPO_URL_HERE> advsafe || true
cd advsafe
sudo -u ubuntu bash scripts/setup_env.sh
EOF
)

echo "[aws] Launching $INSTANCE_TYPE in $AWS_REGION..."
INSTANCE_ID=$(aws ec2 run-instances \
    --region "$AWS_REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_PAIR_NAME" \
    --security-group-ids "$SECURITY_GROUP_ID" \
    --subnet-id "$SUBNET_ID" \
    --user-data "$USER_DATA" \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=500,VolumeType=gp3}' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Project,Value=advsafe}]' \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "[aws] Instance launched: $INSTANCE_ID"
echo "[aws] Waiting for running state..."

aws ec2 wait instance-running --region "$AWS_REGION" --instance-ids "$INSTANCE_ID"

PUBLIC_DNS=$(aws ec2 describe-instances \
    --region "$AWS_REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicDnsName' \
    --output text)

echo "[aws] Instance running: $PUBLIC_DNS"
echo ""
echo "[aws] Connect:"
echo "    ssh -i ~/.ssh/${KEY_PAIR_NAME}.pem ubuntu@$PUBLIC_DNS"
echo ""
echo "[aws] To start the sweep on the box:"
echo "    cd advsafe"
echo "    source .venv/bin/activate"
echo "    export HF_TOKEN=<your-hf-token>          # required (gated models)"
echo "    export OPENAI_API_KEY=<your-openai-key>  # required for D1/D2/D4 judge cells"
echo "    bash scripts/download_models.sh"
echo "    bash scripts/download_datasets.sh"
echo "    nohup advsafe-sweep --config configs/experiments/sweep.yaml > sweep.log 2>&1 &"
echo ""
echo "[aws] IMPORTANT — terminate when done:"
echo "    aws ec2 terminate-instances --region $AWS_REGION --instance-ids $INSTANCE_ID"
