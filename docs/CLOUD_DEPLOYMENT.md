# Cloud Deployment Guide

Two paths are documented: **Lambda Labs** (recommended for cost) and **AWS**
(documented because the project lead asked, but ~3-8× more expensive for
the same workload). RunPod is a drop-in Lambda Labs alternative.

---

## Cost comparison (60-hour sweep, single GPU)

| Provider | Instance | GPU | $/hr | 60-hr cost |
|---|---|---|---|---|
| **Lambda Labs** | `gpu_1x_a100_sxm4` | 1× A100 80GB | $1.29 | **~$77** |
| RunPod Community | A100 PCIe community | 1× A100 80GB | $0.79 | ~$48 |
| RunPod Secure | A100 PCIe secure | 1× A100 80GB | $1.89 | ~$114 |
| AWS (cheapest A100 option) | `p4de.24xlarge` | 8× A100 80GB | $40.97 | ~$2,458 |
| AWS (cost-rational alt) | `g5.48xlarge` | 8× A10G 24GB | $16.29 | ~$978 |

**Bottom line**: Lambda Labs is the right choice on cost. AWS is included
for compliance / familiarity reasons but burns budget.

---

## Path A — Lambda Labs (recommended)

### Prerequisites

1. Lambda Labs account: <https://lambdalabs.com>
2. Add a payment method.
3. Generate an API key under "API keys".
4. Upload your SSH public key under "SSH keys".
5. HuggingFace token with access to Llama / Gemma gated models.

### Launch

```bash
export LAMBDA_API_KEY=<your-key>
export LAMBDA_SSH_KEY_NAME=<your-key-name>
export HF_TOKEN=<your-hf-token>

bash scripts/launch_lambda.sh
```

The script prints SSH instructions and the instance ID when ready.

### Run the sweep

```bash
# from your local machine:
ssh ubuntu@<instance-ip>

# on the instance:
git clone <your-repo-url> advsafe
cd advsafe
export HF_TOKEN=<your-hf-token>

bash scripts/setup_env.sh
bash scripts/download_models.sh         # ~30 min, 150 GB of weights
bash scripts/download_datasets.sh

# Background the sweep so it survives SSH disconnects
nohup advsafe-sweep --config configs/experiments/sweep.yaml > sweep.log 2>&1 &
echo $! > sweep.pid

# Monitor:
tail -f sweep.log
```

### Tear down (do not forget — billing continues!)

```bash
curl -X POST -u $LAMBDA_API_KEY: \
    -H 'Content-Type: application/json' \
    -d '{"instance_ids":["<instance-id>"]}' \
    https://cloud.lambdalabs.com/api/v1/instance-operations/terminate
```

Or from the dashboard: <https://cloud.lambdalabs.com/instances>.

### Pull results back

```bash
# from your local machine:
rsync -avz ubuntu@<instance-ip>:advsafe/results/ ./results/
```

---

## Path B — AWS

### Why this is more expensive

AWS only sells A100 GPUs in 8-GPU instances (`p4d`, `p4de`). Even if you only
need one GPU, you pay for eight. The cost-rational AWS option is
`g5.48xlarge` (8× A10G 24GB) — A10G has 24GB which is enough for 14B fp16
but tight for 27/32B QLoRA. Plan accordingly.

### Prerequisites

1. AWS account with quota for the chosen instance type (file an increase request if needed).
2. EC2 key pair created in your target region.
3. Security group with port 22 open to your IP.
4. Subnet ID in the target region.
5. AWS CLI installed and `aws configure` run.

### Launch

```bash
export AWS_REGION=us-east-1
export KEY_PAIR_NAME=<your-keypair>
export SECURITY_GROUP_ID=sg-xxxxxxxx
export SUBNET_ID=subnet-xxxxxxxx
export HF_TOKEN=<your-hf-token>
export INSTANCE_TYPE=g5.48xlarge  # cost-rational; for true A100 use p4de.24xlarge

bash scripts/launch_aws.sh
```

### Run the sweep

```bash
# from your local machine:
ssh -i ~/.ssh/$KEY_PAIR_NAME.pem ubuntu@<public-dns>

# on the instance:
cd advsafe
source .venv/bin/activate
export HF_TOKEN=<your-hf-token>
bash scripts/download_models.sh
bash scripts/download_datasets.sh

nohup advsafe-sweep --config configs/experiments/sweep.yaml > sweep.log 2>&1 &
tail -f sweep.log
```

### Tear down

```bash
aws ec2 terminate-instances --region $AWS_REGION --instance-ids <instance-id>
```

### S3 results sync (optional)

Push results to S3 from the instance:

```bash
aws s3 cp --recursive results/ s3://<your-bucket>/advsafe-runs/$(date -I)/
```

---

## Path C — RunPod (Lambda Labs alternative)

If Lambda Labs is out of A100 capacity in your region, RunPod Secure Cloud
is the cleanest backup. Workflow is nearly identical:

```bash
# Via RunPod dashboard:
# - Choose: 1x A100 80GB PCIe, Secure Cloud
# - Template: PyTorch 2.5 + CUDA 12.4
# - Volume: 200 GB

# Then on the pod:
git clone <repo-url> advsafe && cd advsafe
export HF_TOKEN=<token>
bash scripts/setup_env.sh
bash scripts/download_models.sh
bash scripts/download_datasets.sh
nohup advsafe-sweep --config configs/experiments/sweep.yaml > sweep.log 2>&1 &
```

---

## Cost-control checklist

Before launching ANY cloud instance:

- [ ] Have I tested locally on M5 Pro (pilot.yaml) end-to-end?
- [ ] Did the pilot succeed on at least Llama 3.1 8B?
- [ ] Is my HF_TOKEN valid and have I accepted licenses for all 4 models?
- [ ] Have I set a calendar reminder to tear down?
- [ ] Is the budget cap configured at the provider (Lambda: dashboard; AWS: Budgets alert)?
- [ ] Do I know how to SSH in and view logs *without* re-launching?

After the sweep:

- [ ] Results synced back to local? (`rsync` or `aws s3 cp`)
- [ ] Manifest files complete? (Check `results/sweep/sweep_summary.json`)
- [ ] Instance terminated and confirmed?
- [ ] Billing dashboard sanity-checked?

---

## Roadmap: 2-3 day cloud sweep

This sweep is **designed** to fit in 2-3 days on a single A100 80GB:

| Phase | Hours | Note |
|---|---|---|
| Setup + downloads | 1-2 | one-time |
| Training (12 LoRA fine-tunes) | ~30-40 | 4 models × 3 attack-budget levels |
| Inference (400 cells) | ~20-30 | 4 models × 5 attacks × 5 defenses × 1 eval |
| Judge passes | included in inference | Llama Guard 3 loaded once |
| **Total** | **~50-72 hrs** | **= 2-3 days** |
| Buffer for debugging | +5-10 | recommended |

Budget: ~$77 on Lambda Labs, ~$48 on RunPod community, ~$978 on AWS g5.48xlarge.

If you need to compress to 1 day: rent 4× A100 in parallel (~$310 on Lambda)
and modify `advsafe-sweep` to shard cells across GPUs.
