#!/usr/bin/env bash
# Manual Cloud Run deploy (no CI deploy). Run one step at a time:  bash deploy/gcp.sh <step>
#   project -> billing -> apis -> budget -> infra -> secret -> push -> deploy
# Needs: gcloud (logged in), docker, and `python -m sdrag.export --out build/data` before `push`.
set -euo pipefail

PROJECT_ID=${PROJECT_ID:?set PROJECT_ID}
REGION=${REGION:-europe-west1}
BILLING_ACCOUNT=${BILLING_ACCOUNT:-}           # only needed for `billing` and `budget`
SERVICE=sdrag-api
REPO=sdrag
SA="sdrag-run@${PROJECT_ID}.iam.gserviceaccount.com"
BUCKET="${PROJECT_ID}-spend"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:${TAG:-latest}"
CLOUD_BUDGET_USD=${CLOUD_BUDGET_USD:-0.50}    # OpenAI cap for Cloud Run (local .env keeps the rest of $2)

case "${1:-}" in
  project)
    gcloud projects create "$PROJECT_ID" --name="sdrag arXiv RAG" ;;
  billing)
    gcloud billing projects link "$PROJECT_ID" --billing-account="${BILLING_ACCOUNT:?set BILLING_ACCOUNT}" ;;
  apis)
    gcloud services enable run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com \
      storage.googleapis.com iam.googleapis.com billingbudgets.googleapis.com --project "$PROJECT_ID" ;;
  budget)
    # alerts (email to billing admins) at 50/90/100% of gross cost, i.e. before free-trial credits
    gcloud billing budgets create --billing-account="${BILLING_ACCOUNT:?set BILLING_ACCOUNT}" \
      --display-name="sdrag monthly" --budget-amount="${BUDGET_AMOUNT:-3EUR}" \
      --filter-projects="projects/${PROJECT_ID}" --credit-types-treatment=exclude-all-credits \
      --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 --threshold-rule=percent=1.0 \
      --billing-project="$PROJECT_ID" ;;
  infra)
    gcloud artifacts repositories create "$REPO" --repository-format=docker --location="$REGION" --project "$PROJECT_ID"
    # keep only the 2 newest images so storage stays near the 0.5 GB free tier
    mkdir -p build && printf '[{"name":"keep-2","action":{"type":"Keep"},"mostRecentVersions":{"keepCount":2}},
{"name":"delete-rest","action":{"type":"Delete"},"condition":{"tagState":"any"}}]' > build/cleanup-policy.json
    gcloud artifacts repositories set-cleanup-policies "$REPO" --location="$REGION" --project "$PROJECT_ID" \
      --policy=build/cleanup-policy.json --no-dry-run
    gcloud iam service-accounts create sdrag-run --display-name="sdrag Cloud Run" --project "$PROJECT_ID"
    gcloud storage buckets create "gs://${BUCKET}" --location="$REGION" --project "$PROJECT_ID" \
      --uniform-bucket-level-access --public-access-prevention
    gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" --member="serviceAccount:${SA}" \
      --role=roles/storage.objectUser ;;
  secret)
    # reads OPENAI_API_KEY from .env without printing it
    grep -m1 '^OPENAI_API_KEY=' .env | cut -d= -f2- | tr -d '\r\n' |
      gcloud secrets create openai-api-key --data-file=- --replication-policy=automatic --project "$PROJECT_ID"
    gcloud secrets add-iam-policy-binding openai-api-key --member="serviceAccount:${SA}" \
      --role=roles/secretmanager.secretAccessor --project "$PROJECT_ID" ;;
  push)
    gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
    docker build -t "$IMAGE" .
    docker push "$IMAGE" ;;
  deploy)
    gcloud run deploy "$SERVICE" --image "$IMAGE" --region "$REGION" --project "$PROJECT_ID" \
      --service-account "$SA" --allow-unauthenticated \
      --min-instances 0 --max-instances 1 --cpu 1 --memory 2Gi --concurrency 4 --cpu-boost --timeout 120 \
      --set-secrets OPENAI_API_KEY=openai-api-key:latest \
      --set-env-vars "SPEND_PATH=gs://${BUCKET}/openai_spend.json,OPENAI_BUDGET_USD=${CLOUD_BUDGET_USD},RATE_LIMIT_PER_MIN=20" ;;
  *)
    echo "usage: PROJECT_ID=... bash deploy/gcp.sh {project|billing|apis|budget|infra|secret|push|deploy}"; exit 1 ;;
esac
