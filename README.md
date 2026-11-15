# Training Data Generator

Generates chat-format JSONL training files from validated inference samples and uploads them to
S3-compatible storage for model fine-tuning. Port 8152.

| | |
|---|---|
| **Stack** | Python 3.14 · FastAPI · SQLAlchemy · aioboto3 |
| **Database** | PostgreSQL 18 for batch metadata |

## What Training Data Generator Does

When a training trigger fires or an operator calls the check-and-generate endpoint, the training
data generator queries the data-collector service for validated inference samples grouped by
schema ID. Each schema represents a distinct function-calling pattern that has its own fine-tuned
LoRA adapter. The generator evaluates four retraining trigger conditions per schema — initial
sample threshold reached, enough new samples accumulated, model quality dropped below target, or
too much time elapsed since the last training run — and builds batches only for schemas that need
fresh training data.

For each triggered schema, the service fetches unused validated samples from the data-collector,
then runs them through a multi-stage quality filter. The filter strips markdown code fences from
JSON responses, rejects samples that are too short or too long (with per-schema length overrides),
flags LLM refusal patterns like "I cannot" or "As an AI", and validates that each response is
well-formed JSON. When the pool of new samples is thin, the service can supplement batches by
reusing previously-consumed teacher samples — high-quality responses from stronger fallback
providers — up to a configurable ratio cap, deduplicating by content hash. It also fetches the
system prompt from the model-registry routing rule so training data includes the same system
message the model sees at inference time.

The surviving samples are shuffled, split into train/valid sets (default 90/10), serialized as
chat-format JSONL (`{"messages": [...]}`), and uploaded to object storage under a structured path
like `training-data/{schema_id}/batch-{uuid}/`. The service then marks the consumed samples as
used in the data-collector. Batch metadata — sample counts, filter reasons, provider breakdown,
storage paths — is persisted to PostgreSQL so operators can audit generation history.

## Features

- **Trigger-based batch generation** — four retraining conditions evaluated per schema
- **Multi-stage quality filtering** — removes short/long responses, refusal patterns, invalid
  JSON, and markdown code fences
- **Per-schema length overrides** — different max response lengths per schema
- **Teacher sample reuse** — supplements batches with previously-used high-quality fallback
  responses, capped by ratio and deduplicated by content hash
- **System prompt injection** — fetches the system prompt from model-registry routing rules
- **Provider exclusion** — excludes named providers so the student model never trains on its own
  outputs
- **Chat JSONL format** — outputs `{"messages": [...]}` directly consumable by a PEFT pipeline
- **Train/valid splitting** — shuffles and splits samples into configurable sets
- **Object-storage upload** — uploads JSONL files with structured paths
- **Batch audit trail** — persists generation metadata for the list and detail endpoints

## Quick Start

With Docker Compose (service + PostgreSQL 18 + MinIO):

```bash
docker compose up --build
curl -s http://localhost:8152/health/ready
```

With `uv` and services you already run:

```bash
uv sync
TRAINING_DATA_GEN_ENVIRONMENT=development \
TRAINING_DATA_GEN_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5455/training_data_generator \
TRAINING_DATA_GEN_DATA_COLLECTOR_URL=http://localhost:8151 \
TRAINING_DATA_GEN_MODEL_REGISTRY_URL=http://localhost:8155 \
TRAINING_DATA_GEN_MINIO_ENDPOINT=localhost:9000 \
  uv run uvicorn src.main:app --host 0.0.0.0 --port 8152
```

Migrations run automatically to `head` on startup.

## API Endpoints

### Batches `/api/batches`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/generate` | Generate a training batch for a schema |
| GET | `/` | List generated batches (filter by schema_id) |
| GET | `/{batch_id}` | Get details for a specific batch |

### Triggers `/api/triggers`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/check` | Check which schemas need retraining |
| POST | `/check-and-generate` | Check triggers and auto-generate batches |

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info and endpoint listing |
| GET | `/health` | Basic health check |
| GET | `/health/live` | Liveness probe |
| GET | `/health/ready` | Readiness probe (data-collector and object storage) |

## Configuration

All variables are prefixed with `TRAINING_DATA_GEN_`.

| Variable | Default | Description |
|----------|---------|-------------|
| `TRAINING_DATA_GEN_SERVICE_NAME` | `training-data-generator` | Service name for telemetry |
| `TRAINING_DATA_GEN_PORT` | `8152` | HTTP listen port |
| `TRAINING_DATA_GEN_DATABASE_URL` | `""` | PostgreSQL DSN (`postgresql+psycopg://…`); required in production |
| `TRAINING_DATA_GEN_DATA_COLLECTOR_URL` | `http://data-collector.example.com:8151` | Data-collector URL |
| `TRAINING_DATA_GEN_MODEL_REGISTRY_URL` | `http://model-registry.example.com:8155` | Model-registry URL |
| `TRAINING_DATA_GEN_MINIO_ENDPOINT` | `minio.example.com:9000` | S3/MinIO endpoint |
| `TRAINING_DATA_GEN_MINIO_ACCESS_KEY` | `""` | S3/MinIO access key |
| `TRAINING_DATA_GEN_MINIO_SECRET_KEY` | `""` | S3/MinIO secret key |
| `TRAINING_DATA_GEN_MINIO_BUCKET` | `training-data` | Bucket for training data |
| `TRAINING_DATA_GEN_MINIO_USE_SSL` | `false` | Use TLS for object storage |
| `TRAINING_DATA_GEN_INITIAL_SAMPLE_THRESHOLD` | `50` | Minimum quality samples for first training |
| `TRAINING_DATA_GEN_SUBSEQUENT_SAMPLE_THRESHOLD` | `30` | New samples needed since the last training |
| `TRAINING_DATA_GEN_QUALITY_THRESHOLD` | `0.80` | Quality below which retraining triggers |
| `TRAINING_DATA_GEN_RETRAIN_INTERVAL_DAYS` | `14` | Maximum days between retraining runs |
| `TRAINING_DATA_GEN_MIN_SAMPLES` | `10` | Minimum samples required to generate a batch |
| `TRAINING_DATA_GEN_MAX_RESPONSE_LENGTH` | `10000` | Default maximum response length |
| `TRAINING_DATA_GEN_MAX_RESPONSE_LENGTH_OVERRIDES` | `{}` | JSON dict of schema_id → max length |
| `TRAINING_DATA_GEN_REUSE_TEACHER_SAMPLES` | `true` | Reuse teacher samples to supplement batches |
| `TRAINING_DATA_GEN_MAX_TEACHER_RATIO` | `0.7` | Maximum fraction of reused teacher samples |
| `TRAINING_DATA_GEN_TRAIN_SPLIT` | `0.9` | Fraction of samples used for training |
| `TRAINING_DATA_GEN_OTLP_TRACES_ENDPOINT` | `http://localhost:4318/v1/traces` | OTLP HTTP trace endpoint |
| `TRAINING_DATA_GEN_ENVIRONMENT` | `production` | `production` enables the required-field check |
| `TRAINING_DATA_GEN_LOG_LEVEL` | `INFO` | Root log level |
| `TRAINING_DATA_GEN_LOG_FORMAT` | `json` | Log format (`json` or `console`) |

## Architecture

```
src/
├── main.py                      # FastAPI app, lifespan, health endpoints
├── settings.py                  # Pydantic BaseSettings (TRAINING_DATA_GEN_ prefix)
├── db.py                        # async SQLAlchemy engine + session factory
├── logging_config.py            # structured logging (structlog)
├── telemetry.py                 # OpenTelemetry tracing setup
├── models/
│   ├── db_models.py             # SQLAlchemy ORM table (training_batches)
│   └── domain.py                # Pydantic request/response schemas
├── routers/
│   ├── batches.py               # /api/batches
│   └── triggers.py              # /api/triggers
└── services/
    ├── batch_service.py         # fetch, filter, format, upload
    ├── collector_client.py      # HTTP client for data-collector
    ├── registry_client.py       # HTTP client for model-registry
    ├── filter_service.py        # sample quality filtering
    ├── minio_client.py          # S3-compatible upload (aioboto3)
    └── trigger_service.py       # retraining trigger evaluation
```

## Testing

The suite requires PostgreSQL. The DSN is resolved by `tests/conftest.py` from
`TEST_DATABASE_URL` → `DATABASE_URL`, with no host:5432 fallback.

```bash
docker run -d --name tdg-testpg -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=training_data_generator -p 5455:5432 postgres:18-alpine

TEST_DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5455/training_data_generator" \
  uv run pytest
```

Lint and format:

```bash
uv run ruff check .
uv run ruff format --check .
```

The suite enforces an 80% coverage floor (`pyproject.toml`).

## CI

`.github/workflows/ci-public.yml` runs on GitHub-hosted runners: betterleaks secret scan, the
private-reference guard, ruff, the full pytest suite against a PostgreSQL 18 service container,
and `pip-audit`. The image build/publish job is off unless the repository variable
`PUBLISH_IMAGE=true` is set.

## Docker

```bash
docker build -t training-data-generator .
docker run --rm -p 8152:8152 \
  -e TRAINING_DATA_GEN_ENVIRONMENT=development \
  -e TRAINING_DATA_GEN_DATABASE_URL=postgresql+psycopg://postgres:postgres@host.docker.internal:5455/training_data_generator \
  training-data-generator
```

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Licensing

There is no licence gate in this service: it needs no licence key, token, or network call to
build, test or run. Every feature is available by default.
