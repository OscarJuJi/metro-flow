# Deploying

The image is plain Docker and runs anywhere. What follows is the path this
project is set up for, and what to change for the others.

## What ships

The image contains the venv and `artifacts/` — four small files totalling under
half a megabyte. It does **not** contain the panel, the training code's inputs,
or `model.joblib`.

That is deliberate. Training runs in the monthly GitHub Actions job, which gates
on the backtest and commits the artifacts; the image build then needs no network
access, no credentials and no LightGBM run, and the model that ships is exactly
the one the gate measured.

## Hugging Face Spaces

Spaces reads configuration from YAML front matter in the repository's own
`README.md`, which would put deployment metadata at the top of the project's
front page. So the Space gets its own README, kept in `deploy/huggingface/`, and
is pushed to a separate remote:

```bash
# once
huggingface-cli login
huggingface-cli repo create metro-pulse --type space --space_sdk docker
git remote add space https://huggingface.co/spaces/<user>/metro-pulse

# each release
git checkout -B space
cp deploy/huggingface/README.md README.md     # only on this branch
git commit -am "Space README"
git push --force space space:main
git checkout main
```

Spaces builds the `Dockerfile` and expects the app on port 7860, which is what
`PORT` defaults to.

## Fly.io

```bash
fly launch --no-deploy --name metro-pulse
fly deploy
```

Set `[http_service] internal_port = 7860` in `fly.toml`. The service holds
everything in memory and needs no volume: a redeploy is how new data arrives.

## Render

A Docker web service pointed at this repository. Set the health check path to
`/health`; it returns 503 until the artifacts are loadable, which is the
behaviour you want during a bad deploy.

## Running it locally

```bash
docker build -t metro-pulse .
docker run --rm -p 7860:7860 metro-pulse
```

Then open http://127.0.0.1:7860. Without Docker:

```bash
uvicorn metro_pulse.api.app:app --reload --port 7860
```

The service refuses to guess where the artifacts are when it is installed rather
than run from a checkout, so the image sets `METRO_PULSE_ARTIFACTS_DIR`
explicitly. Override it the same way if you keep artifacts somewhere else.
