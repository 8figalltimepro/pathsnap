# PathSnap UI

Vanilla HTML/CSS/JS + bundled Leaflet. Calls the FastAPI service for trips, predictions and metrics.

## Run (from project root)

```bash
scripts/run_api.sh   # API :8000
scripts/run_ui.sh    # UI :8501  ->  http://localhost:8501
```

API base URL defaults to `http://localhost:8000`; edit it in the API popover or open `?api=http://host:8000` (saved in localStorage).

## Model picker
Header select or a RECORDED table row re-runs GET /evaluate/{trip_id}?model=<id>. The table lists CV macro F1, student-test top-1 and HMM top-1; RESULTS names the model the live metrics belong to.
