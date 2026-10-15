# Public demo document set

This folder is a curated, public-safe subset of Insti Buddy's full campus
document collection, used only for the live Streamlit Community Cloud demo.

It intentionally excludes internal/sensitive material (BoG minutes, student
council rulebooks, election management records, disciplinary records, and
parent-safety handbooks). The full private collection (`campus_docs/`) stays
local and is never committed to this repository.

Set the `DOCS_FOLDER` secret to `public_docs` in the deployed app's settings
so `ingest.py` builds the demo's knowledge base from this folder instead of
the full local one.
