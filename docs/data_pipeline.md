# Dataset pipeline strategy

## Design position

This prototype is format-flexible, not schema-agnostic.

It supports CSV, Excel, JSON, and Parquet, but meaningful screening depends on mapping incoming columns to the canonical digital trade compliance schema used by the rule engine and ML pipeline.

## Recommended ingestion flow

1. Load source file.
2. Infer probable column mapping from common aliases.
3. Let the user confirm or override required fields.
4. Canonicalise into the internal screening schema.
5. Validate required screening fields.
6. Run rule engine, anomaly model, classifier, and hybrid scoring.
7. Save mapping trace and screening report.

## Why not claim universal arbitrary input support

A compliance system should not pretend to produce reliable outputs from unknown, unstructured, or semantically ambiguous data. That would be weak science and weak governance.

The correct posture is:
- support multiple file formats
- support field alias inference
- support manual mapping override
- require canonicalisation before screening

## Extension path

For later SaaS deployment, add:
- persisted mapping profiles per customer
- confidence-based mapping approval workflow
- schema registry and version control
- OCR / document extraction as a separate upstream service
