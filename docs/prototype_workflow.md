# Prototype workflow

## End-to-end logic

1. Ingest source dataset in CSV, Excel, JSON, or Parquet.
2. Infer likely source-to-canonical mappings.
3. Require user confirmation for required fields when mapping confidence is incomplete.
4. Canonicalise into the internal trade screening schema.
5. Validate completeness and basic field quality.
6. Run deterministic compliance rules.
7. Run anomaly detection and supervised ML scoring.
8. Fuse outputs into a hybrid risk score and triage band.
9. Return report, explanations, and recommended review action.

## Real-world operating posture

The prototype is designed for pre-screening. It should sit upstream of formal compliance workflows and support, rather than replace, human review.

## Stakeholder value

- Exporters and importers: reduce preventable submission errors.
- Front desk and compliance teams: prioritise workload.
- Analysts and policy stakeholders: surface trend signals and recurring failure modes.
- Researchers: validate hybrid screening performance and interpretability.
