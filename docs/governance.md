# Data Governance Framework

Builds on the Ethical AI Charter (Module 1) and the Privacy Plan (Module 2).
It sets out who may touch which data, how long each layer is kept, and how
compliance is evidenced. The project uses OULAD as a structural analogue for
learner data at TaRL Africa, so the framework is written for the deployment
case, where the data would be personal data about real learners, and applied
in full to OULAD now.

## Principles

| Principle | Source | How the pipeline applies it |
|---|---|---|
| Data minimisation | GDPR Art. 5(1)(c); NDPA 2023 s.24 | The model-ready table carries 19 features and no demographics. Sensitive attributes travel separately and only for fairness audit |
| Storage limitation | GDPR Art. 5(1)(e); NDPA 2023 s.24 | Retention periods per layer, below |
| Data protection by design | GDPR Art. 25 | Pseudonymisation, PII guard and the validation gate are code that runs on every run, not policy that relies on memory |
| Pseudonymised data is still personal data | EDPB (2025) | `model_ready` is treated as personal data and access-controlled, not published |
| Accountability | GDPR Art. 5(2); NDPA 2023 s.24 | Privacy audit log of every read, write and transformation |

## Roles

| Role | Held by | Responsibilities |
|---|---|---|
| Data Owner | The Open University for OULAD (CC BY 4.0). In deployment, the programme organisation | Authorises use; sets purpose |
| Data Steward | Project lead (Desmond Amos Bature) | Runs the pipeline; holds `PIPELINE_SALT`; approves schema and threshold changes; reviews the audit log |
| Analyst | Model developers (Module 4) | Reads `model_ready` and `early_leavers` only |
| Fairness Auditor | Named reviewer, separate from the analyst in deployment | Reads `audit_attributes` to compute subgroup metrics |
| Ethics Committee | As constituted in the Module 1 Charter | Release gate: approves the DPIA and any model release |
| Coach | End user (Module 5 dashboard) | Sees a ranked caseload and cohort aggregates. No raw data, no sensitive attributes, no learner keys |

## Access tiers

| Tier | Data | Contains | Access |
|---|---|---|---|
| 0 | `data/raw/` | `id_student`, all demographics | Data Steward |
| 1 | `data/interim/`, `data/quarantine/` | `id_student`, all demographics | Data Steward |
| 2 | `data/processed/model_ready`, `early_leavers` | Pseudonymous key, features, labels | Analyst, Data Steward |
| 3 | `data/processed/audit_attributes` | Pseudonymous key and sensitive attributes | Fairness Auditor, Data Steward |
| 4 | Dashboard, `reports/` | Aggregates only | Coach, Ethics Committee, public repository |
| Secret | `PIPELINE_SALT` | The pseudonymisation key | Data Steward only, stored outside the repository |

**How this is enforced in the capstone.** Tiers 0 to 3 never enter version
control (`.gitignore`) or the Docker image (`.dockerignore`); data is mounted at
run time; the container runs as a non-root user; the salt is an environment
variable and the run fails without it. **In deployment** each tier maps to its own
storage location with role-based permissions, so that the separation above is
enforced by the platform rather than by folder convention.

## Retention

| Data | Retained for | Then |
|---|---|---|
| Raw source extract | Duration of the programme cycle it describes, plus 12 months for audit | Deleted |
| Interim and quarantine | Overwritten on every run | Purged at project close |
| `model_ready`, `audit_attributes` | Life of the model version trained on them, versioned by DVC | Deleted with the retired model |
| `PIPELINE_SALT` | One data cycle | Rotated. Rotation deliberately makes old and new pseudonyms unlinkable |
| Privacy audit log | At least 12 months | Archived, then deleted |
| Validation results and reports | Life of the release they support | Kept with the release record |

## Change control

* Thresholds (`params.yaml`), expectation suites (`gx/expectations/`) and the
  feature list (`src/features/build_features.py`) change only by commit, with CI
  passing.
* The recall-parity tolerance is pinned by a test that fails if it is relaxed.
* Adding a source field requires a data dictionary entry and an expectation.

## Evidence produced on every run

| Evidence | File |
|---|---|
| Audit trail of every data access | `logs/privacy_audit.jsonl` |
| Validation result, 92 expectations | `reports/validation_summary.json`, GX Data Docs |
| Representation bias report | `reports/representation_bias.json` |
| Re-identification risk | `reports/privacy_report.json` |

## References

European Data Protection Board. (2025). *Guidelines 01/2025 on pseudonymisation*
(Version for public consultation, adopted 16 January 2025).
https://www.edpb.europa.eu/system/files/2025-01/edpb_guidelines_202501_pseudonymisation_en.pdf

Nigeria Data Protection Act, 2023. Federal Republic of Nigeria.

Regulation (EU) 2016/679 of the European Parliament and of the Council (General
Data Protection Regulation). https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32016R0679
