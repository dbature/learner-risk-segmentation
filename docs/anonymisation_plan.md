# Data Anonymisation Plan

Implemented in `src/governance/anonymise.py` and tested in
`tests/test_privacy_bias.py`.

## 1. What identifies a learner in this data

| Type | Fields | Present in OULAD |
|---|---|---|
| Direct identifiers (name, email, phone, address, date of birth) | none | **No.** Checked on every run by the PII guard |
| Source-system identifier | `id_student` | Yes, a number assigned by the Open University |
| Quasi-identifiers | gender, region, age_band, disability, imd_band, highest_education | Yes |

OULAD was released already de-identified. The honest risk is therefore not a
name in a column. It is (a) re-linking through `id_student` and (b) singling a
person out through a rare combination of quasi-identifiers (Sweeney, 2002).

## 2. Controls

| Control | Mechanism | Result on the October 2026 run |
|---|---|---|
| PII guard | Fails the run if any column name matches a direct-identifier pattern or any text value looks like an email or phone number. Tested with Faker-generated names, emails and phone numbers | 0 columns flagged |
| Pseudonymisation | `id_student` replaced by HMAC-SHA256 with a secret salt, truncated to 16 hex characters. Same learner, same key, so splits stay grouped by person | 24,710 distinct learners keyed; `id_student` absent from every published file |
| Secret handling | Salt read from `PIPELINE_SALT`; run fails before reading data if absent; never committed | Enforced |
| Attribute separation | Sensitive attributes written only to `audit_attributes` (Tier 3). The model table carries none | `model_ready` has 0 sensitive columns |
| Write, audit, publish | Files are staged, validated by GX, and only then promoted | A failed run leaves `data/processed/` empty |
| Audit log | Metadata only (counts, column names, hashes), never a learner value | Tested: no learner value appears in the log |

## 3. Re-identification risk, measured

k-anonymity counted on distinct people, not rows, so a learner on two
presentations cannot make up two members of one group.

| Quasi-identifier set | Smallest group | People in groups smaller than 5 |
|---|---|---|
| All six | 1 | 1,142 (4.6% of 24,710) |
| Without region | 2 | 132 (0.5%) |

Counting rows instead of people had reported a smallest group of 2. The
correction found learners who are unique on their attributes.

**Decision.** Region is the attribute that drives most of the risk, and it is
neither a model feature nor one of the fairness attributes in `params.yaml`.
Any release beyond Tier 3, including dashboard cohort views, therefore
excludes region and suppresses any cell under 5 learners. `audit_attributes`
keeps region for the steward, under Tier 3 access.

## 4. What this plan does not claim

Pseudonymised data remains personal data, because whoever holds the salt can
re-link it (EDPB, 2025). The plan reduces exposure and makes misuse
detectable. It does not make the data anonymous, and the governance framework
treats it accordingly.

## References

European Data Protection Board. (2025). *Guidelines 01/2025 on pseudonymisation*
(Version for public consultation, adopted 16 January 2025).

Sweeney, L. (2002). k-anonymity: A model for protecting privacy. *International
Journal of Uncertainty, Fuzziness and Knowledge-Based Systems, 10*(5), 557-570.
https://doi.org/10.1142/S0218488502001648
