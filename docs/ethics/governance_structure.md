# Ethical governance structure

| Body or role | Who | Authority | Meets |
|---|---|---|---|
| Programme Data Ethics Committee | Country director (chair), measurement lead, one instructional coach, a government partner representative, an external member independent of the programme (Module 1) | Approves the DPIA and every release; can suspend deployment; ratifies changes to fairness rules | Quarterly, and within 10 working days of a high fairness incident |
| Data Protection Officer | Designated by the programme (NDPA/GAID) | Privacy notices, data subject requests, compliance audit return | As needed; annual return |
| Measurement lead (model owner) | Programme measurement lead | Retraining decisions; owns monitoring KPIs | Monthly monitoring review |
| Measurement analyst | Pipeline maintainer | Runs pipeline and monitoring; first responder to alerts | Each course run |
| Fairness auditor | Named reviewer independent of model development | Signs the fairness section of each release | Each release |
| Coaches | Users | Override any flag; report harms | Weekly use |

**Decision rights.** No script promotes a model to Production. Promotion
needs (1) every gate passed in code, (2) the fairness auditor's sign-off and
(3) a recorded committee decision. Changing a fairness rule (such as the
pending chi-square rule for IMD) needs a committee vote, recorded in the
repository as a change to `params.yaml` with the minute referenced.

**Escalation.** Coach or analyst → measurement lead (2 working days) →
committee chair (5 working days) → full committee. Any member can call an
immediate suspension pending review.

The structure follows the internal audit approach of Raji et al. (2020):
documented scope, independent review and an artefact trail at every stage.
