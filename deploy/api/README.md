---
title: Learner Risk API
emoji: 📈
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: cc-by-4.0
short_description: Day-30 learner non-completion risk, with reasons
---

# Learner risk API (BAN6800 capstone)

Public `/predict` endpoint for the day-30 non-completion risk model built in
the [learner-risk-segmentation](https://github.com/dbature/learner-risk-segmentation)
repository. Interactive docs are at `/docs`.

* `POST /predict`: one learner's day-30 features, returns a risk score, the flag
  at the validated threshold, the top three reasons and an `explanation_id`.
* `GET /explanations/{id}`: the stored audit record for that prediction.
* `GET /health`, `GET /model-info`.

The model is registered in **Staging** and has not been released for use with
real learners: it fails the fairness release gate on disability and
deprivation. Trained on the Open University Learning Analytics Dataset
(Kuzilek et al., 2017, CC BY 4.0) as an analogue. This Space is deployed
automatically by GitHub Actions from the repository's `main` branch.
