"""Learner engagement segments (Module 1 segmentation layer, ticket #14).

Learners are grouped by the shape of their first four weeks of activity in
the virtual learning environment, measured against other learners on the same
module. Raw clicks are not comparable across modules: some courses are built
around the VLE and others are not, so raw-click clusters mostly recover the
module. Each week's clicks are therefore log-transformed and standardised
within module (mean and standard deviation from the training set only), and
K-means groups learners on those four module-relative values.

k = 4 was fixed in advance by the Module 1 design (steady, slipping,
disengaged, recovering). Silhouette scores for k = 3 to 5 are close (about
0.27 to 0.29), so the data does not argue strongly for a different k. That
modest separation is reported, not hidden: these are regions of a continuum,
useful for choosing the kind of support, not natural types of learner.

Segment names come from the centroids by a fixed rule, never by hand:
lowest overall activity -> Disengaged; highest -> Steady; of the other two,
the one whose week 1 is above its weeks 3 and 4 -> Slipping, the other ->
Late starter (the closest the data comes to Module 1's "recovering").
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

WEEKS = ["clicks_wk1", "clicks_wk2", "clicks_wk3", "clicks_wk4"]
K = 4
SEED = 42
ORDER = ["Steady", "Slipping", "Late starter", "Disengaged"]
DESCRIPTION = {
    "Steady": "Active above the module norm in every one of the first four weeks.",
    "Slipping": "Started above the module norm, then fell below it by weeks 3 and 4.",
    "Late starter": "Far below the module norm in week 1, close to it by weeks 3 and 4.",
    "Disengaged": "Far below the module norm in every one of the first four weeks.",
}


@dataclass
class TrajectorySegmenter:
    k: int = K
    seed: int = SEED
    reference: dict = field(default_factory=dict)  # module -> {week: (mean, std)}
    km: KMeans | None = None
    names: dict = field(default_factory=dict)       # cluster id -> segment name
    silhouette: dict = field(default_factory=dict)

    def _z(self, X: pd.DataFrame) -> np.ndarray:
        logc = np.log1p(X[WEEKS].astype(float).to_numpy())
        out = np.zeros_like(logc)
        mods = X["code_module"].to_numpy()
        for i, w in enumerate(WEEKS):
            mu = np.array([self.reference[m][w][0] for m in mods])
            sd = np.array([self.reference[m][w][1] for m in mods])
            out[:, i] = (logc[:, i] - mu) / sd
        return out

    def fit(self, X_train: pd.DataFrame, sample: int = 5000) -> "TrajectorySegmenter":
        logc = pd.DataFrame(np.log1p(X_train[WEEKS].astype(float).to_numpy()), columns=WEEKS)
        logc["code_module"] = X_train["code_module"].to_numpy()
        g = logc.groupby("code_module")
        self.reference = {m: {w: (float(g[w].mean()[m]), float(g[w].std()[m])) for w in WEEKS}
                          for m in g.groups}
        Z = self._z(X_train)
        rng = np.random.default_rng(0)
        idx = rng.choice(len(Z), min(sample, len(Z)), replace=False)
        for k in (3, 4, 5):
            lab = KMeans(k, n_init=10, random_state=self.seed).fit_predict(Z)
            self.silhouette[k] = float(silhouette_score(Z[idx], lab[idx]))
        self.km = KMeans(self.k, n_init=10, random_state=self.seed).fit(Z)
        self.names = self._name(self.km.cluster_centers_)
        return self

    @staticmethod
    def _name(centres: np.ndarray) -> dict:
        level = centres.mean(axis=1)
        low, high = int(np.argmin(level)), int(np.argmax(level))
        names = {low: "Disengaged", high: "Steady"}
        for c in range(len(centres)):
            if c in names:
                continue
            names[c] = "Slipping" if centres[c, 0] > centres[c, 2:].mean() else "Late starter"
        if sorted(names.values()) != sorted(ORDER):
            raise ValueError(f"centroids do not give the four expected shapes: {names}")
        return names

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Nearest centroid, which is exactly what KMeans.predict does."""
        d = ((self._z(X)[:, None, :] - self.centres[None, :, :]) ** 2).sum(axis=2)
        return np.array([self.names[int(c)] for c in d.argmin(axis=1)])

    @property
    def centres(self) -> np.ndarray:
        return self._centres if self.km is None else self.km.cluster_centers_

    def to_dict(self) -> dict:
        """Everything needed to assign a segment: module reference statistics
        (aggregates over the training set) and four centroids. No learner rows."""
        return {"weeks": WEEKS, "k": self.k, "seed": self.seed, "reference": {m: {w: list(v) for w, v in ws.items()} for m, ws in self.reference.items()},
                "centres": self.centres.tolist(), "names": {str(c): n for c, n in self.names.items()},
                "silhouette": {str(k): v for k, v in self.silhouette.items()}, "description": DESCRIPTION}

    @classmethod
    def from_dict(cls, d: dict) -> "TrajectorySegmenter":
        s = cls(k=d["k"], seed=d["seed"])
        s.reference = {m: {w: tuple(v) for w, v in ws.items()} for m, ws in d["reference"].items()}
        s._centres = np.array(d["centres"])
        s.names = {int(c): n for c, n in d["names"].items()}
        s.silhouette = {int(k): v for k, v in d["silhouette"].items()}
        return s

    def profile(self) -> pd.DataFrame:
        """Module-relative centroid of each segment (z-scores by week)."""
        rows = {self.names[c]: self.centres[c] for c in range(self.k)}
        return pd.DataFrame(rows, index=WEEKS).T.loc[ORDER]


def summary(segments: np.ndarray, y: np.ndarray, scores: np.ndarray, threshold: float,
            X: pd.DataFrame) -> pd.DataFrame:
    t = pd.DataFrame({"segment": segments, "y": y, "score": scores, "flag": scores >= threshold,
                      "submitted": X["submit_rate_by_30"].to_numpy(),
                      "clicks": X["clicks_0_29"].to_numpy()})
    out = t.groupby("segment").agg(
        learners=("y", "size"), non_completion=("y", "mean"), mean_risk=("score", "mean"),
        flagged=("flag", "mean"), share_submitted=("submitted", "mean"), median_clicks=("clicks", "median"))
    out["share"] = out["learners"] / out["learners"].sum()
    reached = t[t.y == 1].groupby("segment")["flag"].mean()
    out["non_completers_flagged"] = reached
    return out.loc[ORDER]
