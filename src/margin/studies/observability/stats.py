"""Small-domain sensitivity summaries for observability study."""

from __future__ import annotations

import numpy as np
import pandas as pd


def domain_sensitivity_summary(
    rows: pd.DataFrame,
    value_column: str,
    *,
    confidence_level: float,
    wild_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return domain points plus equal-domain jackknife and wild-cluster inference."""

    clean = rows[["domain_id", value_column]].replace([np.inf, -np.inf], np.nan).dropna()
    domains = (
        clean.groupby("domain_id", observed=True)[value_column]
        .agg([("estimate", "mean"), ("n_rows", "size")])
        .reset_index()
    )
    if domains.empty:
        return domains, pd.DataFrame()
    values = domains["estimate"].to_numpy(dtype=float)
    estimate = float(values.mean())
    leave_one_out = np.array(
        [np.delete(values, index).mean() for index in range(len(values))], dtype=float
    )
    jackknife_se = float(
        np.sqrt(
            (len(values) - 1) / len(values) * np.sum((leave_one_out - leave_one_out.mean()) ** 2)
        )
    )
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(wild_replicates, len(values)), replace=True)
    wild = estimate + np.mean(signs * (values - estimate), axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    ci_low, ci_high = np.quantile(wild, [alpha, 1.0 - alpha])
    summary = pd.DataFrame(
        [
            {
                "metric": value_column,
                "estimate": estimate,
                "wild_ci_low": float(ci_low),
                "wild_ci_high": float(ci_high),
                "jackknife_se": jackknife_se,
                "leave_one_domain_out_min": float(leave_one_out.min()),
                "leave_one_domain_out_max": float(leave_one_out.max()),
                "positive_domains": int((values > 0).sum()),
                "negative_domains": int((values < 0).sum()),
                "zero_domains": int((values == 0).sum()),
                "n_domains": int(len(values)),
                "n_rows": int(len(clean)),
            }
        ]
    )
    return domains, summary
