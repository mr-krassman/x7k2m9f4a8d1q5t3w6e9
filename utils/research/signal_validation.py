"""Общая логика подтверждения сигналов: permutation, Bonferroni, 60% val."""

from __future__ import annotations

from enum import Enum

import numpy as np

ALPHA = 0.05
VAL_CONFIRM_RATIO = 0.60
_PERM_N = 20_000
_PERM_RNG = np.random.default_rng(42)


class ConfirmationMode(Enum):
    """cohort — знак pooled train vs val-единицы; per_pair — знак train vs val у каждой пары."""

    COHORT = "cohort"
    PER_PAIR = "per_pair"


def bonferroni_alpha(n_tests: int) -> float:
    if n_tests <= 0:
        return ALPHA
    return ALPHA / n_tests


def permutation_p(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 2:
        return float("nan")
    obs = float(arr.mean())
    if obs == 0.0:
        return 1.0
    signs = _PERM_RNG.choice([-1.0, 1.0], size=(_PERM_N, n))
    perm_means = (signs * arr).mean(axis=1)
    return float(np.mean(np.abs(perm_means) >= abs(obs)))


def fmt_p(p: float) -> str:
    if p != p:
        return "n/a"
    if p <= 0:
        return "<0.0001"
    text = f"{p:.4f}"
    if text == "0.0000":
        return "<0.0001"
    return text


def fmt_delta_pp(delta: float | None) -> str:
    if delta is None or delta != delta:
        return "—"
    return f"{delta:+.2f} п.п."


def signal_status(
    p_value: float,
    val_agree: int | None,
    val_total: int,
    *,
    bonferroni_n: int,
    min_effect_pp: float | None = None,
    train_effect_pp: float | None = None,
) -> str:
    if min_effect_pp is not None and train_effect_pp is not None:
        if train_effect_pp != train_effect_pp or abs(train_effect_pp) < min_effect_pp:
            return "не значим"
    if p_value != p_value or p_value >= ALPHA:
        return "не значим"
    if val_total == 0 or val_agree is None:
        return "не значим"
    if p_value < bonferroni_alpha(bonferroni_n) and val_agree / val_total >= VAL_CONFIRM_RATIO:
        return "значим"
    return "не значим"


def val_confirm_text(val_agree: int | None, val_total: int) -> str:
    if val_total == 0 or val_agree is None:
        return "—"
    pct = round(val_agree / val_total * 100)
    mark = "✅" if val_agree / val_total >= VAL_CONFIRM_RATIO else "❌"
    return f"{val_agree}/{val_total} ({pct}%) {mark}"


def intersect_status(*statuses: str) -> str:
    if statuses and all(s == "значим" for s in statuses):
        return "значим"
    return "не значим"


def count_cohort_sign_confirm(
    train_effect: float,
    val_unit_effects: dict[str, float],
) -> tuple[int | None, int]:
    if not val_unit_effects:
        return None, 0
    if train_effect != train_effect or train_effect == 0:
        return None, len(val_unit_effects)
    sign = 1 if train_effect > 0 else -1
    agree = sum(
        1
        for effect in val_unit_effects.values()
        if (effect > 0 and sign > 0) or (effect < 0 and sign < 0)
    )
    return agree, len(val_unit_effects)


def count_per_pair_sign_confirm(
    train_unit_effects: dict[str, float],
    val_unit_effects: dict[str, float],
) -> tuple[int | None, int]:
    common = sorted(set(train_unit_effects) & set(val_unit_effects))
    if not common:
        return None, 0
    agree = 0
    for unit in common:
        train_e = train_unit_effects[unit]
        val_e = val_unit_effects[unit]
        if train_e == 0 or val_e == 0:
            continue
        if (train_e > 0 and val_e > 0) or (train_e < 0 and val_e < 0):
            agree += 1
    return agree, len(common)


def count_confirm(
    mode: ConfirmationMode,
    train_effect: float,
    train_unit_effects: dict[str, float],
    val_unit_effects: dict[str, float],
) -> tuple[int | None, int]:
    if mode is ConfirmationMode.PER_PAIR:
        return count_per_pair_sign_confirm(train_unit_effects, val_unit_effects)
    return count_cohort_sign_confirm(train_effect, val_unit_effects)
