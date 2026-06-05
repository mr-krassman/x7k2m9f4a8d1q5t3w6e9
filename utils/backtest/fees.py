"""Комиссии Bybit для бэктеста."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeeSchedule:
    name: str
    maker_pct: float
    taker_pct: float
    region: str

    @property
    def round_trip_taker_pct(self) -> float:
        return 2.0 * self.taker_pct

    @property
    def round_trip_maker_pct(self) -> float:
        return 2.0 * self.maker_pct


# USDT perpetual, non-VIP, RU/CIS с 07.03.2024 (Bybit help center).
BYBIT_RU_USDT_PERP = FeeSchedule(
    name="Bybit USDT Perpetual (RU/CIS, non-VIP)",
    maker_pct=0.0360,
    taker_pct=0.1000,
    region="RU/CIS",
)

DEFAULT_FEE = BYBIT_RU_USDT_PERP
