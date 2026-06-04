from datetime import datetime, timezone


def parse_iso_utc(value: str) -> datetime:
    norm = value.replace("Z", "+00:00")
    if not norm.endswith(("+00:00", "-00:00", "Z")):
        norm = norm + "+00:00"
    dt = datetime.fromisoformat(norm)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def datetime_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)
