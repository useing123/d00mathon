"""Deterministic scoring (no LLM)."""
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal


def criterion_points(cr, value):
    """Return (earned, max) Decimal for one criterion, or None if not applicable/unweighted."""
    w = Decimal(str(cr.get("weight", 0)))
    if cr.get("gross") or w == 0 or value is None or value == "NA":
        return None
    if cr.get("type", "binary") == "binary":
        if value not in ("YES", "NO"):
            raise ValueError(f"criterion {cr['id']}: binary value must be YES/NO/NA, got {value!r}")
        return (w if value == "YES" else Decimal(0)), w
    scale = sorted(int(x) for x in cr["scale"])
    v = int(value)
    if v not in scale:
        raise ValueError(f"criterion {cr['id']}: {v} not in scale {scale}")
    top = Decimal(scale[-1])
    return (w * Decimal(v) / top if top else Decimal(0)), w


def compute_total(criteria, values, rounding="half_up", decimals=2):
    """values: {criterion_id: 'YES'|'NO'|'NA'|int}. Returns dict with total (float|None) and gross flag.
    NA criteria are dropped and the remaining weights rescaled so the max is 100."""
    earned = Decimal(0)
    maximum = Decimal(0)
    gross = False
    for cr in criteria:
        v = values.get(cr["id"])
        if cr.get("gross"):
            gross = gross or v == "YES"
            continue
        pts = criterion_points(cr, v)
        if pts is None:
            continue
        earned += pts[0]
        maximum += pts[1]
    if maximum == 0:
        return {"total": None, "gross_error": gross, "max_weight_applicable": 0.0}
    q = Decimal(1).scaleb(-decimals)
    mode = ROUND_HALF_EVEN if rounding == "bankers" else ROUND_HALF_UP
    total = (earned * Decimal(100) / maximum).quantize(q, rounding=mode)
    return {"total": float(total), "gross_error": gross, "max_weight_applicable": float(maximum)}
