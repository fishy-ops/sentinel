import re
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Evidence(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    claim: str
    refs: list[str] = Field(min_length=1)


class ExplanationText(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    summary: str
    evidence: list[Evidence] = Field(min_length=1, max_length=8)
    risk_level: str = Field(pattern="^(low|medium|high)$")
    recommended_action: str = Field(pattern="^(approve|review|block_and_contact)$")
    confidence: float = Field(ge=0, le=1)
    limitations: str

    @classmethod
    def model_validate_json(cls, json_data: str | bytes, **kwargs: Any) -> "ExplanationText":
        result = super().model_validate_json(json_data, **kwargs)
        if len(result.summary.split()) > 80:
            raise ValueError("summary exceeds 80 words")
        return result


# Capitalised words that are ordinary vocabulary rather than names of merchants or places.
_PLAIN_WORDS = frozenset(
    "The This That It A An No Not Account Takeover Velocity Burst Amount Spike Structuring "
    "Dormant Drain Transaction Transactions Transfer Transfers Device Country Risk Score "
    "High Medium Low Flag Flagged Review Monday Tuesday Wednesday Thursday Friday Saturday "
    "Sunday January February March April May June July August September October November "
    "December".split()
)


def _values(record: dict[str, Any]) -> list[Any]:
    result: list[Any] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key not in ("ref", "meaning"):
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif value is not None and not isinstance(value, bool):
            result.append(value)

    collect(record)
    return result


def _number_matches(token: str, values: list[Any], ratio: bool = False) -> bool:
    try:
        target = Decimal(token.replace(",", "").replace("$", ""))
    except InvalidOperation:
        return False
    digits = len(token.split(".", 1)[1].rstrip("x%")) if "." in token else 0
    quantum = Decimal(1).scaleb(-digits)
    nums = [Decimal(str(value)) for value in values if type(value) in (int, float, Decimal)]
    candidates = nums + [a / b for a in nums for b in nums if b != 0] if ratio else nums
    return any(value.quantize(quantum, rounding=ROUND_HALF_UP) == target for value in candidates)


def check_grounding(
    explanation: ExplanationText, refs: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    all_values = [value for record in refs.values() for value in _values(record)]
    known_text = [str(value) for value in all_values if isinstance(value, str)]
    timestamps = []
    for value in known_text:
        try:
            timestamps.append(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            pass
    known_devices = {value for value in known_text if re.fullmatch(r"[\w-]+-D\d+", value)}
    known_merchants = {
        value
        for record in refs.values()
        for value in [record.get("untrusted_text", {}).get("merchant_name")]
        if isinstance(value, str)
    }
    known_merchants.update(
        merchant
        for record in refs.values()
        for merchant in record.get("untrusted_text", {}).get("merchant_names", [])
    )
    verdicts = []
    ungrounded = []
    for index, (claim, cited) in enumerate(
        [(explanation.summary, list({ref for item in explanation.evidence for ref in item.refs}))]
        + [(item.claim, item.refs) for item in explanation.evidence]
    ):
        issues = []
        missing = [ref for ref in cited if ref not in refs]
        for ref in missing:
            issues.append({"span": ref, "reason": "unknown_ref"})
        cited_values = [value for ref in cited if ref in refs for value in _values(refs[ref])]
        remaining = claim
        for match in re.finditer(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?\b", claim):
            span = match.group()
            dates = [str(value) for value in cited_values if isinstance(value, str)]
            if not any(date.startswith(span) for date in dates):
                reason = (
                    "ref_mismatch"
                    if any(str(value).startswith(span) for value in all_values)
                    else "not_found"
                )
                issues.append({"span": span, "reason": reason})
            remaining = remaining.replace(span, " ")
        for match in re.finditer(r"\b\d{1,2}:\d{2}(?:\s*(?:AM|PM))?\b", remaining, re.I):
            span = match.group()
            cited_times = [
                datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                for value in cited_values
                if isinstance(value, str) and re.match(r"\d{4}-\d{2}-\d{2}T", value)
            ]

            def matches_time(point: datetime, token: str) -> bool:
                return (
                    point.strftime("%H:%M") == token
                    or point.strftime("%I:%M %p").lstrip("0") == token.upper()
                )

            if not any(matches_time(point, span) for point in cited_times):
                reason = (
                    "ref_mismatch"
                    if any(matches_time(point, span) for point in timestamps)
                    else "not_found"
                )
                issues.append({"span": span, "reason": reason})
            remaining = remaining.replace(span, " ")
        for match in re.finditer(r"\b(?:at\s+)?(\d{1,2})\s*(AM|PM)\b", remaining, re.I):
            span = match.group()
            hour = int(match.group(1)) % 12 + (12 if match.group(2).upper() == "PM" else 0)
            cited_dates = [
                datetime.fromisoformat(value.replace("Z", "+00:00"))
                for value in cited_values
                if isinstance(value, str) and re.match(r"\d{4}-\d{2}-\d{2}T", value)
            ]
            if not any(point.hour == hour for point in cited_dates):
                reason = (
                    "ref_mismatch"
                    if any(point.hour == hour for point in timestamps)
                    else "not_found"
                )
                issues.append({"span": span, "reason": reason})
            remaining = remaining.replace(span, " ")
        for ref in cited:
            remaining = re.sub(rf"\brefs?\s*:\s*`?{re.escape(ref)}`?", " ", remaining, flags=re.I)
        for match in re.finditer(
            r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", claim, re.I
        ):
            span = match.group()
            cited_dates = []
            for value in cited_values:
                if isinstance(value, str) and re.match(r"\d{4}-\d{2}-\d{2}T", value):
                    cited_dates.append(datetime.fromisoformat(value.replace("Z", "+00:00")))
            if not any(point.strftime("%A").lower() == span.lower() for point in cited_dates):
                reason = (
                    "ref_mismatch"
                    if any(point.strftime("%A").lower() == span.lower() for point in timestamps)
                    else "not_found"
                )
                issues.append({"span": span, "reason": reason})
        for match in re.finditer(r"(?<![\w-])\$?\d[\d,]*(?:\.\d+)?(?:x|%)?(?![\w-])", remaining):
            span = match.group()
            ratio = span.endswith("x")
            token = span.rstrip("x%")
            values = cited_values
            if span.endswith("%"):
                values = [
                    float(value) * 100 for value in cited_values if type(value) in (int, float)
                ]
            if not _number_matches(token, values, ratio):
                anywhere = _number_matches(token, all_values, ratio)
                issues.append({"span": span, "reason": "ref_mismatch" if anywhere else "not_found"})
        for span in sorted(known_devices | known_merchants, key=len, reverse=True):
            if span and re.search(rf"(?<!\w){re.escape(span)}(?!\w)", claim, re.IGNORECASE):
                if span not in [str(value) for value in cited_values]:
                    issues.append({"span": span, "reason": "ref_mismatch"})
        for match in re.finditer(r"\b[\w-]+-D\d+\b", claim):
            span = match.group()
            if span not in [str(value) for value in cited_values]:
                issues.append(
                    {
                        "span": span,
                        "reason": "ref_mismatch" if span in known_devices else "not_found",
                    }
                )
        for match in re.finditer(
            r"\b(?:at|merchant)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\b", claim
        ):
            span = match.group(1)
            if span not in known_merchants and span not in [str(value) for value in cited_values]:
                issues.append({"span": span, "reason": "not_found"})
        for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", claim):
            span = match.group()
            if all(word in _PLAIN_WORDS for word in span.split()):
                continue
            if span not in known_merchants and span not in [str(value) for value in cited_values]:
                issues.append({"span": span, "reason": "not_found"})
        for match in re.finditer(r"\b[A-Z]{2}\b", claim):
            span = match.group()
            if span in {"AM", "PM", "ID", "OK"}:
                continue
            if span not in [str(value) for value in cited_values]:
                issues.append(
                    {"span": span, "reason": "ref_mismatch" if span in known_text else "not_found"}
                )
        verdicts.append({"claim_index": index, "grounded": not issues, "issues": issues})
        ungrounded.extend({"claim_index": index, **issue} for issue in issues)
    return {"grounded": not ungrounded, "verdicts": verdicts, "ungrounded_items": ungrounded}
