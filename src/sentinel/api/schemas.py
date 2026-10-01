import re
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Literal, Self

import pycountry
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

ID = r"^[A-Za-z0-9_-]{1,64}$"
CURRENCIES = frozenset(
    {
        "USD",
        "EUR",
        "GBP",
        "CAD",
        "AUD",
        "JPY",
        "CHF",
        "NZD",
        "SEK",
        "NOK",
        "DKK",
        "PLN",
        "CZK",
        "HUF",
        "SGD",
        "HKD",
        "INR",
        "BRL",
        "MXN",
        "ZAR",
    }
)
COUNTRIES = frozenset(country.alpha_2 for country in pycountry.countries)


class TransactionIn(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    transaction_id: str = Field(pattern=ID)
    account_id: str = Field(pattern=ID)
    timestamp: datetime
    amount: str
    currency: str
    merchant_name: str = Field(min_length=1, max_length=120)
    merchant_category: str = Field(min_length=1, max_length=64)
    country: str
    device_id: str = Field(pattern=ID)
    channel: Literal["card_present", "online", "transfer"]
    memo: str | None = Field(default=None, max_length=280)

    @field_validator("amount")
    @classmethod
    def valid_amount(cls, value: str) -> str:
        if not re.fullmatch(r"(?:0|[1-9]\d*)(?:\.\d{1,2})?", value):
            raise ValueError("must be a decimal string with at most two decimal places")
        if not 0 < Decimal(value) <= Decimal("1000000"):
            raise ValueError("must be greater than zero and at most 1000000")
        return value

    @field_validator("currency")
    @classmethod
    def valid_currency(cls, value: str) -> str:
        if value not in CURRENCIES:
            raise ValueError("unsupported ISO 4217 currency")
        return value

    @field_validator("country")
    @classmethod
    def valid_country(cls, value: str) -> str:
        if value not in COUNTRIES:
            raise ValueError("unsupported ISO alpha-2 country")
        return value

    @field_validator("merchant_name", "merchant_category", "memo")
    @classmethod
    def no_controls(cls, value: str | None) -> str | None:
        if value is not None and any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("control characters are not allowed")
        return value

    @model_validator(mode="after")
    def valid_timestamp(self, info: ValidationInfo) -> Self:
        now = (
            info.context.get("now", datetime.now(UTC))
            if isinstance(info.context, dict)
            else datetime.now(UTC)
        )
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must include a timezone")
        if self.timestamp > now + timedelta(minutes=5):
            raise ValueError("timestamp is too far in the future")
        try:
            oldest = now.replace(year=now.year - 5)
        except ValueError:
            oldest = now.replace(year=now.year - 5, day=28)
        if self.timestamp < oldest:
            raise ValueError("timestamp is older than five years")
        return self
