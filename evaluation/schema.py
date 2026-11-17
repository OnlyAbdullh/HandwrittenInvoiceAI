"""The ground-truth format: what one invoice actually says.

One JSON file per labelled image under `evaluation/ground_truth/`. A field the
paper does not have (form B has no city box) is null, and a null truth is not
scored -- an extraction is neither right nor wrong about a field that does not
exist.

Values are what the *paper* says, normalized the way the API reports them:
ISO dates, plain decimal numbers, names exactly as the catalog spells them.

**A number may also be typed as it is written**, as a string of ASCII digits
with the separator the writer used: `"unit_price": "5,70"`. Its value is still
5.7 for scoring; the string is kept (`written`) and becomes the label of that
cell's crop. That matters whenever the written form is not the shortest one:
`5.70` in JSON is the number 5.7, and a crop showing ٥,٧٠ labelled "5.7"
teaches a recognizer to drop a zero it can see -- on handwriting where a zero
is a dot and the commonest misreading is a zero lost or invented. Type the
string whenever the paper shows a decimal part.
"""

from __future__ import annotations

import re

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from string_matching.normalization import fold_digits

# Arabic decimal separator and Arabic comma, both written where a Latin comma
# would be; a labeller typing from an Arabic keyboard produces either.
_ARABIC_SEPARATORS = str.maketrans({"٫": ",", "،": ","})
_WRITTEN_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def written_form(text: str) -> str:
    """A number typed as written, in the one spelling labels use: ASCII digits,
    the separator as typed."""
    return fold_digits(text).translate(_ARABIC_SEPARATORS).strip()


def parse_written_number(text: str) -> float:
    """The value of a number typed as it is written on the paper.

    One separator at most, and it is the decimal one: these invoices write no
    thousands separator, and guessing which of two marks is which would put a
    wrong value in the truth without anyone being told.
    """
    spelled = written_form(text)
    if not _WRITTEN_NUMBER.fullmatch(spelled):
        raise ValueError(
            f"{text!r} is not a number as written: digits, with at most one '.' or ',' "
            "before the decimal part (e.g. \"1080\", \"5,70\")."
        )
    return float(spelled.replace(",", "."))


def plain_number(value: float | None) -> str | None:
    """A value's shortest decimal spelling: what a label is when the truth does
    not say how the number is written.

    Never ``:g``: it keeps six significant digits, so 214812.05 became
    "214812" and every model was scored against a label that dropped the
    cents.
    """
    if value is None:
        return None
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _keep_written(data: object, columns: tuple[str, ...]) -> object:
    """Move numbers typed as strings into `written`, leaving their values."""
    if not isinstance(data, dict):
        return data
    data = dict(data)
    written = dict(data.get("written") or {})
    for column in columns:
        value = data.get(column)
        if isinstance(value, str):
            written[column] = written_form(value)
            data[column] = parse_written_number(value)
    if written:
        data["written"] = written
    return data


class LineItemTruth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_name: str
    quantity: float | None = None
    unit_price: float | None = None
    total_price: float | None = None

    # column -> the number exactly as written, for the columns typed as strings.
    written: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _numbers_as_written(cls, data: object) -> object:
        return _keep_written(data, ("quantity", "unit_price", "total_price"))

    def text_of(self, column: str) -> str | None:
        """What the paper shows in `column`: as typed, else the shortest spelling."""
        return self.written.get(column) or plain_number(getattr(self, column))


class InvoiceTruth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Path of the image, relative to the --images directory given to evaluate.
    image: str

    # Which printed form this is; informational, used to group results.
    form: str | None = None

    invoice_number: str | None = None
    customer_name: str | None = None
    date: str | None = None  # ISO-8601, YYYY-MM-DD
    city: str | None = None
    total: float | None = None
    items: list[LineItemTruth] = Field(default_factory=list)

    # The date as the paper shows it, when that is not day/month/year
    # ("7/8/2026"), e.g. "2026-08-07" or "7.8.26". Only the crop's label uses it.
    date_written: str | None = None

    # Who filled the invoice in (any stable name or number). Invoices by one
    # writer share handwriting, so a recognizer tested on a writer it was
    # trained on looks better than it is: the fine-tuning split keeps each
    # writer's invoices together. Leave null when unknown.
    writer: str | None = None

    # Which digits the handwriting uses: ٠١٢٣٤٥٦٧٨٩ or 0123456789. Truth files
    # are typed in ASCII digits either way; this says which a training label
    # for a handwritten number or date is spelled in, so the recognizer is
    # taught to write the digits it sees.
    digits: Literal["arabic_indic", "western"] = "arabic_indic"

    # field -> the number exactly as written, for `total` typed as a string.
    written: dict[str, str] = Field(default_factory=dict)

    # How the labels were produced, so a reader knows how far to trust them.
    labelled_by: str | None = None
    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _numbers_as_written(cls, data: object) -> object:
        return _keep_written(data, ("total",))

    @property
    def total_text(self) -> str | None:
        """The invoice total as the paper shows it."""
        return self.written.get("total") or plain_number(self.total)

    @property
    def date_text(self) -> str | None:
        """The date as the paper shows it: as typed, else day/month/year."""
        if self.date_written:
            return written_form(self.date_written)
        if not self.date:
            return None
        year, month, day = self.date.split("-")
        return f"{int(day)}/{int(month)}/{year}"


class CatalogEntryTruth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int | None = None
    name: str
    aliases: list[str] = Field(default_factory=list)


class EvaluationCatalog(BaseModel):
    """The catalogs the desktop app would send, so matching is evaluated too."""

    model_config = ConfigDict(extra="forbid")

    merchants: list[CatalogEntryTruth] = Field(default_factory=list)
    products: list[CatalogEntryTruth] = Field(default_factory=list)
    cities: list[CatalogEntryTruth] = Field(default_factory=list)
