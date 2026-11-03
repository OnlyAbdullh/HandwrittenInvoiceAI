"""The ground-truth format: what one invoice actually says.

One JSON file per labelled image under `evaluation/ground_truth/`. A field the
paper does not have (form B has no city box) is null, and a null truth is not
scored -- an extraction is neither right nor wrong about a field that does not
exist.

Values are what the *paper* says, normalized the way the API reports them:
ISO dates, plain decimal numbers, names exactly as the catalog spells them.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LineItemTruth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_name: str
    quantity: float | None = None
    unit_price: float | None = None
    total_price: float | None = None


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

    # How the labels were produced, so a reader knows how far to trust them.
    labelled_by: str | None = None
    notes: str | None = None


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
