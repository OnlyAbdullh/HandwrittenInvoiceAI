"""Labels the cells of the standard Arabic sales-invoice form.

Ported from `classify_bill_cells` in `temp/table_det.py`, which was tuned
against this project's own sample invoices.

The form it knows (right-to-left, as printed):

    [ رقم الفاتورة : 00008 ]          <- above the table, top-left
    [ الاسم : ... ] [ التاريخ : ... ] [ المدينة : ... ]

    | ملاحظات | السعر الإجمالي | السعر الإفرادي | العدد | اسم المنتج | # |
    |   ...   |      ...       |      ...       |  ...  |    ...     | 1 |
                              ... 10 rows ...
    [ السعر الإجمالي ] [ رقماً : ... ] [ كتابة : ... ]   <- below the table

Nothing here is hardcoded to those six columns or ten rows: `column_roles`,
`table_row_count` and `table_col_count` are parameters, because they are
properties of one supplier's printed form. A second form is a config entry, not
a code change -- which is the whole reason classification is a separate,
registered family rather than part of the extractor.

Everything is measured **relative to the detected table body**, never in
absolute page coordinates. On a photographed page the table's position varies by
hundreds of pixels; its position relative to the header does not.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from core.domain.geometry import BoundingBox
from core.domain.layout import InvoiceLayout, LayoutRegion
from core.domain.roles import CellRole, Zone, content_kind_for
from core.domain.table import TableCell, TableExtractionResult
from table_extraction.classifiers.registry import classifier_registry

# Left to right, which is how the columns of this RTL form appear in pixel
# order. Named as roles so nothing has to translate a string twice.
DEFAULT_COLUMN_ROLES: tuple[str, ...] = (
    CellRole.NOTES.value,
    CellRole.LINE_TOTAL.value,
    CellRole.UNIT_PRICE.value,
    CellRole.QUANTITY.value,
    CellRole.PRODUCT_NAME.value,
    CellRole.LINE_NUMBER.value,
)

# The three boxes beside the invoice number, left to right on the printed form.
DEFAULT_HEADER_ROLES: tuple[str, ...] = (
    CellRole.CITY.value,
    CellRole.INVOICE_DATE.value,
    CellRole.MERCHANT_NAME.value,
)

# The totals strip below the table, left to right in pixel order. On the printed
# form each value box sits to the LEFT of its caption, because the caption is
# read first in RTL:
#
#     [ <words> ] [ كتابة : ] [ <figures> ] [ رقماً : ] [ السعر الإجمالي ]
#
# This used to be (LABEL, TOTAL_IN_WORDS, LABEL, TOTAL_IN_FIGURES, TOTAL_AMOUNT),
# i.e. shifted by one: the captions were read as the values, and the box
# actually holding the handwritten total was labelled a caption and never read.
# On images/sahror that made the invoice total null on every run. LABEL cells
# are never sent to the recognizer.
DEFAULT_FOOTER_ROLES: tuple[str, ...] = (
    CellRole.TOTAL_IN_WORDS.value,
    CellRole.LABEL.value,
    CellRole.TOTAL_IN_FIGURES.value,
    CellRole.LABEL.value,
    CellRole.LABEL.value,
)


# The written columns of an item row -- what item_vertical_shift moves.
_ITEM_CELL_ROLES = frozenset(
    {CellRole.PRODUCT_NAME, CellRole.QUANTITY, CellRole.UNIT_PRICE, CellRole.LINE_TOTAL, CellRole.NOTES}
)


class _Cell:
    """A TableCell with the centres the geometry rules need.

    TableCell is frozen -- correctly, it is a domain value -- so the working
    state of classification lives here instead of being written back onto it.
    """

    __slots__ = ("cell", "x_center", "y_center", "role", "synthesized")

    def __init__(self, cell: TableCell, synthesized: bool = False):
        self.cell = cell
        self.x_center = cell.bbox.x + cell.bbox.w / 2.0
        self.y_center = cell.bbox.y + cell.bbox.h / 2.0
        self.role = CellRole.UNKNOWN
        self.synthesized = synthesized


def _cluster_rows(cells: list[_Cell], y_proximity: float | None = None) -> list[list[_Cell]]:
    """Group cells into printed rows, top to bottom, each sorted left to right.

    The comparison is against the running *mean* of the row so far rather than
    the previous cell, so one tall cell cannot drag the reference point far
    enough to split a printed row in two. `y_proximity` defaults to a fraction
    of the median cell height, so this works at any scan resolution.
    """
    if not cells:
        return []

    if y_proximity is None:
        median_h = float(np.median([c.cell.bbox.h for c in cells]))
        y_proximity = max(10.0, median_h * 0.6)

    ordered = sorted(cells, key=lambda c: c.y_center)

    rows: list[list[_Cell]] = []
    current = [ordered[0]]
    for c in ordered[1:]:
        row_y = float(np.mean([r.y_center for r in current]))
        if abs(c.y_center - row_y) <= y_proximity:
            current.append(c)
        else:
            rows.append(current)
            current = [c]
    rows.append(current)

    for row in rows:
        row.sort(key=lambda c: c.x_center)
    return rows


def _nearest_slot(
    x_center: float,
    slot_centers: list[float],
    avg_col_width: float | None,
    tol_ratio: float = 0.4,
) -> int | None:
    """Index of the closest expected column, or None when nothing is close.

    Returning None matters: a cell that lines up with no column is a stray
    detection, and assigning it to the nearest one anyway would shift a real
    value out of its column.
    """
    if not slot_centers:
        return None

    if avg_col_width is None:
        spread = max(slot_centers) - min(slot_centers)
        avg_col_width = spread / max(1, len(slot_centers) - 1)

    tol = avg_col_width * tol_ratio
    best_idx, best_dist = None, float("inf")
    for i, sc in enumerate(slot_centers):
        d = abs(x_center - sc)
        if d < best_dist:
            best_dist, best_idx = d, i

    return best_idx if best_dist <= tol else None


@classifier_registry.register("bill_layout")
class BillLayoutClassifier:
    name = "bill_layout"

    def __init__(
        self,
        table_row_count: int = 10,
        table_col_count: int = 6,
        column_roles: list[str] | tuple[str, ...] = DEFAULT_COLUMN_ROLES,
        header_roles: list[str] | tuple[str, ...] = DEFAULT_HEADER_ROLES,
        footer_roles: list[str] | tuple[str, ...] = DEFAULT_FOOTER_ROLES,
        zone_tolerance: int = 10,
        slot_tolerance_ratio: float = 0.4,
        ragged_row_slack: int = 5,
        printed_header_row: bool = True,
        footer_from_below: bool = False,
        complete_grid: bool = False,
        item_vertical_shift: float = 0.0,
        profile_name: str = "default",
        profiles: list[dict] | None = None,
    ):
        self.profile_name = profile_name
        self.table_row_count = table_row_count
        self.table_col_count = table_col_count
        self.column_roles = [CellRole(r) for r in column_roles]
        self.header_roles = [CellRole(r) for r in header_roles]
        self.footer_roles = [CellRole(r) for r in footer_roles]

        if len(self.column_roles) != table_col_count:
            raise ValueError(
                f"bill_layout: column_roles has {len(self.column_roles)} entries but "
                f"table_col_count is {table_col_count}; they describe the same columns."
            )

        # Cells routinely bleed a few pixels past the table's outer rule.
        self.zone_tolerance = zone_tolerance
        self.slot_tolerance_ratio = slot_tolerance_ratio

        # A data row can carry MORE cells than there are columns: a handwritten
        # figure broken across a printed dot pattern closes as two contours. The
        # run detector tolerates that, and _nearest_slot puts both halves in the
        # right column.
        self.ragged_row_slack = ragged_row_slack

        # Whether the table's first detected row is the printed column-header row
        # ("الصنف | العدد | السعر"). A form whose header band is not ruled as cells
        # has none to detect, and treating its first item row as the header
        # would silently drop that item.
        self.printed_header_row = printed_header_row

        # Take the totals strip from the cells below the table's outline rather
        # than the row after the items: on a form whose last ruled rows are
        # ragged, "the next row" is a broken item row, not the totals.
        self.footer_from_below = footer_from_below

        # Rebuild item rows the extractor only partly found. A printed form has
        # a known number of rows, evenly spaced, in fixed columns: when one
        # break in a rule (or a rounded box corner) leaves a cell's contour
        # open, that cell -- and with it the row -- is lost, and every row
        # below is labelled one position off. Measured on both real form-B
        # photos: the first item row was lost exactly that way.
        self.complete_grid = complete_grid

        # Move each item cell's box down (positive) or up by this fraction of
        # its height. On a form whose writing sits ON the row rule rather than
        # between two rules, the printed cell holds the bottom of the row above
        # and loses the bottom of its own writing; shifting the box recentres
        # the crop on the writing. Off by default: whether it helps a given
        # model is a measurement (tools/benchmark_recognizers.py), not a guess.
        if not -0.5 <= item_vertical_shift <= 0.5:
            raise ValueError(f"bill_layout: item_vertical_shift must be within [-0.5, 0.5], got {item_vertical_shift}.")
        self.item_vertical_shift = item_vertical_shift

        # Other printed forms this classifier recognises, each a full set of
        # the parameters above. The page is labelled by whichever profile -- this
        # one or an alternative -- fits its detected grid best. A second form is
        # a config entry, not a code change.
        self._alternatives = [BillLayoutClassifier(**dict(p)) for p in (profiles or [])]

    # ------------------------------------------------------------------

    def classify(
        self, table: TableExtractionResult, image_size: tuple[int, int]
    ) -> InvoiceLayout:
        candidates = [self, *self._alternatives]
        if len(candidates) == 1:
            return self._classify_as_profile(table, image_size)

        # Ties go to the first (the primary profile), which is also what a page
        # no profile fits at all falls back to.
        fits = [candidate._fit(table) for candidate in candidates]
        best = candidates[max(range(len(candidates)), key=lambda i: (fits[i], -i))]
        return best._classify_as_profile(table, image_size)

    def _fit(self, table: TableExtractionResult) -> int:
        """How many rows of the detected table have exactly this form's column count.

        Exactly, not "about": a six-column page also satisfies a four-column
        form's tolerance for ragged rows, and only an exact count tells them
        apart.
        """
        structure = table.tables[0] if table.tables else None
        if structure is None or not structure.cells:
            return 0

        top = structure.bbox.y - self.zone_tolerance
        bottom = structure.bbox.y + structure.bbox.h + self.zone_tolerance
        inside = [
            _Cell(c) for c in structure.cells if top <= c.bbox.y + c.bbox.h / 2.0 <= bottom
        ]
        return sum(len(row) == self.table_col_count for row in _cluster_rows(inside))

    def _classify_as_profile(
        self, table: TableExtractionResult, image_size: tuple[int, int]
    ) -> InvoiceLayout:
        structure = table.tables[0] if table.tables else None
        if structure is None or not structure.cells:
            return InvoiceLayout(source=f"{table.extractor_name}+{self.name}")

        cells = [_Cell(c) for c in structure.cells]

        top = structure.bbox.y
        bottom = structure.bbox.y + structure.bbox.h
        tol = self.zone_tolerance

        above = [c for c in cells if c.y_center < top - tol]
        inside = [c for c in cells if top - tol <= c.y_center <= bottom + tol]
        below = [c for c in cells if c.y_center > bottom + tol]

        rows = _cluster_rows(inside)
        totals_row, synthesized = self._label_table(rows, below, top, bottom)
        cells.extend(synthesized)
        self._label_header(above)
        self._label_footer(totals_row)

        regions = [
            LayoutRegion(
                # "s": a cell the grid completion inferred rather than detected.
                id=f"{c.cell.table_id}:r{c.cell.row}c{c.cell.col}" + ("s" if c.synthesized else ""),
                bbox=self._item_box(c),
                role=c.role,
                zone=self._zone_of(c, top, bottom),
                content_kind=content_kind_for(c.role),
                row=c.cell.row,
                col=c.cell.col,
                row_span=c.cell.row_span,
                col_span=c.cell.col_span,
                table_id=c.cell.table_id,
            )
            for c in cells
        ]

        return InvoiceLayout(
            regions=regions,
            table_bbox=structure.bbox,
            table_id=structure.table_id,
            source=(
                f"{table.extractor_name}+{self.name}"
                + ("" if self.profile_name == "default" else f"[{self.profile_name}]")
            ),
            raw_output=table.raw_extractor_output,
        )

    def _item_box(self, c: _Cell) -> BoundingBox:
        box = c.cell.bbox
        if not self.item_vertical_shift or c.role not in _ITEM_CELL_ROLES:
            return box
        return replace(box, y=box.y + int(round(self.item_vertical_shift * box.h)))

    def _zone_of(self, c: _Cell, top: int, bottom: int) -> Zone:
        if c.y_center < top - self.zone_tolerance:
            return Zone.HEADER
        if c.y_center > bottom + self.zone_tolerance:
            return Zone.FOOTER
        return Zone.TABLE

    # ------------------------------------------------------------------
    # The item table
    # ------------------------------------------------------------------

    def _label_table(
        self, rows: list[list[_Cell]], below: list[_Cell], top: float, bottom: float
    ) -> tuple[list[_Cell], list[_Cell]]:
        """Label the column-header row and the data rows.

        Returns the totals row and any cells the grid completion synthesized.

        The table block is found as the longest **run** of rows holding about
        the right number of cells, rather than by counting from the top. A stray
        detection above or below the table would break counting; it cannot break
        a run.
        """
        run_start, run_len = self._longest_full_run(rows)
        if run_start is None:
            # No recognizable product table. Everything stays UNKNOWN and the
            # parser falls back to its heuristics -- which is the correct
            # outcome for a form this classifier does not actually fit, and far
            # better than labelling arbitrary cells with confident-looking roles.
            return self._totals_from_below(below), []

        table_rows = rows[run_start : run_start + run_len]
        if self.printed_header_row:
            header_row, data_rows = table_rows[0], table_rows[1 : 1 + self.table_row_count]
        else:
            header_row, data_rows = [], table_rows[: self.table_row_count]

        slot_centers, avg_col_width = self._column_template(table_rows)
        valid = [c for c in slot_centers if c is not None]

        # Index in `rows` just past the last data row; the totals strip is
        # looked for from there.
        first_data = run_start + (1 if self.printed_header_row else 0)
        after_data = first_data + len(data_rows)
        synthesized: list[_Cell] = []
        if self.complete_grid and len(data_rows) < self.table_row_count:
            data_rows, after_data, synthesized = self._complete_rows(
                rows, data_rows, first_data, table_rows, slot_centers, avg_col_width, top, bottom
            )

        for row, is_header in [(header_row, True)] + [(r, False) for r in data_rows]:
            for cell in row:
                slot = _nearest_slot(
                    cell.x_center, valid, avg_col_width, self.slot_tolerance_ratio
                )
                if slot is None:
                    continue
                col_index = slot_centers.index(valid[slot])
                cell.role = (
                    CellRole.COLUMN_HEADER if is_header else self.column_roles[col_index]
                )

        # The totals strip is the row straight after the data rows when the
        # table's outer rule enclosed it, and a separate group below when it did
        # not. Both happen on real scans depending on how the form is printed.
        if self.footer_from_below:
            return self._totals_from_below(below), synthesized

        after_run = max(run_start + run_len, after_data)

        if after_data < len(rows):
            return rows[after_data], synthesized
        if after_run < len(rows):
            return rows[after_run], synthesized
        return self._totals_from_below(below), synthesized

    # ------------------------------------------------------------------
    # Grid completion
    # ------------------------------------------------------------------

    def _complete_rows(
        self,
        rows: list[list[_Cell]],
        data_rows: list[list[_Cell]],
        first_data: int,
        table_rows: list[list[_Cell]],
        slot_centers: list[float | None],
        avg_col_width: float | None,
        top: float,
        bottom: float,
    ) -> tuple[list[list[_Cell]], int, list[_Cell]]:
        """Extend the data rows to `table_row_count`, filling missing cells.

        Works outward from the run of fully detected rows, one row pitch at a
        time: a detected (possibly partial) row at the expected height is
        taken as the next item row; where none was detected but the band still
        lies inside the table outline, an empty band is assumed. It never
        reaches above the printed column-header row or outside the table's
        outline, so neither the header nor the totals strip can become an item.

        Returns (data rows, index in `rows` past the last one, synthesized cells).
        """
        clean = [r for r in table_rows if len(r) == self.table_col_count] or table_rows
        centers = [float(np.mean([c.y_center for c in r])) for r in clean]
        pitch = (
            float(np.median(np.diff(centers))) if len(centers) >= 2
            else float(np.median([c.cell.bbox.h for r in clean for c in r]))
        )
        if pitch <= 0:
            return data_rows, first_data + len(data_rows), []

        def row_center(row: list[_Cell], fallback: float) -> float:
            return float(np.mean([c.y_center for c in row])) if row else fallback

        bands: list[tuple[list[_Cell], float]] = [(r, row_center(r, 0.0)) for r in data_rows]
        need = self.table_row_count - len(bands)
        lo, hi = first_data - 1, first_data + len(data_rows)
        margin = 0.25 * pitch

        # Upward -- only when the form has no printed header row in the grid,
        # since otherwise the row above the first item IS that header.
        expected = bands[0][1] - pitch
        while need > 0 and not self.printed_header_row and expected - pitch / 2 >= top - margin:
            if lo >= 0 and abs(row_center(rows[lo], expected) - expected) < pitch / 2:
                bands.insert(0, (rows[lo], expected))
                lo -= 1
            elif lo < 0 or row_center(rows[lo], expected) < expected - pitch / 2:
                bands.insert(0, ([], expected))
            else:
                break
            expected -= pitch
            need -= 1

        expected = bands[-1][1] + pitch
        while need > 0 and expected + pitch / 2 <= bottom + margin:
            if hi < len(rows) and abs(row_center(rows[hi], expected) - expected) < pitch / 2:
                bands.append((rows[hi], expected))
                hi += 1
            elif hi >= len(rows) or row_center(rows[hi], expected) > expected + pitch / 2:
                bands.append(([], expected))
            else:
                break
            expected += pitch
            need -= 1

        # Fill each band's missing columns from the nearest clean row's cell in
        # that column, moved by the vertical distance between the two -- which
        # also carries the page's slight rotation across, cell by cell.
        valid = [c for c in slot_centers if c is not None]

        def column_of(cell: _Cell) -> int | None:
            slot = _nearest_slot(cell.x_center, valid, avg_col_width, self.slot_tolerance_ratio)
            return None if slot is None else slot_centers.index(valid[slot])

        table_id = next(c.cell.table_id for r in clean for c in r)
        used_rows = {c.cell.row for r in rows for c in r}
        fresh_row = max(used_rows, default=0) + 1
        synthesized: list[_Cell] = []
        completed: list[list[_Cell]] = []

        # A column's normal cell size, from the clean rows. A detected box far
        # smaller than that is not the printed cell -- on a real form B it was
        # a closed loop of the handwriting itself (77x74 inside a 766-wide
        # product cell) -- so it does not stand in for the column.
        typical = {}
        for col in range(self.table_col_count):
            sizes = [(c.cell.bbox.w, c.cell.bbox.h) for r in clean for c in r if column_of(c) == col]
            if sizes:
                typical[col] = (float(np.median([w for w, _ in sizes])), float(np.median([h for _, h in sizes])))

        def stands_for_column(cell: _Cell) -> bool:
            col = column_of(cell)
            if col is None or col not in typical:
                return False
            width, height = typical[col]
            return cell.cell.bbox.w >= 0.5 * width and cell.cell.bbox.h >= 0.5 * height

        for row, center in bands:
            fragments = [c for c in row if not stands_for_column(c)]
            row = [c for c in row if stands_for_column(c)]
            center = row_center(row, center) if row else center
            present = {column_of(cell) for cell in row} - {None}
            for fragment in fragments:
                fragment.role = CellRole.UNKNOWN
            anchor = row or fragments
            row_index = anchor[0].cell.row if anchor else fresh_row
            if not anchor:
                fresh_row += 1

            new_cells = list(row)
            for col in range(self.table_col_count):
                if col in present or slot_centers[col] is None:
                    continue
                reference_row, reference_center = min(
                    zip(clean, centers), key=lambda pair: abs(pair[1] - center)
                )
                reference = next((c for c in reference_row if column_of(c) == col), None)
                if reference is None:
                    continue
                shift = int(round(center - reference_center))
                box = reference.cell.bbox
                cell = _Cell(
                    TableCell(
                        bbox=BoundingBox(x=box.x, y=box.y + shift, w=box.w, h=box.h),
                        row=row_index,
                        col=col,
                        table_id=table_id,
                    ),
                    synthesized=True,
                )
                new_cells.append(cell)
                synthesized.append(cell)

            # One band is one item: every cell of it carries the same row, or
            # the parser -- which groups an item by row -- would split it.
            for cell in new_cells:
                if cell.cell.row != row_index:
                    cell.cell = replace(cell.cell, row=row_index)
            new_cells.sort(key=lambda c: c.x_center)
            completed.append(new_cells)

        return completed, max(hi, lo + 1 + len(bands)), synthesized

    def _longest_full_run(self, rows: list[list[_Cell]]) -> tuple[int | None, int]:
        low, high = self.table_col_count, self.table_col_count + self.ragged_row_slack

        best_start, best_len = None, 0
        cur_start, cur_len = None, 0

        for i, row in enumerate(rows):
            if low <= len(row) <= high:
                if cur_start is None:
                    cur_start = i
                cur_len += 1
            else:
                if cur_len > best_len:
                    best_start, best_len = cur_start, cur_len
                cur_start, cur_len = None, 0

        if cur_len > best_len:
            best_start, best_len = cur_start, cur_len

        return best_start, best_len

    def _column_template(
        self, table_rows: list[list[_Cell]]
    ) -> tuple[list[float | None], float | None]:
        """Where each column sits, as the median x-centre over clean rows.

        Only rows holding exactly `table_col_count` cells vote: a row whose
        figure split into two contours has its cells shifted, and letting it
        vote would drag the whole template sideways.
        """
        clean = [r for r in table_rows if len(r) == self.table_col_count] or table_rows

        centers: list[float | None] = []
        for col in range(self.table_col_count):
            xs = [r[col].x_center for r in clean if len(r) > col]
            centers.append(float(np.median(xs)) if xs else None)

        valid = [c for c in centers if c is not None]
        avg_col_width = (
            (max(valid) - min(valid)) / (self.table_col_count - 1)
            if len(valid) >= 2 and self.table_col_count > 1
            else None
        )
        return centers, avg_col_width

    @staticmethod
    def _totals_from_below(below: list[_Cell]) -> list[_Cell]:
        rows = _cluster_rows(below)
        return rows[0] if rows else []

    # ------------------------------------------------------------------
    # Header and footer
    # ------------------------------------------------------------------

    def _label_header(self, above: list[_Cell]) -> None:
        """Invoice number, then city / date / merchant across the row below it.

        The invoice number is the box nearest the top-left corner of the header
        block, scored on normalized x + y so neither axis dominates on a page of
        any aspect ratio. It is scored across **the whole header block**, not
        within its topmost row: a row holding a single cell normalizes to a
        score of zero on both axes -- `max(range, 1.0)` divided into a spread of
        zero -- so whatever sat in it won by default and its position was never
        consulted at all. A page-edge sliver above the real header row is exactly
        that case, and it took the invoice number on a real scan while the box
        actually reading "رقم الفاتورة : 00010" was left unlabelled.

        City / date / merchant are bucketed across **the header row nearest the
        table**, not across every cell above it. Bucketing everything is what the
        original did, and on a real scan it mislabels: the letterhead's phone
        numbers close as their own small boxes, land in the rightmost bucket,
        and win the merchant field over the box that actually says الاسم. The
        three fields are printed on one row directly above the table, and that
        row is the only place they can be.
        """
        if not above or not self.header_roles:
            return

        rows = _cluster_rows(above)

        xs = [c.x_center for c in above]
        x_min, x_range = min(xs), max(max(xs) - min(xs), 1.0)
        ys = [c.y_center for c in above]
        y_min, y_range = min(ys), max(max(ys) - min(ys), 1.0)

        invoice_number = min(
            above,
            key=lambda c: (c.x_center - x_min) / x_range + (c.y_center - y_min) / y_range,
        )
        invoice_number.role = CellRole.INVOICE_NUMBER

        # The row immediately above the table. With only one header row it is
        # the same row the invoice number came from, and that cell is skipped.
        field_row = [c for c in rows[-1] if c is not invoice_number]
        if not field_row:
            return

        r_xs = [c.x_center for c in field_row]
        r_min = min(r_xs)
        width = max(max(r_xs) - r_min, 1.0)
        buckets = len(self.header_roles)

        for c in field_row:
            frac = (c.x_center - r_min) / width
            c.role = self.header_roles[min(buckets - 1, int(frac * buckets))]

    def _label_footer(self, totals_row: list[_Cell]) -> None:
        if not totals_row or not self.footer_roles:
            return

        ordered = sorted(totals_row, key=lambda c: c.x_center)

        if len(ordered) == len(self.footer_roles):
            for cell, role in zip(ordered, self.footer_roles):
                cell.role = role
            return

        # A different cell count means the strip merged or split somewhere.
        # Bucketing by position degrades gracefully instead of shifting every
        # label by one, which is what zipping a short list would do.
        x_min, x_max = ordered[0].x_center, ordered[-1].x_center
        width = max(x_max - x_min, 1.0)
        buckets = len(self.footer_roles)

        for c in ordered:
            frac = (c.x_center - x_min) / width
            c.role = self.footer_roles[min(buckets - 1, int(frac * buckets))]
