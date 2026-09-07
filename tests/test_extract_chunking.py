"""Chunking of tabular content.

A table's figures are only meaningful under their column names, so the prose
splitter must not touch a grid: no part may start mid-row, every part carries
the header, and no row may be dropped on the way through the quality filter.
"""

from footnote_mcp.extract import _table_rows, chunk_text, filter_low_quality_chunks

HEADER = "| Year | Share | Growth | Units |"
SEPARATOR = "| --- | --- | --- | --- |"
ROWS = [f"| {1900 + i} | {60 + i * 2}.4 | {12 + i}.1 | {i * 137} |" for i in range(120)]
TABLE = "\n".join([HEADER, SEPARATOR, *ROWS])

ALIGNED_HEADER = "Year   Share   Growth   Units"
ALIGNED_ROWS = [f"{1900 + i}   {60 + i * 2}.4   {12 + i}.1   {i * 137}" for i in range(120)]
ALIGNED_TABLE = "\n".join([ALIGNED_HEADER, *ALIGNED_ROWS])


def test_table_is_large_enough_to_force_a_split():
    """Guards the fixture: the assertions below are vacuous on a single part."""
    assert len(chunk_text(TABLE)) > 1


def test_every_table_part_repeats_the_header():
    for part in chunk_text(TABLE):
        assert part.splitlines()[0] == HEADER


def test_table_is_split_on_row_boundaries_only():
    for part in chunk_text(TABLE):
        for line in part.splitlines():
            assert line.startswith("|") and line.endswith("|")


def test_no_row_is_lost_or_duplicated():
    emitted = [line for part in chunk_text(TABLE) for line in part.splitlines()
               if line not in (HEADER, SEPARATOR)]
    assert sorted(emitted) == sorted(ROWS)


def test_numeric_table_survives_the_quality_filter():
    """A grid of digits fails every prose vocabulary heuristic, and used to be
    discarded in full — the figures vanished before the model ever saw them."""
    chunks = chunk_text(TABLE)
    assert chunks
    assert filter_low_quality_chunks(chunks) == chunks


def test_prose_around_a_table_is_not_merged_into_it():
    text = "Revenue grew steadily over the period covered by the survey below.\n\n" + TABLE
    parts = chunk_text(text)
    assert parts[0].startswith("Revenue grew steadily")
    assert all(part.splitlines()[0] == HEADER for part in parts[1:])


def test_a_table_gets_a_wider_budget_than_prose():
    """Rows under a repeated header buy far less per character than sentences
    do, so a grid that fits the prose budget still arrived in fragments."""
    from footnote_mcp import core
    assert core.TABLE_CHUNK_SIZE > core.CHUNK_SIZE
    assert max(len(part) for part in chunk_text(TABLE)) > core.CHUNK_SIZE


def test_a_wrapped_header_is_repeated_in_full():
    """Wikipedia wraps a wide table's column names across two lines. Taking
    only the first left every continuation part naming four of seven columns."""
    wrapped = "\n".join([
        "| Rank | Name | Industry | Revenue |",
        "Growth | Employees | Headquarters |",
        "|---|---|---|---|---|---|---|",
        *[f"| {i} | Co {i} | Retail | {i * 1000} | {i}.1% | {i * 90} | Town {i} |" for i in range(120)],
    ])
    parts = chunk_text(wrapped)
    assert len(parts) > 1
    for part in parts:
        assert part.splitlines()[:3] == wrapped.splitlines()[:3]


# ── space-aligned grids ────────────────────────────────────────────────────

def test_space_aligned_table_is_detected_and_keeps_its_header():
    parts = chunk_text(ALIGNED_TABLE)
    assert len(parts) > 1
    for part in parts:
        assert part.splitlines()[0] == ALIGNED_HEADER


def test_space_aligned_table_loses_no_row():
    emitted = [line for part in chunk_text(ALIGNED_TABLE) for line in part.splitlines()
               if line != ALIGNED_HEADER]
    assert sorted(emitted) == sorted(ALIGNED_ROWS)


def test_indented_code_is_not_mistaken_for_a_table():
    code = "def f(x):\n    if x > 2:\n        return x * 2\n    return x"
    assert _table_rows(code) is None


def test_wrapped_prose_is_not_mistaken_for_a_table():
    prose = ("The report was long.  It covered many topics.  Readers found it dense.\n"
             "Each section opened with a summary of the argument that followed.\n"
             "The appendix listed every source the authors had consulted.")
    assert _table_rows(prose) is None


def test_prose_with_an_incidental_pipe_still_chunks_as_prose():
    text = ("The command is written as `a | b` in the shell, and the pipe "
            "character separates the two stages of the pipeline. " * 4)
    parts = chunk_text(text)
    assert parts
    assert all(not part.startswith("|") for part in parts)
