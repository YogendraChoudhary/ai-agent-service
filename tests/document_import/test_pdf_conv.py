"""
pdf_conv.py: PDF to Markdown from the text layer.

PDFs are built by hand (builders.make_pdf: Helvetica and Helvetica-Bold at chosen sizes and positions), so each case is exactly what it says: heading
sizes, running headers, wrapped and hyphenated lines, bullets, a paragraph that crosses a page, two columns, a table-like page, a scan, a protected file.
Real papers were converted separately (see the live evidence); these tests pin the rules.
"""
from __future__ import annotations

import time

import pytest

import src.document_import.pdf_conv as pdf_conv
from src.document_import.errors import ImportRefused
from src.document_import.pdf_conv import pdf_to_markdown
from tests.document_import.builders import column_page, make_pdf

LOREM = "The committee reviews the quarterly figures and considers the proposed changes to the schedule. "
LONG = LOREM * 3


def convert(pages, **kw):
    pdf_kw = {k: kw.pop(k) for k in ("title", "image_only", "encrypt") if k in kw}
    return pdf_to_markdown(make_pdf(pages, **pdf_kw), **kw)


def one(blocks, **kw):
    return convert([column_page(blocks)], **kw)


def refusal(data: bytes) -> ImportRefused:
    with pytest.raises(ImportRefused) as e:
        pdf_to_markdown(data)
    return e.value


def codes(r) -> set:
    return {w["code"] for w in r.warnings}


def headings(r) -> list:
    return [line for line in r.markdown.splitlines() if line.startswith("#")]


# ---------------------------------------------------------------------------------------------------------------------------------------

class TestHeadings:
    def test_sizes_map_to_levels(self):
        r = one([("B", "Big Title", 24), ("R", LONG, 11), ("B", "Section", 16), ("R", LONG, 11), ("B", "Smaller section", 13), ("R", LONG, 11)])
        assert headings(r) == ["# Big Title", "## Section", "### Smaller section"]

    def test_a_bold_line_at_body_size_is_a_heading_below_the_sized_ones(self):
        r = one([("B", "Title", 20), ("R", LONG, 11), ("B", "A bold subheading", 11), ("R", LONG, 11)])
        assert headings(r) == ["# Title", "## A bold subheading"]

    def test_a_bold_lead_in_sentence_is_not_a_heading(self):
        r = one([("R", LONG, 11), ("B", "This whole sentence is bold and ends with a full stop.", 11), ("R", LONG, 11)])
        assert headings(r) == []

    def test_numbered_bold_line_ending_in_a_period_is_a_heading(self):
        r = one([("R", LONG, 11), ("B", "3.1 Encoder and decoder.", 11), ("R", LONG, 11)])
        assert headings(r) == ["## 3.1 Encoder and decoder."]

    def test_a_document_that_is_mostly_bold_gets_no_bold_headings(self):
        r = one([("B", f"Bold line number {i}", 11) if i % 2 == 0 else ("R", LONG, 11) for i in range(10)])
        assert headings(r) == []

    def test_a_long_large_block_is_a_paragraph(self):
        r = one([("R", LONG, 11), ("R", LOREM * 6, 16)])
        assert headings(r) == []

    def test_a_split_section_number_rejoins_its_heading(self):
        page = column_page([("B", "Title", 20), ("R", LONG, 11)]) + [(72, 560, 14, "B", "1"), (100, 560, 14, "B", "Introduction")] + column_page([("R", LONG, 11)], top=540)
        assert "## 1 Introduction" in headings(convert([page]))

    def test_a_short_heading_that_leads_into_a_paragraph_stays_a_heading(self):
        page = [(72, 740, 22, "B", "Report Title"), (72, 700, 14, "B", "Background")] + column_page([("R", LONG, 10)], top=680)
        assert headings(convert([page])) == ["# Report Title", "## Background"]

    def test_author_lines_under_the_title_are_text_not_headings(self):
        page = [(72, 740, 22, "B", "A Paper Title"), (72, 700, 14, "R", "Ada Lovelace"), (72, 680, 14, "R", "Charles Babbage"), (72, 660, 14, "R", "Analytical Society")]
        page += column_page([("B", "Abstract", 14), ("R", LONG, 10), ("R", LONG, 10)], top=620)
        r = convert([page])
        assert headings(r) == ["# A Paper Title", "## Abstract"] and "Ada Lovelace" in r.markdown


class TestTextFlow:
    def test_wrapped_lines_become_one_paragraph(self):
        r = one([("R", LOREM * 3, 11), ("R", "A second paragraph.", 11)])
        assert r.markdown.count("\n\n") == 1 and r.markdown.startswith(LOREM.strip())

    def test_hyphenated_line_end_is_rejoined(self):
        page = [(72, 720, 11, "R", "A paragraph about inter-"), (72, 706, 11, "R", "national trade and more words to follow here.")]
        assert "international trade" in convert([page]).markdown

    def test_a_real_compound_is_not_glued(self):
        page = [(72, 720, 11, "R", "It beat the English-"), (72, 706, 11, "R", "to-German baseline by a wide margin overall.")]
        assert "English-to-German" in convert([page]).markdown

    def test_a_paragraph_continues_across_a_page_break(self):
        p1 = column_page([("R", LOREM + "The sentence does not end at the bottom of the page and carries on with the", 11)])
        p2 = column_page([("R", "next page, where it finishes. A new sentence starts here.", 11)])
        out = convert([p1, p2]).markdown
        assert "carries on with the next page, where it finishes." in out and out.count("\n\n") == 0

    def test_a_finished_paragraph_is_not_joined_to_the_next_page(self):
        out = convert([column_page([("R", LOREM, 11)]), column_page([("R", "lowercase start on the next page.", 11)])]).markdown
        assert "\n\nlowercase start" in out

    def test_markdown_punctuation_is_escaped(self):
        out = one([("R", "1999. A year of change " + LOREM, 11), ("R", "Stars *x* and _under_ and [link] " + LOREM, 11)]).markdown
        assert "1999\\. A year" in out and "\\*x\\*" in out and "\\_under\\_" in out and "\\[link\\]" in out

    def test_result_reparses_as_paragraphs(self):
        from markdown_it import MarkdownIt
        out = one([("R", "# looks like a heading but is text " + LOREM, 11), ("R", "+ looks like a list marker " + LOREM, 11)]).markdown
        assert [t.type for t in MarkdownIt("commonmark").parse(out) if t.type.endswith("_open")] == ["paragraph_open"] * 2

    def test_ligatures_and_invisible_characters(self):
        assert pdf_conv._clean("ﬁnd the ﬂow of oﬀice​­\U000e0041") == "find the flow of office"

    def test_text_turned_on_its_side_is_skipped(self):
        page = column_page([("R", LONG, 11)]) + [(30, 300, 10, "RR", "arXiv:1706.03762v7 stamp down the margin")]
        r = convert([page])
        assert "arXiv" not in r.markdown and r.stats["rotated_lines_skipped"] >= 1


class TestLists:
    def test_bullets_and_numbers(self):
        out = one([("R", LONG, 11), ("R", "• first point", 11), ("R", "• second point", 11), ("R", "1. step one", 11), ("R", "2. step two", 11), ("R", LONG, 11)]).markdown
        assert "- first point\n- second point\n\n1. step one\n2. step two" in out

    def test_a_wrapped_bullet_is_one_item(self):
        out = one([("R", "• " + "a long bullet that needs more than one line to say what it means " * 3, 11), ("R", "• short", 11)]).markdown
        assert out.count("- ") == 2 and out.splitlines()[0].startswith("- a long bullet") and out.rstrip().endswith("- short")

    def test_a_dash_or_a_number_at_a_line_start_is_a_list_item(self):
        out = one([("R", "- a dashed item", 11), ("R", "1) a numbered item", 11)]).markdown
        assert "- a dashed item" in out and "1. a numbered item" in out

    def test_a_year_at_the_start_of_a_sentence_is_text(self):
        out = one([("R", "2024 was a good year for the committee and its figures " * 3, 11)]).markdown
        assert not out.startswith("2024.") and "2024 was" in out


class TestFurniture:
    def pages(self, n=4, header="Acme Corp Confidential"):
        out = []
        for i in range(1, n + 1):
            lines = column_page([("R", f"Page {i} body. " + LOREM * 2, 11)])
            lines += [(72, 40, 9, "R", header), (300, 28, 9, "R", str(i))]
            out.append(lines)
        return out

    def test_running_headers_and_page_numbers_are_removed(self):
        out = convert(self.pages()).markdown
        assert "Acme" not in out and "\n\n1\n" not in out and "Page 1 body." in out and "Page 4 body." in out

    def test_a_footer_on_one_page_only_is_kept(self):
        pages = self.pages()
        pages[1].append((72, 50, 9, "R", "Only here: a footnote reference"))
        assert "Only here" in convert(pages).markdown

    def test_page_numbers_go_even_on_a_two_page_document(self):
        out = convert(self.pages(2)).markdown
        assert "Acme" in out and "\n\n1\n" not in out and not out.rstrip().endswith("2")

    def test_body_text_that_is_a_number_in_the_middle_is_kept(self):
        page = column_page([("R", LONG, 11)]) + [(72, 400, 11, "R", "42")]
        assert "42" in convert([page]).markdown


class TestColumns:
    def test_two_columns_are_read_left_then_right(self):
        left = column_page([("B", "Left heading", 14)] + [("R", f"LEFTWORD{i} " * 30, 10) for i in range(4)], left=60, width_chars=40)
        right = column_page([("B", "Right heading", 14)] + [("R", f"RIGHTWORD{i} " * 30, 10) for i in range(4)], left=330, width_chars=40)
        title = column_page([("B", "Two Column Paper", 22)], top=770, left=72)
        out = convert([title + left + right]).markdown
        assert out.index("Left heading") < out.index("LEFTWORD3") < out.index("Right heading") < out.index("RIGHTWORD0")
        assert headings(convert([title + left + right]))[:3] == ["# Two Column Paper", "## Left heading", "## Right heading"]

    def test_a_right_column_heading_does_not_jump_ahead_of_left_column_text(self):
        left = column_page([("R", f"LEFTWORD{i} " * 12, 10) for i in range(5)], left=60, top=700, width_chars=40)
        right = column_page([("B", "Right heading", 14)] + [("R", f"RIGHTWORD{i} " * 12, 10) for i in range(4)], left=330, top=740, width_chars=40)
        out = convert([left + right]).markdown
        assert out.index("LEFTWORD4") < out.index("Right heading") < out.index("RIGHTWORD0")        # the heading sits higher on the page but is in the right column

    def test_a_single_column_page_keeps_the_order_pdfminer_gives(self):
        out = one([("B", "One", 16), ("R", "FIRST " * 30, 11), ("B", "Two", 16), ("R", "SECOND " * 30, 11)]).markdown
        assert out.index("FIRST") < out.index("Two") < out.index("SECOND")


class TestWhatItReports:
    def test_a_title_comes_from_the_metadata(self):
        assert one([("R", LONG, 11)], title="Quarterly Review").title == "Quarterly Review"

    def test_junk_metadata_titles_are_ignored(self):
        for junk in ("Microsoft Word - report.docx", "Untitled", "draft.pdf", "ab"):
            assert one([("B", "Real Title", 20), ("R", LONG, 11)], title=junk).title == "Real Title"

    def test_table_like_pages_are_named(self):
        rows = [("Region", "Q1", "Q2", "Q3")] + [(f"Area {i}", str(10 * i), str(11 * i), str(12 * i)) for i in range(1, 8)]
        grid = [(72 + c * 90, 700 - r * 18, 10, "B" if r == 0 else "R", cell) for r, row in enumerate(rows) for c, cell in enumerate(row)]
        r = convert([column_page([("R", LONG, 11)], top=770) + grid, column_page([("R", LONG, 11)])])
        warning = next(w for w in r.warnings if w["code"] == "tables_not_rebuilt")
        assert "Page(s) 1 " in warning["message"] and r.stats["table_pages"] == 1

    def test_ordinary_text_pages_are_not_table_like(self):
        assert "tables_not_rebuilt" not in codes(one([("B", "Title", 20), ("R", LONG * 3, 11), ("R", "• a", 11), ("R", "• b", 11)]))

    def test_images_are_counted_not_imported(self):
        r = convert([column_page([("R", LONG, 11)])], image_only=True)
        assert "images_omitted" in codes(r) and r.stats["images"] == 1

    def test_a_clean_pdf_has_no_warnings(self):
        assert one([("B", "Title", 20), ("R", LONG, 11)]).warnings == []

    def test_page_limit(self):
        r = convert([column_page([("R", f"Page {i}. " + LONG, 11)]) for i in range(1, 9)], max_pages=3)
        assert r.stats["pages"] == 3 and "pages_truncated" in codes(r) and "Page 4." not in r.markdown
        assert "first 3 of 8" in next(w for w in r.warnings if w["code"] == "pages_truncated")["message"]

    def test_time_budget_stops_reading_and_says_so(self, monkeypatch):
        clock = iter(range(0, 10_000))
        monkeypatch.setattr(pdf_conv.time, "monotonic", lambda: next(clock))
        r = convert([column_page([("R", f"Page {i}. " + LONG, 11)]) for i in range(1, 12)], time_budget=3.5)
        assert 1 <= r.stats["pages"] < 11 and "pages_truncated" in codes(r)

    def test_output_cap(self):
        r = convert([column_page([("R", f"Page {i}. " + LONG * 3, 11)]) for i in range(1, 20)], max_chars=3000)
        assert len(r.markdown) <= 3100 and "output_truncated" in codes(r)


class TestRefusals:
    def test_not_a_pdf(self):
        assert refusal(b"hello").code == "not_a_pdf"

    def test_damaged(self):
        assert refusal(b"%PDF-1.4\nthis is not a pdf at all").code == "pdf_unreadable"
        assert refusal(make_pdf([column_page([("R", LONG, 11)])])[:200]).code in ("pdf_unreadable", "no_text")

    def test_password_protected(self):
        assert refusal(make_pdf([column_page([("R", LONG, 11)])], encrypt=True)).code == "pdf_encrypted"

    def test_scanned(self):
        e = refusal(make_pdf([[], []], image_only=True))
        assert e.code == "scanned_pdf" and "OCR" in e.message

    def test_no_text_at_all(self):
        assert refusal(make_pdf([[]])).code == "no_text"

    def test_a_few_stray_characters_are_not_a_text_layer(self):
        assert refusal(make_pdf([[(72, 700, 10, "R", "x")] for _ in range(4)], image_only=True)).code == "scanned_pdf"

    def test_copy_protected(self, monkeypatch):
        from pdfminer.pdfpage import PDFTextExtractionNotAllowed
        import pdfminer.high_level as hl

        def forbidden(*a, **kw):
            raise PDFTextExtractionNotAllowed("no")
            yield
        monkeypatch.setattr(hl, "extract_pages", forbidden)
        assert refusal(make_pdf([column_page([("R", LONG, 11)])])).code == "pdf_copy_protected"

    def test_a_later_damaged_page_keeps_the_pages_before_it(self, monkeypatch):
        import pdfminer.high_level as hl
        real = hl.extract_pages

        def failing(*a, **kw):
            for i, page in enumerate(real(*a, **kw)):
                if i == 2:
                    raise ValueError("page 3 is damaged")
                yield page
        monkeypatch.setattr(hl, "extract_pages", failing)
        r = convert([column_page([("R", f"Page {i}. " + LONG, 11)]) for i in range(1, 6)])
        assert r.stats["pages"] == 2 and "Page 2." in r.markdown


class TestScale:
    def test_sixty_pages(self):
        pages = [column_page([("B", f"Section {i}", 16), ("R", LONG * 4, 11)] * 3, top=690) for i in range(60)]
        t = time.perf_counter()
        r = convert(pages)
        assert time.perf_counter() - t < 20 and r.stats["pages"] == 60 and r.stats["headings"] == 180

    def test_a_page_with_thousands_of_tiny_lines(self):
        lines = [(10 + (i % 20) * 28, 780 - (i // 20) * 4, 3, "R", f"w{i}") for i in range(2000)]
        t = time.perf_counter()
        try:
            convert([lines + column_page([("R", LONG, 11)])])
        except ImportRefused:
            pass
        assert time.perf_counter() - t < 15
