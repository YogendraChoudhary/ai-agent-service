"""Button links stay exactly what the user gave (src/design_canvas/links.py)."""
import json

import pytest

from src.design_canvas.links import check_links, requested_links
from src.design_canvas.schema import DesignDocumentAST


def design(*hrefs):
    return DesignDocumentAST.model_validate({"design_id": "d", "title": "t", "theme_palette": ["#000000", "#FFFFFF"],
        "root": {"node_type": "container", "id": "root", "children": [
            {"node_type": "button", "id": f"b{i}", "label": f"B{i}", "href": h} for i, h in enumerate(hrefs)]}})


def hrefs(ast):
    return [b.href for b in ast.root.children]


@pytest.mark.parametrize("query, href", [
    ("A button linking to https://example.com/buy", "https://example.com/buy"),
    ("A button to our site example.com", "https://example.com"),
    ("A button to www.example.com/pricing", "https://www.example.com/pricing"),
    ("An email button to mailto:hi@example.com", "mailto:hi@example.com"),
    ("A call button for tel:+15550100", "tel:+15550100"),
    ("Jump to #pricing", "#pricing"),
])
def test_links_the_user_gave_are_kept(query, href):
    ast = design(href)
    assert check_links(ast, query) == [] and hrefs(ast) == [href]


def test_a_link_changed_into_another_is_removed_and_reported():
    ast = design("#checkout")
    warnings = check_links(ast, "Link the Buy now button to /checkout")
    assert hrefs(ast) == [None]
    assert [w["kind"] for w in warnings] == ["link_removed", "link_not_set"]
    assert warnings[1]["link"] == "/checkout" and "full https://" in warnings[1]["detail"]


def test_an_invented_link_is_removed():
    ast = design("https://example.com/subscribe")
    warnings = check_links(ast, "A newsletter card with a Subscribe button")
    assert hrefs(ast) == [None] and warnings[0]["kind"] == "link_removed" and warnings[0]["node_id"] == "b0"


def test_links_already_in_the_edited_design_survive_an_edit():
    existing = json.dumps({"root": {"children": [{"node_type": "button", "href": "https://example.com/buy"}]}})
    ast = design("https://example.com/buy")
    assert check_links(ast, "Change the label to Start now", existing) == [] and hrefs(ast) == ["https://example.com/buy"]


@pytest.mark.parametrize("query", ["Link the button to javascript:alert(1)", "Use data:text/html,hi as the link"])
def test_unsafe_links_the_user_asked_for_are_reported(query):
    warnings = check_links(design(None), query)
    assert [w["kind"] for w in warnings] == ["link_not_set"]


@pytest.mark.parametrize("query", ["use and/or wording", "add 1/2 cup of sugar", "the #1 product", "a sign-up card"])
def test_ordinary_words_are_not_links(query):
    assert requested_links(query) == [] and check_links(design(None), query) == []
