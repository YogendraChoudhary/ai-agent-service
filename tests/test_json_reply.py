"""Claude Opus 5.5 sometimes fences its JSON; the chart extractor and key-point highlighter must accept that without a second call."""
import json

import pytest

from src.charts.extract import suggest_chart
from src.json_reply import strip_json_fence
from src.key_points import highlight_key_points
from tests.charts.test_charts import EXTRACTED, SOURCE as CHART_SOURCE
from tests.test_key_points import SELECTION, SOURCE as KEY_SOURCE

PAYLOAD = json.dumps({"a": 1, "b": [1, 2]})


@pytest.mark.parametrize("reply", [
    f"```json\n{PAYLOAD}\n```", f"```\n{PAYLOAD}\n```", f"  ```JSON\r\n{PAYLOAD}\r\n```  \n", f"```json\n{PAYLOAD}```", f"```json   \n{PAYLOAD}\n```\n\n"])
def test_a_reply_that_is_one_fenced_block_is_unwrapped(reply):
    assert json.loads(strip_json_fence(reply)) == {"a": 1, "b": [1, 2]}


@pytest.mark.parametrize("reply", [PAYLOAD, "", "Here you go:\n```json\n{}\n```", "```json\n{}\n```\nThanks!", "```python\nx = 1\n```\n```json\n{}\n```", None])
def test_anything_else_is_returned_unchanged(reply):
    assert strip_json_fence(reply) == reply


def test_chart_extraction_accepts_a_fenced_reply():
    fenced = lambda *a: f"```json\n{json.dumps(EXTRACTED)}\n```"
    plain = lambda *a: json.dumps(EXTRACTED)
    via_fence, via_plain = suggest_chart(CHART_SOURCE, fenced), suggest_chart(CHART_SOURCE, plain)
    assert via_fence is not None
    assert via_fence.model_dump(exclude={"chart_id"}) == via_plain.model_dump(exclude={"chart_id"})      # chart_id is generated per call


def test_key_points_accepts_a_fenced_reply_on_the_first_call():
    calls = []

    def llm(system, messages, spec):
        calls.append(spec.stage)
        return f"```json\n{json.dumps(SELECTION)}\n```"

    result = highlight_key_points(KEY_SOURCE, llm)
    assert result is not None and calls == ["key_points"]          # no repair round trip
