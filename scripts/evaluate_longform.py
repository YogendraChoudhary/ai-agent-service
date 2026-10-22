"""
Long-document refinement (Mode 1): a free structural check, or an opt-in live evaluation.

  python scripts/evaluate_longform.py                                       # offline: 10 documents x 4 operations, scripted models
  python scripts/evaluate_longform.py --live --providers openai --corpus docs --limit 3 --ops grammar enhance
  python scripts/evaluate_longform.py --live --providers openai claude --corpus posts --ops tone --tone "friendly and warm"
  python scripts/evaluate_longform.py --live --files report.md notes.md --ops simplify --parallel 2

Offline proves the safety net on every document for nothing: a well-behaved scripted model loses nothing, and a damaging one
(drops a locked element in one chunk, summarises another) still returns every document complete, the damaged chunks restored as
written. --live makes PAID provider calls with the configured default keys and, for each (document, operation, provider),
records the structural loss (measured here, independently of the pipeline's own checks), the word ratio, chunks repaired and
restored, how many injected typos were fixed, latency (first text and total), tokens and cost, then writes the refined
document and a diff (local_run/longform_eval/) so providers can be read side by side. Nothing is ever uploaded.

Corpora: builtin (10 synthetic documents, 800-5,000 words, every kind of locked element), docs (this repository's Markdown,
sliced at section headings), posts (blog posts already generated under local_run/), or --files. Every document gets a few
deterministic spelling slips in its prose (--typo-every), which is what the grammar and enhance runs are scored on.

Targets from docs/rich_content_engine_plan.md (Phase 5): 0 structural loss on every run; word ratio 0.9-1.1 for enhance and
grammar; restored chunks <= 5%; p50 <= 45 s for 5,000 words.
"""

import argparse
import difflib
import json
import logging
import random
import re
import statistics
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings                                                                  # noqa: E402
from src.longform.markdown_ast import heading_levels, list_shape, parse, prose_words, shape, word_count   # noqa: E402
from src.longform.prompts import OPERATIONS                                                  # noqa: E402
from src.longform.refine import RefineConfig, qualifies, refine_stream                       # noqa: E402
from src.models import ADVANCED_MODEL_IDS, BASIC_MODEL_IDS                                   # noqa: E402
from src.observability import finish_trace, start_trace                                      # noqa: E402
from src.provider_errors import status_code                                                  # noqa: E402

RUN_DIR = ROOT / "local_run"
PROTECTED = ("table", "blockquote", "image", "fence", "html")
TYPES = {"grammar": "fix_grammar", "simplify": "simplify", "tone": "change_tone", "enhance": "enhance_with_ai"}
PREFIX = {"grammar": "Fix grammar: ", "simplify": "Simplify: ", "tone": "Change tone to {instruction}: ",
          "enhance": "Enhance: {instruction}, Text: "}                  # what api.build_full_query sends the single call
DEFAULT_INSTRUCTION = {"tone": "friendly and warm", "enhance": ""}
URL = re.compile(r"https?://[^\s)>\]\"']+")
TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>]*)?/?>|<!--.*?-->", re.S)       # inline HTML: a tag or a comment

TARGET_RATIO = (0.9, 1.1)                    # enhance and grammar
TARGET_RESTORED = 0.05
TARGET_P50_SECONDS = 45                      # for 5,000 words
LONG_DOCUMENT_WORDS = 4000                   # the documents the latency target speaks about
TYPO_EVERY_WORDS = 150
BUILTIN = [("d01", 800), ("d02", 1200), ("d03", 1500), ("d04", 2000), ("d05", 2500), ("d06", 3000), ("d07", 3500),
           ("d08", 4000), ("d09", 4500), ("d10", 5000)]


@contextmanager
def overriding(**values):
    """Settings changed for the length of a run only (the evaluation turns the pipeline on without touching the environment)."""
    old = {name: getattr(settings, name) for name in values}
    for name, value in values.items():
        setattr(settings, name, value)
    try:
        yield
    finally:
        for name, value in old.items():
            setattr(settings, name, value)


@dataclass
class Doc:
    id: str
    title: str
    text: str                                                # what is sent: the original with its spelling slips
    original: str
    typos: Dict[str, str] = field(default_factory=dict)      # misspelling -> the word it was
    source: str = ""

    @property
    def words(self) -> int:
        return word_count(self.text)


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

_SLIP_WORD = re.compile(r"(?<![\w'’./:@#\[(`-])[a-z]{6,12}(?![\w'’./:@#\])`-])")
_CODE_SPAN = re.compile(r"`[^`\n]*`")


def misspell(word: str, rng: random.Random) -> str:
    """A slip people make that is rarely another word: two neighbouring letters swapped, or one doubled."""
    i = rng.randrange(1, len(word) - 2)
    if rng.random() < 0.5 and word[i] != word[i + 1]:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word[:i] + word[i] + word[i:]


def fenced_lines(lines: List[str]) -> set:
    """Indexes of the lines of a fenced code block (a list item may hold one)."""
    inside, marker, found = False, "", set()
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        fence = re.match(r"`{3,}|~{3,}", stripped)
        if fence and not inside:
            inside, marker = True, fence.group(0)[0]
            found.add(i)
        elif inside:
            found.add(i)
            inside = not (fence and fence.group(0)[0] == marker)
    return found


def slip_sites(block_text: str) -> List[Tuple[int, int, int]]:
    """(line, start, end) of every whole lowercase word of 6-12 letters that may take a slip: never code, a name or a link target."""
    lines = block_text.split("\n")
    skip = fenced_lines(lines)
    sites = []
    for n, line in enumerate(lines):
        if n in skip:
            continue
        spans = [m.span() for m in _CODE_SPAN.finditer(line)]
        sites += [(n, m.start(), m.end()) for m in _SLIP_WORD.finditer(line)
                  if not any(a <= m.start() < b for a, b in spans)]
    return sites


def inject_typos(text: str, every: int = TYPO_EVERY_WORDS, seed: int = 0) -> Tuple[str, Dict[str, str]]:
    """
    About one spelling slip per `every` prose words, deterministic for a seed. Only whole lowercase words in paragraphs and
    lists change, so a heading, table, quote, code (also inside a list), link target or emphasis is never touched.
    Returns the text and {misspelling: the word it was}.
    """
    blocks = parse(text)
    candidates = [(b, *site) for b, block in enumerate(blocks) if block.kind in ("paragraph", "list")
                  for site in slip_sites(block.text)]
    wanted = min(len(candidates), prose_words(blocks) // every) if every > 0 else 0
    if not wanted:
        return text, {}
    rng = random.Random(seed)
    existing = set(re.findall(r"[A-Za-z]+", text))
    typos: Dict[str, str] = {}
    edits: Dict[int, List[Tuple[int, int, int, str]]] = {}
    for b, n, a, z in sorted(rng.sample(candidates, wanted)):
        word = blocks[b].text.split("\n")[n][a:z]
        slip = misspell(word, rng)
        if slip != word and slip not in existing and slip not in typos:
            typos[slip] = word
            edits.setdefault(b, []).append((n, a, z, slip))
    rebuilt = []
    for b, block in enumerate(blocks):
        lines = block.text.split("\n")
        for n, a, z, slip in sorted(edits.get(b, []), reverse=True):               # right to left: earlier offsets stay valid
            lines[n] = lines[n][:a] + slip + lines[n][z:]
        rebuilt.append("\n".join(lines))
    return "".join(rebuilt), typos


def typos_left(output: str, typos: Dict[str, str]) -> int:
    """How many of the injected misspellings are still in the output as whole words."""
    return sum(bool(re.search(rf"(?<![A-Za-z]){re.escape(slip)}(?![A-Za-z])", output)) for slip in typos)


def sections_of(text: str) -> List[str]:
    """The text cut before every H1 and H2 heading (as the parser sees headings: never inside a code fence)."""
    sections, current = [], []
    for block in parse(text):
        if block.kind == "heading" and block.level <= 2 and current:
            sections.append("".join(b.text for b in current))
            current = []
        current.append(block)
    if current:
        sections.append("".join(b.text for b in current))
    return sections


def slice_document(text: str, max_words: int = 5000, min_words: int = 800) -> List[str]:
    """Whole sections packed into parts of at most `max_words` (a section longer than that is a part of its own)."""
    parts: List[List[str]] = []
    current, size = [], 0
    for section in sections_of(text):
        words = word_count(section)
        if current and size + words > max_words:
            parts.append(current)
            current, size = [], 0
        current.append(section)
        size += words
    if current:
        parts.append(current)
    if len(parts) > 1 and sum(word_count(s) for s in parts[-1]) < min_words:     # a short tail joins the part before it
        tail = parts.pop()
        parts[-1] += tail
    return ["".join(p) for p in parts]


def synthetic_document(words: int, seed: int) -> str:
    """Word-salad prose carrying every kind of locked element, about `words` words long."""
    from tests.longform.fixtures import FENCE, FIGURE, NESTED_QUOTE, QUOTE, TABLE, paragraph

    rng = random.Random(seed)
    sections = max(2, round(words / 350))
    per = max(40, int(words * 0.8) // sections // 3)          # the tables, quotes, code and lists add the rest
    links = ["[the guide](https://example.com/guide)", "[a study](https://example.org/study \"Study title\")",
             "[our policy][pol]", "https://example.net/plain.", "<https://auto.example.com/x>", "[ref](https://example.com/two?a=1&b=2)"]
    parts = [f"# Working document {seed}", "", "*A subtitle line that is italic.*", "", paragraph(rng, 50), ""]
    for i in range(sections):
        parts += [f"## Section {i + 1}", "", paragraph(rng, per), ""]
        parts += [f"{paragraph(rng, per // 2)} See {links[i % len(links)]} for details. {paragraph(rng, per // 2)}", ""]
        if i % 3 == 0:
            parts += [TABLE, ""]
        if i % 2 == 1:
            parts += [QUOTE if i % 4 == 1 else NESTED_QUOTE, ""]
        if i % 4 == 2:
            parts += [FIGURE, ""]
        if i % 5 == 4:
            parts += [FENCE, ""]
        if i % 3 == 1:
            parts += [f"- {paragraph(rng, 12)}", f"- {paragraph(rng, 10)}", f"  - {paragraph(rng, 8)}", f"- {paragraph(rng, 9)}", ""]
        parts += [paragraph(rng, per), ""]
    parts += ['[pol]: https://example.com/policy "Policy"', ""]
    return "\n".join(parts)


def make_doc(doc_id: str, title: str, original: str, source: str, every: int = TYPO_EVERY_WORDS) -> Doc:
    seed = int(re.sub(r"\D", "", doc_id) or 0)
    text, typos = inject_typos(original, every, seed)
    return Doc(doc_id, title, text, original, typos, source)


def builtin_corpus(every: int = TYPO_EVERY_WORDS) -> List[Doc]:
    return [make_doc(doc_id, f"synthetic, ~{words} words", synthetic_document(words, int(doc_id[1:]) + 20), "builtin", every)
            for doc_id, words in BUILTIN]


def docs_corpus(every: int = TYPO_EVERY_WORDS) -> List[Doc]:
    """This repository's own Markdown: tables, code blocks, links and nested lists as people really write them."""
    docs = []
    for path in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        parts = slice_document(path.read_text(encoding="utf-8"))
        for n, part in enumerate(parts, 1):
            label = path.name + (f" (part {n}/{len(parts)})" if len(parts) > 1 else "")
            docs.append(make_doc(f"r{len(docs) + 1:02d}", label, part, str(path.relative_to(ROOT)), every))
    return docs


def posts_corpus(every: int = TYPO_EVERY_WORDS) -> List[Doc]:
    """The blog posts Mode 2 wrote under local_run/ (natural model prose, tables, quotes, pictures), alone and three together."""
    runs = {}
    for pattern in ("blog_eval/*.md", "blog_9to5/*.md", "eval_1450*/blog_eval/*.md"):
        for path in sorted(RUN_DIR.glob(pattern)):
            runs.setdefault(path.name.split("_")[0], path)               # one run per brief (t01, t04, ...)
    posts = [re.sub(r"\A<!--.*?-->\s*", "", p.read_text(encoding="utf-8"), flags=re.S) for p in runs.values()]
    posts = [p for p in posts if word_count(p) >= 800]
    groups = [("post", [p]) for p in posts] + [("three posts", posts[i:i + 3]) for i in range(len(posts) - 2)]
    return [make_doc(f"p{n:02d}", f"{label}, {word_count(chr(10).join(g))} words", "\n\n".join(g).rstrip("\n") + "\n", "local_run", every)
            for n, (label, g) in enumerate(groups, 1)]


def files_corpus(paths: List[str], every: int = TYPO_EVERY_WORDS) -> List[Doc]:
    return [make_doc(f"f{n:02d}", Path(p).name, Path(p).read_text(encoding="utf-8"), p, every) for n, p in enumerate(paths, 1)]


# ---------------------------------------------------------------------------
# Measuring a result
# ---------------------------------------------------------------------------

def collapsed(blocks) -> List[str]:
    """The block kinds in order, a run of paragraphs counting once: merging or splitting paragraphs is not a structural change."""
    out: List[str] = []
    for kind in shape(blocks):
        if not (kind == "paragraph" and out and out[-1] == "paragraph"):
            out.append(kind)
    return out


def urls_of(text: str) -> Counter:
    return Counter(u.rstrip(".,;:!?") for u in URL.findall(text))


def structure_loss(original: str, output: str, freeze_quotes: bool = True) -> List[str]:
    """
    What the output lost or moved, judged from the two Markdown texts alone (not by the pipeline's own checks): tables, quotes,
    pictures, code and HTML byte for byte, every URL and inline tag, heading levels, list shapes, and the order of the blocks.
    Empty = intact.
    """
    before, after = parse(original), parse(output)
    kinds = PROTECTED if freeze_quotes else tuple(k for k in PROTECTED if k != "blockquote")
    want, got = [b.text for b in before if b.kind in kinds], [b.text for b in after if b.kind in kinds]
    lost = []
    if want != got:
        first = next((i for i, (a, b) in enumerate(zip(want, got)) if a != b), min(len(want), len(got)))
        lost.append(f"protected blocks: {len(got)} of {len(want)} kept, first difference at #{first + 1}")
    if urls_of(original) != urls_of(output):
        diff = (urls_of(original) - urls_of(output)) + (urls_of(output) - urls_of(original))
        lost.append(f"urls changed: {sorted(diff)[:3]}")
    if Counter(TAG.findall(original)) != Counter(TAG.findall(output)):
        lost.append("inline HTML tags changed")
    if heading_levels(before) != heading_levels(after):
        lost.append(f"headings: {len(heading_levels(after))} of {len(heading_levels(before))}, or their levels changed")
    if list_shape(before) != list_shape(after):
        lost.append("a list changed shape")
    if collapsed(before) != collapsed(after):
        lost.append("the order or kind of the blocks changed")
    return lost


def percentile(values: List[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]


def row_of(doc: Doc, op: str, instruction: str, output: str, summary: dict, *, provider: str, model: str,
           first_chunk: Optional[float], seconds: float, warnings: list, error: Optional[str] = None) -> dict:
    """One result, whoever produced it (a scripted model or a provider), in the shape the summary reads."""
    longform = summary.get("longform") or {}
    before, after = prose_words(parse(doc.original)), prose_words(parse(output))
    loss = structure_loss(doc.original, output) if output else ["no text"]
    row = {
        "provider": provider, "model": model, "id": doc.id, "title": doc.title, "op": op, "instruction": instruction,
        "words": doc.words, "pipeline": summary.get("mode") == "longform",
        "chunks": longform.get("chunks", 0), "repaired": longform.get("repaired", 0), "restored": longform.get("restored", 0),
        "passthrough": longform.get("passthrough", 0), "repair_reasons": longform.get("repair_reasons", {}),
        "frozen": longform.get("frozen"), "words_in": before, "words_out": after,
        "ratio": round(after / before, 3) if before else 1.0, "loss": loss,
        "typos": {"injected": len(doc.typos), "left": typos_left(output, doc.typos)},
        "first_chunk": round(first_chunk, 2) if first_chunk is not None else None, "seconds": round(seconds, 2),
        "served_by": summary.get("model_used"), "fallbacks": summary.get("fallbacks"), "llm_calls": summary.get("llm_calls"),
        "tokens_in": summary.get("tokens_in"), "tokens_out": summary.get("tokens_out"), "est_cost_usd": summary.get("est_cost_usd"),
        "warnings": [w.get("kind") for w in warnings], "error": error, "content": output,
    }
    row["failures"] = failures(row)
    return row


def failures(row: dict) -> List[str]:
    """What a run must satisfy whoever wrote it. The offline run fails on any of these; a live run reports them."""
    out = []
    if row["error"]:
        out.append(f"error: {row['error']}")
    elif not row["pipeline"]:
        out.append("the chunked path did not run")
    out += [f"structural loss: {item}" for item in row["loss"]]
    return out


# ---------------------------------------------------------------------------
# Offline
# ---------------------------------------------------------------------------

def fixer(typos: Dict[str, str]) -> Callable[[str], str]:
    """A model that corrects the misspellings it is shown."""
    if not typos:
        return lambda passage: passage
    pattern = re.compile(r"(?<![A-Za-z])(" + "|".join(map(re.escape, sorted(typos, key=len, reverse=True))) + r")(?![A-Za-z])")
    return lambda passage: pattern.sub(lambda m: typos[m.group(1)], passage)


def trim(passage: str, every: int = 12) -> str:
    """A simplifier: drops every 12th word of the plain prose lines and leaves everything else as it is."""
    lines = []
    for line in passage.split("\n"):
        if line and not re.search(r"\[\[|link://|^\s*(?:[#>|`~*-]|\d+\.)", line):
            line = " ".join(w for i, w in enumerate(line.split(" "), 1) if i % every)
        lines.append(line)
    return "\n".join(lines)


def scripted(op: str, typos: Dict[str, str], damaging: bool):
    from tests.longform import fakes

    fix = fixer(typos)
    behaviours = {"grammar": lambda p, s: fix(p), "simplify": lambda p, s: fix(trim(p)),
                  "tone": lambda p, s: fix(fakes.polish(p, s)), "enhance": lambda p, s: fix(fakes.polish(p, s))}
    per_unit = {"c2": fakes.drop_a_token, "c3": fakes.summarise} if damaging else {}
    return fakes.ScriptedRefiner(behaviours[op], per_unit)


def run_one_offline(doc: Doc, op: str, damaging: bool) -> dict:
    instruction = DEFAULT_INSTRUCTION.get(op, "")
    request_id = f"eval-longform-offline-{doc.id}-{op}-{'bad' if damaging else 'good'}"
    start_trace(request_id, "longform_evaluation", type=TYPES[op], chat_model="scripted")
    started, first, chunks, warnings = time.perf_counter(), None, [], []
    for item in refine_stream(doc.text, instruction, OPERATIONS[op], scripted(op, doc.typos, damaging), RefineConfig(),
                              request_id=request_id):
        if isinstance(item, str):
            first = first if first is not None else time.perf_counter() - started
            chunks.append(item)
        elif "__warnings__" in item:
            warnings = item["__warnings__"]
    seconds = time.perf_counter() - started
    summary = finish_trace(request_id, "ok") or {}
    row = row_of(doc, op, instruction, "".join(chunks), summary, provider="scripted",
                 model="damaging" if damaging else "well-behaved", first_chunk=first, seconds=seconds, warnings=warnings)
    row["expected_bad"] = damaging                                       # its restored chunks are the point, not a miss
    return row


def run_offline(corpus: Optional[List[Doc]] = None) -> List[dict]:
    """Every document x operation with a well-behaved model (nothing changes but prose) and a damaging one (the net must hold)."""
    rows = []
    for doc in corpus or builtin_corpus():
        for op in OPERATIONS:
            good, bad = run_one_offline(doc, op, False), run_one_offline(doc, op, True)
            if good["restored"]:
                good["failures"].append(f"{good['restored']} chunks restored although the model did nothing wrong")
            if good["typos"]["left"]:
                good["failures"].append(f"{good['typos']['left']} of {good['typos']['injected']} spelling slips not fixed")
            if good["op"] in ("enhance", "grammar") and not TARGET_RATIO[0] <= good["ratio"] <= TARGET_RATIO[1]:
                good["failures"].append(f"word ratio {good['ratio']}")
            if bad["chunks"] >= 3 and not bad["restored"]:
                bad["failures"].append("a damaging model went unnoticed")
            rows += [good, bad]
    return rows


# ---------------------------------------------------------------------------
# Live
# ---------------------------------------------------------------------------

def run_one_live(provider: str, key: str, doc: Doc, op: str, instruction: str, resolver=None) -> dict:
    from src.text_processing_agent import process_user_query_stream

    request_id = f"eval-longform-{provider}-{doc.id}-{op}"
    query = PREFIX[op].format(instruction=instruction or "") + doc.text
    start_trace(request_id, "longform_evaluation", type=TYPES[op], chat_model=provider)
    started, first, chunks, warnings, failure = time.perf_counter(), None, [], [], None
    try:
        for chunk in process_user_query_stream(
                query, provider, key, request_id=request_id, force_agent_type=op, is_byok=False, key_resolver=resolver,
                instruction=instruction or None, user_text=doc.text):
            if isinstance(chunk, str):
                first = first if first is not None else time.perf_counter() - started
                chunks.append(chunk)
            elif isinstance(chunk, dict) and "__warnings__" in chunk:
                warnings = chunk["__warnings__"]
    except Exception as error:
        failure = error
    seconds = time.perf_counter() - started
    summary = finish_trace(request_id, "error" if failure else "ok") or {}
    ids = ADVANCED_MODEL_IDS if op == "enhance" else BASIC_MODEL_IDS
    row = row_of(doc, op, instruction, "".join(chunks), summary, provider=provider, model=ids[provider], first_chunk=first,
                 seconds=seconds, warnings=warnings, error=str(failure) if failure else None)
    served_by = summary.get("model_used")
    if served_by and not served_by.startswith(f"{provider}/"):
        row["failures"].append(f"served by {served_by}, not {provider} (a fallback answered)")      # never score the wrong model
    return row


def write_outputs(row: dict, doc: Doc, out_dir: Path) -> Path:
    """The refined document, and a diff against what was sent, named for the run so providers can be compared."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inputs").mkdir(exist_ok=True)
    (out_dir / "inputs" / f"{doc.id}.md").write_text(doc.text, encoding="utf-8")
    stem = f"{row['id']}_{row['op']}_{row['provider']}"
    path = out_dir / f"{stem}.md"
    path.write_text(row["content"] or f"(no text: {row['error']})", encoding="utf-8")
    diff = difflib.unified_diff(doc.text.splitlines(), (row["content"] or "").splitlines(), "sent", "refined", lineterm="", n=0)
    (out_dir / f"{stem}.diff").write_text("\n".join(diff) + "\n", encoding="utf-8")
    return path


def summarise(rows: List[dict]) -> List[str]:
    """Printable lines comparing the run against the plan's Phase 5 targets."""
    ran = [r for r in rows if r.get("pipeline")]
    if not ran:
        return ["no results"]
    answered_elsewhere = [r for r in ran if any(f.startswith("served by") for f in r.get("failures", []))]
    scored = [r for r in ran if not r.get("expected_bad") and r not in answered_elsewhere]     # neither a model scripted to misbehave
    if not scored:                                                                              # nor one that another model answered for
        return [f"no results: {len(answered_elsewhere)} runs were answered by a fallback provider, not the one asked for"]
    chunks, restored = sum(r["chunks"] for r in scored), sum(r["restored"] for r in scored)
    ratioed = [r for r in scored if r["op"] in ("enhance", "grammar")]
    in_window = sum(TARGET_RATIO[0] <= r["ratio"] <= TARGET_RATIO[1] for r in ratioed)
    lines = [f"runs: {len(ran)} of {len(rows)} used the chunked path   structural loss: {sum(bool(r['loss']) for r in ran)} runs (target 0)",
             f"word ratio {TARGET_RATIO[0]}-{TARGET_RATIO[1]} for enhance/grammar: {in_window}/{len(ratioed)}",
             f"restored chunks: {restored}/{chunks} = {restored / chunks if chunks else 0:.1%} (target <= {TARGET_RESTORED:.0%})   "
             f"repaired: {sum(r['repaired'] for r in scored)}"]
    long = [r["seconds"] for r in scored if r["words"] >= LONG_DOCUMENT_WORDS and r["provider"] != "scripted"]
    if long:
        lines.append(f"documents of {LONG_DOCUMENT_WORDS}+ words: p50 {statistics.median(long):.1f}s p95 {percentile(long, 0.95):.1f}s "
                     f"(target p50 <= {TARGET_P50_SECONDS}s for 5,000 words)")
    injected, left = sum(r["typos"]["injected"] for r in scored), sum(r["typos"]["left"] for r in scored)
    if injected:
        lines.append(f"spelling slips left in the output: {left}/{injected}")
    if answered_elsewhere:
        lines.append(f"NOT COUNTED: {len(answered_elsewhere)} runs of {sorted({r['provider'] for r in answered_elsewhere})} were answered by "
                     f"{sorted({r['served_by'] for r in answered_elsewhere})} (a fallback), so they say nothing about the provider asked for")
    bad = [r for r in ran if r.get("expected_bad")]
    if bad:
        lines.append(f"scripted to misbehave: {len(bad)} runs, {sum(r['restored'] for r in bad)} of {sum(r['chunks'] for r in bad)} chunks "
                     f"restored as written, structural loss in {sum(bool(r['loss']) for r in bad)} runs (target 0)")
    for key in dict.fromkeys((r["provider"], r["model"], r["op"]) for r in scored):
        mine = [r for r in scored if (r["provider"], r["model"], r["op"]) == key]
        secs, firsts = [r["seconds"] for r in mine], [r["first_chunk"] for r in mine if r["first_chunk"] is not None]
        cost = [r["est_cost_usd"] for r in mine if r.get("est_cost_usd") is not None]
        lines.append(
            f"  {key[0]:9} {key[1]:20} {key[2]:9} n={len(mine):<3} ratio {statistics.mean(r['ratio'] for r in mine):.3f}  "
            f"restored {sum(r['restored'] for r in mine)}/{sum(r['chunks'] for r in mine)}  repaired {sum(r['repaired'] for r in mine)}  "
            f"slips left {sum(r['typos']['left'] for r in mine)}/{sum(r['typos']['injected'] for r in mine)}  "
            f"p50 {statistics.median(secs):.1f}s p95 {percentile(secs, 0.95):.1f}s"
            + (f"  first {statistics.median(firsts):.1f}s" if firsts else "")
            + (f"  ${sum(cost):.3f}" if cost else ""))
    reasons = Counter(kind for r in scored for kinds in r["repair_reasons"].values() for kind in kinds)
    if reasons:
        lines.append("why chunks needed a repair: " + ", ".join(f"{k} {n}" for k, n in reasons.most_common()))
    return lines


def pick(corpus: List[Doc], ids: Optional[List[str]], limit: int) -> List[Doc]:
    chosen = [d for d in corpus if not ids or d.id in ids]
    return chosen[:limit or None]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="Make PAID provider calls using the configured default keys")
    parser.add_argument("--providers", nargs="+", choices=["openai", "claude", "gemini", "grok"], default=["openai"])
    parser.add_argument("--corpus", choices=["builtin", "docs", "posts"], default=None,
                        help="builtin (the default offline), docs (the default with --live) or posts")
    parser.add_argument("--files", nargs="+", metavar="FILE.md", help="Evaluate these Markdown files instead of a corpus")
    parser.add_argument("--ids", nargs="+", help="Only these document ids (d01, r03, p02, f01 ...)")
    parser.add_argument("--limit", type=int, default=0, help="Documents per provider; 0 runs them all")
    parser.add_argument("--ops", nargs="+", choices=list(OPERATIONS), default=list(OPERATIONS))
    parser.add_argument("--tone", default=DEFAULT_INSTRUCTION["tone"], help="The tone the tone runs ask for")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION["enhance"], help="The instruction the enhance runs get")
    parser.add_argument("--typo-every", type=int, default=TYPO_EVERY_WORDS, metavar="WORDS",
                        help="One spelling slip per this many prose words (0 = none)")
    parser.add_argument("--parallel", type=int, default=0, help="Chunks in flight (LONGFORM_PARALLEL) for this run")
    parser.add_argument("--chunk-words", type=int, default=0, help="Target chunk size (LONGFORM_CHUNK_WORDS) for this run")
    parser.add_argument("--first-chunk", type=int, default=None, metavar="WORDS",
                        help="Size of the first chunk (LONGFORM_FIRST_CHUNK_WORDS) for this run; 0 = same as the others")
    parser.add_argument("--list", action="store_true", help="Print the corpus (ids, titles, sizes) and stop")
    parser.add_argument("--debug", action="store_true", help="Log what a model answered when a passage needed a repair")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.limit < 0 or args.typo_every < 0 or args.parallel < 0 or args.chunk_words < 0 or (args.first_chunk or 0) < 0:
        parser.error("--limit, --typo-every, --parallel, --chunk-words and --first-chunk must be non-negative")
    args.out = args.out or RUN_DIR / "longform_evaluation.json"

    name = args.corpus or ("docs" if args.live else "builtin")
    if args.files:
        corpus = files_corpus(args.files, args.typo_every)
    else:
        corpus = {"builtin": builtin_corpus, "docs": docs_corpus, "posts": posts_corpus}[name](args.typo_every)
    corpus = pick(corpus, args.ids, args.limit)
    single_call = [d for d in corpus if not qualifies(d.text)]
    if single_call and not args.list:
        print("Skipped (they would keep the single call, which this evaluation does not measure): "
              + ", ".join(f"{d.id} {d.title}" for d in single_call))
        corpus = [d for d in corpus if d not in single_call]
    if not corpus:
        parser.error(f"no documents to evaluate in {'--files' if args.files else 'the ' + name + ' corpus'} (--ids/--limit too narrow?)")
    if args.list:
        for d in corpus:
            print(f"{d.id}  {d.words:>6} words  {len(d.typos):>2} slips  {d.title}  [{d.source}]"
                  + ("" if d not in single_call else "  (single call: HTML or token look-alikes)"))
        return 0

    if not args.live:
        logging.getLogger("src.longform").setLevel(logging.ERROR)           # the damaging model's restores are expected
        rows = run_offline(corpus)
        bad = [r for r in rows if r["failures"]]
        print(f"Offline: {len(corpus)} documents x {len(OPERATIONS)} operations x 2 scripted models through the real pipeline, "
              f"{len(bad)} runs with failures")
        for r in bad:
            print(f"  {r['id']} {r['op']} {r['model']}: {r['failures']}")
        print("\n".join(summarise(rows)))
        return int(bool(bad))

    if args.debug:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        logging.getLogger("src.longform").setLevel(logging.DEBUG)
    rows = run_live(args, corpus)
    print("\n".join(summarise(rows)))
    print(f"Wrote {len(rows)} results to {args.out} and the documents to {RUN_DIR / 'longform_eval'}")
    return int(any(r.get("error") or r.get("loss") for r in rows))


def run_live(args, corpus: List[Doc]) -> List[dict]:
    keys = {"openai": settings.default_openai_api_key, "claude": settings.default_anthropic_api_key,
            "gemini": settings.default_gemini_api_key, "grok": settings.default_grok_api_key}
    from scripts.evaluate_blog import key_resolver

    resolver = key_resolver()
    out_dir = RUN_DIR / "longform_eval"
    overrides = {"longform_refine_enabled": True}
    if args.parallel:
        overrides["longform_parallel"] = args.parallel
    if args.chunk_words:
        overrides["longform_chunk_words"] = args.chunk_words
    if args.first_chunk is not None:
        overrides["longform_first_chunk_words"] = args.first_chunk
    rows: List[dict] = []
    with overriding(**overrides):
        for provider in args.providers:
            if not keys[provider]:
                rows.append({"provider": provider, "error": "No configured key", "pipeline": False, "failures": ["no key"],
                             "warnings": []})
                continue
            stop = False
            for doc in corpus:
                for op in args.ops:
                    instruction = {"tone": args.tone, "enhance": args.instruction}.get(op, "")
                    row = run_one_live(provider, keys[provider], doc, op, instruction, resolver)
                    rows.append(row)
                    write_outputs(row, doc, out_dir)
                    args.out.parent.mkdir(parents=True, exist_ok=True)       # saved after every run: an interrupted paid
                    args.out.write_text(json.dumps([{k: v for k, v in r.items() if k != "content"} for r in rows], indent=2),
                                        encoding="utf-8")                      # evaluation loses no finished work
                    print(f"{provider:7} {doc.id} {op:9} {row['seconds']:>6}s first={row['first_chunk']}s chunks={row['chunks']} "
                          f"repaired={row['repaired']} restored={row['restored']} ratio={row['ratio']} "
                          f"slips_left={row['typos']['left']}/{row['typos']['injected']} failures={row['failures']}")
                    if row["error"] and status_code(Exception(row["error"])) in (400, 401, 403, 404):
                        stop = True                                              # a dead key or retired model: stop paying for it
                        break
                    if any(f.startswith("served by") for f in row["failures"]):
                        print(f"{provider} did not answer (a fallback did): skipping its remaining runs")
                        stop = True                                              # every further run would be paid for on another model
                        break
                if stop:
                    break
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
