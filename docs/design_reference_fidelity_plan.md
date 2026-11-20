# Design canvas — reference-image fidelity plan

Status: **Backend, fixtures, and local renderer implemented.** Schema 1.1 and two-pass vision default to off.
Live baseline, budget comparison, and editor acceptance are rollout gates. Section 1 describes the pre-change code.

## 0. The report

A user attached `Promotion Designs.png` (Thanksgiving / Black Friday promo: gradient headline, two-column card with
the offer on the left and a big `50%` on the right) and asked for an accurate design. Two generations came back:

| Run | Layout | Typography / colour |
|---|---|---|
| Earlier run (screenshot 3) | **Wrong.** The `50%` panel sits *below* the offer instead of beside it | Editor font, solid colours |
| Later run (screenshot 2) | Two columns, roughly right | **Wrong.** Different font (editor's DM Sans), no gradient, "Half the" white instead of lavender, eyebrow and "OFF FOR 3 MONTHS" not monospaced/letter-spaced, `%` not superscript, vertical rule replaced by a second filled card, no copy icon, no button glow |

Goal: when a reference image is attached, the design must reproduce **layout, colours and type** as closely as the
editable AST can express, the result must be checked against the reference before it is returned, and nothing that
works today (prompt designs, AI edits of existing designs, the current frontend) may break on the way.

## 1. What the code does today (root causes)

### RC1 — The AST cannot express the reference's typography or decoration
`src/design_canvas/schema.py`

* `TextSpan` has only `text, color, font_weight, italic, underline`. There is **no font family, letter spacing,
  text transform (uppercase), gradient fill or superscript.**
* `DesignDocumentAST` has no font tokens, so the frontend renders design text in whatever the editor uses
  (screenshot 2's toolbar shows **DM Sans**). This is why "the fonts are different": the model has no field to put a
  font in.
* `TextNode.font_size_px` caps at **96**. The reference's `50%` is ~200 px tall at 2000 px wide (~150 px at the
  1500 px the server downsizes to), so it can't be matched.
* `DividerNode` is horizontal only — the reference's vertical rule between the columns can't be expressed, so the
  model invents a second card.
* `NodeStyle` has no gradient background, no shadow/glow; `IconButtonNode.icon_name` has no `copy` or
  `arrow_up_right`; `ButtonNode` has no trailing icon.
* `font_weight` is `normal|medium|semibold|bold` — no `light` / `extrabold`.

### RC2 — The colour extractor throws away the accents, then the prompt forbids using any others
`src/design_canvas/vision_extractor.py` quantizes the whole image to 8 colours with median cut and sorts them **by
pixel count**. Measured on the reported reference (`1.webp`, 2000×1162):

```
extracted_palette: #1B1923 #2B2637 #211E2A #151619 #1C1C22 #101014 #191720 #887F97
pixels actually in the image:
  button lavender  #BD9CF4   ← missing
  headline mint    #BED5D4   ← missing
  card surface     #282333   (≈ #2B2637, present)
```

Seven of eight colours are near-black shades of the background; **every accent is missing**. Then
`src/design_canvas/prompts.py` (`build_initial_design_prompt`, `create_from_vision`) says:

> Use the extracted palette colours exactly — do NOT invent new colours.

So the model is told to choose between obeying the prompt (dark mush) and obeying the picture. Either way the
validator never checks colours: `validate_design_ast(..., vision_meta=...)` takes `vision_meta` but marks it
"reserved" and does nothing with it (`src/design_canvas/validate.py`).

### RC3 — The vision task has almost no instructions about layout
The whole vision task is two sentences (`prompts.py`). Nothing tells the model to keep side-by-side regions
side-by-side, to keep column proportions, reading order, text verbatim, or relative sizes. The schema gives it no
way to say "these two columns, 58/42, never wrap": `ContainerNode` has `flex_row` and `grid` but no column ratios,
and the rendering guide (`docs/FRONTEND_HANDOFF.md` §4.5) says that with `columns: null` the frontend "pick[s] your
own rule (e.g. auto-fit)" — an auto-fit grid or a wrapping flex row **stacks the `50%` panel under the offer** on a
narrow editor column. That is screenshot 3.

### RC4 — Repair rounds lose the image
`src/design_canvas/service.py` passes `image_bytes=reference_image_bytes if attempt == 0 else None`, and
`_run_design_json_completion` (`src/text_processing_agent.py`) rebuilds the provider messages on each call and only
attaches the image when `image_bytes` is set. So on a repair round (schema or contrast error), **the model is asked
to fix the design without the reference in front of it.**

### RC5 — The auto-healer can destroy accent colours
`_fix_contrast` (`validate.py`) recolours any span below 3:1 to pure black or white. A dim-but-correct accent from
the reference (or a gradient approximation) becomes white — exactly "Half the" in screenshot 2 would look like.

### RC6 — Nothing compares the result to the reference
The harness checks schema, ids, SVG safety, contrast and depth. Nothing checks: are all texts present, is the
two-column region still two columns, are the colours from the image, is the font family set. A design that ignores
the reference passes as `pass` in `report.attempts` — which is how a wrong design gets reported as done.

### RC7 — Vision runs on the speed-tuned setting
`src/models.py`: `DESIGN_MODEL_IDS = {"openai": "gpt-6.1-sol"}`, `OPENAI_DESIGN_EFFORT = "low"` — chosen for a
9.8 s prompt-design median. Recreating a dense screenshot is a harder perception task; it gets the same budget,
and `_DESIGN_MAX_OUTPUT_TOKENS = 8192` leaves little room once the AST gets richer.

### Not the cause (checked)
* Routing: a request with an attachment plus a UI cue goes to `create_from_vision` (`src/design_intent.py`,
  rule 2), and the image is fetched and sent on the first call for all three providers.
* Image size: `_design_reference_image` downsizes to 1500 px on the long side — enough to read this design.

## 2. What to build

Seven work items. Order matters: **W1–W2 change nothing the frontend sees**, W3 is additive behind a flag, and the
frontend renderer (W7) must ship before the flag goes on.

| # | Item | Fixes | Frontend change? | Risk to existing behaviour |
|---|---|---|---|---|
| W0 | Fidelity eval harness + fixtures (first, so every later step is measured) | RC6 | no | none (dev only) |
| W1 | Role-based colour extraction + honest prompt wording | RC2 | no | low |
| W2 | Keep the image on repair rounds; palette-aware contrast heal | RC4, RC5 | no | low |
| W3 | Schema 1.1: fonts, tracking, case, gradient text, superscript, vertical divider, column ratios, no-wrap rows, shadow, icons | RC1, RC3 | **yes** | medium → flag + downgrade |
| W4 | Two-pass vision: `ReferenceSpec` extraction, then AST from spec + image | RC3 | no | +1 call latency, flag |
| W5 | Deterministic fidelity checks against the spec and palette, one bounded repair | RC6 | no | could add a repair round |
| W6 | Vision-specific model effort / output budget | RC7 | no | latency/cost, measured |
| W7 | Frontend renderer + handoff doc for 1.1 | RC1, RC3 | **yes** (other repo) | renderer must ignore unknown fields |

### W0 — Eval harness first (proves or disproves every later claim)

* Fixtures: `tests/design/fixtures/reference/promo_blackfriday.webp` (this report) plus 3–4 more varied references
  (light theme, a 3-card pricing row, a serif editorial hero, a mobile screen). Each gets a hand-written
  `*.expect.json`: required texts, regions with `side_by_side` + approximate ratio, expected accent colours, font
  class per text role.
* `scripts/design_fidelity_eval.py` (paid, live): runs `create_from_vision` on each fixture N times, saves the AST and
  report to `local_run/design_fidelity/run_<ts>.json`, and scores with the same checks W5 adds (text recall, layout
  match, accent ΔE, fonts set). Optional `--render`: renders the AST to HTML with a small Python reference renderer
  (mirrors `FRONTEND_HANDOFF.md` §4.5 + the 1.1 additions) and screenshots it with Playwright (dev dependency only,
  never imported by the server) so before/after images can be shown side by side.
* **Record a baseline on today's code before changing anything.** Every later item reports its delta against it.

### W1 — Role-based colour extraction

Replace frequency-only quantization with a role-aware palette. Pure Pillow, no new runtime dependency.

1. Downscale to ≤ 400 px long side (speed: keeps the < 15 ms budget).
2. Quantize to 64 colours and get counts. Keep the full, unmerged palette for W5. The promo measurements favoured
   Pillow's fast octree over median cut for the small lavender and mint regions.
3. Merge near-duplicates (ΔE76 < 6 in Lab — a 20-line helper, no numpy needed).
4. Assign roles:
   * `backgrounds`: top 1–3 by area.
   * `surfaces`: next by area whose luminance differs from the background by a small step (cards, panels).
   * `text`: colours with contrast ≥ 4.5:1 against the dominant background and low area.
   * `accents`: area ≥ 0.05 % and chroma (Lab √(a²+b²)) ≥ 20, sorted by chroma × log(area). Also include
     small swatches (≤ 5 % area) with chroma ≥ 6 and contrast ≥ 3:1 against the dominant background.
     Keep a leading swatch per hue in the prompt accent list. The reference lavender `#BD9CF4` has chroma 49.5 and
     the mint `#BED5D4` has chroma 8.1.
5. Keep `top_hex_colors` (unchanged field, for compatibility) and add `roles` and `quantized_hex_colors`.

Prompt wording changes from "do NOT invent new colours" to:

```
Colours: prefer the measured colours below; use the role that matches (background, surface, text, accent).
If the image clearly shows a colour that is not listed (a small accent, a gradient stop), use its true hex.
Never replace a coloured element with white, black or grey.
```

Reference code (extractor):

```python
@dataclass(frozen=True)
class PaletteRoles:
    backgrounds: List[str]
    surfaces: List[str]
    text: List[str]
    accents: List[str]

@dataclass(frozen=True)
class VisionMetadata:
    width: int
    height: int
    top_hex_colors: List[str]          # unchanged: frequency order, kept for compatibility
    is_dark_theme: bool
    dominant_background: Optional[str]
    roles: Optional[PaletteRoles] = None   # new
    quantized_hex_colors: Optional[List[str]] = None

def _lab(rgb):  # sRGB → CIE Lab (D65), plain-Python
    ...

def _chroma(lab):
    return (lab[1] ** 2 + lab[2] ** 2) ** 0.5

def _assign_roles(colors_with_counts, total_px) -> PaletteRoles:
    merged = _merge_close(colors_with_counts, delta_e=6.0)
    by_area = sorted(merged, key=lambda c: c.count, reverse=True)
    bg = by_area[0]
    accents = [c for c in merged
               if c.count / total_px >= 0.0005 and (
                   _chroma(c.lab) >= 20
                   or (c.count / total_px <= 0.05 and _chroma(c.lab) >= 6 and _contrast(c, bg) >= 3)
               )]
    accents.sort(key=lambda c: _chroma(c.lab) * math.log1p(c.count), reverse=True)
    ...
```

Acceptance (unit test on the fixture): `roles.accents` contains a colour within ΔE 8 of `#BD9CF4` and one within
ΔE 8 of `#BED5D4`; `dominant_background` stays within ΔE 5 of `#1B1923`; runtime < 15 ms on a 1080×1920 image.

### W2 — Keep the reference on repairs; heal contrast without killing accents

* `service.py`: send `reference_image_bytes` on **every** attempt, not only attempt 0. The provider adapters already
  attach it to the first user turn only (`not any(isinstance(m.get("content"), list) ...)` / `not conversation_turns`),
  so the history stays well-formed; the image simply stays in context.

  ```python
  raw_output = await agent_runner.run_json_completion(
      messages=conversation,
      image_bytes=reference_image_bytes,          # was: only on attempt 0
      temperature=0.1 if attempt > 0 else 0.2,
  )
  ```
  Cost: image tokens on each repair round (Claude's system prompt is already cached; the image is ~1.5k tokens at
  1500 px). Test: a fake runner records `image_bytes` per attempt and asserts it is present on attempts 0..n.

* `_fix_contrast`: before falling back to black/white, try the closest palette colour (same hue family, from
  `roles.text` then `roles.accents`) that reaches 3:1. Only if none does, use black/white. Needs `vision_meta`,
  which `auto_heal_design_ast` already receives. Check the solid fallback and every gradient stop independently.
  Preserve stops that pass and repair each low-contrast stop while keeping the gradient angle and stop order.

### W3 — Schema 1.1 (additive, flag-gated, with a downgrade path)

All new fields are optional with defaults equal to today's rendering, so **every 1.0 AST is a valid 1.1 AST and
renders the same.**

```python
FontRole = Literal["display", "heading", "body", "mono"]

# Fonts the frontend can load (Google Fonts, OFL). The model picks the closest match;
# an exact commercial font in a screenshot can't be identified reliably, so this is "closest", not "same".
ALLOWED_FONT_FAMILIES = (
    "Inter", "Inter Tight", "DM Sans", "Manrope", "Plus Jakarta Sans", "Outfit", "Sora", "Space Grotesk",
    "Instrument Sans", "Geist", "Urbanist", "Poppins", "Montserrat",
    "Playfair Display", "Fraunces", "DM Serif Display", "Instrument Serif",
    "JetBrains Mono", "IBM Plex Mono", "Space Mono", "Geist Mono",
)

class FontToken(BaseModel):
    family: str                                    # validated against ALLOWED_FONT_FAMILIES
    fallback: Literal["sans-serif", "serif", "monospace"] = "sans-serif"

class DesignFonts(BaseModel):
    display: Optional[FontToken] = None   # big headlines, numerals
    heading: Optional[FontToken] = None
    body: Optional[FontToken] = None
    mono: Optional[FontToken] = None

class LinearGradient(BaseModel):
    angle_deg: int = Field(90, ge=0, le=360)
    stops: List[str] = Field(..., min_length=2, max_length=5)   # each HEX_COLOR_PATTERN

class TextSpan(BaseModel):
    ...existing fields...
    gradient: Optional[LinearGradient] = None       # gradient-filled text; `color` stays the solid fallback
    baseline: Literal["normal", "super", "sub"] = "normal"
    size_scale: float = Field(1.0, ge=0.3, le=1.0)  # e.g. 0.45 for the superscript "%"

class TextNode(BaseModel):
    ...existing fields...
    font_role: FontRole = "body"
    letter_spacing_em: float = Field(0.0, ge=-0.1, le=0.5)   # -0.04 tight display, 0.2 eyebrow
    text_transform: Literal["none", "uppercase"] = "none"
    font_size_px: int = Field(16, ge=10, le=200)              # was le=96

class NodeStyle(BaseModel):
    ...existing fields...
    background_gradient: Optional[LinearGradient] = None
    shadow: Literal["none", "soft", "glow"] = "none"
    shadow_color: Optional[str] = Field(None, pattern=HEX_COLOR_PATTERN)
    flex_grow: Optional[int] = Field(None, ge=1, le=100)      # proportional width inside a flex_row, e.g. 58/42

class DividerNode(BaseModel):
    ...existing fields...
    orientation: Literal["horizontal", "vertical"] = "horizontal"

class ContainerNode(BaseModel):
    ...existing fields...
    column_ratios: Optional[List[int]] = Field(None, min_length=2, max_length=6)  # grid only, e.g. [3, 2]
    wrap: bool = False            # flex_row: never wraps unless True; explicit "side by side stays side by side"

class ButtonNode(BaseModel):
    ...existing fields...
    font_role: FontRole = "body"
    trailing_icon: Optional[Literal["arrow_up_right", "arrow_right", "copy", "chevron_down"]] = None

IconName = Literal[...existing 10..., "copy", "arrow_up_right", "chevron_down"]

class DesignDocumentAST(BaseModel):
    version: Literal["1.0", "1.1"] = "1.0"
    fonts: Optional[DesignFonts] = None
    ...
```

Validators to add: `column_ratios` only with `layout == "grid"` and length == `columns`; font family in the allow-list
(unknown → schema error, repaired by the model); every gradient stop matches the hex pattern; a vertical divider
must be a direct child of a non-wrapping `flex_row`. Text contrast uses the **lowest contrast ratio** over the
solid fallback and all gradient stops against the effective background.

**Compatibility guard (the "nothing breaks" part):**

* Config flag `DESIGN_SCHEMA_V11_ENABLED` (default **on** since 2026-11-20). The JSON schema embedded in `DESIGN_SYSTEM_PROMPT` is
  generated from the model, so it would grow automatically — guard it: build the prompt schema from the 1.0 model
  when the flag is off, the 1.1 model when on. Both strings stay byte-stable (prompt cache).
* `downgrade_to_v10(ast) -> DesignDocumentAST` runs on the way out when the flag is off (or the request says
  `schema_version: "1.0"`): gradient → `color` = first stop; `baseline/size_scale/letter_spacing/text_transform`
  dropped (uppercase is applied to the text itself so the words still read right); vertical divider → removed
  (its container keeps a `gap_px`); `fonts`, `font_role`, `shadow`, `background_gradient`, `trailing_icon` dropped;
  `column_ratios` → `columns = len(ratios)`; `font_size_px > 96` → 96; `version = "1.0"`. A test feeds every 1.1
  fixture through it and asserts the result validates against the **1.0** model.
* Edit mode: an incoming 1.0 AST stays 1.0 unless the flag is on; an incoming 1.1 AST with the flag off is refused
  with a clear error instead of being silently stripped.

### W4 — Two-pass vision recreation (layout fidelity)

Behind `DESIGN_VISION_TWO_PASS` (default on since 2026-11-20). Only for `create_from_vision`.

**Pass A — describe** (small JSON, image attached). Pydantic model `ReferenceSpec`:

```python
class SpecText(BaseModel):
    id: str                              # t1, t2, ...
    text: str                            # verbatim, including case and punctuation
    role: Literal["eyebrow", "display", "heading", "subheading", "body", "label", "button", "caption", "numeral"]
    approx_size_px: int                  # at the supplied image scale
    color_hex: str
    gradient_hex: Optional[List[str]] = None
    font_class: Literal["geometric_sans", "grotesk_sans", "humanist_sans", "serif", "display_serif", "mono"]
    weight: Literal["light", "normal", "medium", "semibold", "bold", "extrabold"]
    uppercase: bool
    letter_spacing: Literal["tight", "normal", "wide"]

class SpecRegion(BaseModel):
    id: str
    parent_id: Optional[str] = None       # None for top-level regions
    arrangement: Literal["stack", "side_by_side", "grid"]
    ratio: Optional[List[int]] = None    # side_by_side widths, e.g. [55, 45]
    separator: Literal["none", "vertical_rule", "gap", "card_edges"] = "none"
    background_hex: Optional[str] = None
    has_border: bool = False
    corner_radius_px: int = 0
    text_ids: List[str]                   # direct SpecText ids in reading order

class ReferenceSpec(BaseModel):
    canvas_background_hex: str
    regions: List[SpecRegion]           # flat, siblings in reading order
    texts: List[SpecText]
    font_guess: Dict[str, str]          # role -> family from ALLOWED_FONT_FAMILIES
```

Each horizontal panel is a region with `parent_id` set to the horizontal region id. Reject duplicate ids,
missing parents, cycles and references to unknown text ids before using the spec.

**Pass B — build**: the existing harness, with the spec and W1 roles added to the prompt and the image still
attached. New vision task text:

```
TASK: Recreate the attached reference image as a DesignDocumentAST, as exactly as the schema allows.
Follow <reference_spec> for structure:
- For a "side_by_side" region with separator "gap" or "none", use layout "grid", columns = number of panels,
  and column_ratios = its ratio. Keep each panel in a separate grid cell.
- For a "side_by_side" region with separator "vertical_rule", use layout "flex_row", wrap false, and set
  style.flex_grow to the ratio on the content panels (e.g. 58 and 42). Place a vertical DividerNode between them.
- Keep "card_edges" as separately styled panels in a non-wrapping flex_row with the matching proportions.
- Use every text in <reference_spec> verbatim, once, in reading order. Do not add or drop text.
- Font: set `fonts` from font_guess; give every text node the matching font_role; uppercase → text_transform,
  wide tracking → letter_spacing_em 0.12–0.25, tight → -0.02 to -0.05.
- Sizes: keep relative sizes; font_size_px ≈ approx_size_px × (target width / supplied image width).
- Gradient text in the spec → span.gradient with those stops; numerals with a small raised symbol → baseline "super".
```

Why two passes and not one bigger prompt: the spec is a checkable intermediate. W5 can compare the AST against it
deterministically, and a wrong layout is caught *before* returning instead of being returned as `pass`. Cost: one
extra call (~3–8 s on the measured models); W0 decides whether it earns that.

### W5 — Deterministic fidelity checks (bounded repair)

New checks in `validate.py`, run only when a `ReferenceSpec` (W4) or an unmerged palette (W1) exists. They are **repair
hints, not hard failures**: at most one fidelity repair round (`DESIGN_MAX_FIDELITY_REPAIRS = 1`), after which the
best design is returned with the remaining findings in `report["fidelity"]` and in `complete.warnings`, so the
result is never claimed as exact when it isn't.

```python
def check_fidelity(ast, spec, quantized_hex_colors) -> List[str]:
    errors = []
    ast_texts = _normalised_texts(ast)                     # span texts joined per node, whitespace-folded
    for t in spec.texts if spec else ():
        if _norm(t.text) not in ast_texts:
            errors.append(f"FIDELITY_TEXT_MISSING: \"{t.text[:40]}\" from the reference is not in the design")
    for region in _side_by_side_regions(spec):
        node = _container_holding(ast, panel_text_ids(spec, region))
        if node is None or not _matches_horizontal_layout(node, region):
            errors.append(f"FIDELITY_LAYOUT_STACKED: keep separate panels in one row for region `{region.id}` "
                          f"and follow the {region.separator!r} separator rule")
    for node, color in _all_colours(ast):
        if quantized_hex_colors and _min_delta_e(color, quantized_hex_colors) > 12:
            errors.append(f"FIDELITY_COLOUR_OFF_PALETTE: `{node.id}` uses {color}, nearest reference colour is "
                          f"{_nearest(color, quantized_hex_colors)}")
    if spec and spec.font_guess and (ast.fonts is None):
        errors.append("FIDELITY_FONTS_MISSING: set `fonts` from <reference_spec> font_guess")
    return errors
```

Only structural findings start the repair call (`FIDELITY_REPAIR_TRIGGERS` in `service.py`): `LAYOUT_STACKED`,
`TEXT_MISSING`, `FONTS_MISSING`, `FONT_CLASS`. Tracking, gradient, raised-symbol and off-palette findings on their own
are returned as warnings; when a repair does run they are sent with it. Text matching ignores symbol glyphs (Unicode
`So`/`Sm`: arrows, check marks, `⧉`, `|`) on both sides, because a design draws them as icons or dividers. Live on
Claude, 2026-11-19: the first run repaired for tracking alone (53.5 s, 3 calls); a later run repaired because the
description wrote "Find my plan ↗" and the arrow was a `trailing_icon`. That one mismatch also made the correct
55/45 row read as stacked. After both changes, three runs took 32.7–33.0 s with 2 calls each, and each left
one tracking warning.

**Tracking and inline rules (2026-11-19).** Three causes. Buttons had no `letter_spacing_em`, so BLACKFRIDAY could
never be tracked; 1.1 buttons now take `letter_spacing_em` and `text_transform`. The prompt's tight range
(-0.02 to -0.05) overshot: matching each line's ink width/height against Chromium renders gave -0.03 for the 96 px
display line, -0.01 for the 46 px heading, and +0.10 to +0.20 for the mono labels. `tracking_range()` in
`reference.py` encodes that: tight -0.035 to -0.025 at ≥ 72 px, -0.025 to -0.015 at 48–71 px, -0.015 to -0.005
below; wide 0.08 to 0.25; normal -0.01 to 0.02. The prompt states those values. The spec could not describe the "|"
between inline items: `SpecText.rule_before` now does, and the build prompt turns it into a vertical divider.
`heal_reference_details()` runs after every passing build with no model call. It clamps out-of-range tracking on
nodes whose whole text is the reference text, and inserts a missing rule when the item already sits in a
non-wrapping row. `FIDELITY_SEPARATOR_MISSING` reports the rest. Two live Claude runs: 34.6 and 34.7 s, 2 calls,
clean first pass, rule present, tracking -0.03 / -0.01 / +0.08–0.10. The heal had nothing to change.

**Dash and headline weight (2026-11-19).** The "14–day" en dash in `promo_blackfriday.expect.json` was an error in
the expectation, not in generation. The reference draws "14-day" and "add-ons" with the same glyph, 10 px wide and
0.625 of the glyph height at 2512 px, and "add-ons" is always hyphenated. Corrected; the two latest live runs now
score text recall 1.0 with no findings. Headline weight is open. Measuring stroke against line height puts all three
large texts at Inter Tight ~550 (600 is +6%, 500 is −11%). A second, x-height-normalised method gives 495–595,
because one pixel of stroke is ~80 weight units at 1500 px. Pixel overlap with the reference headline ranks families:
DM Sans 550 0.634, Urbanist 0.620, Instrument Sans 0.582, Geist 0.580, Inter Tight 0.552–0.558 at any weight.
Family matters more than weight. Claude's text boxes from the describe call overlap the real glyphs by 0.97
and 0.95 on the headline lines and 0.80 on the heading. That is precise enough to measure from. Options: a server-side
typography matcher with the allow-listed font files bundled, or a weight-only measurement from offline per-family tables.

**Measured typography (2026-11-19, built).** `src/design_canvas/typography_match.py` + the 21 allow-listed families'
upright OFL files in `src/design_canvas/fonts/` (31 files, 6.5 MB, each with its `OFL.txt`). The describe call
now returns a `box` per display/heading/subheading/numeral text (first line). The harness measures those lines on the
original upload, capped at 2,600 px (`_design_measure_image`), and falls back to the model's image if their aspect
ratios differ. Measurement runs between the two calls (`asyncio.to_thread`).
For each candidate family: weight from ink area at the line's height (overlap barely moves with weight; area grows
steadily and ignores spacing), letter spacing solved from ink width, then mask IoU ranks families. Optical size is
applied as browsers do. Ink is colour distance from the box background, normalised by the strongest nearby ink, so
gradient lines measure evenly. Box-edge slivers of neighbouring lines are dropped. Headings take the headline family
when within 0.03 IoU. Texts with a raised suffix or best IoU < 0.45 are skipped with a reason.
Results feed `font_guess` and a `<measured_typography>` prompt block. The post-build step enforces them: font tokens,
`font_role`, exact `font_weight_value` (new 1.1 field on spans and buttons) plus nearest named weight, spacing.
`FIDELITY_TYPOGRAPHY` reports misses.
Known-answer tests recover family, weight within ±50 and spacing within ±0.012 for synthetic lines. That covers light
and dark backgrounds, static fonts, a gradient and a neighbouring-line sliver.
Promo: DM Sans 550 for headline and heading (-0.04 / -0.019 em). Inter Tight ranks 4th for the headline.
Live Claude, two runs: 35.6 / 35.8 s, 2 calls, measurement 0.8 / 1.7 s, clean first pass, no findings. Headline pixel
overlap with the reference, measured on Chromium renders: Inter Tight 0.494 / 0.406 before, DM Sans 0.518–0.541 /
0.525–0.534 after. The large-text tight range widened to -0.045..-0.025 (DM Sans measures -0.04). Remaining: word gaps
in the headline are slightly narrower than the reference (one spacing value per line also narrows spaces), and DM Sans
is the closest allowed family, not the reference's own typeface.

**Second reference: food-delivery listing (2026-11-19).** Prompt "replicate this design 100%" with a 1298×1096
screenshot. It routes to `create_from_vision`, Claude first. Run outputs, renders, `PROMPTS.md` and a hand transcription
(`expected_texts.json`) are in `local_run/design_fidelity/live_claude_20261119_burger/`. Defects found and fixed:
1. The describe call sent the filter chips as `side_by_side` with seven `text_ids` and no panels. The strict
   validator rejected the whole description, so the build ran from the image alone with no checks and no
   measurement. `ReferenceSpec._normalise_hints` now turns such a region into the new `row` arrangement (≥ 2 texts) or
   a `stack`, and drops a ratio that doesn't match the panels. Rows and card grids (`columns`) are now checked.
2. Typography: "Burger", described as tight, measured as Inter 800 at +0.056 em. Spacing stood in for a wider
   typeface and the letters spread visibly. Solved spacing is now bounded by the described tracking class
   ± 0.01 em. The size for that range and for optical size now comes from the box ink height against the candidate
   font, not the model's `approx_size_px`. Result, stable across runs: title Geist 650 +0.005, heading Outfit 675
   -0.02. The promo is unchanged (DM Sans 550).
Independent score, two final runs: 46/46 texts, 9/9 cards intact, 5/5 "Ad" labels, 4-column × 9 grid.
Today's default scores the same on text and structure, but merges cuisine and area lines and uses the system font.
Time: new pipeline 87.5 / 89.5 s (describe 35 s, measure 0.4 s, build 52–54 s) vs 41.7 s for the default's one call.
The build used 7,814–8,177 output tokens of a 12,288 cap: a denser page needs `DESIGN_VISION_MAX_OUTPUT_TOKENS`
raised. Not reproducible in this format: the restaurant photos, which have no image node and are drawn as colour
blocks with ovals. Also not reproduced: the circled rating star and the filter icon. The orange "NEW" badge colour
warning is a false alarm: the badge is too small for the 64-colour palette.

**Same layout, new content (2026-11-19).** "Like this, but for pizzas" was checked as a copy. Every replaced
burger name was a `FIDELITY_TEXT_MISSING`, and the grid, located by its text, read as stacked. That cost a 48 s repair
asking to restore "Burger" and returned 24 false warnings.

`design_intent.wants_new_content()` now sets `report["reference_use"]` to `adapt`, triggered by "instead of", "but
for", "like this … for", "inspired by", "same layout but", "this design for our…", "replace the dishes". Adapt mode:
* The build prompt treats the screenshot as a layout and style template, with new text per element role, photo
  prompts for the new content, and measured styles moved to whatever replaces each text.
* `check_fidelity(content="new")` checks only structure (grid column count, side-by-side panel count, photo-slot
  count), fonts and colours.

Routing: Rule 2b accepts "Design a page like this for…" (screenshot + design verb + template cue + page noun), which
was answered as chat. Description placement slips (unknown or duplicate text ids, unplaced texts) are normalised in
code instead of costing a repair.

Live, "Design a page like this for pizzas and Asian fusion dishes": no fidelity repair, no false warnings.
* Run A: 125 s, with one description repair for an unknown text id; that case is now normalised.
* Run B: 48.8 s, 2 calls, but its description stopped after the header, so the checks were thin. The describe
  prompt now asks for the whole page; a completeness guard is still open.

"add real photos" took 6–7 s with no Claude call in every run.

**Completeness guard (2026-11-19, built).** `reference.description_coverage()` compares the lowest box the description
gives (regions now carry `box` too) with the last content row of the screenshot. The undescribed span counts as
short when it is over 15% of the page height with region boxes, or over 50% when only text and photo boxes exist.
That second case is a stricter bar, because large texts are boxed and small ones are not.

A short description gets one request for the whole page (`build_reference_coverage_prompt`). This shares the
describe call's single repair with invalid JSON. If the description is still short, or the retry breaks, it is used
anyway: `reference_analysis: partial`, plus a `FIDELITY_ANALYSIS_PARTIAL` warning naming the y-limit of what was
checked. `reference_coverage` and `reference_use` are now traced.

On recorded descriptions: header-only 0.83 (flagged); complete burger ones 0.04; the complete promo ones 0.32 on the
text basis, which passes but would be a false alarm at 15%. Two live runs had all 13 of 13 regions boxed and 0.0
undescribed. They took 87.8 / 89.7 s with 2 calls; photos took 5.3 / 6.1 s.

Match panel text groups to separate direct children of the same container, in reading order. With a vertical rule,
check for a direct vertical divider between those panels. Use the unmerged palette even when a shade has no prompt
role. Compare the repaired AST with the previous valid AST and keep the better match.

Score recorded per run (`report["fidelity"]`): text recall, layout matches / regions, accent coverage, fonts set.
These are the same numbers W0 prints, so the harness and production measure the same thing.

### W6 — Vision budget

* `create_from_vision` can test OpenAI effort `medium` with a 12,000 token output cap. Claude keeps its configured effort.
* Compare fidelity and latency using W0. The current default remains `low` and 8,192 pending the live comparison.
  See `docs/MODEL_UPGRADE_BENCHMARK.md` §4.5.1 for the local settings and measured extractor results.

### W7 — Frontend renderer (separate repo) + handoff

Add a §4.8 "Schema 1.1" section to `docs/FRONTEND_HANDOFF.md`. Renderer rules:

```ts
// 1. Isolate the canvas from the editor font (today's DM Sans leak).
canvasRoot.style.fontFamily = cssFamily(ast.fonts?.body) ?? 'system-ui, sans-serif';

// 2. Load only allow-listed fonts, once per family.
function ensureFont(token?: FontToken) {
  if (!token || !ALLOWED_FONT_FAMILIES.has(token.family)) return;
  loadGoogleFont(token.family, [300, 400, 500, 600, 700, 800]);  // <link> to fonts.googleapis.com
}

// 3. Text node
el.style.fontFamily   = cssFamily(ast.fonts?.[node.font_role ?? 'body']);
el.style.letterSpacing = `${node.letter_spacing_em ?? 0}em`;
el.style.textTransform = node.text_transform === 'uppercase' ? 'uppercase' : 'none';

// 4. Gradient span (solid `color` stays as the fallback)
if (span.gradient) {
  s.style.backgroundImage = `linear-gradient(${span.gradient.angle_deg}deg, ${span.gradient.stops.join(',')})`;
  s.style.backgroundClip = 'text'; s.style.webkitBackgroundClip = 'text';
  s.style.color = 'transparent';
}
if (span.baseline === 'super') { s.style.verticalAlign = 'super'; s.style.fontSize = `${span.size_scale}em`; }

// 5. Layout: side by side must stay side by side
if (node.layout === 'grid' && node.column_ratios)
  el.style.gridTemplateColumns = node.column_ratios.map(r => `minmax(0, ${r}fr)`).join(' ');
if (node.layout === 'flex_row') el.style.flexWrap = node.wrap ? 'wrap' : 'nowrap';
if (child.style?.flex_grow) {
  childEl.style.flex = `${child.style.flex_grow} 1 0`;
  childEl.style.minWidth = '0';
}

// 6. Vertical divider
if (divider.orientation === 'vertical') {
  d.style.borderLeft = `${t}px solid ${c}`;
  d.style.alignSelf = 'stretch';
  d.style.flexShrink = '0';
}
```

Plus: shadow tokens (`soft` → `0 8px 24px rgb(0 0 0 / .35)`, `glow` → `0 0 24px {shadow_color}66`), background
gradients, the three new icons, and **scale-to-fit**: keep the design at its natural width and use the editor's
canvas zoom or viewport mechanism in a narrower column. A raw CSS transform can put the caret, drag handles and
hit tests in the wrong coordinate space. Verify all three at reduced zoom, along with the two panels and divider
occupying the same row. Renderers must ignore unknown fields.

## 3. Order of work and gates

Each step ends with evidence. Steps 6–7 run alongside Step 5 using the Python reference renderer and Playwright.
Frontend confirmation gates Step 8 only.

| Step | Work | Evidence required before moving on |
|---|---|---|
| 1 | W0 harness + fixtures + **baseline run on today's code** | `local_run/design_fidelity/run_<ts>.json` with per-fixture scores; side-by-side renders saved |
| 2 | W1 extractor + prompt wording | New unit tests pass; full `tests/design` suite passes unchanged; eval: accent coverage up vs baseline, no fixture regresses on text recall |
| 3 | W2 image on repairs + palette heal | Test asserts image present on every attempt; contrast-heal test keeps a lavender span lavender when a palette colour reaches 3:1 |
| 4 | W3 schema 1.1 + downgrade, flag **off** | All existing 1.0 tests pass byte-for-byte; prompt with flag off is byte-identical to today's (`test_design_prompts`); every 1.1 fixture downgrades to a valid 1.0 AST |
| 5 | W7 frontend renderer ships (other repo) | Their acceptance checklist §4.7 + new 1.1 items; **cannot be verified from this repo — report it as pending until the frontend confirms** |
| 6 | W4 two-pass + W5 checks, flags on in local only | Eval: layout match = 100 % on the promo fixture across 5 runs (no stacked `50%`), text recall ≥ 0.98, accents within ΔE 8, fonts set; latency median recorded |
| 7 | W6 budget tuning | Chosen effort/cap with measured fidelity vs latency table |
| 8 | Turn flags on in production (`DESIGN_SCHEMA_V11_ENABLED`, `DESIGN_VISION_TWO_PASS`) | Only after step 5 is confirmed by the frontend |

Regression guards that run at every step: `pytest tests/design` (all of it), the existing live script
`scripts/local_design_requests.py` (prompt designs + 4 edits must still pass 34/34 as on 2026-10-27), and the prompt
cache test (`tests/core/test_prompt_cache_usage.py`) since the system prompt changes.

## 4. What this will and will not achieve

* **Will:** side-by-side layouts stay side by side; text appears verbatim; colours come from the image, including
  small accents; gradient headlines, tracked uppercase eyebrows, monospaced labels and superscript symbols are
  expressible and checked; fonts are set explicitly instead of inheriting the editor's.
* **Will not:** pixel-identical output. The exact typeface in a screenshot (here a tight display grotesk) can't be
  identified with certainty and may be a commercial font; the design gets the **closest allow-listed family** with
  matching weight and tracking, and the warnings say so. Soft background glows/noise textures are approximated with
  a gradient or omitted. Photos and logos inside a reference are out of scope for this AST (no image node) — they
  would need a W8 "image node" with uploaded assets, not planned here.

## 5. Files touched (for the implementer)

| File | Change |
|---|---|
| `src/design_canvas/vision_extractor.py` | Role-based palette (W1) |
| `src/design_canvas/prompts.py` | Vision task text, palette roles block, spec block, version-gated schema string (W1, W3, W4) |
| `src/design_canvas/schema.py` | 1.1 fields, allow-list, validators, `ReferenceSpec` (W3, W4) |
| `src/design_canvas/validate.py` | Palette-aware `_fix_contrast`, gradient contrast, `check_fidelity`, `downgrade_to_v10` (W2, W3, W5) |
| `src/design_canvas/service.py` | Image on every attempt, Pass A call, fidelity repair round, `report["fidelity"]` (W2, W4, W5) |
| `src/text_processing_agent.py` | Vision-only effort / output cap, pass `complete.warnings` for fidelity findings (W5, W6) |
| `src/models.py`, `config.py` | `OPENAI_DESIGN_VISION_EFFORT`, `DESIGN_SCHEMA_V11_ENABLED`, `DESIGN_VISION_TWO_PASS`, `DESIGN_MAX_FIDELITY_REPAIRS` |
| `tests/design/*` | New tests per step; fixtures under `tests/design/fixtures/reference/` |
| `scripts/design_fidelity_eval.py` | New live eval (W0) |
| `docs/FRONTEND_HANDOFF.md` | §4.8 Schema 1.1 rendering rules (W7) |
