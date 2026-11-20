# Design canvas — real photos in image placeholders (plan)

Status: **implemented 2026-11-19 except I5** (image-to-image from the screenshot's crops: not built, a rights
decision). Code: `src/design_canvas/images.py`, `ImageNode` in `schema_v11.py`, `SpecImage` and the photo heal in
`reference.py`, Rule F in `design_intent.py`, the fill branch in `text_processing_agent.py`, pricing in
`credit_manager.design_credits_scaled` + `api.py`. Written 2026-11-19 after the burger-listing live run
(`local_run/design_fidelity/live_claude_20261119_burger/`). Builds on schema 1.1 (`docs/design_reference_fidelity_plan.md`).

## Live result (2026-11-19, production `.env`, Claude)

Burger screenshot, run as "replicate this design 100%" then "add real photos". Outputs are in
`local_run/design_fidelity/live_claude_20261119_burger/photo_fill/` (`SUMMARY.md`, `run.json`, renders).

* **Create:** 9 photo slots with the photos' exact shapes (`140:93`, `280:187`, taken from the boxes) and prompts
  with no brand names.
* **Fill:** placeholders arrived in 0.01 s with no Claude call. All 9 photos were done 6.5 s in (each 5.4–6.5 s,
  `gemini-3.1-flash-lite-image`, 1200x896, generated at 4:3 and cropped to the slot).
* **Independent score of the final design:** 46/46 texts, 5/5 "Ad" labels, 4x9 grid, 9/9 photos.

Fixed along the way, each found live:
1. The model wrote invalid JSON while choosing an aspect from the list. Shapes now come from the box, and the
   describe call gets one repair turn (`build_reference_repair_prompt`, `reference_analysis: pass_after_repair`).
2. Brand names appeared in photo prompts. The build rules now forbid them too.
3. The description missed the "Ad"/"NEW" labels and the build dropped them. The image is now the source of truth for
   text, and the describe call is told to list even tiny labels.
4. A 9 px badge cost a 44 s schema repair. Text and button sizes now clamp in schema 1.1.
5. `TASK_ROUTE_DEFAULTS` rejected `design_canvas`, so the design default could not be pinned. The validator now accepts it.

Test-run isolation: no credits moved, and the backend quota and file-tracking calls were stubbed. 27 test images
(3 runs × 9) were on the CDN under `design-fill-test/ai-images/`. All 72 test images from the session were permanently deleted on 2026-11-19, all versions; the CDN URLs now return 403.

## The behaviour we want

1. **Create from a screenshot:** every photo in the reference becomes an **image placeholder** of the right size and
   shape, carrying a description of what the photo shows. No images are generated yet, so creating costs nothing extra.
2. **The user then asks for real photos** with the design selected ("add real photos", "fill the images", "use real
   pictures here"). The system skips the layout model and sends each empty placeholder through the image
   generation flow we already have. Each photo appears in its own placeholder as soon as it is ready.
3. **Optional, off by default:** "use the photos from my screenshot" regenerates each photo from its crop of the
   reference (image-to-image). It looks closest to the original, but it reproduces someone else's photo, so it is
   only for images the user owns (see I5).

## What the code does today

| Area | Today | File |
|---|---|---|
| Photos in a recreated design | No image node exists, so photos are drawn as `vector_illustration` colour blocks with ovals (burger run: 9 of 9) | `src/design_canvas/schema.py`, `schema_v11.py` |
| "add real photos" with a design selected | Rule 1b routes it to `edit_existing_ast`. The layout model can only edit the AST, so no photo is generated: the request silently does nothing useful | `src/design_intent.py` |
| Image generation | `_build_image_tool` reserves credits, picks the provider and model (`plan_image_target`), and generates in a background thread through `generate_image_unified`. It uploads to the CDN and records the result in `pending_images` | `src/text_processing_agent.py:1059`, `src/image_gen.py:980` |
| Delivering images to the editor | After the main stream, the API polls `pending_images` and emits `image_ready {placeholder_id, url, width, height, format}` or `image_failed` per image. This works for any route that yields `__pending_images__` | `api.py` ~1546, ~1630–1660 |
| Price | 5 credits per image (`IMAGE_CREDITS_SCALED = 50`), per-model overrides via `IMAGE_CREDITS_BY_MODEL`. Gemini default `gemini-3.1-flash-lite-image`: 3.5–4.7 s, $0.0336 per image (`docs/image_models.md`) | `src/credit_manager.py` |

## Work items

### I1 — `ImageNode` in schema 1.1

```python
class ImageNode(BaseModel):
    node_type: Literal["image"] = "image"
    id: str = Field(..., pattern=NODE_ID_PATTERN)
    alt: str = Field(..., min_length=1, max_length=300)
    prompt: str = Field(..., min_length=3, max_length=600)     # what the photo shows: the generation prompt
    src: Optional[str] = None                                    # set once generated; our CDN host only
    placeholder_id: Optional[str] = None                         # set while generating; matches image_ready
    aspect_ratio: Literal["1:1", "4:3", "3:4", "16:9", "9:16"] = "4:3"   # the image flow's own aspects
    width_percent: int = Field(100, ge=10, le=100)
    object_fit: Literal["cover", "contain"] = "cover"
    source_box: Optional[list[int]] = None                       # where the photo sits in the reference (I5)
    style: NodeStyle = Field(default_factory=NodeStyle)          # border_radius etc.
    editable: bool = True

    @field_validator("src")
    @classmethod
    def _our_cdn_only(cls, value):
        # Never an arbitrary URL from the model: https on settings.cdn_url's host, else rejected.
        ...
```

* Add it to the 1.1 `DesignNode` union. `downgrade_to_v10` turns it into a `vector_illustration` with one rect in the
  photo's dominant colour, so a 1.0 editor shows a placeholder box, not an error.
* The aspect is the nearest of the five the image flow supports. The node's CSS `aspect-ratio` uses the same value,
  so a generated photo always fits its slot without letterboxing.

### I2 — Recreation emits placeholders, not drawings

* Describe call (`REFERENCE_SYSTEM_PROMPT`): add `images: [{id, box, description, aspect}]`, one per photo, with a
  plain description of the subject ("crispy chicken burger with lettuce on a red background, studio product shot").
  Text, logos and brand marks stay out of the description.
* Build prompt (1.1): "Every entry in reference_spec.images becomes an ImageNode with that description as its prompt,
  in the same panel and order. Never draw a photo as a vector_illustration." The server copies `source_box` from the
  spec onto the node.
* Check (`check_fidelity`): `FIDELITY_IMAGE_MISSING` when a spec photo has no ImageNode in its panel. The
  post-build step can swap a `vector_illustration` sitting where a spec photo belongs for an ImageNode, with no
  model call.

### I3 — Route "add real photos" to a fill, not an edit

New sub-mode `fill_design_images` in `detect_design_intent`, checked before Rule 0 and Rule 1b when
`existing_design_ast` is present and the query is not a question:

```python
_PHOTO_FILL_RE = re.compile(
    r"\b(?:add|use|put|insert|fill|generate|create|replace|swap|show)\b.{0,40}?"
    r"\b(?:real\s+|actual\s+|proper\s+|stock\s+)?(?:photos?|images?|pictures?|pics|photographs?)\b",
    re.IGNORECASE)
```

* Placeholders without `src` exist: fill them. "replace / regenerate / new photos": fill all image nodes again.
* No ImageNodes yet (a design created before I2): one `edit_existing_ast` call with "turn the photo areas into
  ImageNodes", then the fill, in the same request.
* A request naming a part ("only the KFC card", "the first photo"): fill the nodes whose card text matches, or the
  node the editor sends as selected (`selected_node_id`, a new optional request field).

### I4 — The fill itself (no layout model call)

New `src/design_canvas/images.py`:

```python
def fill_targets(ast: DesignDocumentASTV11, replace: bool, only_ids: set[str] | None) -> list[ImageNode]:
    """Empty image nodes (or all, when replacing), in reading order, capped at settings.design_max_fill_images."""

def photo_prompt(node: ImageNode) -> str:
    """node.prompt + 'photograph, no text, no logos, no watermark'. The model never picks the image model:
    selection_text="" as in the blog tool."""
```

In the design route (`text_processing_agent.py`, Mode 8 block), for `fill_design_images`:

```python
tool = _build_image_tool(provider, key, pending_images, request_id, user_id, credit_manager, is_byok,
                         image_ctx, ToolBudget(len(targets), "image"))
for node in targets:
    placeholder = tool(photo_prompt(node), aspect_ratio=node.aspect_ratio)   # new optional per-call aspect
    node.placeholder_id = placeholder_id_of(placeholder)                     # "ERROR:"/"skipped" -> warning
yield {"__event__": {"event": "design_ready", "data": ast.model_dump()}}     # placeholders show at once
yield {"__pending_images__": pending_images}                                 # api.py streams image_ready
```

* Credits: each image is reserved and charged by the existing tool code. The fill makes no design-model call, so the
  design price reservation in `api.py` is skipped for this sub-mode. When credits run out mid-list, the images
  already reserved still generate, the rest stay placeholders, and `complete.warnings` says how many.
* Images generate in parallel (one background thread each, as for the blog). Nine Gemini images should take
  about 5–10 s in total, if the provider does not rate-limit nine calls at once. That needs measuring; a small
  concurrency limit is the fallback.
* Abort: the existing image-wait loop in `api.py` already stops on `/agent/abort`.

### I5 — Optional: regenerate from the screenshot's own photos

Behind `DESIGN_FILL_FROM_REFERENCE_ENABLED` (default **off**), and only when the user asks for "the photos from my
screenshot". The reference upload's CDN URL is kept in the AST (`reference_image_url`). Each node's `source_box` crop
goes to `generate_image_unified(reference_image_bytes=crop, prompt=...)` as an image-to-image edit. This gives the
closest match, but it reproduces the source photo, so it is only appropriate for images the user has the rights to.
That is a product and legal decision, not a technical one.

### I6 — Renderer and frontend contract

* Reference renderer (`scripts/design_reference_render.py`) and `docs/FRONTEND_HANDOFF.md` §4.8:
  * an `<img>` with `object-fit` and CSS `aspect-ratio`;
  * a neutral placeholder showing the `alt` text while `src` is empty, shimmering while `placeholder_id` is set;
  * never render a `src` outside the CDN host.
* On `image_ready`, the editor sets `src = url` on the node whose `placeholder_id` matches and clears
  `placeholder_id`. On `image_failed`, it clears `placeholder_id` and keeps the placeholder. It saves the AST either way.

### I7 — Proof

* Unit tests for each of these:
  * the ImageNode schema, its CDN-only `src` and its downgrade;
  * the fill intent: positives, questions, no design selected, "replace", one named card;
  * `fill_targets`, including the cap;
  * the aspect mapping and `photo_prompt`;
  * running out of credits partway;
  * renderer output.
* Harness test: a vision creation with spec images yields ImageNodes, not vector illustrations.
* API stream test: `design_ready` with placeholder ids, then one `image_ready` per placeholder.
* Live run on the burger screenshot:
  1. Create the design: 9 placeholders, no image cost.
  2. Ask "add real photos": 9 images.
  3. Report the per-image and total time, the credits, the provider and model, the exact image prompts, and the
     render beside the reference.

## Settings

| Setting | Default | Purpose |
|---|---|---|
| `DESIGN_IMAGE_FILL_ENABLED` | on (since 2026-11-20) | The `fill_design_images` route. Needs schema 1.1 and the editor change in I6 |
| `DESIGN_MAX_FILL_IMAGES` | 12 | Most images one fill request generates |
| `DESIGN_FILL_FROM_REFERENCE_ENABLED` | off | I5 image-to-image from the screenshot's crops |
