"""
Visually rich blog generation (Mode 2 of docs/rich_content_engine_plan.md).

  request -> plan (code) -> outline (LLM, JSON, validated) -> sections (LLM, parallel, ordered) -> lint -> Markdown

Code owns structure: how long the post is, which tables / quotes / images it must contain, the headings, the
image markup. The model owns prose. Everything the model writes passes a validator before it reaches the editor.

  plan      BlogPlan: lengths and visual quotas decided from the user's words and options
  schema    Outline models (pydantic, extra="forbid") and the limits the prompts are generated from
  outline   outline call -> validate -> one repair -> rule-based fill
  media     image directives -> placeholders; generation starts right after the outline validates
  draft     one section: model body + code-written heading + code-inserted image block
  validate  lint of a drafted section (assigned table / quote, density, length, stray markup)
  service   the pipeline as a generator of editor events and Markdown (no app imports; takes an injectable LLM)
  runtime   binds the pipeline to this app: provider candidates, keys, image tool, credits
"""
