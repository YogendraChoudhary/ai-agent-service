"""
Building blocks shared by the two long-form modes (docs/rich_content_engine_plan.md).

  markdown_ast    top-level Markdown blocks with exact line ranges (the parser both modes lint with)
  ordered_stream  run units in parallel, hand their results back in document order

Mode 1 (structure-preserving refinement) adds freeze / chunk / verify / refine here later; Mode 2 (src/blog)
already uses the two modules above.
"""
