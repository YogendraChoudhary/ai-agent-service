"""
Labelled corpus for the image intent router.

Each case: (group, query, has_reference, has_previous_image, expected_kind)

The corpus is hand-written, so passing it does not prove real-world precision. It exists so
that every misroute found in review stays fixed: the "text_payload" and "text_followup"
groups are queries that once generated an image instead of editing text. When a new
misroute is reported, add it here first.
"""

CASES = []


def _add(group, expected, rows, has_ref=False, has_prev=False):
    for q in rows:
        CASES.append((group, q, has_ref, has_prev, expected))


# --- should generate -------------------------------------------------------------------
_add("generate", "generate", [
    "draw a mosque at dusk",
    "create an image of a futuristic neon city",
    "generate a photo of a golden retriever in snow",
    "design a modern minimalist logo for a coffee shop",
    "render a 3d isometric illustration of a computer room",
    "illustrate a children's storybook dragon",
    "paint a watercolor landscape of mountains",
    "sketch a portrait of an elderly man with wisdom",
    "produce an epic wallpaper of galaxy stars",
    "generate 4k high quality photo of red sports car",
    "draw a funny cartoon cat eating pizza",
    "create a banner for twitter account",
    "make an icon for a mobile music player",
    "paint a vibrant oil painting of sunflowers",
    "generate a poster for jazz festival",
    "sketch a cyberpunk robotic arm",
    "design a flat vector graphic of rocket launch",
    "render an architectural exterior of a glass villa",
    "create a photorealistic avatar of female astronaut",
    "draw a cute baby panda holding bamboo",
    "please generate an image of a red barn",
    "can you create a logo for my bakery?",
    "I want you to draw a dragon",
    "image of a cat on a skateboard",
    "photo of a sunset over the ocean in 16:9",
    "Hey, make a poster for a book fair",
])

# --- text work that merely mentions images: must NOT generate --------------------------
_add("text_request", "none", [
    "fix grammar in this sentence",
    "make this text shorter",
    "summarize the meeting notes",
    "translate this paragraph to Spanish",
    "create a markdown table comparing iPhone models",
    "write an article with 2 images about AI development",
    "write a blog post and generate an image for it",
    "write a function that draws a circle",
    "how do I create a logo in Figma?",
    "what is the best way to design a poster?",
    "draw conclusions from the quarterly data",
    "make a list of icons for the app",
    "create a table of photos with captions",
    "make a summary of the image analysis",
    "generate ideas for logo designs",
    "create a poster design brief",
    "write a prompt for an image generator",
    "give me ideas for a logo",
    "Expand on the graphics discussion in my essay",
    "convert this list of icons to a table",
])

# --- user text wrapped by a text operation (api.build_full_query prefixes) -------------
_add("text_payload", "none", [
    "Fix grammar: Please create a logo for our new brand",
    "Simplify: We need to design a banner and generate images for the launch",
    "Enhance: make it punchier, Text: Our poster design improves the photo quality",
    "Make shorter: The infographic shows how to create a poster",
    "Complete: To draw a picture of the mountain, first",
    "Translate to French: Please generate an image of a red barn",
])

# --- vision Q&A / captions / OCR --------------------------------------------------------
_add("vision_text_no_ref", "none", [
    "write a caption for this photo",
    "generate an alt text for this image",
    "extract text from this screenshot",
    "ocr this receipt picture",
    "summarize the chart in this image",
    "describe what is in this picture",
    "what does this image show?",
    "explain this diagram",
    "analyze this medical scan photo",
])
_add("vision_text_with_ref", "none", [
    "who is the person in this photo?",
    "what is this landmark?",
    "where was this photo taken?",
    "can you describe what is happening in this image?",
    "summarize this screenshot",
    "write a caption for this",
    "extract the text from this image",
    "translate the text in this image to French",
    "make a list of the objects in this picture",
    "give me a title for this photo",
    "write a poem about this picture",
    "is this a good logo?",
], has_ref=True)

# --- edits of an attached image ---------------------------------------------------------
_add("edit_with_ref", "edit", [
    "refine the image with more indepth and taj-mahal vibe",
    "add a Taj Mahal vibe",
    "refine this with golden lighting",
    "make it look like an oil painting",
    "change the style to sketch",
    "improve the background details",
    "remove background from this photo",
    "turn this image into anime style",
    "restyle this picture into watercolor",
    "recolor the car in the image to matte black",
    "upscale this photo to 4k",
    "cutout the background of this product image",
    "touch up the portrait face in this photo",
    "make the sky more dramatic",
    "improve the image",
], has_ref=True)

# --- edits that name an image but none is available -> defer, never generate -----------
_add("edit_no_image_available", "none", [
    "refine the image with more indepth and taj-mahal vibe",
    "enhance the photo lighting and contrast",
    "remove background from this photo",
])

# --- follow-ups with a previous image in memory ----------------------------------------
_add("redo_with_prev", "generate", [
    "regenerate", "redo", "again", "try again", "another version", "give me another one", "same again",
], has_prev=True)
_add("delta_with_prev", "edit", [
    "make it night",
    "make it darker and rainier",
    "same but in autumn colors",
    "change the mood to dramatic sunset",
    "more contrast and vibrant lighting",
    "less saturated colors",
    "make it a wide 16:9 banner",
    "give it a vintage look",
    "refine the image with a taj mahal vibe",
], has_prev=True)

# --- the same phrases with NO previous image must not generate -------------------------
_add("followup_no_prev", "none", [
    "regenerate", "redo", "try again", "make it night", "same but in autumn colors",
    "change the mood to dramatic sunset", "make it darker and rainier",
])

# --- ordinary text edits, even when an image exists in memory --------------------------
_add("text_followup", "none", [
    "make it more formal",
    "change the tone to friendly",
    "more concise please",
    "give it a catchy title",
    "make it shorter",
    "make it longer and add examples",
    "change the style to formal",
    "change the tone of the email",
], has_prev=True)
_add("text_followup_no_prev", "none", [
    "make it more formal",
    "change the tone to friendly",
    "more concise please",
    "give it a catchy title",
])
