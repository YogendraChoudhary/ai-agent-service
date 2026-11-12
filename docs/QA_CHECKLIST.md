# Eddyter AI: Test Checklist

For anyone testing the new AI features in the Eddyter editor. No technical knowledge is needed: each row says what to
do, what to type, what you should see, and roughly how long it should take.

**How to use it**

- Tick the Status box when a row behaves as described. If it doesn't, note what you saw, what you typed and the time.
- Times are what we measured during development. A little slower is fine; much slower (two or three times) is worth
  reporting.
- "Instant" means under a second, with no credits used.
- Button and tab names may differ slightly in your editor. Use the closest one.
- Where a row says "credits", check your credit balance before and after. Plain text costs about **1 credit per 1,000
  characters** (minimum 1).

## Sample document

Several tests use this document. Create it in a new page (headings as headings, the table as a real table):

> **Launch guide** *(main heading)*
>
> **Introduction** *(heading)*
> This introduction is very very useful for readers who want to understand the launch and what it means for the team.
> A second intro paragraph about goals and timelines.
>
> **Benefits** *(heading)*
> The benefits are clear and clear for every team that adopts the new workflow.
> *(a quote under it, formatted as a quote:)* Customers say the new workflow saves them time every week.
>
> **Pricing** *(heading)*
> The price is USD 20 per seat per month in 2026.
> *(table with two columns, Plan and USD: Team 20, Business 35)*
>
> **Discounts** *(smaller heading under Pricing)*
> Students get 50% off.
>
> **Conclusion** *(heading)*
> We recommend launching in May with a small cohort.

---

## Part 1: Writing something new (Write tab, no document needed)

| Status | Feature | What to do and type | What you should see | Time |
|---|---|---|---|---|
| [ ] | Clean writing | Type: *Write a 3-item checklist for onboarding a new engineer* | Just the checklist. No opening line like "Sure! Here's…", no closing line like "Feel free to adjust…" or "Let me know if…", and no questions back to you | 1–3 s |
| [ ] | Blanks instead of questions | Type: *Write a product description for our app* | A full description with blanks in square brackets such as **[App Name]**. The AI does not ask you for details | 3–5 s |
| [ ] | Asking about a document it can't see (rewrite) | Type: *Rewrite this* | No text is written. A message says: "This refers to your document, which wasn't included. Use Edit to change existing text." No credits used | Instant |
| [ ] | Same, adding to a document | Type: *Add 3 more items to the list*, then *continue my draft* | Message ends with "Use Draft to add new content to it." Nothing written | Instant |
| [ ] | Same, asking about a document | Type: *Summarize the meeting notes into action items* | Message ends with "Use Ask to ask about it." Nothing written | Instant |
| [ ] | Text included in the request still works | Type *Rewrite this:*, press Shift+Enter for a new line, then type *The dog ran fast and it was very very quick and fast.* | A cleaner version of the sentence | ~2 s |
| [ ] | Email requests are kept | Type: *Write an email asking the team for their availability next week* | A complete email. Its own question to the team ("Could you share your availability?") stays in | 1–2 s |
| [ ] | Chart in Write | Type: *Make a bar chart of quarterly revenue: Q1 10, Q2 14, Q3 9, Q4 17* | A bar chart (not a table) | ~2 s |
| [ ] | Code | Type: *Write a Python function that checks if an email address is valid* | Code shown in a code block. It may refresh once with a corrected version | 5–12 s |
| [ ] | Plain question | Type: *What is the capital of Australia?* | A short, direct answer (Canberra) | ~1 s |

---

## Part 2: Blog posts (Write tab)

| Status | Feature | What to do and type | What you should see | Time |
|---|---|---|---|---|
| [ ] | Full blog post | Type: *Write a blog post about remote work for small teams* | A title and subtitle first, then section headings, then the sections. The post includes at least one table and a couple of highlighted quotes. Pictures appear a little after the text finishes | Title in ~2 s, text done in ~30–40 s, pictures a few seconds later |
| [ ] | Requested length | Type: *Write a blog post about healthy sleep habits, about 2,000 words* | A noticeably longer post, about 2,000 words, with two tables | Longer than the standard post (not measured) |
| [ ] | No pictures | Type: *Write a blog post about time management, text-only with no images* | A blog post with no pictures | ~30 s |
| [ ] | Not everything is a blog | Type: *Write a short essay about cats* | A normal short essay, without the blog layout | ~3 s |

---

## Part 3: Asking about your document (Ask mode)

Open the sample document and use Ask mode.

| Status | Feature | What to do and type | What you should see | Time |
|---|---|---|---|---|
| [ ] | Question about the document | Ask: *What does Pricing cost?* | The answer uses the document: USD 20 per seat per month | 2–4 s |
| [ ] | Empty page | Open an empty page and ask: *What is this about?* | Message: "There is no document content to answer questions about. Add some text first." | Instant |
| [ ] | Mode tip | Ask: *Insert a quote after the Benefits paragraph* | You get an answer, and the editor may suggest switching to Draft | ~3 s |

---

## Part 4: Adding new text at a place you name (Draft mode)

Open the sample document and use Draft mode. Check **where** the new text lands.

| Status | Feature | What to do and type | What you should see | Time |
|---|---|---|---|---|
| [ ] | Below a heading | *Draft a one-sentence summary below the Pricing heading* | New sentence right under the "Pricing" heading. The heading is not repeated | ~1.5 s |
| [ ] | End of a section | *Add a note about annual billing at the end of the Pricing section* | New text after "Students get 50% off." (the Discounts part belongs to Pricing) | ~1.5 s |
| [ ] | After a section | *Add a recommendation after the Conclusion* | New text after "We recommend launching in May…" | ~3 s |
| [ ] | Above a heading | *Write a short transition paragraph above the Conclusion heading* | New paragraph just before the "Conclusion" heading | ~2 s |
| [ ] | Between two sections | *Add a support section between Pricing and Conclusion* | New section just before "Conclusion" | ~2.5 s |
| [ ] | Top of the document | *Add a one-line summary at the top of the document* | New line above "Launch guide" | ~1 s |
| [ ] | After a numbered paragraph | *Draft a sentence after the second paragraph* | New sentence after "A second intro paragraph about goals and timelines." | ~1.5 s |
| [ ] | Below the table | *Write a caption below the table* | Caption directly under the table | ~1.5 s |
| [ ] | No place named | Click in the middle of the document, then: *Write the next paragraph* | New paragraph where your cursor is | ~2 s |
| [ ] | Exact text only | *Add Acme Inc. below the Pricing heading* | Just **Acme Inc.** appears right under the "Pricing" heading. No sentence about Acme, no extra paragraph | Instant, no credits |
| [ ] | Exact text, other kinds | *Add 555-0100 below the Pricing heading*, then *Add "Built to last." above the Conclusion heading* | Only the number, then only the quoted words (without the quote marks), each at its place. No "Phone:" in front, no bold | Instant, no credits |
| [ ] | Exact text with formatting | *Add Acme Inc. in bold below the Pricing heading*, then *Add Acme Inc. as a heading below the Contact heading* | Only "Acme Inc.", in bold, then only "Acme Inc." as a heading, each at its place. Nothing else is written | ~1.5 s |
| [ ] | Asking for writing still writes | *Add a short note about Acme below the Pricing heading* | A short note is written (not just "Acme") | ~1.5 s |
| [ ] | Blanks instead of made-up facts | *Add a short bio for Priya Shah below the Introduction heading* | A short bio. The document does not say who she is, so a blank in square brackets such as "[role]" stands where the missing fact would be; no invented job title | ~1.5 s |
| [ ] | "Next to" and "beside" | *Add a note next to the Pricing heading*, then *Add a line beside the Conclusion heading* | New text right after the "Pricing" heading, then right after the "Conclusion" heading | ~1.5 s each |
| [ ] | Next to a table | *Add a caption next to the table* | Caption directly under the table | ~1.5 s |
| [ ] | Heading not named, cursor in the section | Click inside the Benefits paragraph, then: *Add a line below the heading* | New line right under the "Benefits" heading | ~1.5 s |
| [ ] | Heading not named, no cursor | Without clicking in the document: *Add a line below the heading* | Nothing written. Message: "The document has more than one heading. Name the one you mean (for example 'below the Pricing heading'), or place the cursor in its section." | Instant |
| [ ] | Below the title | *Add a subtitle below the title* | New line right under "Launch guide", above "Introduction" | ~1.5 s |
| [ ] | "Top" in a name is not the top | *Add a note next to the Top Benefits table* in a document that has such a table, or any heading with "Top" in its name (for example "Top 10 list") | The note goes next to that table or heading, **not** at the top of the document | ~1.5 s |
| [ ] | Heading that doesn't exist | *Draft a paragraph below the Risks heading* | Nothing written. Message: "There is no heading named 'Risks'. Name an existing heading, or place the cursor where the draft should go." | Instant |
| [ ] | Unclear which paragraph | *Add a sentence after the paragraph about the launch* | Nothing written. Message: "More than one paragraph mentions 'launch'. Quote its opening words, or place the cursor there." | Instant |
| [ ] | "This paragraph" without clicking first | Without clicking in the document: *Add a note below this paragraph* | Nothing written. Message: "Place the cursor in the paragraph you mean, or name its heading." | Instant |
| [ ] | Paragraph that doesn't exist | *Draft a sentence after paragraph 9* | Nothing written. Message: "There is no paragraph 9. Name the heading, or place the cursor there." | Instant |

---

## Part 5: Changing existing text (Edit mode)

Open the sample document and use Edit mode. Edit only rewrites what is already there.

| Status | Feature | What to do and type | What you should see | Time |
|---|---|---|---|---|
| [ ] | Rewrite one section | *Make the Introduction more concise and remove repetition* | Proposed changes **only** in the Introduction ("very very" is gone). Other sections untouched. You can accept or reject | ~5 s |
| [ ] | Change the tone | *Add more warmth to the Introduction* | A warmer Introduction with the same facts | 7–9 s |
| [ ] | Small fix | *Remove the repeated words in Benefits* | "clear and clear" is fixed; nothing else changes | ~3 s |
| [ ] | Accepting the changes | Accept a proposed change | The document updates exactly as shown in the proposal | Instant |
| [ ] | Small addition inside a line | *Add (2024) next to the title* | Proposed change: the title becomes "Launch guide (2024)". Nothing else changes. You can accept or reject | 4–6 s |
| [ ] | Small addition, no value given | *Add a date after the title* | Proposed change: the title gets a blank in square brackets, such as "Launch guide [Date]". The AI does not make up a date | 4–6 s |
| [ ] | Small addition keeps numbers honest | *Add (2024) next to the title*, accept it, then check the Pricing paragraph | Only the title has "(2024)". "USD 20", "2026" and "50%" are unchanged | — |
| [ ] | Edit the title | *Make the title more engaging* | Proposed change to the title only. It is still a heading of the same size | 5–8 s |
| [ ] | Rename a heading | *Rename the Benefits heading to Why it matters* | Proposed change to that heading only; the text under it is untouched | 4–6 s |
| [ ] | Edit a quote | *Make the quote shorter* | Proposed change to the quote only; it is still a quote (still has its quote bar) | 4–6 s |
| [ ] | Headings are left alone otherwise | *Make the Introduction more concise* and *Tighten the paragraphs but leave the headings unchanged* | Headings and the quote never appear among the proposed changes | ~5 s |
| [ ] | Asking Edit to add something | *Add a short FAQ below the Pricing heading* (also try *Insert a table comparing the plans*, *Add a date below the title* or *Add (2024) after the table*; adding *inside* a line, as above, is fine) | Nothing changes. Message: "Edit only rewrites existing text; use Draft to add new content." No credits used | Instant |
| [ ] | Asking Edit for new facts | *Add two examples to the Benefits paragraph* (also try *Make the Benefits paragraph more detailed*) | Nothing changes. Message: "Edit only rewrites what is already there and can't add new details, examples or facts. Use Draft to add them." No credits used | Instant |
| [ ] | Asking Edit to delete or move | *Delete the Conclusion section* (also try *Move the table below the intro*) | Nothing changes. Message: "Edit only rewrites existing text; it can't delete, move, merge or split sections, paragraphs or tables. Make that change directly in the document." No credits used | Instant |
| [ ] | Writing a brand-new sentence | *Write a closing sentence for the Conclusion* | Nothing changes. The message says to use Draft | ~3 s |
| [ ] | Numbers are protected | Any Edit request on the Pricing section | "USD 20", "2026" and "50%" never change | — |
| [ ] | Credits | Compare your balance | A proposed change is charged. A request that changes nothing costs nothing | — |

---

## Part 5b: Letting the AI pick the mode (Auto)

Open the sample document and choose **Auto** instead of Ask, Draft, Edit or Write. The result should show which mode the AI picked
(your editor may call it a label or badge). The pick should match what you asked for.

| Status | Feature | What to do and type | What you should see | Time |
|---|---|---|---|---|
| [ ] | A question | *What does Pricing cost?* | Mode **Ask**. The answer in the chat: USD 20 per seat per month. Nothing in the document changes | 2–4 s |
| [ ] | New text at a place | *Add a one-sentence summary below the Pricing heading* | Mode **Draft**. New sentence under the "Pricing" heading | ~2 s |
| [ ] | Changing existing text | *Make the Introduction more concise* | Mode **Edit**. Proposed changes in the Introduction only | ~5 s |
| [ ] | A small addition | *Add (2024) next to the title* | Mode **Edit**. The title becomes "Launch guide (2024)" | 4–6 s |
| [ ] | Exact text | *Add Acme Inc. below the Pricing heading* | Mode **Draft**. Only "Acme Inc." under "Pricing" | Instant |
| [ ] | Something standalone | *Write a Python function that checks if an email address is valid* | Mode **Write**. A code block, as in the Write tab. Nothing is read from your document | 5–12 s |
| [ ] | Unclear wording | *Write a closing sentence for the Conclusion* | Mode **Draft**. One sentence for the Conclusion. The pick may take about 1 s longer than the rows above | ~3 s |
| [ ] | A big new section is not squeezed into Edit | *Write a new section about security after the Pricing section* | Mode **Draft** (never Edit). New section after Pricing | ~2–3 s |
| [ ] | Empty page | On an empty page: *Make it shorter* | Mode **Ask**. Message: "There is no document content to answer questions about. Add some text first." | Instant |
| [ ] | Manual choice still works | Choose Edit yourself and type *Add a short FAQ below the Pricing heading* | No label from Auto. Same "use Draft" message as in Part 5 | Instant |
| [ ] | Credits | Compare your balance after an Auto Ask, an Auto Draft and an Auto Edit | Charged exactly as if you had chosen that mode yourself. The step that picks the mode costs nothing | — |

---

## Part 6: Enhance and charts

Select this text in a document: *Our store earned USD 12,000 in January, USD 18,000 in February, and USD 15,000 in March.*
Then use Enhance and type the instruction shown.

| Status | Feature | What to do and type | What you should see | Time |
|---|---|---|---|---|
| [ ] | Chart on request | Instruction: *Make it clearer and add a pie chart* | Improved sentence (no table added to the text), then a pie chart suggestion you can insert. It shows January, February and March with the right amounts and says "Built only from values stated in your content. Review before inserting." | ~3.5 s |
| [ ] | Chart type follows your words | Instruction: *Rewrite this and add a chart* | A bar chart suggestion | ~3 s |
| [ ] | Chart asked in passing | Instruction: *Tighten it, and chart the monthly revenue* | Improved text plus a chart suggestion | ~4 s |
| [ ] | Nothing to chart | Select *Our store had a strong start to the year and customers were happy.* Instruction: *Improve it and add a bar chart* | Improved text, then a message: "This can't be charted: no set of two or more comparable values, each stated with its label, was found in your text…" No chart, no table | ~1–2 s |
| [ ] | Numbers that can't be compared | Select *We hired 12 people and spent USD 40,000 on ads in 2025.* Instruction: *Add a bar chart* | Same "can't be charted" message | ~2 s |
| [ ] | Instruction not copied into the text | Any Enhance instruction | The result never starts with your instruction or with "Text:" | — |
| [ ] | Chart offered without asking | Sales sentence, instruction: *Make it more professional* | Improved text. A chart suggestion may or may not appear; both are fine | ~3.5 s |
| [ ] | Change chart style | On an inserted chart, ask: *change graph type to line and change color to red* | Same numbers, now a red line chart | Instant |
| [ ] | Other tools never chart | Change Tone to formal on the sales sentence | Only a tone change, no chart | ~2 s |

---

## Part 7: Other writing tools

| Status | Feature | What to do and type | What you should see | Time |
|---|---|---|---|---|
| [ ] | Key points | Select a paragraph with several findings and click Highlight Key Points | A one-sentence summary followed by bullet points. Every point comes from your text | Not measured |
| [ ] | Key points with nothing to find | Select a single plain sentence, Highlight Key Points | "No key points found." | Not measured |
| [ ] | Long document | Paste a long article (2+ pages) with a table, a quote and a code block. Click Fix Grammar | Text is corrected section by section, with a progress indicator. Table, quote and code block stay exactly as they were. Headings, lists and links kept | ~12 s per 1,000 words. First text in ~10–18 s on very long documents |
| [ ] | Short text | Fix Grammar on one short paragraph | A single quick correction | A few seconds (not measured) |

---

## Part 8: Designs

| Status | Feature | What to do and type | What you should see | Time |
|---|---|---|---|---|
| [ ] | New design | Create a design: *Design an onboarding card UI for a meditation app: dark background, two-tone headline, a Start button* | A "designing…" state, then a card with a two-colour headline and a Start button. Text is readable. Each part can be clicked and edited | 12–47 s (usually ~25 s) |
| [ ] | Button link and columns | *Design a pricing card with a 'Buy now' button linking to https://example.com/buy and a 3-column grid of feature tiles* | The Buy now button opens example.com/buy; tiles in 3 columns | ~20 s |
| [ ] | Website without "https" | *…a Pricing button that links to example.com/pricing* | Button opens https://example.com/pricing | ~20 s |
| [ ] | No made-up links | *Design a newsletter signup card with a Subscribe button* | The Subscribe button has no link | ~25 s |
| [ ] | Unsafe link refused | *Design a banner with a button that links to javascript:alert(1)* | The button has no link, and a note says the link can't be used | 15–55 s |
| [ ] | Incomplete link refused | *Design a checkout card with a button that links to /checkout* | No link, and a note: "'/checkout' can't be used as a button link. Use a full https:// address, a mailto: or tel: link, or a #section anchor." | ~25 s |
| [ ] | AI edit | On a design with a linked button, ask: *Change the button label to Get started* | Label changes. Link, layout and everything else stay the same | 20–30 s |
| [ ] | AI edit of columns | On the pricing card: *Change the grid to 2 columns* | Tiles now in 2 columns | ~20 s |
| [ ] | Stop | Start a new design and click Stop during "designing…" | Stops right away; no design appears | Instant |
| [ ] | Credits | Compare balance | A new design and an AI edit each cost credits; a design that fails costs nothing | — |

---

## Part 9: Pictures

| Status | Feature | What to do and type | What you should see | Time |
|---|---|---|---|---|
| [ ] | Picture stays with its page | In page A: *Generate an image of a cat on a sofa*. Open page B and type: *Make it night* | Page A's cat picture does **not** change. Typing *Make it night* in page A does turn it into a night scene | Not measured |

---

## Part 10: Email templates (if your editor has this option)

| Status | Feature | What to do and type | What you should see | Time |
|---|---|---|---|---|
| [ ] | Category list | Open the email template option | Seven types: Reminder / deadline, Policy notice, Invitation, Product launch, Feature update, Announcement, General | Instant |
| [ ] | Create an email | Pick Reminder, brief: *Remind staff the safety training closes Friday* | A complete, nicely laid-out email about the training deadline | Not measured |
| [ ] | Credits | Compare balance | Charged by the amount of text; your own-key accounts are not charged | — |

---

## Things that are expected (please don't report these)

| What you may notice | Why |
|---|---|
| In Edit, *Expand the Conclusion a little* sometimes changes nothing | Lengthening without new facts is hit and miss; the "no changes" result costs nothing |
| A "no changes" message in Edit is worded differently each time | The wording comes from the AI |
| In Auto, a request the AI cannot decide on (for example when its helper is down) is answered as **Ask**, with a note that it was not sure | Ask never changes the document. Choose Draft or Edit yourself and send it again |
| In Auto, "Write a job description for a junior designer" may come back as Draft or as Write | Both write new text; either is fine |
| A small addition without a value gives "[Phone number]" or "[Date]" | The AI never invents a phone number or date. Fill the blank in yourself |
| Draft ignores the place when written in another language (e.g. Spanish) | Only English place words are understood; the text goes where your cursor is |
| *Add a section between X and Y* when neither section exists | The text goes where your cursor is instead of asking |
| A design asked for in the normal chat box comes back as text or a picture | Use the design option to create designs |
| Stopping a design still uses about 1 credit | Known issue, being fixed |
