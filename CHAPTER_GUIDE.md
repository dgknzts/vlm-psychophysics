# How to write a chapter

Rules learned while making the crowding chapter (`chapters/crowding/`), which is the model to
follow. Read this before working on any chapter. CLAUDE.md has the site's design decisions;
this file is about the chapters' content: writing, figures, the experiment, and how to work.

## Writing

- **Short, plain and direct.** One idea per paragraph, two or three sentences. If a part feels
  long, it is: cut it.
- **Numbers go in figures, not in the text.** A paragraph that lists "21% at 0.29°, 68% at
  0.58°, 92% at 1.17°…" is too much; nobody reads it. Say what happened in words
  ("at 10° the model needed about twice the space") and point to the figure.
- **Itemize comparisons.** "Like humans / Unlike humans" as short numbered points with a bold
  lead ("**A much smaller crowding zone.** …"), one or two sentences each.
- **Sound human-written.** No filler, no hype words, no "it is worth noting". Write a draft
  the user can rewrite in their own voice; tell them which parts are your drafts.
- **Never invent numbers.** Human results come from the papers, with citations. If a number
  cannot be found in a source you have read, give no number. Say so to the user.
- **Correct wrong premises, with a source.** Example: the user thought flanker reports
  (substitutions) are rare in humans; the literature (Strasburger & Malania 2013, cited in
  Coates 2021) says they are common. The text then said what is really different: the model
  made *only* flanker errors.
- **Explain before you show.** Every figure is introduced by the text before it, and the text
  after it carries on from it. Watch the transitions between parts: after a technical detour
  (the tokenizer tutorial) add a short bridge back to the topic ("Back to crowding: what the
  grid does to our letters").
- **Link the steps.** When one figure follows a single example through several steps, name it
  the same way everywhere ("our square", "square 4 of its token", "token 58") so the reader
  can follow it.
- The chapter opens with "What we will do here", a numbered list of the parts. Keep it up to
  date when a part is added (e.g. the tokenizer tutorial became item 4).

## Figures

- **Real data over drawings.** When a figure shows what the model computes, use the model's
  real numbers (`code/model_internals.py` pulls them out on a laptop CPU). If something must
  be made up or drawn, say so in the figure itself ("values made up", "a drawing, not
  measured").
- One figure, one idea. Short captions: what is shown, not what to conclude.
- Figures are white with dark ink (IBM Plex Sans), fixation-dot red for the model or for
  "ours", grey for human data. The page puts them on the mid-grey display.
- Number figures in reading order; sub-figures as 6a, 6b, … for one walkthrough. Renumber the
  text references when a figure is added.
- Plots: proportion correct against spacing on a log axis, guessing level marked; model and
  human data on the same axes when they can be compared.
- Squares stay square (equal aspect). Labels must not overlap anything; look at every figure
  before showing it.

### Animations

- Use them only where motion explains something (a matrix becoming a list, information mixing
  across layers, tokens entering the input). Keep a still picture for steps that don't need
  motion.
- Save as **MP4** (`figures.save_video`), not GIF: the page gives it play/pause and a time
  bar. In the Markdown: `![640](img/name.mp4)`.
- **Slow, one thing at a time.** Each caption stays on screen about 2 seconds before the next
  change. Elements fade in; nothing appears suddenly. When many similar things happen (128
  tokens), show the first few slowly, then speed up, and pause on the one that matters.
- Pauses must land on something meaningful: an outline or a highlight during a pause has to
  point at what the caption is talking about.
- Pale background (`ANIM_PALE`) so white pixels are visible.

## The experiment

- Recreate the human paradigm as closely as a model allows, in the same units: degrees of
  visual angle, via a stated pixels-per-degree (crowding: a 24-inch, 1920-px monitor at the
  paper's viewing distance, 32 px/deg).
- **Keep it simple.** Few conditions, clearly motivated; drop what the user doesn't need
  (e.g. extra resolutions). Earlier exploratory runs go to `archive/`.
- Same task and answer set as the humans, as a forced choice: read the model's score for each
  allowed answer as its first word and take the highest; guessing level = 1 / number of
  answers.
- About 100 trials per point, as in classic psychophysics.
- Fit the same psychometric function to model and human data, and compare at the **same
  cutoff**; report a range over cutoffs rather than one number.
- Look at the errors, not only accuracy (crowding: every error was a flanker).
- Say what had to change from the human experiment, briefly, where it matters.
- Model runs over all trials go to Colab and are run by the user (`run_model.ipynb`);
  analysis and figures run locally (`analysis.ipynb`, `figures.py`). Single-image looks
  inside the model can run locally on CPU.

## Files for a chapter

`chapters/<name>/`: `<name>.md` (the text, which the user edits), `<name>.html` (made by
`python build.py`, never edited by hand), `img/`, `code/` (`stimuli.py`, `figures.py`,
`run_model.ipynb`, `analysis.ipynb`, `model_internals.py`), `data/`, `sources/` (PDFs,
git-ignored), `archive/`. Notebook cells carry `metadata.section` so they show in the
Notebook view. Markdown rules are at the top of `build.py`.

## Working with the user

- Change one thing at a time and show it. Do what was asked, nothing more.
- Before showing anything, render it and look at it yourself (figures, video frames, a
  headless-Chrome screenshot of the page). Check the right file was written.
- Keep replies short: what changed, what to look at, any caveat. Quote the new text when the
  user asked for a rewrite.
- When unsure between options, give a recommendation, or a small temporary comparison page.
- The user often sends new requests while you work: handle them in order, and say which
  ones are done.
