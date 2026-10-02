"""Turn chapters/<name>.md into design/chapter.html.

Run:  python build.py

What the Markdown file can contain:
  # Title                      the chapter title
  anything before the first ## goes at the top of the article
  # Big title                  any later # line: a big title above the next part
  ## Heading                   starts the next part
  ## Long heading | Short      the short label is used in the side list
  ### Heading                  a smaller heading inside a part
  1. item                      numbered list lines
  **bold**  *italic*           as usual
  (Author, 1970)               any (...) with a year becomes a citation
  ![label](img/file.png) ...   pictures on one line become one grey display row
  ![label]()                   an empty placeholder tile
  ![](img/drawing.svg)         one .svg on its own line: a full-width drawing
  ![540](img/drawing.svg)      the same, at most 540 px wide (also works for one .png)
  [demo: bouma]                a playable figure (written in design/demos.js)
  **Figure 1.** text           the caption under the figure above it
  *whole paragraph*            a "work in progress" note
  - item                       reference list lines
"""

import html
import os
import re

# work from the folder this script is in, wherever it is run from
os.chdir(os.path.dirname(os.path.abspath(__file__)))

chapterFile  = "chapters/crowding.md"
templateFile = "templates/chapter.html"
outputFile   = "design/chapter.html"

# the six parts of every chapter, in order; the notebook view uses the same names
sectionIds = ["classic", "human", "vlm", "model", "compare", "meaning"]


def inline(text, citations=True):
  """Bold, italic and citations inside one paragraph."""
  text = html.escape(text, quote=False)
  text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
  text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
  # a bracket that contains a year, e.g. (Bouma, 1970)
  if citations:
    text = re.sub(r"\(([^()]*\b(?:1[89]|20)\d\d[a-z]?)\)", r'<span class="cite">(\1)</span>', text)
  return text


def withCaption(inside, caption, width=""):
  """Wrap a figure's content and its caption in <figure>; width (pixels) limits both."""
  style = ""
  if width != "":
    style = ' style="max-width: ' + width + 'px"'
  out = "<figure" + style + ">\n" + inside + "\n"
  if caption:
    out += "<figcaption>" + inline(caption) + "</figcaption>\n"
  return out + "</figure>"


def demoHtml(name, caption):
  """A playable figure; design/demos.js draws into the empty box."""
  return withCaption('<div class="drawing" data-demo="' + name + '"></div>', caption)


def figureHtml(imageLine, caption):
  """One line of ![label](src) images, plus its caption, as a figure."""
  images = re.findall(r"!\[(.*?)\]\((.*?)\)", imageLine)

  # a single .svg is a drawing: put the svg itself in the page
  # a number as the label, e.g. ![540](img/x.svg), sets its largest width in pixels
  if len(images) == 1 and images[0][1].endswith(".svg"):
    svg = open("design/" + images[0][1], encoding="utf-8").read().strip()
    width = ""
    if images[0][0].isdigit():
      width = images[0][0]
    return withCaption('<div class="drawing">\n' + svg + '\n</div>', caption, width)

  tiles = []
  for label, src in images:
    if src == "":
      tiles.append('<div class="tile"><div class="empty">' + html.escape(label).replace(" (", "<br>(") + '</div></div>')
    elif label.isdigit() or label == "":
      tiles.append('<div class="tile"><img src="' + src + '" alt=""></div>')
    else:
      tiles.append('<div class="tile"><img src="' + src + '" alt=""><span>' + html.escape(label) + '</span></div>')

  rowClass = "display-row"
  if len(images) == 1:
    rowClass = "display-row one"
  if len(images) == 2:
    rowClass = "display-row two"
  if len(images) == 3:
    rowClass = "display-row three"

  # one picture with a number as its label, e.g. ![640](img/x.png): at most that many pixels wide
  rowStyle = ""
  if len(images) == 1 and images[0][0].isdigit():
    rowStyle = ' style="grid-template-columns: minmax(0, ' + images[0][0] + 'px)"'
  inside = '<div class="display">\n<div class="' + rowClass + '"' + rowStyle + '>\n' + "\n".join(tiles) + '\n</div>\n</div>'
  return withCaption(inside, caption)


def blockHtml(block):
  """One paragraph-sized block of Markdown (no headings) as HTML."""
  lines = block.split("\n")
  if all(line.startswith("- ") for line in lines):
    return '<p class="cite">' + "<br>\n".join(inline(line[2:], citations=False) for line in lines) + "</p>"
  if all(re.match(r"\d+\. ", line) for line in lines):
    items = ["<li>" + inline(re.sub(r"^\d+\. ", "", line)) + "</li>" for line in lines]
    return "<ol>\n" + "\n".join(items) + "\n</ol>"
  text = " ".join(lines)
  if text.startswith("*") and text.endswith("*") and not text.startswith("**"):
    return '<p class="pending">' + inline(text[1:-1]) + "</p>"
  return "<p>" + inline(text) + "</p>"


# ---- read the Markdown and split it into blocks separated by blank lines ----
markdown = open(chapterFile, encoding="utf-8").read()
blocks   = [b.strip() for b in re.split(r"\n\s*\n", markdown) if b.strip() != ""]

title    = ""
intro    = []   # html pieces before the first ## heading
sections = []   # list of [heading, list of html pieces, big title shown above it]
bigTitle = ""   # a later "# " line: a big title before the next part

i = 0
while i < len(blocks):
  block = blocks[i]
  if block.startswith("# ") and title == "":
    title = block[2:].strip()
  elif block.startswith("# "):
    bigTitle = block[2:].strip()
  elif block.startswith("## "):
    heading = block[3:].strip()
    shortLabel = heading
    if " | " in heading:
      heading, shortLabel = [part.strip() for part in heading.split(" | ", 1)]
    sections.append([heading, [], bigTitle, shortLabel])
    bigTitle = ""
  elif block.startswith("### "):
    if len(sections) == 0:
      intro.append("<h3>" + inline(block[4:].strip()) + "</h3>")
    else:
      sections[-1][1].append("<h3>" + inline(block[4:].strip()) + "</h3>")
  else:
    # where this block goes: the intro, or the current part
    if len(sections) == 0:
      pieces = intro
    else:
      pieces = sections[-1][1]

    isImages = re.fullmatch(r"(\s*!\[.*?\]\(.*?\))+\s*", block)
    isDemo   = re.fullmatch(r"\[demo:\s*(\w+)\]", block)
    if isImages or isDemo:
      # the next block is the caption if it starts with **Figure
      caption = ""
      if i + 1 < len(blocks) and blocks[i + 1].startswith("**Figure"):
        caption = " ".join(blocks[i + 1].split("\n"))
        i = i + 1
      if isDemo:
        pieces.append(demoHtml(isDemo.group(1), caption))
      else:
        pieces.append(figureHtml(block, caption))
    else:
      pieces.append(blockHtml(block))
  i = i + 1

# ---- build the table of contents and the article ----
tocLines     = []
articleParts = []
for number, (heading, pieces, bigTitle, shortLabel) in enumerate(sections):
  sectionId = sectionIds[number]
  tocLines.append('        <li><a href="#" data-section="' + sectionId + '">' + html.escape(shortLabel) + '</a></li>')
  bigTitleHtml = ""
  if bigTitle != "":
    bigTitleHtml = '<h1 class="big-title">' + inline(bigTitle) + '</h1>\n'
  articleParts.append(
    bigTitleHtml
    + '<section data-section="' + sectionId + '">\n'
    + '<h2>' + html.escape(heading) + '</h2>\n'
    + "\n".join(pieces) + "\n</section>")

introHtml = ""
if len(intro) > 0:
  introHtml = '<div class="intro">\n' + "\n".join(intro) + "\n</div>\n\n"

page = open(templateFile, encoding="utf-8").read()
page = page.replace("{{title}}", html.escape(title))
page = page.replace("{{toc}}", "\n".join(tocLines))
page = page.replace("{{article}}", introHtml + "\n\n".join(articleParts))
page = page.replace("<!-- Template for chapter pages. build.py fills the {{...}} slots from chapters/*.md -->\n", "")
page = page.replace("<!doctype html>", "<!doctype html>\n<!-- Made by build.py from " + chapterFile + ". Edit that file, not this one. -->", 1)

open(outputFile, "w", encoding="utf-8", newline="\n").write(page)
print("wrote", outputFile, "from", chapterFile, "-", len(sections), "parts")
