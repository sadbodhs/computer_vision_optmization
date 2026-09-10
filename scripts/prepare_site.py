#!/usr/bin/env python3
"""Stage the repo's markdown for MkDocs without duplicating anything in git.

The committed docs use relative links that are correct when browsing the repo on
GitHub (../scripts/foo.sh, ../README.md). On a published site those resolve
outside the site root and 404. This rewrites them at BUILD time only:

  ../README.md            -> index.md            (site home)
  ../<non-md path>        -> absolute GitHub blob URL
  docs/foo.md (in README) -> foo.md              (README becomes the index)

Committed files are never modified.
"""
import os
import re
import shutil
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/site/src"
BLOB = "https://github.com/sadbodhs/computer_vision_optmization/blob/main"

shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)

link_re = re.compile(r"\]\(([^)]+)\)")


def fix_docs_link(m):
    href = m.group(1)
    if href.startswith(("http://", "https://", "#", "mailto:")):
        return m.group(0)
    if href == "../README.md":
        return "](index.md)"
    if href.startswith("../"):
        rest = href[3:]
        # .md files that live outside docs/ (e.g. ../results/README.md) also
        # aren't in the site tree -> send them to GitHub too
        return f"]({BLOB}/{rest})"
    return m.group(0)


def fix_readme_link(m):
    href = m.group(1)
    if href.startswith(("http://", "https://", "#", "mailto:")):
        return m.group(0)
    if href.startswith("docs/"):
        return f"]({href[5:]})"          # docs/precision.md -> precision.md
    if href.endswith(".md"):
        return f"]({BLOB}/{href})"       # STORY.md etc. stay on GitHub
    return f"]({BLOB}/{href})"           # scripts/, results/, triton/


# docs/*.md -> site root
n = 0
for name in sorted(os.listdir(os.path.join(REPO, "docs"))):
    if not name.endswith(".md"):
        continue
    src = os.path.join(REPO, "docs", name)
    text = open(src, encoding="utf-8").read()
    open(os.path.join(OUT, name), "w", encoding="utf-8").write(link_re.sub(fix_docs_link, text))
    n += 1

# non-markdown assets under docs/ (figures) must come along, or they 404
for sub in ("img",):
    src_dir = os.path.join(REPO, "docs", sub)
    if os.path.isdir(src_dir):
        shutil.copytree(src_dir, os.path.join(OUT, sub))
        print(f"copied docs/{sub}/")

# README.md -> index.md
readme = open(os.path.join(REPO, "README.md"), encoding="utf-8").read()
open(os.path.join(OUT, "index.md"), "w", encoding="utf-8").write(link_re.sub(fix_readme_link, readme))

print(f"staged {n} docs + index.md into {OUT}")
