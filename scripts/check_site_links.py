#!/usr/bin/env python3
"""Stand-in for `mkdocs build --strict`, for checking a change before pushing.

The site is built in CI (.github/workflows/pages.yml) and mkdocs is not installed
on the benchmark host, so there is otherwise no way to catch a broken link until
after a deploy. This checks what --strict checks:

  * every nav entry resolves to a staged file, and every staged file is in nav
  * every internal .md link resolves

plus one thing --strict does NOT check and which has broken this site twice:

  * anchor targets (#some-heading), including those into the README-derived index

Run scripts/prepare_site.py first; this reads the staged tree, not docs/.

Usage: python3 scripts/check_site_links.py [site_src_dir]
Exit:  0 clean, 1 problems (listed).
"""
import os
import re
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "site_src"
bad = []

files = {n for n in os.listdir(SRC) if n.endswith(".md")}

nav = re.findall(r"^\s+- .+?: (\S+\.md)$", open("mkdocs.yml").read(), re.M)
for n in nav:
    if n not in files:
        bad.append("nav -> missing file: %s" % n)
for n in sorted(files - set(nav)):
    bad.append("staged but not in nav: %s" % n)
print("nav: %d entries, %d staged .md files" % (len(nav), len(files)))

link_re = re.compile(r"\]\((?!https?:|#|mailto:)([^)\s]+)\)")
head_re = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.M)


def slugs(path):
    out = set()
    for h in head_re.findall(open(os.path.join(SRC, path), encoding="utf-8").read()):
        s = re.sub(r"[^\w\s-]", "", h.lower().replace("`", "")).strip()
        out.add(re.sub(r"[\s_]+", "-", s))
    return out


anchors = {f: slugs(f) for f in files}

for f in sorted(files):
    for href in link_re.findall(open(os.path.join(SRC, f), encoding="utf-8").read()):
        target, _, frag = href.partition("#")
        if not target:
            continue
        if target.endswith(".md"):
            if target not in files:
                bad.append("%s -> %s (no such page)" % (f, href))
            elif frag and frag not in anchors[target]:
                bad.append("%s -> %s (no such anchor)" % (f, href))
        elif not os.path.exists(os.path.join(SRC, target)):
            bad.append("%s -> %s (missing asset)" % (f, target))

for b in bad:
    print("FAIL", b)
print("%d problems" % len(bad))
sys.exit(1 if bad else 0)
