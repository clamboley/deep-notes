# Deep Notes

A small library of visual paper notes and companion videos, published at [clamboley.github.io/deep-notes](https://clamboley.github.io/deep-notes/).

The site is plain HTML, CSS, and JavaScript. GitHub Pages publishes the generated `public/` directory whenever the `v5` branch is pushed.

## Add a paper

1. Put the standalone HTML note in `site/papers/<slug>/index.html`.
2. Put its cover image in `site/assets/`. If there is a companion video, put it in `site/videos/`.
3. Add an entry to `site/papers.json`. Use the **paper's original publication date** in `paperDate` (`YYYY-MM-DD`), not the date the note was written. Put the paper's filter categories in `labels` and omit `video` when there isn't one.
4. Run `node scripts/build.mjs` and open `public/index.html` to check the result.

The build sorts cards by paper publication date, newest first, and checks that every referenced local file exists. It adds the shared site header to the published copy of each standalone note; the source HTML stays untouched. When a paper has a video, the build creates a small video page with a link back to the library. The year and label filters are generated from `papers.json`; search covers title, description, labels, and paper date.

The previous Quartz notes are no longer part of the site; they remain available in Git history.
