# deep-notes

My personal notes and research, published as a digital garden.

🔗 https://clamboley.github.io/deep-notes

## Stack

- [Quartz v5](https://quartz.jzhao.xyz/) — static site generator for digital gardens
- GitHub Pages — hosting, deployed automatically on push

## Adding a note

Create a Markdown file in `content/` with frontmatter:

```markdown
---
title: Note title
tags: [tag1, tag2]
date: YYYY-MM-DD
draft: false
---

Your content here.
```

Set `draft: true` to keep a note local without publishing it.

## Deploying

```bash
npx quartz sync
```

This commits, pushes to the `v5` branch, and triggers the GitHub Actions deploy.
