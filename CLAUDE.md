# deep-notes

Personal deep learning research note system published at https://clamboley.github.io/deep-notes.

**Stack:** Quartz v5 + GitHub Pages. Pushing to the `v5` branch triggers GitHub Actions and deploys automatically.

## Content layout

```
content/
├── index.md
├── templates/note-template.md   # reference before creating a new note
├── scaling-laws/
├── architectures/
├── pretraining/
├── training/
├── quantization/
└── math/
```

Images live in the same folder as the note and are embedded with `![[filename.png]]`.

`content/sw.js` is the PWA service worker — it lives in `content/` intentionally so the Assets emitter places it at the root of `public/` (required for service worker scope). Do not move it to `quartz/static/`.

## Note conventions

**Frontmatter fields:** `title`, `tags`, `date`, `draft`, `source` (paper URL), `aliases`.

**Callout types in use:**
- `[!abstract]` — TL;DR at the top
- `[!warning]` — caveats, revisions, known limitations
- `[!question]` / `[!tip]` — for untangling confusions (only add these when the user has discussed the topic)

**Figures:** always include the verbatim figure caption from the paper in italics below the embed:
```
![[kaplan-fig5.png]]
*Figure 5: [exact paper caption text]*
```

**Writing style:** the audience is a senior DL practitioner. Skip introductory motivation. Use precise terminology (FLOPs, MFU, FSDP, TP, etc.) without defining it unless the note specifically covers it.

**Sections like "Things I Had to Untangle" and "Open Questions" are only added after discussing the topic** — do not pre-populate them.

## Customized Quartz files

`quartz/components/Head.tsx` has been edited beyond the Quartz default:
- SVG favicon (`icon.svg`) with PNG fallback (`icon.png`) for Safari
- `<link rel="apple-touch-icon">` pointing to `quartz/static/apple-touch-icon.png`
- `<link rel="manifest">` pointing to `quartz/static/manifest.json`
- `<meta name="theme-color">` and inline service worker registration script

Do not overwrite this file with the upstream default.

## Static assets

| File | Purpose |
|------|---------|
| `quartz/static/icon.svg` | PWA + browser favicon (Chrome/Firefox) |
| `quartz/static/icon.png` | Favicon fallback (Safari) |
| `quartz/static/apple-touch-icon.png` | iOS home screen icon (180×180) |
| `quartz/static/manifest.json` | Web App Manifest |
| `content/sw.js` | Service worker (served at `/deep-notes/sw.js`) |

To regenerate `apple-touch-icon.png` from an SVG:
```bash
node -e "require('sharp')('input.svg').resize(180,180).png().toFile('apple-touch-icon.png')"
```
