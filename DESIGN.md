---
name: Deep Notes
description: A quiet index for visual paper explanations.
colors:
  primary: "#2f62d6"
  neutral-bg: "#f3f5f8"
  neutral-surface: "#ffffff"
  neutral-line: "#d3dae3"
  neutral-ink: "#152030"
  neutral-muted: "#5a6676"
  neutral-cover: "#e7ebf0"
  dark-bg: "#0e141b"
  dark-surface: "#151d27"
  dark-line: "#2a3542"
  dark-ink: "#e3e9f0"
  dark-muted: "#a7b3c1"
  dark-primary: "#8aaaff"
typography:
  title:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "24px"
    fontWeight: 700
    lineHeight: 1.16
    letterSpacing: "-0.02em"
  body:
    fontFamily: "IBM Plex Sans, system-ui, sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.55
  control:
    fontFamily: "IBM Plex Sans, system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 500
    lineHeight: 1.4
  label:
    fontFamily: "IBM Plex Mono, ui-monospace, monospace"
    fontSize: "12px"
    fontWeight: 400
    lineHeight: 1.4
rounded:
  input: "8px"
  image: "10px"
  video-mobile: "6px"
---

# Design System: Deep Notes

## Overview

**Creative North Star: "The visual paper index"**

The library is a quiet entrance to self-contained paper explanations. Its cool surfaces, precise lettering, and restrained diagrams come from the V-JEPA 2 note. Covers carry the paper's subject; the surrounding interface stays spare.

## Colors

Blue marks links and observed information. The ground, surfaces, text, and dividers use the note's cool neutral palette. Paper-specific cover diagrams may also use its rose prediction and amber goal colors. Dark mode uses the note's dark counterparts.

**The Meaning Rule.** Accent colors express roles or actions, not decoration.

## Typography

Archivo gives the title and card headings their firm shape. IBM Plex Sans carries prose and controls. IBM Plex Mono is reserved for dates, counts, and diagram labels.

## Layout

The index sits in a centered container capped at 1180px. Papers form a three-column grid on wide screens, two columns below 850px, and one column below 620px. Year and label selectors sit alongside search above the cards; on phones the selectors share a row and search fills the next one.

## Elevation & Depth

The site is flat. Thin separators and changes in surface color establish hierarchy; cards do not use shadows.

## Shapes

The search field has gently rounded corners. Cover images have a slightly larger radius. Diagram geometry inside covers remains crisp.

## Components

Each card has one main link around its image and title, with a separate video link when available. The paper date sits beneath the title. Hover lifts only the cover image slightly; keyboard focus remains clearly outlined. The header uses an outlined snare-drum link to the author's main site.

The video page is a focused continuation of the library: a text back link first, then the paper title, native video controls, and a small link to the note. It reuses the same cool ground and type families.

## Do's and Don'ts

- **Do** give every paper a specific visual cover tied to its subject.
- **Do** display the original paper publication date.
- **Don't** add tags, abstracts, or decorative badges to the card surface; search can use that metadata without showing it.
