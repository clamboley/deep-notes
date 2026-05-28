---
title: <Concept name — e.g. Flash Attention>
tags: [<topic>, <subtopic>, <method-or-architecture>]
date: 2026-05-28
draft: false
source: <url of the paper or blog post>
aliases: [<alternative names or abbreviations people search for>]
---

<!-- Quartz reads the frontmatter above:
     - tags become clickable tag pages + feed search
     - draft: true hides this note from the published site (use for half-baked captures)
     - source/aliases are optional but handy
     Delete these comments once you're comfortable; they don't render on the site. -->

> [!abstract] TL;DR
> <One or two sentences: what this is and why it matters. Write this last, after the rest is clear.>

## Context

<Where does this fit? What problem was the field stuck on before this, and what does this unlock? Link the lineage with wikilinks, e.g. builds on [[Attention is All You Need]] and motivates [[Paged Attention]].>

## Core idea

<The single load-bearing insight, in plain language — the thing you'd say out loud to explain it to a colleague. Intuition first, formalism later.>

## How it works

<Walk through the mechanism step by step. Keep it skimmable with short paragraphs or a numbered list.>

Inline math reads naturally: the scores are $QK^\top / \sqrt{d_k}$ before the softmax.

Display math for the load-bearing equations:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

Code or pseudocode where it clarifies more than prose:

```python
# Sketch the core computation, not the whole repo
def attention(q, k, v):
    scores = (q @ k.transpose(-2, -1)) / k.shape[-1] ** 0.5
    return scores.softmax(dim=-1) @ v
```

Embed a figure (drop the image in the content/ folder next to this note):

![[diagram.png]]
<!-- caption: redraw or screenshot the one figure that made it click -->

## Things I had to untangle

<This is the high-value section for you — the bits the LLM helped clarify. Capture the
misconception AND the correction, so future-you doesn't relearn the same trap.>

> [!question] What confused me
> <The thing that didn't add up at first.>

> [!tip] What actually clicked
> <The resolution, in your own words.>

## Open questions

- [ ] <Something still fuzzy — a TODO you can revisit later.>
- [ ] <A follow-up paper or experiment worth chasing.>

## Related

<Wikilinks here populate Quartz's graph view and backlinks, so your notes interconnect.>

- [[<related concept>]] — <one line on how it connects>
- [[<contrasting approach>]] — <what it does differently>

## References

1. <Author(s)>, *<Title>* (<year>) — <source url>
2. <Any blog post, video, or thread that helped>
