import { readFile, writeFile, mkdir, cp, rm, stat } from "node:fs/promises"
import { join } from "node:path"

const root = new URL("../", import.meta.url).pathname
const source = join(root, "site")
const output = join(root, "public")
const papers = JSON.parse(await readFile(join(source, "papers.json"), "utf8"))
const template = await readFile(join(source, "index.template.html"), "utf8")
const videoTemplate = await readFile(join(source, "video.template.html"), "utf8")
const headerTemplate = await readFile(join(source, "header.template.html"), "utf8")
const headerFor = (rootPath) => headerTemplate.replace("{{ROOT}}", rootPath)

const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
})[char])
const labelKey = (label) => encodeURIComponent(label.toLocaleLowerCase())

async function localAsset(path) {
  if (typeof path !== "string" || path.startsWith("/") || path.split("/").includes("..")) throw new Error(`Invalid local path: ${path}`)
  const file = join(source, path)
  if (!(await stat(file)).isFile()) throw new Error(`Missing file: ${file}`)
  return escapeHtml(path)
}

const slugs = new Set()
const cards = []
const videoPages = []
const notePages = []
for (const paper of papers.sort((a, b) => b.paperDate.localeCompare(a.paperDate))) {
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(paper.slug) || slugs.has(paper.slug)) throw new Error(`Invalid or duplicate slug: ${paper.slug}`)
  slugs.add(paper.slug)
  if (!/^\d{4}-\d{2}-\d{2}$/.test(paper.paperDate) || Number.isNaN(Date.parse(`${paper.paperDate}T00:00:00Z`))) throw new Error(`Invalid paper date: ${paper.paperDate}`)
  const date = new Intl.DateTimeFormat("en", { month: "long", day: "numeric", year: "numeric", timeZone: "UTC" }).format(new Date(`${paper.paperDate}T00:00:00Z`))
  const cover = await localAsset(paper.cover)
  if (paper.note !== `papers/${paper.slug}/index.html`) throw new Error(`Expected note at papers/${paper.slug}/index.html`)
  const note = await localAsset(paper.note)
  notePages.push(paper.note)
  const video = paper.video ? await localAsset(paper.video) : null
  const videoPage = video ? `papers/${paper.slug}/video.html` : null
  if (video) videoPages.push({ slug: paper.slug, title: escapeHtml(paper.title), cover, note, video })
  const labels = paper.labels || []
  const searchable = [paper.title, paper.description, ...labels, paper.paperDate].join(" ").toLocaleLowerCase()

  cards.push(`<article class="paper-card" data-search="${escapeHtml(searchable)}" data-year="${paper.paperDate.slice(0, 4)}" data-labels="${escapeHtml(labels.map(labelKey).join(" "))}">
  <a class="card-main" href="${note}" aria-label="Read ${escapeHtml(paper.title)} notes">
    <span class="card-cover"><img src="${cover}" alt="${escapeHtml(paper.coverAlt || "")}" loading="lazy"></span>
    <span class="card-title">${escapeHtml(paper.title)}</span>
  </a>
  <div class="card-meta">
    <time class="card-date" datetime="${escapeHtml(paper.paperDate)}">${escapeHtml(date)}</time>
    ${video ? `<a class="video-link" href="${videoPage}" aria-label="Watch ${escapeHtml(paper.title)} video"><svg viewBox="0 0 16 16" fill="none" aria-hidden="true"><circle cx="8" cy="8" r="7" stroke="currentColor"/><path d="m6.5 4.9 4.2 3.1-4.2 3.1V4.9Z" fill="currentColor"/></svg>Watch video</a>` : ""}
  </div>
</article>`)
}

await rm(output, { recursive: true, force: true })
await mkdir(output, { recursive: true })
for (const name of ["assets", "papers", "videos", "header.css", "styles.css", "app.js"]) {
  await cp(join(source, name), join(output, name), { recursive: true })
}
for (const note of notePages) {
  const file = join(output, note)
  const html = await readFile(file, "utf8")
  if (!/<\/head>/i.test(html) || !/<body(?:\s[^>]*)?>/i.test(html)) throw new Error(`Note must be a full HTML page: ${note}`)
  await writeFile(file, html
    .replace(/<\/head>/i, '<link rel="stylesheet" href="../../header.css"></head>')
    .replace(/<body(?:\s[^>]*)?>/i, (body) => `${body}\n${headerFor("../../")}`))
}
const years = [...new Set(papers.map((paper) => paper.paperDate.slice(0, 4)))].sort().reverse()
const labels = [...new Set(papers.flatMap((paper) => paper.labels || []))].sort((a, b) => a.localeCompare(b))
const yearOptions = `<option value="">All years</option>${years.map((year) => `<option value="${escapeHtml(year)}">${escapeHtml(year)}</option>`).join("")}`
const labelOptions = `<option value="">All labels</option>${labels.map((label) => `<option value="${escapeHtml(labelKey(label))}">${escapeHtml(label)}</option>`).join("")}`
await writeFile(join(output, "index.html"), template
  .replace("{{SITE_HEADER}}", headerFor("./"))
  .replace("{{YEAR_OPTIONS}}", yearOptions)
  .replace("{{LABEL_OPTIONS}}", labelOptions)
  .replace("{{PAPER_CARDS}}", cards.join("\n")))
for (const page of videoPages) {
  await writeFile(join(output, "papers", page.slug, "video.html"), videoTemplate
    .replace("{{SITE_HEADER}}", headerFor("../../"))
    .replaceAll("{{TITLE}}", page.title)
    .replaceAll("{{COVER}}", page.cover)
    .replaceAll("{{VIDEO}}", page.video)
    .replaceAll("{{NOTE}}", page.note))
}
console.log(`Built ${cards.length} paper${cards.length === 1 ? "" : "s"} in ${output}`)
