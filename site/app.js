const input = document.querySelector("#paper-search")
const yearFilter = document.querySelector("#year-filter")
const labelFilter = document.querySelector("#label-filter")
const cards = [...document.querySelectorAll(".paper-card")]
const status = document.querySelector("#result-status")
const empty = document.querySelector("#empty-state")

function updateResults() {
  const query = input.value.trim().toLocaleLowerCase()
  const year = yearFilter.value
  const label = labelFilter.value
  let shown = 0

  for (const card of cards) {
    const matches = card.dataset.search.includes(query)
      && (!year || card.dataset.year === year)
      && (!label || card.dataset.labels.split(" ").includes(label))
    card.hidden = !matches
    if (matches) shown++
  }

  status.textContent = `${shown} ${shown === 1 ? "paper" : "papers"} shown`
  empty.hidden = shown !== 0
}

input.addEventListener("input", updateResults)
yearFilter.addEventListener("change", updateResults)
labelFilter.addEventListener("change", updateResults)

document.addEventListener("keydown", (event) => {
  if (event.key !== "/" || event.altKey || event.ctrlKey || event.metaKey) return
  if (document.activeElement?.matches("input, textarea, [contenteditable]")) return
  event.preventDefault()
  input.focus()
})
