const CACHE = 'deep-notes-v1'
const BASE = '/deep-notes'

self.addEventListener('install', () => self.skipWaiting())

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  )
})

// Cache-first for static assets; passthrough everything else
self.addEventListener('fetch', e => {
  const url = new URL(e.request.url)
  if (!url.pathname.startsWith(BASE + '/static/')) return

  e.respondWith(
    caches.match(e.request).then(cached => {
      const network = fetch(e.request).then(res => {
        if (res.ok) caches.open(CACHE).then(c => c.put(e.request, res.clone()))
        return res
      })
      return cached ?? network
    })
  )
})
