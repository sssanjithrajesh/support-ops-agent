---
id: KB-010
title: Slow performance and timeouts
queues: [Technical Support, Product Support, IT Support]
keywords: slow performance latency timeout loading lag optimization speed dashboard report analytics
---
Most slowness reports are either a single very large query or a browser extension, and both are quick to rule out.

Steps:
1. Try the same action in a private browsing window with extensions disabled.
2. Narrow the date range or filters on the slow dashboard or report; anything over one million rows is computed on demand.
3. If it is still slow, send us the request id shown in the browser's network tab and the time it took; we can see the server-side timing for that exact request.

Performance incidents affecting many users are posted on the status page.
