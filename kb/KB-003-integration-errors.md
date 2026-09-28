---
id: KB-003
title: Integration and API connection errors
queues: [Technical Support, IT Support, Product Support]
keywords: integration api connection sync webhook token 401 403 timeout connect third-party plugin compatibility configuration
---
Integration failures come from one of three places: an expired API token, a permission the integration user does not have, or a network rule blocking our IP range.

Steps:
1. Regenerate the API token under Settings > Integrations and paste it into the connected system; tokens expire after 90 days.
2. Confirm the integration user has the "Integration" role. A 403 with a valid token means a permission is missing.
3. If requests time out, ask your network team to allow our published IP ranges (Settings > Integrations > Network).

If the error persists after all three steps, send us the request id from the failing response and we will trace it on our side.
