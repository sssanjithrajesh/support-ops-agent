# Support Ops Agent

**Type:** 48-hour prototype.
**Target roles:** Customer-ops AI startups (Sierra, Decagon, Cresta), Solutions Engineer roles.

An agent that triages support tickets and drafts replies against a Zendesk or ServiceNow developer
sandbox, with Slack escalation. Reuses the integration patterns from `contract-compliance-reviewer`.

## Deliverables

- One MCP server for the ticketing system
- Idempotent writes, webhook signature verification, retry with jitter
- 30-case eval set with a reported pass rate
- Short write-up: what broke in the integration and how it was handled

## Status

Not started.
