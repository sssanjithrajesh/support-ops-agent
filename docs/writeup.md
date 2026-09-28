# Decomposition: an agent in the support queue

## 1. Who is the user, and what do they do today?

Two people.

The **support agent** opens the ticketing system in the morning to a queue sorted by age. For each
ticket they read it, decide which team owns it, decide how urgent it is, look up the article that
answers it, and write a reply that is mostly that article with a greeting. Around one ticket in ten
needs someone else right now: a security report, a legal threat, a company-wide outage. Those get
found when the agent gets to them, which on a busy day is hours later.

The **support lead** owns the rules: what counts as urgent, which tickets can never be answered by
a template, who gets paged. Today those rules live in a wiki page and in people's heads. When they
change, retraining is a meeting.

## 2. What does success look like?

In numbers a support lead would accept:

- Zero customer replies sent by software in situations where a human should have read the ticket
  first. This is the gate; nothing else matters if it is not zero.
- Every ticket that a human must see today is flagged the minute it arrives, not when someone
  reaches it in the queue. Measured as escalation recall on a labelled set.
- The draft attached to a review item is right often enough that approving is faster than writing:
  the right article chosen most of the time, and never a claim that is not in the article.
- No duplicate writes to the ticketing system, ever, including during rate limits and partial
  failures. A duplicate public reply is a customer-visible bug.
- The lead can change a rule without a deploy and see, per ticket, which rule fired.

## 3. What is the smallest thing that delivers that?

- A webhook receiver that checks the signature and ignores replays.
- Triage that produces a queue, a priority and a set of named signals, each with the text that
  matched.
- A knowledge base of a dozen articles and a search that says which words matched.
- A draft that is the article plus a template, so "grounded" is a property, not a hope.
- A decision function with three outcomes in a fixed order: escalate beats everything, auto-send
  must pass every check, review is the default.
- A review page where a named human approves, edits or rejects.
- An outbox where every write has an idempotency key, and a client that can tell whether a failed
  write actually landed.
- A sandbox that behaves like the real ticketing API, including its failures, so all of this can be
  tested end to end without credentials.

Not built, on purpose: a fine-tuned classifier, a vector database, multi-turn conversations, a
Zendesk app, per-agent permissions, a real Slack app with interactive buttons.

## 4. What did the data teach us that changed the design?

**The labels in the public dataset are not the truth.** The dataset labels each ticket with a queue
and a priority. A naive Bayes classifier trained on 1,500 tickets agrees with those labels on 36% of
500 held-out tickets. Training on 11,400 tickets moves that to 38%. Then 23 of the held-out tickets
were labelled by hand, and the dataset agrees with the human on 30% of them. A report of medical
records leaked through an insecure link is labelled "Human Resources, low". The classifier was not
the ceiling; the labels were.

This changed two things. First, the eval is built on human labels and reports the dataset agreement
as a separate, lower number, instead of pretending the dataset is ground truth. Second, routing got
a second input: **queue hints**, words a support lead associates with a queue, in the policy file.
Hints took agreement with the human labels from 47% to 80% and moved agreement with the dataset
labels from 36% to 33%. That trade is the right one and the write-up says so rather than hiding it.

**Naive Bayes is absurdly overconfident.** On a 60-token ticket the posterior is 99.99% for one
class, which drowned any hint. Dividing the log-likelihood by (tokens / 6) put the classifier and
the hints on the same scale. The number 6 was chosen on the 30 hand-labelled cases; the 500
held-out tickets are the check that it did no damage there (33% vs 36%).

**The dataset is one-ninth data breaches.** Over 2,000 tickets, 221 fire the security signal, and
sampling them shows they really are breach reports (medical records, phishing, unauthorised access).
The 15% overall escalation rate is therefore correct for this data and would be unacceptable in a
real queue, where the fix would be a separate security intake rather than a looser signal. The
signal does have one known false positive in the eval set (case-06, a feature request that mentions
"unauthorized access" hypothetically), kept in the set deliberately.

**Priority labels are close to noise.** The dataset's priority is 41% "medium"; the classifier gets
42% on held-out and 53% on the hand labels, with the boosts from signals doing most of the work on
the cases that matter (outage, breach, legal). The design leans on signals for urgency and treats
the classifier's priority as a weak prior.

**"Retry on 5xx" is wrong for ticket updates.** The first end-to-end run against the sandbox with
apply-then-500 fault injection produced four copies of the same comment. See section 6.

## 5. Tradeoffs we made on purpose

**Rules over learning for the decision.** The classifier suggests a queue; it never decides whether
a customer hears from software. That decision is a short list of named checks in YAML. Less clever,
fully explainable, and a support lead can change it at 4pm on a Friday.

**Review as the default.** 84% of the sample lands in review. That is deliberate for a first
deployment: the cost of a wrong auto-send is a customer incident; the cost of a needless review is a
click. The auto-send list is meant to grow queue by queue as the review verdicts show the drafts are
being approved unchanged.

**Grounded by construction, not by judging.** The deterministic draft is literally the article. The
LLM planner may write differently, but its draft must cite an article and the tool layer will route
an uncited draft to review. We did not build an LLM judge for groundedness; for this scope a string
check on the template is stronger and cheaper.

**Marker tags over a distributed lock or a dedupe service.** Zendesk updates have no idempotency
key, so the client writes one into the ticket's tags in the same request as the comment and reads
it back before any retry. It costs one GET per write and survives crashes between "remote accepted"
and "row marked sent". It also leaves a visible audit trail on the ticket.

**An in-process outbox worker.** A daemon thread drains the outbox every five seconds. In a real
deployment it is a separate process or a cron; here the point is that the agent never calls the
ticketing system inline, so a rate limit cannot fail a webhook.

**A sandbox instead of a live Zendesk developer account.** No credentials were available, and a
sandbox that injects faults on a schedule is a better test than a real API that mostly works. The
client is written against Zendesk's real endpoints, auth scheme, idempotency header and webhook
signature, so pointing it at a subdomain is a config change.

## 6. Failure modes and what we did about them

| Failure | Consequence | Mitigation | Status |
|---|---|---|---|
| Update applied, response lost (500 / timeout) | Duplicate comment on retry | Marker tag in the same PUT; read-back before retry; single PUT per call | Tested against sandbox; naive version kept as a failing-by-design test |
| Rate limit during a burst | Writes fail, agent errors | Retry-After honoured; jittered backoff; outbox reschedules with a longer curve | Tested |
| Forged or replayed webhook | Fake tickets, double processing | HMAC check with 5-minute window; event-id dedupe table | Tested (positive, flipped-character negative, stale timestamp, replay) |
| Planner ignores the protocol (skips decide, or asks to send anyway) | Unsupervised reply | Tools check the recorded decision; no decision or wrong decision -> review item | Tested with a scripted LLM client |
| Card number in ticket | PII in notes, Slack, drafts | Redaction on evidence, notes, drafts, summaries; `pii_in_ticket` blocks auto-send | Tested |
| Security signal fires on prevention advice | Needless escalation | Known; one case in the eval set; fix is a narrower pattern in policy.yaml | Documented, not fixed |
| Dataset queue labels noisy | Classifier looks bad, hints look like overfitting | Human labels for the eval; hints and temperature chosen on them; held-out reported alongside | Reported |
| Slack not configured | Lost escalations | Dispatcher records the fact; internal note still written; row visible in UI | Tested |
| Outbox row stuck failing | Silent backlog | 8 attempts then `failed`; count shown on the queue page | Tested; no alerting yet |
| Two review verdicts on one item | Double send | Item status checked; second verdict is a no-op; reply key is per ticket | Tested |
| Knowledge base has no matching article | Draft is a generic acknowledgement | `grounded=False`, kb score 0, auto-send impossible; goes to review | Covered by decision checks |

## 7. What we would do next, in order

1. Run the review queue with real agents for two weeks and record approve / edit / reject per
   queue and per article. Widen the auto-send list only where drafts are approved unchanged above
   an agreed rate.
2. Split security intake from support: a `security` action that pages a different channel and
   never drafts a customer reply, and a narrower pattern that distinguishes "we were breached" from
   "how do we prevent breaches".
3. Replace the naive Bayes with a small fine-tuned classifier once there are a few hundred human
   labels from step 1, keeping the hints as a veto layer and the eval as the gate.
4. A Slack app with approve / edit buttons so escalations and reviews can be handled where the
   on-call already is. Signature verification for inbound Slack is a copy of what the webhook does.
5. Multi-turn: when the customer replies, re-run triage on the thread and let the agent propose a
   follow-up, still through the same decision function.
6. Move the outbox worker to its own process, add a `failed > 0` alert, and put the sandbox fault
   schedule in CI so every PR runs the end-to-end test with faults on.
