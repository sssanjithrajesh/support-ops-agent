# Attribution

## Ticket dataset

`data/raw/tickets_sample.json.gz` is a 2,000-row sample of
[Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)
(file `dataset-tickets-multi-lang-4-20k.csv`), licensed CC BY-NC 4.0. English rows only, whitespace
normalised, sampled with seed 42 by `scripts/fetch_tickets.py`. The dataset is synthetic; it contains
no real customers. This repository is a non-commercial portfolio project.

Columns kept: subject, body, answer (as `reference_answer`), type, queue, priority, tags. The `queue`
and `priority` columns are used as training labels for the classifier and as a comparison point in the
evals; the write-up discusses how noisy they are.

## Everything else

Knowledge-base articles in `kb/`, the seven handwritten eval cases in `evals/cases/hand_labelled.yaml`,
the human labels on the 23 dataset cases, `config/policy.yaml`, and all code are original to this
project (MIT).

Patterns reused from the sibling project `contract-compliance-reviewer`: the HTTP client with retry
and jitter, the durable outbox, and the eval scorecard/report structure.
