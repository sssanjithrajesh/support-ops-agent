### Scorecard: `deterministic` planner

| expected action | n | queue acc | priority acc | action acc | escalation recall | kb top-1 | unsafe auto-send |
|---|---:|---:|---:|---:|---:|---:|---:|
| auto_send | 2 | 100% | 50% | 100% | 0% | 100% | 0% |
| escalate | 9 | 100% | 89% | 100% | 100% | 86% | 0% |
| review | 19 | 68% | 37% | 95% | 0% | 88% | 0% |
| **overall** | **30** | 80% | 53% | 97% | 100% | 88% | 0% |

Dataset label agrees with the human label on 30% of the dataset-sourced cases.

Held-out 500 tickets against the dataset's own labels: queue accuracy 33% (majority-class baseline 28%), priority accuracy 42%. Actions: auto_send 2, escalate 77, review 421.
