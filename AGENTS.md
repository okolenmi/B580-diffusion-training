# Rules for AI agents working in this repository

## What matters
1. **Do not "overtest".** Tests usage should have sense. Don't run them when unnecessary (for example, after docs edit) 
2. **Do not simplify the goals.** Good result after a long time wins over fast simple solutions without much future.
3. **OOP approach is recommended.** Avoid any "spaghetti code". Separeted functional independent classes is a better 
   approach if you know what you are doing. Medium efforts in this direction.

4. **The previous implementation is good.** `nodes/`, `manager/`, `backend/` and `docs/` hold a polished,
   hardware-tested system (memory levers, B580 findings, known issues). Only its structure was messy. Before
   writing or "investigating" anything performance- or memory-related, read what the old code and `docs/` already
   say, and port that knowledge; do not rediscover it by testing again.

## Tests: keep it clean
- Remove what is already useles or have almost no value. Don't be "sentimental" about it.
- It's recommended to group tests somehow, or at least write purpose of a test in docstring at top.
- Do your best even if it's already a mess. Step by step situation will improve.

## Code and comments
- Comments explain a non-obvious *why*. No history ("moved from"), no measurements, no incident stories in code.
  Use appropriate doc files instead.
- Keep names honest and short; fix a bad name now, not later.

## If you have no access to XPU (sandbox case)
- You can rely on user to help you with testing. Add items which need to be tested in summary.
- (For Claude specifically) At most one sub-agent at a time. User have about 100k tokens per 5 hours. Do not burn them 
  on tests, re-reading, or long reports.
