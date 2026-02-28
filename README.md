# Pokemon Red RL with Mistral

The project explores LLM-guided battle policy behavior in Pokemon Red, with:
- runtime battle loops in `run.py`
- isolated evaluation harness under `evals/`
- cumulative run stats in `artifacts/campaign_log.json`

## Phases
- Phase 0/1: Setup of the emualator + save states
- Phase 2: Battle(s) + Battle Memory
- Phase 3: Navigation (RL)
- Phase 4: Beating Brock with the combination of all above

## Mistral Usage
- Evals: Landed on XXX model 
- LLM reasoning learning: What does it learn from each battles. (llm_decision_calls, llm_reflection_calls)
- Policy updates on what to do for battles + navigation
- Reward mechanism learning (EUREKA reward iteration)