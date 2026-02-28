# Pokémon Red RL Agent
**Mistral AI Hackathon — Project Specification**
`API Track` · `Mistral Large 3` · `Ministral 8B` · `PyBoy Emulator`

---

> **One-liner:** Build a Pokémon Red battle agent that genuinely learns from experience using Mistral — a self-improving system where the strategy prompt is the policy, battle outcomes are the reward, and Mistral's reflection is the policy update.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Tech Stack](#2-tech-stack)
3. [Phase 0 — Environment Setup](#phase-0--environment-setup-night-1-2-hrs)
4. [Phase 1 — Core RL Loop](#phase-1--core-rl-loop-saturday-8-hrs)
5. [Phase 2 — Richer Battle State](#phase-2--richer-battle-state-saturday-evening-2-hrs)
6. [Phase 3 — EUREKA Reward Design](#phase-3--eureka-reward-design-sunday-morning-3-hrs)
7. [Phase 4 — Fine-Tuning Track](#phase-4--fine-tuning-track-sunday-3-hrs-stretch)
8. [Weekend Timeline](#8-weekend-timeline)
9. [Decision Gates](#9-decision-gates)
10. [Deliverables](#10-deliverables)
11. [Hackathon Pitch](#11-hackathon-pitch)
12. [Risks](#12-risks)
13. [Appendix](#13-appendix)

---

## 1. Project Overview

### The Core Idea

Traditional RL trains a neural network by repeatedly nudging its weights towards actions that earned higher rewards. This project applies the same loop to a language model — except the "weights" are the system prompt, and the "gradient update" is Mistral reflecting on its battle history and rewriting its own strategy.

| RL Concept | Our Implementation |
|---|---|
| Policy | Mistral system prompt (the battle strategy) |
| State | Battle state read from RAM via PyBoy emulator |
| Action | Move choice: slots 0–3 (Scratch, Ember, Growl, Tackle) |
| Reward | HP remaining + win/loss outcome + turns taken |
| Policy Update | Mistral reflects on history, rewrites strategy |
| Episode | One complete battle from start to KO |
| Learning Signal | Win rate improving over 30 episodes |

### Why This Is Real RL

Three things make this a genuine reinforcement learning system, not just an LLM chatbot:

- The agent receives **no human guidance** during battles — only a reward signal after each episode ends
- The policy (prompt) changes **only based on observed outcomes**, not hand-crafted rules
- The system **improves measurably** over time: win rate increases as the strategy evolves

### Hackathon Track

| Track | Rationale |
|---|---|
| **API Track** | Primary — uses Mistral Large 3 + Ministral 14B via API |
| **Fine-Tune Track** | Stretch goal — Ministral 8B fine-tuned on synthetic data (+20 pts) |

---

## 2. Tech Stack

| Component | Technology |
|---|---|
| Emulator | PyBoy (Python Game Boy emulator, headless mode) |
| Game | Pokémon Red ROM + `battle_rattata.state` save file |
| Battle Agent | Mistral Large 3 via API (`mistral-large-latest`) |
| Strategy Updater | Mistral Large 3 — reflects and rewrites prompt |
| Fine-tune model | Ministral 8B (stretch goal) |
| Hardware | M4 MacBook Pro 24GB — MPS-accelerated, Ollama for local LLM |
| Language | Python 3.11+ |
| Key libraries | `pyboy`, `mistralai`, `numpy` |

### Key RAM Addresses (Pokémon Red)

| Address | Meaning |
|---|---|
| `0xD057` | In-battle flag (0 = overworld, 1 = in battle) |
| `0xD16C` | Player current HP |
| `0xD18D` | Player max HP |
| `0xD18C` | Player level |
| `0xD16B` | Player Pokémon species ID |
| `0xD01C–0xD01F` | Player move IDs (slots 0–3) |
| `0xCFE6` | Enemy current HP |
| `0xCFE8` | Enemy level |
| `0xCFE5` | Enemy Pokémon species ID |
| `0xD356` | Badge bitmask (bit 0 = Boulder Badge) |

---

## Development Phases

> **Rule:** Each phase must be working and demo-able before attempting the next. A polished Phase 1 beats a broken Phase 3 every time.

---

> ⚠️ **Goal:** Verify everything works before the hackathon starts. Do not skip this.

### 0.1 Install Dependencies

```bash
brew install sdl2
pip install pyboy gymnasium mistralai numpy
export MISTRAL_API_KEY=your_key_here
```

### 0.2 Verify PyBoy

```python
from pyboy import PyBoy

pyboy = PyBoy("pokemon_red.gb", window_type="headless")
pyboy.set_emulation_speed(0)

for _ in range(1000):
    pyboy.tick()

print(pyboy.get_memory_value(0xD362))  # x position — should be non-zero
print(pyboy.get_memory_value(0xD16C))  # player HP — should be non-zero
print("✅ PyBoy working")
```

### 0.3 Create Save State

Manually play Pokémon Red to the moment just before Brock's first Pokémon appears. Then:

```python
with open("battle_rattata.state", "wb") as f:
    pyboy.save_state(f)
print("✅ Save state created")
```

### 0.4 Success Criteria

- [ ] RAM address `0xD16C` prints a non-zero number (player HP)
- [ ] `battle_rattata.state` file exists and is > 0 bytes
- [ ] Mistral API key returns a valid response from a test call

---

## Phase 1 — Core RL Loop _(Saturday, ~8 hrs)_

> 🎯 **Goal:** A working end-to-end system where Mistral fights real battles in the real emulator and measurably improves over 30 episodes.

### 1.1 Emulator Wrapper — `PokemonEmulator`

Build a class wrapping PyBoy with these methods:

| Method | Description |
|---|---|
| `read(addr)` | Reads a single RAM address |
| `press(button, frames)` | Sends button input and ticks |
| `reset()` | Loads the save state |
| `in_battle()` | Returns bool from `0xD057` |
| `get_battle_state()` | Returns full state dict with player + enemy info |
| `execute_move(slot)` | Navigates fight menu and confirms move |
| `wait_for_battle(timeout)` | Ticks until battle flag activates |
| `wait_for_battle_end(timeout)` | Ticks until battle flag clears |

### 1.2 Mistral Battle Agent — `MistralBattleAgent`

| Attribute / Method | Description |
|---|---|
| `strategy` | String — starts generic, evolves each update cycle |
| `pick_move(state)` | Calls Mistral Large 3, parses `ACTION: <0-3>` from response |
| `record_battle(moves, outcome, hp, turns)` | Stores episode data to history |
| `update_strategy()` | **The RL update step** — Mistral reads history, rewrites prompt |
| `print_summary()` | Win rate graph + full strategy evolution log |

### 1.3 Reward Function

| Event | Reward |
|---|---|
| Win battle | +100 base |
| HP remaining on win | +1 per HP (up to +50 bonus) |
| Fewer turns on win | −2 per turn (efficiency bonus) |
| Lose battle | −100 base |
| HP remaining on loss | +0.5 per HP (partial credit for lasting longer) |

### 1.4 Main Loop

```python
agent = MistralBattleAgent(api_key, model="mistral-large-latest")
emu   = PokemonEmulator("pokemon_red.gb", "battle_rattata.state")

for episode in range(30):
    outcome = run_episode(emu, agent)

    if episode % 5 == 0 and episode > 0:
        agent.update_strategy()   # ← the RL update

agent.print_summary()
```

Strategy updates every 5 battles. Print the full strategy diff at each update so evolution is visible.

### 1.5 Success Criteria

- [ ] Agent fights 30 complete battles without crashing
- [ ] Strategy prompt changes at least 3 times over 30 episodes
- [ ] Win rate in episodes 26–30 is higher than episodes 1–5
- [ ] Mistral's reasoning prints to terminal during each battle

### 1.6 Common Issues + Fixes

| Problem | Fix |
|---|---|
| Battle never detected | Save state isn't close enough to battle start — replay and save later |
| Move execution wrong | Increase `hold_frames` in `execute_move()`, add more `tick()` after `'a'` |
| `ACTION` not parsed | Add fallback: return `0` if regex fails, log the raw response |
| API rate limit hit | Add `time.sleep(1)` between `pick_move()` calls |
| Wrong HP values | Verify ROM is Pokémon Red not Blue — some addresses differ |

---

## Phase 2 — Richer Battle State _(Saturday Evening, ~2 hrs)_

> 🎯 **Goal:** Give Mistral actual move names, types, and type effectiveness — so its reasoning becomes genuinely good, not just HP-based guessing.

### 2.1 Move + Species Lookup Tables

Extend the emulator to decode:

- Player move names, base power, and type from move ID (addresses `0xD01C`–`0xD01F`)
- Player and enemy species names from species ID (`0xD16B`, `0xCFE5`)
- Type effectiveness multiplier (2× / 0.5× / 1×) using Gen 1 type chart

### 2.2 Enhanced Prompt

Updated battle state passed to Mistral includes:

- Full move names with power, type, and effectiveness vs enemy type
- Pokémon species names (e.g. `Charmander vs Geodude`)
- HP as percentage as well as absolute values
- Turn count (to penalise stalling in prompt)

### 2.3 Success Criteria

- [ ] Mistral mentions move names in its reasoning (not just "move 0")
- [ ] Mistral references type matchups in at least 50% of decisions
- [ ] Strategy updates reference specific moves by name

---

## Phase 3 — EUREKA Reward Design _(Sunday Morning, ~3 hrs)_

> 🎯 **Goal:** Add the EUREKA loop for overworld navigation — Mistral iteratively writes and improves the PPO reward function.

> ⚠️ **Only attempt if Phase 1 + 2 are fully working and demo-able.**

### 3.1 PPO Navigation Environment

Wrap PyBoy as a Gymnasium environment:

| Property | Value |
|---|---|
| Observation | 8 RAM values: x, y, map ID, badges, HP, level, in_battle flag, visited tile count |
| Action space | 8 discrete buttons: up, down, left, right, A, B, start, select |
| Battle handling | Skip steps when `in_battle` flag is active — Mistral handles those |
| Reset | Load a navigation save state (e.g. start of Pallet Town) |

### 3.2 EUREKA Loop (4 iterations minimum)

```
1. Describe environment to Mistral Large 3
        ↓
2. Mistral writes reward.py
        ↓
3. Run PPO for 20–30 minutes
   Record: unique tiles visited, furthest map reached
        ↓
4. Show results to Mistral:
   "agent explored X tiles, got stuck at Y"
        ↓
5. Mistral critiques and rewrites reward.py
        ↓
   repeat from step 3
```

### 3.3 Metrics to Capture

| Metric | How to Measure |
|---|---|
| Unique tiles visited per episode | Set of `(map_id, x, y)` tuples |
| Furthest map reached | Maximum `map_id` value seen |
| Average episode length | Steps before reset |
| Reward per episode | Sum of rewards over episode |

### 3.4 Success Criteria

- [ ] Mistral has rewritten `reward.py` at least twice
- [ ] Exploration coverage measurably increases between reward versions
- [ ] Agent reaches at least Route 1 (map ID `0x0C`) in final reward version

---

## Phase 4 — Fine-Tuning Track _(Sunday, ~3 hrs, Stretch)_

> 🎯 **Goal:** Fine-tune Ministral 8B on synthetic battle data from Mistral Large 3. Upload adapter to HuggingFace for +20 bonus points.

> ⚠️ **Only attempt if Phases 1–2 are solid. Fine-tuning job takes 2–3 hrs — kick it off early.**

### 4.1 Synthetic Data Generation

Use Mistral Large 3 to generate 500–2000 labelled battle examples:

```python
# For each scenario, ask Large 3 to reason and pick optimal move
# Save in Mistral fine-tuning JSONL format:

{"messages": [
  {"role": "system",    "content": "You are a Pokémon battle expert..."},
  {"role": "user",      "content": "Battle state: Charmander vs Geodude..."},
  {"role": "assistant", "content": "Reasoning: Geodude is Rock type...\nACTION: 0"}
]}
```

Programmatically generate diverse scenarios: all type combos, various HP levels, level ranges 5–50.

### 4.2 Fine-Tuning Job

```python
# Upload training data
with open("pokemon_battles.jsonl", "rb") as f:
    uploaded = client.files.upload(
        file=("pokemon_battles.jsonl", f, "application/json")
    )

# Start job (runs on Mistral infrastructure, not locally)
job = client.fine_tuning.jobs.create(
    model="open-ministral-8b",
    training_files=[uploaded.id],
    hyperparameters={
        "training_steps": 100,
        "learning_rate": 1e-4
    }
)
print(f"Job started: {job.id}")
# Takes 2–3 hours. Start Saturday evening, check Sunday morning.
```

### 4.3 HuggingFace Upload

- Create public HuggingFace repository
- Upload fine-tuned adapter weights
- Write model card describing training data and task

### 4.4 Swap Into RL Loop

Replace `"mistral-large-latest"` with your fine-tuned model ID in `MistralBattleAgent`. Run 30 episodes with each and compare win rates — this comparison is a result in itself.

### 4.5 Success Criteria

- [ ] Fine-tuning job completes without error
- [ ] Adapter uploaded to HuggingFace with public access
- [ ] Fine-tuned model callable via API
- [ ] Win rate comparison documented: base Ministral 8B vs fine-tuned

---

## 8. Weekend Timeline

| Task | Priority |
|---|---|
| `brew install sdl2` + `pip install pyboy mistralai` | Required |
| Verify PyBoy loads ROM and RAM reads work | Required |
| Manually play to before Brock, create save state | Required |
| Confirm Mistral API key works | Required |

### Saturday (Full Day)

| Time | Task | Phase |
|---|---|---|
| 9am–11am | Build `PokemonEmulator`, test RAM reads | 1 |
| 11am–1pm | Build `MistralBattleAgent`, test `pick_move()` | 1 |
| 1pm–2pm | **Lunch** | — |
| 2pm–4pm | Wire main loop, run first 10 episodes, debug | 1 |
| 4pm–5pm | Add move names + type effectiveness data | 2 |
| 5pm–6pm | Run full 30 episodes, capture win rate data | 1 |
| 6pm–8pm | If solid: generate synthetic data for fine-tuning | 4 |
| 8pm+ | If data ready: kick off fine-tuning job overnight | 4 |

### Sunday (Full Day)

| Time | Task | Phase |
|---|---|---|
| 9am–10am | Review Saturday results, fix overnight issues | — |
| 10am–1pm | Attempt EUREKA loop if Phase 1+2 are solid | 3 |
| 1pm–2pm | **Lunch** | — |
| 2pm–3pm | Upload fine-tuned model to HuggingFace if ready | 4 |
| **3pm** | **Hard stop — polish whatever level you're on** | — |
| 3pm–5pm | Build demo: win rate graph + strategy evolution | — |
| 5pm+ | Record demo video, prepare presentation | — |

---

## 9. Decision Gates

At each checkpoint: **"Is this working well enough to demo confidently?"**
- **No** → stay and fix it, this phase is the project
- **Yes** → proceed to the next phase

| Time | Gate Question |
|---|---|
| Saturday 12pm | Can I run a battle and see Mistral's reasoning in the terminal? |
| Saturday 4pm | Is the 30-episode RL loop running and win rate improving? |
| Saturday 6pm | Is Phase 1+2 solid enough to leave overnight? |
| Sunday 10am | Is Phase 1+2 demo-ready? Only then attempt EUREKA. |
| **Sunday 3pm** | **Hard stop. Whatever phase is working — polish it now.** |

---

## 10. Deliverables

### Minimum (Phase 1+2 complete)
- `pokemon_rl.py` — single runnable file (~400 lines)
- `battle_rattata.state` — PyBoy save state
- `battle_results.json` — 30 episodes of outcome + strategy data
- Win rate progression (ASCII graph in terminal output)
- Strategy evolution log — all prompt versions side by side

### Extended (Phase 3 complete)
- `reward_v1.py` through `reward_v3.py` — EUREKA reward iterations
- `exploration_results.json` — unique tiles visited per reward version
- Exploration coverage comparison across reward versions

### Stretch (Phase 4 complete)
- `pokemon_battles.jsonl` — synthetic training data (500+ examples)
- HuggingFace model card URL
- Win rate comparison: base Ministral 8B vs fine-tuned model

---

## 11. Hackathon Pitch

> *"We built a Pokémon Red battle agent that genuinely learns from experience using Mistral Large 3. The system prompt is the policy. Battle outcomes are the reward. Mistral's reflection on its own history is the policy update. Over 30 episodes, win rate against Brock improves from X% to Y% — with every strategy change driven entirely by observed outcomes, not human-crafted rules."*

### RL Concept Map for Judges

| Traditional RL | This Project |
|---|---|
| Neural network weights | Mistral system prompt |
| Gradient descent update | Mistral reflection + rewrite |
| Gym environment | PyBoy emulator + RAM reads |
| Reward signal | HP remaining + win/loss + efficiency |
| Training loop | 30 battle episodes |
| Policy improvement | Win rate increasing over time |

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| PyBoy setup fails on Mac  |
| RAM addresses wrong for ROM | Verify HP address prints sensible value before running |
| Battle flag timing off | Add generous `tick()` buffer after state transitions |
| API rate limits | Add `sleep(1)` between calls; use Ollama local as backup |
| Win rate doesn't improve | Check reward function logic; lower `UPDATE_EVERY` to 3 |
| Fine-tuning job fails | Skip Phase 4, submit Phase 1+2 — still a strong project |
| Running out of time | Every phase is independently demo-able. Stop at any level. |

---

## 13. Appendix

### Mistral Models Used

| Model | Role |
|---|---|
| `mistral-large-latest` | Battle decisions + strategy reflection |
| `mistral-large-latest` | EUREKA reward function generation (Phase 3) |
| `mistral-large-latest` | Synthetic data generation (Phase 4) |
| `open-ministral-8b` (fine-tuned) | Fine-tuned battle agent — stretch goal |

### Key Files

| File | Purpose |
|---|---|
| `pokemon_rl.py` | Main runnable script — all phases in one file |
| `battle_rattata.state` | PyBoy save state — reset point for every episode |
| `battle_results.json` | All episode data: outcomes, moves, rewards, strategy log |
| `pokemon_battles.jsonl` | Synthetic fine-tuning data (Phase 4 only) |
| `SPEC.md` | This document |

### Useful References

- **EUREKA paper:** [eureka-research.github.io](https://eureka-research.github.io)
- **PyBoy docs:** [github.com/Bonsai/PyBoy](https://github.com/Bonsai/PyBoy)
- **Pokémon Red RAM map:** [datacrystal.tcrf.net](https://datacrystal.tcrf.net/wiki/Pok%C3%A9mon_Red/Blue)
- **Mistral fine-tuning docs:** [docs.mistral.ai/guides/finetuning](https://docs.mistral.ai/guides/finetuning)
- **Mistral Agents API:** [docs.mistral.ai/agents/introduction](https://docs.mistral.ai/agents/introduction)

---

*Pokémon Red RL Agent · Mistral AI Hackathon · API Track*