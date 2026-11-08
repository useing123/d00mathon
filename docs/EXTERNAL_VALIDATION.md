# External validation on a public, human-labelled customer-support dataset

**Question.** Does our evaluator agree with *independent human annotators* on *real* customer-support conversations it has never
seen, when it scores them on *their* rubric? This is a proxy check: different company, domain (e-commerce furniture/retail
chat, not brokerage), language (EN) and rubric. It is not a measurement on the Freedom criteria with Freedom labels (see
docs/HUMAN_VALIDATION.md for that).

> **Validation status (external, public human labels):** On 68 real customer-support chats from MAIA-DQE (CC BY-ND, annotators' own rubric), the default jury reaches κ 0.63 on dropped/concluded (82 % accuracy; human–human 90 %) and Spearman ρ 0.58 on 1–5 task success; the Freedom rubric's 04 tracks their task success with ρ 0.55.
> Politeness is weak (κ 0.23, impolite recall 19 %; our inspection suggests noisy labels). This is a proxy: EN, e-commerce, a different rubric. See docs/EXTERNAL_VALIDATION.md.

## 1. Dataset search (shortlist)

Searched the HF Hub API (`customer support`, `customer service`, `call center`, `contact center`, `support chat`,
`user satisfaction`, `politeness`, `empathy dialog`), GitHub and the web (EN + RU queries). Most HF "customer support" sets are
synthetic (Bitext, many `*synthetic*` dumps) or ticket/intent sets **without quality labels**. No public RU/KZ support
dataset with human quality labels was found (RU contact-center sets on HF are audio or intent-only, e.g.
`mkaiuse/Support_Contact_Center_RU_27000`; Habr posts by banks describe internal, unreleased gold sets).

| rank | dataset | real dialogs? | human labels (level) | match to our criteria | lang | licence |
|---|---|---|---|---|---|---|
| **1** | **MAIA-DQE** — Mendonça et al., GEM 2023, [paper](https://aclanthology.org/2023.gem-1.2) · [data](https://github.com/johndmendonca/MAIA-DQE) | yes — real customer-support chats of 4 companies (Unbabel MAIA), agent EN ↔ customer DE/PT-BR/PT-PT with MT | dialogue: Task Success 1–5, Dropped conversation 0/1; turn: Politeness, Sensibleness, Understanding 0/1, IQ 1–5; sentence: fluency, engagement, emotion (incl. agent empathy) | TS → 04, Politeness → 05/06 (and 08 only for gross rudeness), Dropped → 07 NA, Sensibleness → 04/06 | EN (+DE/PT originals) | CC BY-ND 4.0 |
| 2 | **USS — User Satisfaction Simulation**, Sun et al., SIGIR 2021, [repo](https://github.com/sunnweiwei/user-satisfaction-simulation) | JDDC subset: real JD.com customer service (3,300 dialogs); SGD/MultiWOZ: crowd-written | user satisfaction 1–5 per turn and per dialog, **3 annotators each** (+ written explanations on JDDC) | satisfaction ≈ 04 + 05 (outcome, not agent behaviour) | ZH (JDDC), EN | no licence file in the repo; JDDC has its own terms → research only, not redistributable |
| 3 | ABCD — Chen et al., NAACL 2021, [repo](https://github.com/asappresearch/abcd) | human-human role-play of agents following company guidelines | action/flow annotations (guideline compliance), no quality ratings | process compliance (05) only indirectly | EN | MIT |
| 4 | TWEETSUMM (IBM, EMNLP-F 2021) / Customer Support on Twitter (Kaggle) | real brand–customer tweets | summaries only / none | — | EN | CDLA / CC BY-NC-SA |
| 5 | Politeness-annotated Multi-DoGo (Mishra et al., Neurocomputing 2022) | crowd-written agent–customer | politeness per utterance | 05/06 | EN | not publicly downloadable (on request) |
| – | Stanford politeness, EmpatheticDialogues, Dusha (RU emotion) | not support dialogs | politeness / empathy / emotion | weak | EN/RU | CC BY / CC BY-NC / CC BY |

**Choice: MAIA-DQE.** It is the only one with real support conversations *and* dialogue-level human quality labels that map
onto our criteria, plus published inter-annotator agreement. USS-JDDC is the runner-up (real support, 3 raters per dialog)
but measures *customer satisfaction*, not agent compliance, is in Chinese, and has unclear licensing.

## 2. Method

* **Loader** `cqa/ext_data.py`: MAIA JSON → our transcript format. Agent turns keep the agent's original English (`text_src`),
  customer turns use the dataset's English MT (`text_mt`). Labels are derived only from the source annotations:
  `task_success` = dialogue Task Success (1–5); `dropped` = dialogue Dropped conversation; `agent_polite` / `agent_sensible` =
  minimum of the turn labels over agent turns (NO if *any* agent turn is 0). Rows with source value −1 are dropped.
* **Rubric** `checklists/ext_maia_dqe.yaml`: four criteria whose text is copied from the paper's annotation guidelines
  (§3.2.2–3.2.3): TS (scale 1–5), DC (YES = reached its end), POL, SENS. The model never sees the labels: the evaluated file has
  transcripts only, and the labels are in `results/external/maia_sample_labels.json`.
* **Sample**: 544 dialogs in 5 files; 43 are duplicates (`pt_br_client_02` contains `pt_br_client_04`, with identical labels), leaving 501 unique.
  Fixed-seed (42) stratified sample: all impolite-agent dialogs (rare negative class), then round-robin over
  TS bucket {1–2, 3, 4–5} × dropped {0,1}. **n = 68 unique dialogs** (70 drawn, 2 duplicate ids removed); human labels in the sample: TS 1/2/3/4/5 = 7/15/23/3/20,
  dropped 30, impolite 16, insensible 19.
* **Systems**: (1) default product jury (Haiku + Sonnet → Opus judge) on THEIR rubric; (2) self-hosted Qwen3.8-27B-FP8 on vLLM
  (RunPod A40): `local` (full generation) and `cascade-local` (vLLM guided-choice decision server tier 1 → Haiku → Sonnet);
  (3) our **Freedom rubric** (`freedom_chat_v1`, single Haiku model for cost) on the same dialogs, with agreement measured only on mapped criteria.
* **Metrics**: accuracy, Cohen's κ, quadratic-weighted κ / Spearman ρ / within-1 for the 1–5 scale, precision/recall of the
  negative class; 95 % percentile bootstrap CIs over dialogs (2,000 resamples, seed 7). Script: `scripts/external_bench.py`.

**Human ceiling (from the paper, Table 2 / §3.3).** 3 annotators on 10 PT-PT dialogs, observed % agreement, all three agreeing:
Task Success **30 %** full (60 % partial), binarised **80 %**; Dropped **90 %**; Politeness **98.4 %** (turn level), Sensibleness **92.1 %** (turn level).
The authors do not report κ ("due to the class imbalance … uninformative"). Turn-level agreement is not comparable with our
dialogue-level aggregate, and the n = 10 ceiling is itself very noisy.

## 3. Results (n = 68, 95 % bootstrap CI)

### 3a. Our system on THEIR rubric

| setup | TS ρ | TS QWK | TS exact / ±1 | TS≥4 acc (κ) | DC acc | DC κ | POL acc | POL κ | impolite P / R | SENS acc | SENS κ | $ (70 dialogs) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jury (Haiku+Sonnet→Opus) | 0.578 [0.41, 0.71] | 0.449 [0.29, 0.59] | 0.397 / 0.779 | 0.765 (0.38) | 0.824 [0.73, 0.91] | 0.634 [0.44, 0.81] | 0.794 | 0.227 [-0.01, 0.49] | 0.75 / 0.19 | 0.647 | 0.245 [0.02, 0.47] | 5.1464 |
| local: Qwen3.8-27B-FP8 vLLM generation | 0.616 [0.46, 0.73] | 0.571 [0.41, 0.70] | 0.485 / 0.765 | 0.779 (0.48) | 0.794 [0.69, 0.88] | 0.567 [0.36, 0.76] | 0.779 | 0.093 [0.00, 0.30] | 1.0 / 0.06 | 0.691 | 0.103 [-0.14, 0.34] | 0.0 |
| cascade-local: Qwen3.8 guided-choice → Haiku → Sonnet | 0.638 [0.48, 0.75] | 0.510 [0.34, 0.65] | 0.471 / 0.779 | 0.779 (0.43) | 0.706 [0.60, 0.81] | 0.373 [0.17, 0.57] | 0.779 | 0.093 [0.00, 0.30] | 1.0 / 0.06 | 0.529 | 0.137 [-0.06, 0.33] | 0.05468 |
| human–human (paper, 3 raters, 10 dialogs) | – | – | 0.30 all-3-agree (0.60 two-of-3) | 0.80 | 0.90 | – | 0.984 (turn) | – | – | 0.921 (turn) | – | – |

TS exact / ±1 = exact accuracy / share within one step. "impolite P / R" = precision / recall for the human class "some agent turn impolite" (16/68).
The local runs cost $0 in API fees (6 Haiku/Sonnet escalations in the cascade = $0.055); the RunPod A40 cost **$0.52** for the whole session
(resume, vLLM install + 29 GB weights download, ~10 min to serve, both runs; balance $2.518 → $1.998). Tier 1 decided 274/280 criteria in
the cascade (the router logs tier 1 as "jev-1.14", but `CQA_JEV_*` pointed it at our decision server with Qwen3.8-27B-FP8).
The `local` run was done twice by accident (concurrently, same output path); the run kept here was started with `--fresh` (no cache).
Its duplicate scored a mean total of 78.11 against 77.31, which is a rough sign of run-to-run noise. Each setup otherwise ran once.

### 3b. Our Freedom rubric (freedom_chat_v1, Haiku) vs their labels on mapped criteria

| our criterion ↔ their label | metric | value |
|---|---|---|
| 04 consultation (0–20) ↔ Task Success (1–5) | Spearman ρ | **0.549** [0.339, 0.722] |
| 04 ≥ 15 ↔ TS ≥ 4 ("resolved") | accuracy / κ | 0.721 [0.618, 0.824] / 0.407 [0.185, 0.628] |
| 05 (politeness + empathy) ↔ agent politeness | accuracy / κ; recall of impolite | 0.691 / 0.282 [0.048, 0.516]; recall 0.625 (10/16), precision 0.40 |
| 08 gross error ↔ agent impoliteness | flags | 0 / 68 flagged (by design: no MAIA impoliteness here reaches "insult / off-platform / data leak") |
| 07 = NA ↔ Dropped conversation | accuracy | 0.559: 07 was never NA (36 NO, 32 YES); our 07 rules give NA only if the operator *had no chance* to close |

## 4. Error analysis (10 random disagreements, jury, seed 7)

| id (MAIA) | crit | human | system | system rationale (abridged) | category |
|---|---|---|---|---|---|
| de-01 2021-01-10-47 | DC | ended | dropped | "problem not solved, last operator turn asks for data again without a conclusion" | rubric ambiguity ("without a conclusion" vs "dropped") |
| pt-br-04 2021-01-31-5 | DC | dropped | ended | "operator confirmed e-mail and closed with a promised follow-up" | rubric ambiguity (annotators treat an unresolved hand-off as dropped) |
| de-02 2021-01-31-7 | DC | dropped | ended | "closing line, customer said bye, though problem not solved" | rubric ambiguity (same) |
| pt-br-04 2021-01-17-0 | SENS | yes | no | "template 'chat closing' while client present; [16] generic answer to the credits question" | model stricter than annotators (plausible model error) |
| de-01 2020-12-27-31 | SENS | yes | no | "turn 10 ignored the replacement question, repeated greeting and re-asked details" | label noise likely (the evidence is in the transcript) |
| de-01 2021-02-21-3 | SENS | yes | no | "asks if client is still there right after the client answered, then closes chat" | label noise likely |
| de-01 2021-02-21-3 | TS | 5 | 3 | "colour only with storage; alternative left unanswered; chat closed" | model error / ambiguity (partial vs full) |
| de-01 2021-01-03-6 | TS | 5 | 4 | "solved, but extra verification steps → 4" | rubric ambiguity (4 vs 5 "unnecessary turns") |
| de-01 2021-01-10-18 | TS | 2 | 3 | "gave dispatch date, promised compensation later — partial" | rubric ambiguity (2 vs 3) |
| de-01 2021-01-24-14 | TS | 5 | 4 | "solved, but confusion at the start and unanswered contact question" | rubric ambiguity (4 vs 5) |

Count: rubric ambiguity 7, likely label noise 2, model error 1–2. The sample above contains no POL case, so we inspected all 14 POL
disagreements: 13 are "human impolite / system polite". In most of them the agent turn that annotators marked as impolite is
neutral or polite on its face, e.g. *"Is there anything else I can help you with today please?"*, *"You're welcome! Please let me
know if there's anything else I can do for you."*, *"How can I help with this order please?"*. Others are curt but not rude
(*"The steps are given to follow."*, *"The name on the card?"*). These are a mix of label noise (or labels shifted to the neighbouring
turn) and a rubric difference: MAIA's 0 = "not at worst civil", while our models count curt-but-civil as polite. The one
"system impolite / human polite" case (*"you have already said that"*, *"no this order is not in your name"*) is defensible. All
disagreements: `results/external/maia_disagreements_all.json`.

## 5. Interpretation

**What this shows.** On real support chats from a different company and annotation team, scored on the annotators' own
rubric, the default jury agrees with humans well above chance on outcome and flow:
dropped-vs-concluded κ 0.63 (82 % accuracy; human-human 90 %), task-success rank correlation ρ 0.58 (78 % within ±1; human-human exact agreement is only 30 %),
binarised success 76.5 %. The production Freedom rubric, without any adaptation, ranks the same dialogs in the same order as human Task Success
(04 ↔ TS ρ 0.55). Self-hosted Qwen3.8-27B-FP8 does as well on Task Success (ρ 0.62–0.64, QWK 0.51–0.57; the CIs overlap with the jury) but worse on Dropped (κ 0.57 for generation, 0.37 for the cascade). Its cost is about $0.5 of GPU time compared with $5.1.

**What it does not show.** (1) It is not Freedom data, not RU/KZ, and the human labels are not our checklist: it does not validate the
08 gross-error detector, 01/03 (greeting/identification) or the RU/KZ language handling. (2) Politeness and Sensibleness have low agreement on THEIR rubric: POL κ 0.09–0.23, impolite recall 6–19 %; SENS κ 0.10–0.25. Our inspection points mostly to label noise and a stricter definition of "impolite" in MAIA, but we have not adjudicated it, so this
remains an open weakness. With the Freedom rubric, 05 catches 10/16 impolite dialogs at precision 0.40. (3) The system is
systematically conservative on TS (47/68 rated 3; humans give 5 to 20/68), so its absolute 1–5 calibration differs from these annotators.
(4) n = 68, single run per setup, wide CIs; the ceiling comes from 10 dialogs.

## 6. Reproduce

```bash
unset CQA_MODEL CQA_ENSEMBLE CQA_JURORS CQA_JUDGE
python scripts/external_bench.py prepare --n 70 --seed 42          # downloads MAIA-DQE to data/external/ (git-ignored)
python scripts/external_bench.py run --which ext --preset jury      # their rubric, default jury     (~$5.1)
python scripts/external_bench.py run --which freedom --model claude-haiku-5-5   # Freedom rubric (~$0.56)
# self-hosted (vLLM Qwen3.8-27B-FP8 + decision_server.py vllmchoice, see docs/KZ_MODELS.md "Decision models"):
CQA_OPENAI_BASE_URL=http://127.0.0.1:18000/v1 CQA_OPENAI_API_KEY=$TOK CQA_ROUTER_LOCAL_MODEL=qwen3.8-27b-fp8 \
CQA_JEV_BASE_URL=http://127.0.0.1:18101/v1/decisions CQA_JEV_API_KEY=$TOK CQA_JEV_MODEL=qwen3.8-27b-fp8 \
  python scripts/external_bench.py run --which ext --preset cascade-local   # and --preset local
python scripts/external_bench.py metrics                            # -> results/external/maia_metrics.json
```

Committed: ids + human labels (`maia_sample_labels.json`), per-dialog system values + short rationales (`maia_*_run*.json`),
metrics, disagreements. MAIA transcripts are not redistributed. Data © the MAIA-DQE authors, CC BY-ND 4.0; cite Mendonça et al. 2023.
Note: `long_wait_seconds: 300` was added to `ext_maia_dqe.yaml` after the jury run (the decision-model path needs it; MAIA has no
timestamps, so it does not affect any rubric text); the jury run therefore has a different checklist version hash.

**Cost of this study:** Anthropic $6.53 (jury $5.15, Freedom/Haiku $0.56, cascade escalations $0.055, two 3-dialog pilots $0.78); RunPod $0.52.
