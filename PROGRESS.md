# Project Progress Log — Agent Intent Guard

ML-based detection of manipulated transaction intent in blockchain-connected AI agents. Compiled 2026-09-01, updated 2026-09-03 with the compound-attack / adversarial-training result and the held-out OOD generalization test. Everything below is either a directly measured result from this project's own scripts, or explicitly marked as an approximate figure pending a final confirmatory rerun.

## 1. Research direction and positioning

The project trains and evaluates a machine-learning classifier that detects manipulated transaction intent in the memory/context an AI agent uses when deciding how to act on a crypto/blockchain task — a lightweight, pre-signing content check, distinct from cryptographic authorization or transaction-signing security. It is positioned against three verified 2026 papers, none of which occupy the same space:

| Paper | Approach | Trained ML detector? | Relation to this work |
|---|---|---|---|
| arXiv 2503.16248 | Demonstrates the memory-injection attack; introduces CrAIBench benchmark; evaluates prompt-based and fine-tuning defenses | No | Source of the attack framing and the dataset used here; does not report a detector, so no recall/FPR baseline exists to compare against |
| arXiv 2511.15712 | Non-ML cryptographic / protocol verification of agent transactions | No | Different guarantee class (proof of protocol compliance, not content classification) — complementary, not competing |
| arXiv 2605.01143 | General-purpose behavioral-trajectory XGBoost detector for agent misuse | Yes, but never applied to crypto/blockchain | Closest prior ML work in spirit; this work is the first to apply a trained ML detector specifically to blockchain-agent transaction-intent manipulation |

Data source: SentientAGI/crypto-agent-safe-function-calling (Hugging Face) — 2,199 crypto-domain "memory injection" examples. Manual reading of the underlying text (the dataset's own label field only distinguishes "memory injection" from other categories) surfaced three qualitatively distinct manipulation strategies used as the threat model: fund/data redirection to an attacker-specified address, tool substitution or exclusion, and fabricated security-instruction framing.

## 2. Feature engineering

Six features are computed per (query, memory) pair:

- **memory_len_capped** — raw memory length in characters, capped at 500 (cap alone was later shown NOT to fix the length-padding vulnerability — see Section 4).
- **injection_keyword_score** — regex/keyword-based scoring of known injection trigger phrases.
- **num_foreign_addrs / has_foreign_addr** — count and presence of crypto addresses in the memory that do not match the query's own address; regex covers Ethereum hex, Bitcoin legacy base58, Bitcoin bech32, plus a structural "labeled as an address" cue for non-standard fake addresses.
- **query_memory_similarity** — sentence-embedding cosine similarity between query and memory (all-MiniLM-L6-v2), computed via max-pooling over per-sentence embeddings rather than one whole-text embedding — this specific design choice is the fix for the length/dilution vulnerability in Section 4.
- **starts_with_imperative** — grammatical mood check on the memory's first word; added to close a residual false-positive gap after the domain-confound fix.

## 3. Domain confound — found and fixed

Early models trained with a benign class drawn only from a different topic domain than the attack class learned "is this crypto-topic text" rather than "is this manipulated," caught by a dedicated in-domain benign diagnostic set (300 genuine, non-malicious crypto-domain examples) that both XGBoost variants initially failed almost completely:

| Stage | XGBoost (full features) FPR | XGBoost (no keyword) FPR |
|---|---|---|
| Cross-domain-only benign class (initial, flawed design) | ~100% | ~79% |
| After mixing in genuine in-domain benign examples | 12–15% | 12–15% |
| After fixing template substring bug + adding imperative-mood feature | < 1.5% | < 1.5% |

Fix: rebuilt the benign class so each crypto query pairs with either a cross-domain benign memory (50%) or a genuine in-domain benign memory drawn from neutral / tool-trust / self-consistent templates (50%). The residual 12–15% FPR after that first fix traced to one template ("No overrides or special handling…") accidentally containing the keyword-scored substring "override"; rewording it, plus adding the imperative-mood feature, brought FPR under 1.5%.

## 4. Baseline results and adversarial robustness

### 4.1 Clean-data results and paraphrase robustness

| Model | Attack recall | FPR (in-domain) | Paraphrase recall | Notes |
|---|---|---|---|---|
| Keyword-rule baseline | 0.874 | 0.000 | 0.000 | Illustrative weak baseline — 100% evasion under trigger-word paraphrasing, as expected |
| XGBoost, full 6 features | 0.988 | 0.015 | 0.932 | Strongest raw numbers; relies partly on keyword feature |
| XGBoost, no keyword feature | 0.968 | 0.028 | 0.968 | More robust to paraphrasing; recommended model going forward |

*Precision/recall/F1/AUC/in-domain-FPR are the 10-seed mean from `multi_split_eval.py` (`final_results_20260901_205811.txt`); paraphrase recall is the single confirmatory run from `adversarial_test.py`, same file. Verified 2026-09-03 against the raw results file directly, not taken from any summary.*

### 4.2 Length-padding / context-dilution attack — first fix attempt failed

Appending irrelevant filler text (context-dilution / length-padding) to attack memories was tested as a second adversarial capability. Capping memory_len alone did NOT fix it — recall for the no-keyword XGBoost model still collapsed under heavy padding, nearly identically to the uncapped case:

| Padding level (filler repeats) | 0 | 10 | 25 | 50 | 100 |
|---|---|---|---|---|---|
| XGBoost, no-keyword (raw, uncapped length) | 0.966 | 0.823 | 0.807 | 0.807 | 0.807 |
| XGBoost, no-keyword (length capped at 500 chars only) | 0.970 | 0.818 | 0.807 | 0.807 | 0.807 |

*This was reported as a negative result rather than glossed over — the cap-only "fix" clearly does not solve the underlying problem.*

### 4.3 Root cause isolated: embedding dilution, not length

Direct measurement of the similarity feature itself (not just downstream recall) proved the actual mechanism: the old whole-text-embedding similarity collapses under padding because appending long filler text shifts the whole document's embedding vector away from the short, topically-focused query. The real fix — computing similarity via max-pooling over per-sentence embeddings instead of one whole-text embedding — was then measured directly, before and after, on the same padding schedule:

| Padding level | Old whole-text similarity (mean) | New max-sentence similarity (mean) |
|---|---|---|
| pad = 0 | 0.3549 | 0.3617 |
| pad = 10 | — | 0.3611 |
| pad = 25 | — | 0.3611 |
| pad = 50 | — | 0.3611 |
| pad = 100 | 0.1808 | 0.3611 |

The old similarity feature drops 49% (relative) from pad=0 to pad=100. The new max-sentence similarity moves 0.17% (relative) across the identical range — this is a direct, mechanistic confirmation that the fix addresses the actual cause, not just a symptom.

*Confirmed by the final rerun (`test_length_evasion.py`, `final_results_20260901_205811.txt`, verified 2026-09-03): with the max-sentence-similarity fix applied, padding-only attack recall is stable at 0.952 (XGBoost full) / 0.923 (no-keyword) from pad=10 through pad=100 — no collapse, versus the 0.807 floor measured with the cap-only fix above. In-domain FPR is 0.015 (full) / 0.028 (no-keyword) — the earlier approximate 1.3–3.0% estimate holds. This is the disclosed sensitivity/specificity trade-off from the more sensitive similarity computation, not hidden or averaged away.*

## 5. Model interpretability (SHAP)

Rerun against the final feature set (memory_len_capped + max-sentence query_memory_similarity), matching exactly what `train_model.py` and `multi_split_eval.py` use. Global importance, mean |SHAP value| across the test set:

| Feature | Mean \|SHAP value\| |
|---|---|
| injection_keyword_score | 3.5950 |
| starts_with_imperative | 2.1744 |
| memory_len_capped | 2.1433 |
| query_memory_similarity | 2.0555 |
| num_foreign_addrs | 0.5578 |
| has_foreign_addr | 0.0000 |

**Finding: has_foreign_addr is redundant.** It contributes exactly 0.0000 — not small, zero, across the entire test set. This is consistent with it being a strict binary transform of num_foreign_addrs already in the model, so XGBoost never finds a split where it adds information. Documented rather than silently dropped: retained for analyst interpretability (a binary flag reads faster in an audit log than a count), with SHAP confirming it does not affect classifier decisions.

**Worked example: an attack caught without keywords or addresses.** A correctly-flagged true-positive attack instance with injection_keyword_score = 0.0 and num_foreign_addrs = 0 — an attack the keyword-rule baseline would almost certainly have missed — was still caught by the ML model, driven primarily by imperative sentence structure and memory length:

| Feature | Value | SHAP contribution |
|---|---|---|
| starts_with_imperative | 1.0 | +5.9531 |
| injection_keyword_score | 0.0 | -1.8363 |
| memory_len_capped | 157.0 | +1.7416 |
| query_memory_similarity | 0.2912 | -0.4494 |
| num_foreign_addrs | 0.0 | -0.4170 |
| has_foreign_addr | 0.0 | +0.0000 |

base_value = -0.0527, final_score = 4.9392 (positive → classified as attack). This is a concrete illustration of the ML model generalizing beyond simple pattern-matching, which is the core empirical claim of the paper.

## 6. Methodology note: handling of externally-sourced review content

Several rounds of externally-generated "expert review" content were pasted into this project as candidate guidance during development. Each was treated as unverified data to check, not as instructions to follow, consistent with treating claims about one's own results with the same scrutiny as claims about the literature. Concretely:

- A cited "ERC-8181 session key mechanism" was checked via web search and found to be a real EIP number but titled "Self-Sovereign Agent NFTs" — unrelated to session keys. Not used.
- A full "expected results" table with specific invented numbers (0.985 / 0.925 / 0.952 / 0.001) was presented as a projected outcome and, in a later message, reused as if it were an actual measured result. Identified as fabricated and never used — the real numbers in this document were measured from this project's own scripts instead.
- A claim that the length-evasion recall improvement was "the cap working as designed" was refuted directly using this project's own controlled A/B evidence (Section 4.2: the cap alone measurably did not help — recall stayed at 0.807; the improvement only appeared after the max-sentence-similarity fix in Section 4.3).

## 7. Compound adversarial attack and adversarial training

Paraphrase and length-padding were also tested applied together (a single black-box attacker rewriting trigger phrases and padding the result), a stronger and more realistic test than either vector alone, run against the full 6-feature XGBoost model with the length-dilution fix from Section 4.3 already applied. This combined attack revealed a genuine, non-additive vulnerability that neither single-vector test predicted: recall drops to a floor of 0.859 as padding increases, well below either individual-attack recall.

| Padding (repeats) | Avg. memory length | Baseline recall | Symmetric-adv (v2) recall |
|---|---|---|---|
| 0 | 168 | 0.932 | 0.930 |
| 1 | 262 | 0.877 | 0.902 |
| 2 | 355 | 0.859 | 0.920 |
| 3 | 443 | 0.859 | 0.966 |
| 5 | 500 | 0.859 | 0.936 |
| 25 / 50 / 100 | 500 (capped) | 0.859 | 0.936 |

A first attempted fix (adversarial training by augmenting only the attack class with one fixed padding pattern) was tested and rejected before being reported as a result. Stress testing — recall at low, pre-saturation padding levels, and false-positive rate on padded but genuinely benign memory — showed the model had learned a spurious shortcut: 0% of original training rows of either class reached the 500-character cap, but 100% of the synthetic augmented attack rows did, so the model learned "length == cap" as a near-perfect but fake attack indicator. That version reached 100% false-positive rate on padded genuine benign content and was not deployed.

The corrected approach ("v2") augments both classes symmetrically — attack rows via paraphrase+padding, benign rows via padding alone — which breaks the spurious length-to-label correlation:

| Padding (repeats) | Avg. memory length | Baseline FPR (padded, genuine benign) | Symmetric-adv (v2) FPR |
|---|---|---|---|
| 0 | 162 | 0.0023 | 0.0045 |
| 3 | 423 | 0.0205 | 0.0114 |
| 5 / 25 / 50 / 100 | 500 (capped) | 0.0205 | 0.0341 |

Mechanism check confirms the fix: of training rows sitting exactly at the length cap, 57.3% are augmented attack rows and 66.3% are augmented benign rows — both classes now mixed at the cap, versus the rejected version's 0% / 0% / 100% split. The primary in-domain benchmark (`indomain_benign.csv`, unmodified) shows zero regression: FPR = 0.0133 for both baseline and v2. Net result: combined-attack recall floor raised from 0.859 to 0.936, clean-data recall cost of 0.7% (0.982 → 0.975), padded-benign FPR worst case rising from 2.05% to 3.41% — a disclosed, modest trade-off, not a free win.

## 8. Held-out out-of-distribution (OOD) generalization test

To test generalization beyond the CrAIBench-derived training and test distribution — rather than replacing that dataset, per the author's direction to keep it and add to it — a held-out set of 80 hand-authored query/memory pairs was built: 40 benign examples across 8 blockchain-agent task domains not specifically represented in training (DEX swaps, lending, staking, bridging, NFT trading, recurring payments, portfolio rebalancing, gas management), and 40 attack examples across four subtypes with ground-truth labels — the three from the original threat model (address_redirection, tool_substitution, security_framing) plus one deliberately novel subtype, amount_manipulation, that neither model was ever trained to detect, included specifically as a stress test rather than an expected win. Wording was deliberately kept distinct from the exact trigger phrases used in training-side paraphrase augmentation, so that a good score reflects genuine generalization rather than memorized phrasing.

| Model | Overall recall | Overall FPR | Address redirect. | Tool subst. | Security framing | Amount manip. |
|---|---|---|---|---|---|---|
| Baseline (full 6-feature XGBoost) | 0.900 | 0.150 | 1.000 | 1.000 | 0.600 | 1.000 |
| v2 (symmetric adversarial-trained) | 0.825 | 0.100 | 1.000 | 0.900 | 0.500 | 0.900 |

Two findings, both disclosed rather than smoothed over. First, false-positive rate rises roughly 10x on this novel phrasing relative to the in-domain benchmark (0.0133 → 0.10–0.15) — expected for out-of-distribution text, and a more honest number than reporting only in-domain FPR. Tracing the false positives shows a specific mechanism: every one has no foreign address and no injection keyword, but query_memory_similarity of roughly 0.50–0.54, above the benign-class average of 0.377 — each is a short memory that directly and narrowly answers the query, structurally similar to what a paraphrase attack looks like. This suggests the model has partly learned "high similarity + short + no other signal" as a paraphrase-attack proxy, which also catches legitimately terse, on-topic benign notes.

Second, and more consequential for how the two models should be described in the paper: v2 does not dominate baseline here. Baseline beats v2 on recall in every subtype except address_redirection (tied at 1.0): overall 0.900 vs 0.825, tool_substitution 1.0 vs 0.9, security_framing 0.6 vs 0.5, amount_manipulation 1.0 vs 0.9. v2 only wins on FPR (0.100 vs 0.150). This is consistent with the well-documented accuracy/robustness trade-off in adversarial training: hardening against one specific attack pattern (compound padding+paraphrase, Section 7) made the decision boundary more conservative in a way that helped FPR but cost recall on attack types the hardening was not designed for. Neither model should be described as strictly better; the paper should present this as a disclosed, threat-model-dependent trade-off. security_framing is the weak subtype for both models (0.5–0.6 recall) — it has no address or keyword signal to lean on and is the hardest of the four by construction.

*Caveat: this OOD set was authored by an LLM assisting the project, not independently written by the author. One data-quality bug was already found and fixed during construction (3 of 10 address_redirection addresses had an odd hex-digit count and were silently missed by the has_foreign_addr feature; corrected, and address_redirection recall moved from 0.90 to a clean 1.00 for both models). As of this writing, the author's own read-through of the 80 examples for domain realism is still pending — until that happens, this set should be described in any paper draft as "LLM-drafted, review pending," not as independently human-authored.*

## 9. Current state and open items

- **Done:** threat model (3 manipulation strategies), 6-feature engineering, domain-confound diagnosis and fix, keyword/XGBoost baseline comparison, paraphrase-attack robustness test, length/dilution vulnerability (diagnosed, fixed, mechanistically confirmed), SHAP interpretability, compound paraphrase+padding attack (diagnosed), adversarial training (one attempt rejected on stress test, one validated), and the held-out OOD generalization test (Section 8).
- **Done:** Literature_Review.docx has been rebuilt for the current direction (13+ live-verified Q1 papers, plus additional verified supporting sources) — the earlier note that it still targeted an abandoned quantum-triage direction is now out of date.
- **Done:** code and results pushed to a public GitHub repository (github.com/vk-vaishnavi10/agent-intent-guard).
- **Done:** `final_results_20260901_205811.txt` verified directly (2026-09-03) — the 10-seed baseline figures (0.988 recall / 0.999 AUC / 1.5% FPR) a pasted external summary had cited are confirmed accurate to three decimals; Section 4.1's table matches. One number from that same pasted summary — "v2 padding(100x) recall = 1.000" — is not backed by any script actually run (`test_length_evasion.py` has no v2 column) and should not be used.
- **Pending:** author review of the 80 OOD examples for domain realism (Section 8 caveat).
- **Pending:** fold the compound-attack, adversarial-training, and OOD sections above into `Project_Summary.docx` and `Paper_Sections_Draft.docx`, which currently predate all of it.
- **Open decision:** whether to present the baseline or v2 model as primary in the paper, or present both with the trade-off disclosed (current recommendation, given Section 8's finding that neither dominates).
- **Open decision:** target journal has not been confirmed by the author — TIFS was raised only via pasted external content, not chosen.
- **Open decision:** whether has_foreign_addr is kept (current decision, for analyst interpretability, despite SHAP = 0.0000 — see Section 5) or dropped; a pasted external summary recently suggested dropping it without being aware this decision, and its reasoning, had already been made.
