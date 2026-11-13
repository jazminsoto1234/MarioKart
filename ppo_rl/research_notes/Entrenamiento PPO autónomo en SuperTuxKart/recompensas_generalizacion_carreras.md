# Rewards, curriculum, opponents and generalization for multi-scenario racing RL (SuperTuxKart / PPO)

Research date: 2026-11-11. Verification note: items marked **[verified]** were read from the primary source in this session (paper PDF pages or official page). Items marked **[prior knowledge, not re-fetched]** come from well-known papers whose canonical URLs are given but whose text was not re-read in this session. The report writer should treat them as lower-confidence on exact numbers.

## Q1. Reward shaping for racing: what the top systems actually used

### Takeaway
Every strong racing system uses one main dense term: **progress along the track centerline per step** (Δ arc length). It then adds speed-scaled penalties for wall contact and leaving the track. GT Sophy found progress alone was not enough to *win races*: it needed explicit passing rewards and collision-etiquette penalties. For SuperTuxKart, the pystk2 `distance_down_track` field directly gives the centerline progress term.

### Cited Findings
**Fuchs et al., "Super-Human Performance in Gran Turismo Sport Using Deep RL" (IEEE RA-L, accepted Feb 2021; arXiv 2008.07971, 2020) [verified]**
- Lap time is a sparse objective. They replace it with a course-progress proxy: project the car onto the centerline and use `r_prog_t = cp_t − cp_{t−1}`. They argue that maximizing discounted progress approximates minimizing lap time when γ is high enough. — [arXiv 2008.07971](https://arxiv.org/pdf/2008.07971.pdf)
- Final reward: `r_t = r_prog_t − c_w·||v_t||²` if in contact with a wall, else `r_prog_t`. The penalty scales with kinetic energy. Without it, policies "did not brake and simply grinded along the track's walls." With a *fixed-value* wall penalty, the agent either ignored it or learned to brake fully and stand still, depending on the penalty strength. — [arXiv 2008.07971](https://arxiv.org/pdf/2008.07971.pdf)
- Discounting creates a short-term bias that reduces the incentive to brake, and the wall penalty counteracts it. — [arXiv 2008.07971](https://arxiv.org/pdf/2008.07971.pdf)
- Observations: velocity, acceleration, heading angle relative to the centerline tangent (the "only direct way" to detect wrong-way driving), M rangefinders (max 100 m) over the front 180°, previous steering, a wall-contact flag, and N curvature samples of the centerline ahead. Actions: steering plus one combined throttle/brake scalar, because fast strategies never use both together. SAC with 2×256 MLPs. Data collection used 4 PS4s × 20 cars. — [arXiv 2008.07971](https://arxiv.org/pdf/2008.07971.pdf)
- Robustness tests on a different car, a different track, tyre-friction changes, observation noise and delay, all without retraining. — [arXiv 2008.07971](https://arxiv.org/pdf/2008.07971.pdf)

**Wurman et al., "Outracing champion Gran Turismo drivers with deep RL" (GT Sophy), Nature 602, 10 Feb 2022 [verified, Methods section]**
- The reward is a hand-tuned linear combination of 8 terms computed on (s, s′): course progress R_cp, off-course R_soc/R_loc, wall R_w, tyre slip R_ts, passing bonus R_ps, any-collision R_c, rear-end R_r, and unsporting-collision R_uc. Weights were per track (Extended Data Table 1). — [Nature 2022 PDF](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf)
- **Progress:** `R_cp = l′ − l` (centerline metres, from projecting position onto the centerline). It is **masked while the car is off course** to remove the incentive to cut corners. — [Nature 2022 PDF](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf)
- **Off-course:** `R_soc = −(s′_o − s_o)·(s′_kph)²`, i.e. time off course in that step × speed². On Sarthe they used a linear-in-speed version `R_loc` to avoid value explosion, doubled at difficult chicanes. "Off course" means 3 or more tyres out of bounds. — [Nature 2022 PDF](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf)
- **Wall:** `R_w = −(s′_w − s_w)·(s′_kph)²`, i.e. wall-contact time × speed². — [Nature 2022 PDF](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf)
- **Passing bonus:** `R_ps = Σ_i (s_Li − s′_Li)·max(1_(b,f)(s_Li), 1_(b,f)(s′_Li))`, where s_Li is the signed centerline distance to opponent i. It only applies when the opponent is within b = 20 m behind or f = 40 m ahead. It is negative when the agent loses ground, which "ensured that there were no positive-cycle reward loops to exploit" and encouraged defensive driving. It was masked at chicanes to stop the agent cutting corners to pass. — [Nature 2022 PDF](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf)
- **Collisions:** any-collision is a Boolean penalty `R_c = −max_i c_i`. Rear-end is `R_r = −Σ_i c_i·1[opponent ahead]·||v − v_i||²`. Unsporting-collision is an extra Boolean penalty for rear-ending, side-swiping on straights, or hitting in curves when the agent is at fault. They added it because simply raising the any-collision penalty "resulted in very timid agent behaviour." — [Nature 2022 PDF](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf)
- Large squared-speed penalties destabilized Q learning. They mitigated this with gradient clipping (global norm 10). — [Nature 2022 PDF](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf)
- Training: QR-SAC (M = 32 quantiles, N-step returns), γ = 0.9896, 10 Hz decisions, 150 s episodes. Network 4×2048 with dropout 0.1, replay buffer of 10⁷ transitions. Time trial took ~8 days with 10 PS4s, and traffic policies took 7–12 days with 21 PS4s. — [Nature 2022 PDF](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf)
- Racing features added for opponents: a car-contact flag, a slipstream scalar, and nearby cars within a fixed front/rear distance, ordered by distance, each with relative position, velocity and acceleration. — [Nature 2022 PDF](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf)
- Sony's blog gives the time-trial learning curve: ~1 h to slow laps, 8 h to match the built-in AI, ~1 day to reach the top 10% of humans, and 8–10 more days to reach consistent superhuman driving. Without the off-track penalty the agent learned to cut corners. — [Sony AI blog, Jun 14 2022](https://ai.sony/blog/training-the-worlds-fastest-gran-turismo-racer)
- Coverage quotes the authors: progress reward alone "was not enough to incentivize the agent to win the race." — [summary of Nature paper, via search snippet](https://www.deeplearning.ai/the-batch/fast-and-daring-wins-the-race)

**Ma, Seno, Subramanian, Wurman, Stone, Sherstan, "Automated Reward Design for Gran Turismo" (arXiv 2511.02094, 3 Nov 2025) [verified abstract and appendix]**
- Lists the GT Sophy baseline terms used in GT7 work. They include `steering_change_cost` (magnitude of steering change) and `acted_steering_history_cost` (frequent direction flips in a window), which are smoothness terms not in the 2022 list. — [arXiv 2511.02094](https://arxiv.org/html/2511.02094v1)
- An LLM generates reward functions, a VLM evaluates them by preference, and humans give feedback. The resulting agents are competitive with GT Sophy. Evaluation: 4-lap races against 19 built-in AIs with the agent **starting last**. Metrics are mean final placement, car-collision time and course-out time, and agents over the collision or course-out thresholds are disqualified (set to last). — [arXiv 2511.02094](https://arxiv.org/html/2511.02094v1)

**Kaufmann et al., "Champion-level drone racing using deep RL" (Swift), Nature 2023 [verified via PMC]**
- `r_t = r_prog + r_perc + r_cmd − r_crash`. Progress is λ1 × (reduction in distance to the next gate). Perception is λ2·exp(λ3·δ_cam⁴), keeping the camera on the gate. The command term penalizes body rates and squared action change. Crash is 5.0 with episode termination. — [PMC10468397](https://pmc.ncbi.nlm.nih.gov/articles/PMC10468397/)
- PPO with an asymmetric actor-critic: the **value network gets privileged state and is used only in training**. 100 parallel envs, episodes start at a random gate perturbed around previously observed states, 2×128 MLPs, lr 3e-4, 1e8 steps in ~50 min. — [PMC10468397](https://pmc.ncbi.nlm.nih.gov/articles/PMC10468397/)
- Swift did **not** randomize dynamics. It fine-tuned on residual models (Gaussian processes for observations, k-NN for dynamics) fitted to real data, and domain randomization was only a baseline. — [PMC10468397](https://pmc.ncbi.nlm.nih.gov/articles/PMC10468397/)

**Trackmania**
- tmrl calls a raw-speed reward naive. For TM2020 it records a reference trajectory as points and rewards the number of points passed since the previous step, i.e. discretized centerline progress. — [tmrl on PyPI](https://pypi.org/project/tmrl/0.1.6)
- Linesight (IQN-based, per its object names) beat 10 of 12 official-campaign world records in TM Nations Forever. Custom maps need a "Reference Line" (a centerline) to compute progress. — [Linesight docs](https://linesight-rl.github.io/linesight/build/html)
- A Linesight fork adds "a virtual checkpoint system for dense progress tracking" to explore reward shaping. — [rulka docs](https://artyomzemlyak.github.io/rulka/)

**SuperTuxKart / pystk2-gymnasium [verified README]**
- The default reward is Δ track distance / 10, plus a rank-dependent position term among K karts, −0.1 per step, and +10 on finishing. Env options include `track` (random by default), `num_kart` (default 3), `laps`, and AI `difficulty` 0–2 (default 2). `AgentSpec` has `rank_start` (None = random start position) and `use_ai`. There is a multi-agent env `multi-full-v0`. The README states there are no pixel observations in that wrapper. — [pystk2-gymnasium GitHub](https://github.com/bpiwowar/pystk2-gymnasium)
- pystk2 is based on the STK 1.4 branch and supports races run in parallel for RL. — [pystk2 GitHub](https://github.com/bpiwowar/pystk2); [pystk2 docs](https://pystk2.readthedocs.io/)

### Inferences (concrete recommended reward for STKPixelEnv; my synthesis, not from any one source)
Per step at ~10–20 Hz, using privileged state:
```
d      = distance_down_track (wrap-safe: Δ mod track_length, clipped to [-5, +5] m to kill respawn/lap-wrap spikes)
r_prog = 1.0 * Δd   (masked/0 if off-road or if kart is rescuing; negative Δd naturally penalizes wrong-way)
r_wall = -c_w * |v|^2 * 1[wall/kart-terrain collision this step]          (Fuchs; c_w tuned so a full-speed hit ≈ -(1–3 s of progress))
r_off  = -c_o * |v| * 1[off-road]                                          (GT Sophy linear variant R_loc, more stable than v^2)
r_resc = -R_rescue (≈ value of 3–5 s of progress) when rescue is triggered (manual or auto)  — prevents "rescue as teleport" hacks
r_pass = c_p * Σ_i (gap_i(t-1) - gap_i(t)) for rivals within [-20 m, +40 m] (GT Sophy R_ps; zero-sum per rival, no loops)
r_rank = optional small potential-based term: γΦ(s') - Φ(s), Φ = -k*rank  (Ng et al. 1999 style; avoids per-step rank farming)
r_time = -0.01..-0.1 per step (pystk2 default -0.1 with /10 progress scaling)
r_end  = +B_finish * f(final rank)  e.g. +10 for 1st, scaled down to 0 for last; −B on timeout/DNF
r_smooth = -c_s * |Δsteer| (GT7 Sophy steering_change_cost) — small, mainly for pixel-policy stability
```
- The two rank-based terms (`r_pass` and `r_rank`) should be zero-sum or potential-based. GT Sophy explicitly designed `R_ps` to be negative when losing ground, which prevents loops such as letting a rival pass and repassing it.
- Mask the progress term while off-road. Otherwise STK shortcuts and the auto-rescue that moves the kart forward can be exploited, as GT Sophy found with corner cutting.
- Items: do not reward "fire" directly. Reward the effect, i.e. rival slowdown and rank gained, which `r_pass` already captures. A small direct reward for hitting a rival is optional and risks spamming.

### Gaps
- I did not read the GT Sophy Extended Data Table 1 weights. They are in the Nature supplementary material.
- The exact reward code of Linesight was not found in docs; check `github.com/pb4git/linesight` (config files).
- I did not fetch AWS DeepRacer (reward = Python function of `track_width`, `distance_from_center`, `progress`, `speed`, `heading`; default "follow the centerline"), F1TENTH or Gym-TORCS (commonly `v·cosθ − v·|sinθ| − v·|trackPos|`) in this session. Both are **[prior knowledge, not re-fetched]**. Canonical: [AWS DeepRacer reward params](https://docs.aws.amazon.com/deepracer/latest/developerguide/deepracer-reward-function-input.html), [Gym-TORCS DDPG (Yan Pan Lau blog)](https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html).

## Q2. Generalization: randomization, held-out tracks, augmentation, memory

### Takeaway
Agents overfit strongly to small sets of levels. Procgen needed hundreds to 10,000 levels to close the train/test gap. STK has ~20 tracks, so hold 3–5 out for testing and randomize everything else (kart, start rank, opponent count and difficulty, items, laps, reverse direction). Combine that with image augmentation and an asymmetric critic.

### Cited Findings
- Procgen (Cobbe et al., 2019): the default is 500 training levels (200 for easy mode). Sweeps from 100 to 100,000 levels with PPO for 200M steps showed agents "strongly overfit to small training sets in almost all environments", and some needed ~10,000 levels to close the gap. Larger level sets also improved *training* performance, which the authors attribute to an implicit curriculum. — [Procgen paper (ar5iv)](https://ar5iv.arxiv.org/html/1912.01588); [OpenAI blog](https://openai.com/index/procgen-benchmark/)
- Prioritized Level Replay (Jiang et al., ICML 2021) samples training levels by estimated learning potential (TD-error / GAE magnitude). It significantly improved generalization on 10 of 16 Procgen games, and combined with augmentation (UCB-DrAC) it reached over 76% improvement in test return. It is algorithm-agnostic and works with PPO. — [PLR, PMLR v139](https://proceedings.mlr.press/v139/jiang21b.html); [arXiv 2010.03934](https://arxiv.org/pdf/2010.03934)
- Fuchs et al. tested transfer to unseen cars and tracks and to perturbed friction, noise and delay without retraining. This is the standard robustness protocol for a single-track agent. — [arXiv 2008.07971](https://arxiv.org/pdf/2008.07971.pdf)
- GT Sophy trained **separate policies per car/track** and did not attempt a single cross-track policy. — [Nature 2022 PDF](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf); [Sony AI](https://ai.sony/blog/unveiling-gran-turismo-sophy-an-ai-breakthrough)
- Swift used an asymmetric actor-critic with a privileged-state critic. This is directly applicable to an RGB actor with a privileged-state critic. — [PMC10468397](https://pmc.ncbi.nlm.nih.gov/articles/PMC10468397/)
- **[prior knowledge, not re-fetched]** RAD (Laskin et al., 2020): random crop/translate was the strongest augmentation on DMControl and Procgen. DrQ (Kostrikov et al., 2020) uses random shift (pad 4 px and crop) plus averaging the Q-target over augmentations. For PPO, DrAC (Raileanu et al., 2021) regularizes the policy and value to be invariant to augmentation, rather than naively feeding augmented images, which can hurt PPO. — [RAD arXiv 2004.14990](https://arxiv.org/abs/2004.14990); [DrQ arXiv 2004.13649](https://arxiv.org/abs/2004.13649); [DrAC arXiv 2006.12862](https://arxiv.org/abs/2006.12862)

### Inferences
- STK's track set is tiny compared with Procgen. Generalization to unseen *layouts* will be limited, so maximize within-track diversity: both directions (reverse mode), random start rank, random kart, laps 1–3, num_kart 1–8, difficulty 0–2, and items on and off. Use per-scenario seeds as "levels" for PLR.
- For PPO on pixels, use DrAC-style random shift together with colour jitter (useful because tracks have very different palettes). Avoid crops that remove the HUD's rank and item indicators. Alternatively, crop the HUD region and feed the item and rank from privileged state as scalar actor inputs if fairness allows.
- Frame stack (4) is enough for velocity. A recurrent layer (LSTM/GRU) helps with items (remembering a held item, an incoming projectile) and with lap counting. Starting with a frame stack and adding an LSTM after convergence is a common choice, but I did not find a racing-specific ablation.

### Gaps
- No source found comparing frame stacking with recurrent policies specifically for racing.
- No STK-specific held-out-track generalization study was found.

## Q3. Curriculum and scenario sampling

### Takeaway
GT Sophy did **not** use a sequential curriculum. It trained one policy on a fixed mixture of complementary scenarios with stratified replay sampling, because skills were otherwise *forgotten*. Use a time-trial warm-up, then a mixture over scenarios (with PLR-style prioritization), not hard stages.

### Cited Findings
- GT Sophy scenarios: the agent was launched with 0, 1, 2, 3 or 7 opponents nearby (1v0 to 1v7), with randomized track positions, speeds, spacing and opponent policies, using up to 20 cars per PS4. Base scenarios ran 150 s. They added time- and distance-limited **skill scenarios** on specific sections: 8-car grid starts, 1v1 slipstream passing, and the final chicane in light traffic. — [Nature 2022 PDF](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf)
- "Mistake learning": when an evaluated policy lost control, the state shortly before the event was saved and reused as a training start state. This is a form of reverse curriculum / start-state replay. — [Nature 2022 PDF](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf)
- "Unlike curriculum training where early skills are supplanted by later ones", their scenarios were complementary and trained into a single policy using hand-tuned ratios. Even so, random buffer fluctuations caused skills to be **unlearned** between epochs. They fixed this with **multi-table stratified sampling**, which enforces each scenario's share of every mini-batch. — [Nature 2022 PDF](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf)
- Swift starts each episode at a random gate, perturbed around a previously observed state. This is start-state randomization, which covers the whole track uniformly. — [PMC10468397](https://pmc.ncbi.nlm.nih.gov/articles/PMC10468397/)
- PLR prioritizes levels by learning potential and can serve as an automatic curriculum over (track, difficulty, opponents) seeds. — [PLR](https://proceedings.mlr.press/v139/jiang21b.html)
- **[prior knowledge, not re-fetched]** PAIRED (Dennis et al., NeurIPS 2020) uses an adversary that generates environments to maximize regret. ACCEL (Parker-Holder et al., ICML 2022) edits replayed levels with small mutations, combined with PLR. Both need a parameterizable level generator, which STK lacks beyond the scenario parameters. — [PAIRED arXiv 2012.02096](https://arxiv.org/abs/2012.02096); [ACCEL arXiv 2203.01302](https://arxiv.org/abs/2203.01302)

### Inferences (proposed schedule for STK PPO; synthesis)
1. **Phase A (time trial, ~10–30M steps):** 1 kart, no items, random track from the training set, random start position along the track using pystk `rank_start` plus mid-track restarts if feasible. Reward: progress, wall, off-road and rescue terms. Gate to the next phase when the agent finishes ≥90% of laps on all training tracks.
2. **Phase B (traffic):** add AI opponents. Sample num_kart ∈ {2, 4, 6, 8} and difficulty ∈ {0, 1, 2}, weighted up as the win rate rises. Enable `r_pass` and collision terms, and randomize start rank, including last.
3. **Phase C (items):** turn items on and add the fire, nitro and drift action heads (or unmask them). Keep 20–30% of episodes as time trial or no items, to avoid forgetting (GT Sophy's stratified-mixture lesson).
4. Throughout, use PLR over scenario tuples (track, direction, kart, n_opp, difficulty, items, start rank), saving "mistake states" (crash or rescue) as restart points if STK state save/restore is possible.
- Because PPO is on-policy, GT Sophy's stratified-batch fix translates to **fixed per-scenario quotas across parallel envs**, e.g. N envs split by scenario class, plus per-scenario advantage normalization.

### Gaps
- No published STK curriculum ablation was found.
- I did not verify whether pystk2 supports mid-race state snapshots for start-state replay.

## Q4. Multi-agent / competitive training, etiquette, items

### Takeaway
GT Sophy beat humans while training against randomized opponent policies, not full self-play leagues. It chose release policies through round-robin tournaments and human review. For STK, built-in AI at mixed difficulty plus a pool of frozen past selves (fictitious self-play, AlphaStar-lite) is the cost-effective recipe.

### Cited Findings
- GT Sophy randomized "opponent policies" in training scenarios. It selected the final agent by saving checkpoints, running them in scenario tests (an "n-athlon" of pre-specified situations such as starts and slipstream), then holding a round-robin policy-vs-policy tournament scored like the real event plus collision metrics, with a human committee making the final choice. A special "starter" policy handled the grid start and switched to the race policy after 2,100 m at Sarthe. — [Nature 2022 PDF](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf)
- Etiquette was encoded through rewards (rear-end, unsporting-collision and any-collision penalties). Raising generic collision penalties made the agent timid, so they used fault-specific penalties instead. — [Nature 2022 PDF](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf)
- Head-to-head events against top human drivers were held on 2 Jul 2021 and 21 Oct 2021. GT Sophy won the October team event. — [Sony AI](https://ai.sony/blog/unveiling-gran-turismo-sophy-an-ai-breakthrough); [Scientific American](https://www.scientificamerican.com/article/ai-outraces-human-champs-at-the-video-game-gran-turismo/)
- pystk2-gymnasium supports multi-agent races (`multi-full-v0`), with each kart controlled by a policy or by the built-in AI (`use_ai`). This enables self-play and mixed pools. — [pystk2-gymnasium](https://github.com/bpiwowar/pystk2-gymnasium)
- **[prior knowledge, not re-fetched]** AlphaStar (Vinyals et al., Nature 2019) used league training with main agents, main exploiters and league exploiters, plus prioritized fictitious self-play that samples opponents by win probability, f(p) = (1−p)² for hard opponents. — [Nature 2019](https://www.nature.com/articles/s41586-019-1724-z)

### Inferences
- STK opponents for training: about 60% built-in AI (difficulty sampled toward where the win rate is ~50%), about 30% frozen past checkpoints of the agent, and about 10% the current policy (self-play). The pool keeps the agent from overfitting to the built-in AI's quirks.
- Items: STK items (bowling ball, cake, swatter, bubblegum, zipper, plunger) have strategic value. Reward only outcome-based effects (rank and gap change via `r_pass`). Give the actor the held item from privileged or HUD features. A recurrent policy or item embedding helps with timing (hold a defensive item when leading). Collision penalties on kart contact are optional in STK, since bumping is not penalized by rules. Use a small any-collision penalty to avoid losing speed, not for sportsmanship.

### Gaps
- GT Sophy's exact training opponent policies (built-in AI vs earlier Sophy versions) are not detailed in the pages I read.
- No published STK item-strategy study was found.

## Q5. Prior work on SuperTuxKart / pystk and Mario Kart-like games

### Takeaway
There is no standardized STK RL benchmark or leaderboard (as of this search). Prior STK work is mostly course projects (UT Austin's Krähenbühl course built the original pystk, and a Sorbonne-linked fork, pystk2, has a Gymnasium wrapper), a load-testing paper, and student DQN/PPO repos. Mario Kart work is mostly imitation (MariFlow) or neuroevolution.

### Cited Findings
- pystk2 is a modified STK 1.4 with Python bindings and parallel races for RL. pystk2-gymnasium is in alpha (environments may change abruptly), has a full and simplified observation set, and offers discrete/continuous/multidiscrete action variants. — [pystk2](https://github.com/bpiwowar/pystk2); [pystk2-gymnasium on PyPI](https://pypi.org/project/pystk2-gymnasium/0.4.2/)
- RELINE (load testing of games with RL, 2022) used SuperTuxKart through PySuperTuxKart with a DQN agent. — [arXiv 2201.06865](https://arxiv.org/pdf/2201.06865)
- Informal student projects exist, including DQN racing-kart and PPO notebooks. — [Deep-Q-Learning-Racing-Kart](https://github.com/laythamra28/Deep-Q-Learning-Racing-Kart); [rx-00 blog](https://rx-00.github.io/blog)
- A University of Twente BSc thesis compares GT Sophy-style and Trackmania AI approaches and notes the key differences in algorithm, reward and inputs. — [Neinders, UTwente](https://essay.utwente.nl/96153/1/Neinders_BA_EEMCS.pdf)
- **[prior knowledge, not re-fetched]** The original pystk was built for Philipp Krähenbühl's UT Austin deep learning course (CS 342/394D). Its final projects were 2v2 ice hockey and imitation-learned driving from images (aim-point regression), not RL racing benchmarks. — [pystk docs](https://pystk.readthedocs.io/)
- **[prior knowledge, not re-fetched]** MariFlow (SethBling, 2017) used an LSTM trained by imitation on human Mario Kart play. MarIQ (SethBling, 2019) used Q-learning. NEAT-based Mario Kart agents exist as hobby projects. None are peer-reviewed. — [MariFlow manual](https://docs.google.com/document/d/1p4ZOtziLmhf0jPbZTTaFxSKdYqE91dYcTNqTVdd6es4/)

### Gaps
- No 2023–2026 peer-reviewed STK RL benchmark or thesis was found in this session's searches. The report should state this as absence of evidence, not proof.
- No PySuperTuxKart2 / `STKPixelEnv` publication was found. The user's env appears to be custom.

## Q6. Evaluation protocols

### Takeaway
Report win rate and mean finishing position against the built-in AI at each difficulty, on training and held-out tracks, from random start positions (including last). Use ≥3 training seeds and ≥50–100 races per cell, plus collision, off-road and rescue rates as failure metrics.

### Cited Findings
- GT7 automated-reward paper: 100 races of 4 laps against 19 built-in AIs with the agent starting last. Metrics are mean final placement, collision time and course-out time, with threshold-based disqualification. It compared multiple seeds (five GT Sophy seeds set the thresholds) and kept races in the 50% interquantile range. — [arXiv 2511.02094](https://arxiv.org/html/2511.02094v1)
- GT Sophy time-trial consistency: over 200 laps, best 114.071 s, mean 114.208 s, SD 0.061 s, against a human record of 114.181 s. — [Sony AI blog](https://ai.sony/blog/training-the-worlds-fastest-gran-turismo-racer)
- Fuchs: lap times compared against the built-in AI and against more than 50,000 human players' best laps. Action-rate limits were set to human-feasible values (0.03 rad steering change and 80% throttle/brake change per frame) for fairness, which did not significantly change lap times. — [arXiv 2008.07971](https://arxiv.org/pdf/2008.07971.pdf)
- Procgen protocol: train on N levels and report on held-out levels. — [Procgen](https://ar5iv.arxiv.org/html/1912.01588)

### Inferences
- STK evaluation matrix: {train tracks, held-out tracks} × {difficulty 0, 1, 2} × {n_opp 3, 7} × {items on/off} × {start rank random/last}. For each cell report win rate with Wilson 95% CI, mean rank, finish rate, rescues/lap, and lap time against the built-in AI's best lap. Use 3 or more seeds with IQM aggregation (rliable-style).
- Gate checkpoint selection on a held-out "n-athlon" (GT Sophy-style) rather than training return, because return mixes shaping terms.

### Gaps
- No published human lap-time reference dataset for STK was found.
