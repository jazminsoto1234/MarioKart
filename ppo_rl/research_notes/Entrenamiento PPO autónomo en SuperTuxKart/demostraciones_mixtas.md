# Learning from mixed-quality (good + bad) human demonstrations, then online PPO, for SuperTuxKart

Scope: methods that use human gameplay containing both good and bad plays to pretrain an agent and then fine-tune it with online PPO (asymmetric actor-critic: privileged critic, vision actor, Gaussian steer/gas plus categorical drift/item heads). Research date: 2026-11-11. About 17 tool calls. Most numbers come from primary papers (arXiv/ar5iv); where only secondary summaries were available, that is noted.

## Q1. Behavior cloning with mixed data, filtered/percentile BC, and scoring human segments with privileged state

### Takeaway
Plain BC on mixed-quality data is weak because it imitates the average of good and bad play. Keeping only the best data by return is a very strong, simple baseline. On D4RL "medium-replay" (mixed-quality replay buffers), 10%BC roughly matches Decision Transformer and some offline RL methods, while full-data BC collapses on some tasks. This makes "label the human data with privileged-state scores, then filter or weight it" the cheapest high-value first step.

### Cited Findings
- Standard BC minimizes average negative log-likelihood over all samples, so it has no way to penalize bad actions; it also suffers compounding error under distribution shift — [UNIQ, arXiv 2410.08307](https://arxiv.org/html/2410.08307v1)
- %BC is BC on the top X% of timesteps ranked by episode return. X=100% is ordinary BC, and X near 0% clones only the best trajectory — [Decision Transformer, Chen et al. 2021](https://arxiv.org/pdf/2106.01345)
- D4RL medium-replay-v2 normalized scores (IQL paper, Table 1):
  - BC: halfcheetah 36.6 / hopper 18.1 / walker2d 26.0
  - 10%BC: 40.6 / 75.9 / 62.5
  - DT: 36.6 / 82.7 / 66.6
  - AWAC: 40.5 / 37.2 / 27.0
  - TD3+BC: 44.6 / 60.9 / 81.8
  - CQL: 45.5 / 95.0 / 77.2
  - IQL: 44.2 / 94.7 / 73.9
  - Source: [Kostrikov et al., IQL, ICLR 2022](https://ar5iv.labs.arxiv.org/html/2110.06169)
- The DT authors say DT "either outperforms or is competitive with %BC", without the confound of having to choose the right subset. Note that their reported average pools the medium and medium-replay datasets (DT 56.1 vs 10%BC 56.7) — [Decision Transformer](https://arxiv.org/pdf/2106.01345)
- Filtered BC (Emmons et al., RvS) averages 59.7 on medium-replay. Return-conditioned BC with weighting and conditioning (CWBC) scores 67.0, which the authors describe as about a 12% improvement over Filtered BC on med-replay — [Hong et al., Reliable Conditioning of BC, arXiv 2210.05158](https://arxiv.org/pdf/2210.05158)
- A 2025 study finds MLP Filtered BC competitive with or better than DT in sparse-reward settings — [Should We Ever Prefer Decision Transformer?, arXiv 2507.10174](https://arxiv.org/html/2507.10174v1)
- AlphaStar did quality filtering by skill tier. It used supervised learning on about 971k replays from players above 3500 MMR, then a further supervised fine-tune only on replays above 6200 MMR. Per a secondary summary, this raised the win rate against the built-in elite bot (PvP) from 87% to 96% — [DeepMind AlphaStar blog](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/); summary in [xlnwel AlphaStar notes](https://xlnwel.github.io/blog/reinforcement%20learning/AlphaStar)
- Robomimic's "Multi-Human" datasets combine 6 operators split into "worse", "okay" and "better" tiers (2 operators each, 50 demos each, 300 in total). The authors report a larger performance gap on multi-human data than on single-operator data, and say batch/offline RL methods struggle on mixed human data while BC-RNN holds up better — [robomimic v0.1 docs](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html); [Mandlekar et al., arXiv 2108.03298](https://arxiv.org/pdf/2108.03298)

### Inferences
- For SuperTuxKart, the privileged 60 Hz state lets the user build a dense per-segment score without any learned reward. Candidate signals:
  - Δ track progress (distance along the track centerline) per second.
  - Signed off-track distance.
  - Rescue events and crash/collision flags.
  - Lap time relative to the per-track best.
  - Item outcomes: hit or miss, and position change after an item is used.
- Per-segment returns-to-go computed with the same reward function that PPO will use (e.g. 2–5 s windows) feed several methods at once: %BC, AWR-style weights, DT conditioning, and the T-REX ranking pairs from Q3.
- Filter at the segment level, not the episode level. A human lap with one crash still holds many good corners. Segment-level filtering keeps far more data than whole-episode filtering.
- Bad segments should not be thrown away. They are needed for Q3 methods and for warming up the critic (Q4).

### Gaps
- I found no published numbers on segment-level versus episode-level filtering for racing games specifically.
- I did not verify the AlphaStar 87%→96% figure in the Nature supplement; it comes from a secondary summary.

## Q2. Advantage-weighted and offline-to-online methods (AWR, AWAC, CRR, IQL, Cal-QL, RLPD, JSRL, Kickstarting): which handle bad data well?

### Takeaway
Methods that fit a value or Q function and then do advantage-weighted BC (IQL, CRR, AWR-style) handle mixed data far better than plain BC: medium-replay hopper goes from 18 with BC to 95 with IQL. IQL is also the strongest of these at offline-to-online fine-tuning in its paper. Cal-QL targets the "unlearning dip" at the start of fine-tuning. RLPD and JSRL avoid offline pretraining pitfalls entirely: RLPD mixes the demo data into an off-policy learner's replay, and JSRL uses a guide policy to set the start states for exploration. However, all of IQL, Cal-QL and RLPD are off-policy and Q-based, so they do not plug directly into PPO.

### Cited Findings
- IQL fits an upper-expectile value function, backs it up into a Q-function without querying unseen actions, and extracts the policy with advantage-weighted BC. Defaults: τ=0.7, β=3.0 for locomotion; τ=0.9, β=10.0 for AntMaze — [IQL](https://ar5iv.labs.arxiv.org/html/2110.06169)
- Offline-to-online AntMaze totals, before → after online fine-tuning: AWAC about 107.7→108.3, CQL 151.5→231.1, IQL 370.1→473.7. For example, IQL antmaze-large-play went 25.5→46.0. The authors attribute AWAC's weak result to a much weaker offline initialization — [IQL paper, Table 2 (via search summary of arXiv 2110.06169)](https://arxiv.org/pdf/2110.06169)
- A 2024 study reports that AWAC and IQL with larger networks overcome the "paradoxical failure modes" on mixed datasets and reach state of the art on D4RL. This suggests that network capacity matters on mixed data — [arXiv 2403.13097](https://arxiv.org/pdf/2403.13097v1.pdf)
- CRR's policy update is a loss-weighted form of BC (weighted by the critic's advantage) — summarized in [UNIQ related work, arXiv 2410.08307](https://arxiv.org/html/2410.08307v1)
- Cal-QL:
  - It learns a conservative Q function that is also calibrated, meaning it upper-bounds the value of a reference policy such as the behavior policy.
  - It is a one-line change on top of CQL.
  - It mitigates the "initial unlearning" dip that CQL shows at the start of fine-tuning.
  - It beats prior methods on 9 of 11 fine-tuning benchmarks (the v1 abstract said 10 of 11).
  - Source: [Nakamoto et al., arXiv 2303.05479](https://arxiv.org/abs/2303.05479v4)
- RLPD (Ball et al., ICML 2023):
  - No offline pretraining.
  - "Symmetric sampling": each batch is 50% online replay and 50% offline data.
  - LayerNorm in the critic, plus large critic ensembles.
  - The authors report improvements up to 2.5× over prior results on several domains.
  - Sources: [arXiv 2302.02948](https://arxiv.org/pdf/2302.02948); [RLinf docs](https://rlinf.readthedocs.io/en/latest/rst_source/reference/algorithms/rlpd.html)
- Jump-Start RL (JSRL):
  - A guide policy (which only needs to be better than random) rolls in for the first h steps, and an exploration policy takes over after that.
  - The guide horizon is shrunk as a curriculum, so the final policy is RL-only.
  - Strongest in the small-data regime.
  - In theory it reduces the sample complexity of exploration from exponential to polynomial in the horizon.
  - Source: [Uchendu et al., ICML 2023](https://proceedings.mlr.press/v202/uchendu23a.html)
- Kickstarting (DeepMind 2018):
  - The student optimizes its own RL loss plus a distillation loss toward the teacher's policy, and the teacher's weight is annealed over training.
  - The teacher is "a proxy", not a target, so students can surpass it.
  - On DMLab-30 with IMPALA, the student matched a from-scratch agent with about 10× fewer steps and finished 42% higher.
  - Sources: [Schmitt et al., arXiv 1803.03835](https://arxiv.org/pdf/1803.03835); the annealing detail is from [reading notes](https://yobibyte.github.io/files/paper_notes/Kickstarting_Deep_Reinforcement_Learning_Simon_Schmitt__Jonathan_J__Hudson__Augustin_Zidek_et_al___2018.pdf)

### Inferences
- Several pieces port directly into a PPO pipeline:
  - **Policy pretraining:** offline advantage-weighted BC. Train a privileged-state critic V(s) (and optionally Q) offline on the scored human data using the PPO reward. Then train the vision actor with weights exp(β·A(s,a)), clipped, the AWR/IQL form. This automatically down-weights crashes and wasted items without hand-set thresholds.
  - **Critic initialization:** the asymmetric critic already uses privileged state, so the same offline critic can initialize the PPO critic.
  - **JSRL** is compatible with on-policy PPO, because it only changes who acts during rollout.
  - **Kickstarting** is compatible with PPO, because it is an auxiliary loss.
- RLPD and Cal-QL need an off-policy learner such as SAC. They are relevant if the user is willing to swap PPO for SAC. The tension between RLPD's off-policy data reuse and PPO's on-policy updates is real: PPO cannot directly consume offline transitions without importance-ratio issues.

### Gaps
- I did not independently verify the AntMaze numbers against the final ICLR table; they come from a search summary of the paper.
- I found no head-to-head of AWR-pretrain → PPO versus BC → PPO on mixed human data in games.

## Q3. Explicitly using bad demonstrations (negative demos, DWBC, 2IWIL/IC-GAIL, T-REX/D-REX, preferences, Decision Transformer)

### Takeaway
The "architecture that combines good and bad plays" is most likely one of the following:
- **Confidence- or discriminator-weighted imitation** (2IWIL/IC-GAIL, DWBC), which learns which demos are good and reweights BC.
- **Ranking-based reward learning** (T-REX/D-REX), which learns a reward from "better vs worse" trajectory pairs and can beat the demonstrator.
- **Return-conditioned policies** (Decision Transformer/RvS), which train on everything and condition on a high target return at deploy time.
- **Newer "undesirable demonstration" methods** (SafeDICE, UNIQ, Hoang et al. 2025), which push the policy away from bad data.

### Cited Findings
- **DWBC** (Xu et al., ICML 2022):
  - Trains a discriminator that separates the small expert set from the unlabeled mixed set; its outputs weight the BC loss.
  - Reported better returns and faster training than baselines; code at github.com/ryanxhr/DWBC.
  - Sources: [ar5iv 2207.10050](https://ar5iv.labs.arxiv.org/html/2207.10050); [Tsinghua PDF](https://air.tsinghua.edu.cn/pdf/Discriminator-Weighted-Offline-Imitation-Learning-from-Suboptimal-Demonstrations.pdf)
  - Criticism: the discriminator needs costly hyperparameter tuning and struggles with high-dimensional data — [arXiv 2401.08957](https://www.arxiv.org/pdf/2401.08957)
  - ILID instead selects useful imperfect data using dynamics information — [arXiv 2405.17476](https://arxiv.org/html/2405.17476v3)
- **2IWIL / IC-GAIL** (Wu et al., ICML 2019):
  - Use confidence scores on only a small fraction of demos.
  - 2IWIL: two-step importance weighting. IC-GAIL: end-to-end GAIL.
  - Both significantly outperform GAIL on all data. IC-GAIL usually reaches higher return; 2IWIL converges faster.
  - Sources: [PMLR v97](https://proceedings.mlr.press/v97/wu19a.html); results detail from [secondary review](https://liner.com/review/imitation-learning-from-imperfect-demonstration)
- **T-REX** (Brown et al., ICML 2019):
  - Learns a reward from ranked demonstrations, which allows extrapolating beyond the best demo.
  - Claimed to be the first better-than-demonstrator result on Atari without ground-truth reward.
  - Performance is more than an order of magnitude above the best demonstration on some tasks.
  - Robust to modest ranking noise, including a noisily improving novice.
  - Sources: [arXiv 1904.06387](https://arxiv.org/pdf/1904.06387); [ICML oral page](https://icml.cc/virtual/2019/oral/4615)
- **Undesirable / negative demonstrations:**
  - SafeDICE mixes undesirable and unlabeled data, assigning negative weights to the undesirable part. UNIQ (inverse Q-learning) learns to move away from undesirable demos and extracts the policy with weighted BC — [UNIQ, arXiv 2410.08307](https://arxiv.org/html/2410.08307v1)
  - Hoang et al. 2025 learn from expert plus undesirable demos with Q-weighted BC and no adversarial training — [arXiv 2505.21182](https://arxiv.org/html/2505.21182v1)
  - A 2026 failure-aware offline RL paper splits data into successes and collisions and uses the failures only as a negative signal, never as imitation targets, to avoid copying unsafe actions — [arXiv 2604.23360 (via Emergent Mind)](https://www.emergentmind.com/papers/2604.23360)
  - ORIL learns a reward by contrasting demonstrator observations against unlabeled ones, then runs offline RL — [arXiv 2011.13885](https://arxiv.org/pdf/2011.13885)
- **Return-conditioned policies:** DT and RvS train on all data, conditioned on return-to-go. On medium-replay, DT is about as good as 10%BC (see Q1 table). CWBC improves the reliability of the conditioning, especially for out-of-distribution target returns — [DT](https://arxiv.org/pdf/2106.01345); [CWBC](https://arxiv.org/pdf/2210.05158)

### Inferences
- In SuperTuxKart, the privileged state can label "bad" automatically: rescue triggered, off-track for more than X frames, a large speed drop from a wall hit, an item fired with no target in range. Human confidence labels are therefore not needed, which removes the main practical cost of 2IWIL and DWBC.
- Practical options, ranked from simplest to most complex:
  1. **Weighted BC with a negative term.** Apply the positive weight w=exp(βA) to good/neutral segments. For segments labeled as crashes, add a small, clipped "unlikelihood" term that pushes probability mass away from the action taken in the ~1 s before the crash. For the categorical drift and item heads this is easy: −log(1−p(a)). For the Gaussian steer/gas heads it must be clipped strictly, because the term is unbounded.
  2. **Return-conditioned actor.** Condition the actor on a "skill/quality" token (good vs bad segment) during pretraining, and always condition on "good" at deploy and during PPO. This uses bad data for representation learning without imitating it.
  3. **T-REX-style reward model** from pairs of segments ranked by progress. This is useful mainly if the hand-designed PPO reward turns out to be hard to shape, e.g. for item-use quality. It is optional, because the true reward is computable from privileged state.
- Using bad data as negative imitation targets is risky. "Bad" moments are often caused by earlier states, not by the last action. Valuing them through the critic (advantage-weighting / Q-weighting) is safer than direct unlikelihood.

### Gaps
- I could not identify one specific paper matching the user's description ("an architecture that combines gameplays inclined toward bad plays as well as good plays"). The candidates are listed above.
- I found no quantitative head-to-head results of negative-BC/unlikelihood versus advantage-weighting for continuous-control games.
- The D-REX details (automatic ranking from noise injection) were not fetched in this pass.

## Q4. Keeping demonstration knowledge during PPO fine-tuning (DAPG, KL to the BC policy, BC auxiliary loss, forgetting, critic warm-up)

### Takeaway
There are three proven pieces:
1. A demonstration loss whose weight decays over training (DAPG, Kickstarting).
2. A KL penalty to the frozen pretrained policy (AlphaStar, VPT). In VPT, removing it causes catastrophic forgetting of later skills.
3. Critic warm-up with the actor frozen, followed by a slow actor learning-rate ramp (PIRLNav). Without it, fine-tuning shows an early collapse and small final gains.

### Cited Findings
- **DAPG:**
  - BC pretraining, then natural policy gradient on an augmented objective that includes the demo log-likelihood weighted by w = λ0·λ1^k·max A.
  - The weight decays with the iteration counter k, to avoid biasing the gradient toward demos late in training.
  - Typical values: λ0=0.1 with λ1=0.995, or λ1=0.99.
  - Sources: [arXiv 1910.04281](https://ar5iv.arxiv.org/html/1910.04281); [AAMAS 2020](https://aamas.csc.liv.ac.uk/Proceedings/aamas2020/pdfs/p465.pdf)
  - Criticism: the decay is non-adaptive, regardless of demo quality — [arXiv 2509.15981](https://arxiv.org/html/2509.15981v2)
- **AlphaStar:** the RL loss combined policy-gradient updates (V-trace, TD(λ), UPGO) with a KL loss toward the supervised agent, to aid exploration and preserve strategic diversity. Exploiter agents were periodically reset to the supervised agent — [DeepMind blog](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/); [survey arXiv 2111.07631](https://arxiv.org/pdf/2111.07631)
- **VPT (OpenAI, Minecraft):**
  - RL fine-tuning adds an auxiliary KL loss between the RL policy and the frozen pretrained policy; this replaced the entropy bonus.
  - Without the KL term, the agent only obtained early items (logs, planks, sticks, crafting table), and progress stalled after about 100k episodes. The authors attribute this to catastrophic forgetting.
  - RL from the early-game VPT model reached a reward of about 25, versus about 13 from the foundation model.
  - Source: [Baker et al., arXiv 2206.11795](https://arxiv.org/pdf/2206.11795)
- **PIRLNav (CVPR 2023), BC→PPO-style RL fine-tuning for ObjectNav:**
  - **Phase 1 (critic learning):** the BC actor is frozen and only the critic trains on rollouts sampled from that policy, for about 8M steps until the critic loss plateaus. The critic's final linear layer is initialized near zero.
  - **Phase 2 (interactive learning):** the actor is unfrozen. The actor LR warms up from 0 to 1.5e-5 between 8M and 12M steps, while the critic LR decays from 2.5e-4 to 1.5e-5. The visual encoder stays frozen.
  - **Without the transition**, there is a rapid early performance drop and only small gains.
  - **Ablation** (20k demos, success / SPL):
    - BC: 52.0%
    - Naive BC→RL: 53.6%
    - + critic learning: 56.7%
    - + critic decay: 59.4%
    - Full PIRLNav: 61.9%
  - Source: [Ramrakhya et al., arXiv 2301.07302](https://ar5iv.labs.arxiv.org/html/2301.07302)
- **Kickstarting:** the distillation weight is annealed so the student can surpass the teacher — [arXiv 1803.03835](https://arxiv.org/pdf/1803.03835)

### Inferences
- PPO-specific recommendations:
  - KL(π_θ || π_BC) is cheap to compute for both the Gaussian and the categorical heads. Apply it per head, with a separate coefficient for each. Item and drift heads likely need a stronger prior, because their rewards are sparse.
  - Start the KL coefficient at about 0.1–1 relative to the PPO loss and decay it, either geometrically (DAPG-style) or adaptively (shrink it while the return improves).
  - Keep the PPO entropy bonus small or zero on heads that have a KL prior. VPT replaced entropy with KL.
- The asymmetric critic makes warm-up especially valuable: the critic sees privileged state and can be pretrained to fit returns-to-go on the human data (offline Monte-Carlo targets), then refined online with the actor frozen.
- Freezing the vision encoder early in fine-tuning, as PIRLNav did, also protects the representation learned by BC.

### Gaps
- I found no published KL coefficient values for AlphaStar or VPT in the fetched material.
- I found no data on PPO+BC-auxiliary versus PPO+KL-to-BC head-to-head in racing.

## Q5. Game examples (AlphaStar, VPT, GT Sophy, DQfD, Counter-Strike BC, MineRL, racing) and how much human data is needed / quality vs quantity

### Takeaway
Large game successes that used human data followed the same recipe: supervised pretraining on (filtered) human data, then RL with a regularizer back to the human prior. Data volumes ranged from about 95 hours (Counter-Strike BC) to about 971k replays (AlphaStar) and about 70k hours of video (VPT). PIRLNav shows clear diminishing returns: the RL gain shrinks as BC data grows, and less than half the demos gets within about 4 points of the best result. Human demos beat synthetic shortest-path demos as a starting point for RL.

### Cited Findings
- **DQfD:**
  - Uses a combined loss: double-Q, a supervised large-margin loss on demo actions, and L2.
  - In the first 1M steps it beats Prioritized Dueling Double DQN on 41 of 42 Atari games; PDD DQN takes about 83M steps to catch up.
  - It beats pure imitation on 39 of 42 games, beats the best demo on 14 of 42, and reaches state of the art on 11 of 42 (AAAI version).
  - Source: [Hester et al., AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11757/11616)
- **Counter-Strike deathmatch:** BC on a large noisy dataset scraped from online servers (5.5M frames, about 95 h; the v1 abstract said 4M), plus smaller clean expert datasets — [Pearce & Zhu, arXiv 2104.04258](https://arxiv.org/abs/2104.04258v2); [dataset](https://huggingface.co/datasets/TeaPearce/CounterStrike_Deathmatch)
- **AlphaStar:** about 971k replays (players above 3500 MMR) for supervised learning, a further fine-tune on replays above 6200 MMR, then league RL with KL to the supervised policy — [DeepMind](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/)
- **VPT:** pretrained on unlabeled internet video using an inverse dynamics model, then BC and RL fine-tuning with KL — [arXiv 2206.11795](https://arxiv.org/pdf/2206.11795)
- **GT Sophy:**
  - Trained with QR-SAC (quantile-regression soft actor-critic); racing rules were encoded in the reward.
  - Sony reports it beat all but a few hundred top players after 24 h of training on one track.
  - Sources: [Sony AI blog](https://ai.sony/blog/training-the-worlds-fastest-gran-turismo-racer); [Nature 2022 PDF](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf)
- **PIRLNav, human vs synthetic demos:**
  - BC alone: 77k human demos reach 64.1% success; 70k frontier-exploration demos reach 44.9%; 240k shortest-path demos reach 6.4%.
  - With BC success matched at about 48%, RL fine-tuning reaches 66.1% from human data, 51.3% from frontier exploration, and 43.6% from shortest-path.
  - Source: [PIRLNav](https://ar5iv.labs.arxiv.org/html/2301.07302)
- **PIRLNav scaling:**
  - RL adds +10.1 points of success with 20k demos but only +6.3 with 77k.
  - 35k demos followed by RL reach 66.4%, versus 70.4% with 77k.
  - Frontier-exploration scaling saturates around 70k trajectories.
  - The 77k human demos took about 2,378 annotation hours.
  - Source: [PIRLNav](https://ar5iv.labs.arxiv.org/html/2301.07302)
- **Robomimic:** mixed-operator datasets ("worse", "okay", "better") are harder than single-operator data, and offline RL struggles on them — [arXiv 2108.03298](https://arxiv.org/pdf/2108.03298)

### Inferences
- **Recommended pipeline for SuperTuxKart:**
  1. **Label.** Compute per-step PPO reward from the 60 Hz privileged state on all human episodes (progress, off-track, rescue, item outcomes). Compute discounted returns-to-go, and mark failure windows (about 1–2 s before a rescue or crash).
  2. **Offline critic.** Train the privileged critic V(s) on Monte-Carlo or TD(λ) targets over all data, good and bad. The bad data is valuable here.
  3. **Actor pretraining (vision).**
     - Use advantage-weighted BC, w=clip(exp(β·A),0,w_max) with β≈1–3, on all segments. A = G−V(s), or use IQL-style expectile V and Q if time permits.
     - Optionally add a quality/return token to the actor, or exclude the failure windows (%BC fallback).
     - Baseline to beat: BC on the top 10–30% of segments by return.
  4. **Critic warm-up online.** Freeze the actor and run PPO rollouts updating only the critic until the value loss plateaus. Zero-init the last critic layer if the critic is not reused offline.
  5. **PPO fine-tuning.**
     - Ramp the actor LR from 0.
     - Freeze the vision encoder at first.
     - Add a per-head KL(π||π_BC) with a decaying coefficient (separate coefficients for steer/gas and for drift/item).
     - Optionally add a DAPG-style BC term on the top-percentile human segments, decaying with λ1≈0.99–0.995 per iteration.
  6. **Optional JSRL.** Start episodes with the BC policy driving for the first h seconds and the PPO policy taking over, shrinking h as PPO improves. Alternatively, use the human states as reset points (a cheap "start-state curriculum" across the track).
- **Data budget:** PIRLNav and DQfD suggest that tens of hours of mixed human play are enough to give a strong start. Beyond that, RL gains diminish, so quality and coverage (all tracks, recovery behavior) matter more than raw hours. Recovery-from-mistake segments are especially valuable for BC's covariate shift, which is another reason not to delete bad episodes outright: keep the recovery after a crash and drop or down-weight the moments leading into it.

### Gaps
- The search snippets did not explicitly confirm whether GT Sophy used any human demonstrations. My recollection is that it did not (pure RL with QR-SAC), but this should be checked against the Nature paper.
- MineRL competition findings (e.g. BC+RL winners, ForgER, SQIL) and Trackmania/racing imitation results were not covered in this pass.
- I found no SuperTuxKart-specific imitation or offline-RL results.
