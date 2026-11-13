# Arquitectura actor-critic asimétrica (crítico privilegiado) + espacio de acciones híbrido para PPO en SuperTuxKart

Notes in English (with some Spanish). Proposal under review: critic V(s) on privileged state (position, rivals, track progress, geometry); actor = image encoder (RGB+HUD) → MLP trunk → Gaussian head (steer, gas) + categorical heads (drift, item use); deployment actor sees only player-visible info.

Source-verification legend: items marked (fetched) were checked against the full text or official page in this session; items marked (abstract/search) were verified only from abstract or search snippets; items marked (not fetched) cite well-known primary papers by arXiv ID, but this session did not open their text, so the report writer should treat their detail as lower confidence.

## 1. Asymmetric actor-critic literature: pitfalls and fixes

### Takeaway
The proposal is a direct instance of asymmetric actor-critic (Pinto et al. 2018), which consistently helps image-based policies. A critic on the state ALONE, V(s), is theoretically biased under partial observability. The recommended fix is a critic that sees BOTH the actor's observation/history and the privileged state, V(h, s), which is provably unbiased (Baisero & Amato 2022), and MAPPO's ablations point the same way empirically. In SuperTuxKart the bias is probably small for the core driving task, because the frame+HUD reveals most of the relevant state. It is larger for rival positions behind or out of view and for item outcomes.

### Cited Findings
- Pinto, Andrychowicz, Welinder, Zaremba, Abbeel, "Asymmetric Actor Critic for Image-Based Robot Learning" (RSS 2018, arXiv 1710.06542). The critic is trained on full simulator state while the actor gets only images, using DDPG. Asymmetric inputs "significantly improve performance" in simulation, and with domain randomization they transfer sim-to-real with no real-world data. (abstract/search) — [arXiv](https://arxiv.org/abs/1710.06542); [RSS proceedings](https://roboticsproceedings.org/rss14/p08.html)
- Baisero & Amato, "Unbiased Asymmetric Reinforcement Learning under Partial Observability" (AAMAS 2022, arXiv 2105.11674): (fetched) — [ar5iv](https://ar5iv.labs.arxiv.org/html/2105.11674)
  - For non-reactive (history-dependent / recurrent) policies the state critic V(s) is generally ill-defined (Thm 4.1).
  - For reactive policies it is well-defined but generally biased (Thm 4.2). It averages over observations and histories that share a state, so it can give the same value to histories that call for different behavior.
  - V(s) is unbiased only if observations map bijectively to states (Thm 4.3), i.e. the POMDP is really an MDP, which the authors call unrealistic.
  - The fix is a history-state critic V(h, s). E_s|h[V(h,s)] = V(h) (Thm 5.1), so it gives unbiased policy gradients (Thm 5.2), with TD errors computed on consecutive (h, s) pairs.
  - Experiments: 8 partially observable navigation domains, 20 seeds each. The state critic often failed or converged to suboptimal behavior and was unstable in "Cleaner". V(h,s) was best overall and the only method to solve Heaven-Hell-3/4.
  - Value analysis showed the state critic could not separate histories that share a state.
- MAPPO, Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (NeurIPS 2022 D&B, arXiv 2103.01955): centralized-critic PPO. (fetched) — [ar5iv](https://ar5iv.labs.arxiv.org/html/2103.01955)
  - Suggestion: "include both local, agent-specific features and global features in the value function input", but "check that these features do not unnecessarily increase the input dimension".
  - Also: "Utilize value normalization to stabilize value learning" (running mean/std of targets, PopArt-style).
  - "maintain a clipping ratio ϵ under 0.2".
  - At most 10 epochs on hard environments and 15 on easy ones; prefer not to split data into many minibatches; use large batches.
- Cai et al., "Provable Partially Observable RL with Privileged Information" (NeurIPS 2024, arXiv 2412.00985). Vanilla asymmetric actor-critic can be inefficient. They propose a belief-weighted asymmetric AC with polynomial sample complexity, and show expert distillation is tractable under a "deterministic filter condition". (abstract/search) — [arXiv](https://arxiv.org/abs/2412.00985)
- Practice in legged locomotion (Light-Loco-Parkour, arXiv 2608.02653). The teacher is trained with asymmetric AC, but the teacher ACTOR is restricted to inputs the student can infer, while the critic keeps the privileged info "for more accurate value estimates". (abstract/search) — [arXiv](https://arxiv.org/pdf/2608.02653)

### Inferences
- Concrete fix for the user: build the critic input as concat(stop_grad_or_own(actor_features or small own CNN of the frame), velocity, privileged_state). Do not use privileged_state alone. If the actor uses frame stacking or an RNN, also give the critic the stacked frames or the same recurrent summary, which approximates V(h, s).
- Under clean-sim conditions, using privileged state only in the critic does not bias the actor's deployment behavior. It changes only the baseline and the advantage estimate. The actor still optimizes over what it can see. The real risks are value bias (Baisero & Amato) and the critic "explaining away" variance the actor cannot act on. Including the observation reduces both.
- Privileged features useful for a racing critic: distance_down_track / lap progress, rank, signed lateral offset from the track center (center_path_distance), heading error vs path direction, curvature of the next N path segments, speed, relative positions of the K nearest rivals, current powerup/attachment, and time remaining. pystk2-gymnasium already exposes most of these (see section 3).
- Normalize every privileged feature with a running mean/std, as for observation normalization, before it enters the critic.

### Gaps
- No published asymmetric-AC study specifically on kart racing or vision-based racing games was found. GT Sophy (Nature 2022) used QR-SAC on feature observations, not vision with an asymmetric critic (abstract/search: [Sony AI](https://ai.sony/publications/Outracing-champion-Gran-Turismo-drivers-with-deep-reinforcement-learning)); its observation and frequency details were not verified.
- Quantitative size of V(s) bias in near-fully-observable driving tasks: no source found.

## 2. Hybrid / parameterized action spaces in PPO

### Takeaway
Factor the policy as independent heads conditioned on a shared trunk. Joint log π = Σ head log-probs, and joint entropy = Σ head entropies, optionally with per-head coefficients. Use a Gaussian with clipping in the env (CleanRL/SB3 default) or a Beta distribution for bounded steer/gas; tanh-squashing requires the Jacobian log-prob correction. Use a state-independent log σ for PPO. Mask invalid discrete actions (fire with no item) by setting their logits to −inf.

### Cited Findings
- H-PPO, Fan, Su, Zhang, Yu, "Hybrid Actor-Critic RL in Parameterized Action Space" (IJCAI 2019, arXiv 1903.01344). Multiple parallel sub-actors (discrete + continuous parameters) share one critic; H-PPO outperforms prior parameterized-action RL methods. (abstract/search) — [arXiv](https://arxiv.org/abs/1903.01344); [IJCAI](https://www.ijcai.org/proceedings/2019/316)
- "37 implementation details of PPO" (Huang et al., ICLR Blog Track 2022): (fetched) — [ICLR blog](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
  - Action dimensions are independent, so prob(a) = Π prob(aᵢ) and log-probs are summed over dimensions (MultiDiscrete and continuous alike).
  - Continuous actions: a Gaussian whose mean comes from the network and whose log std is a state-independent parameter initialized to 0. Sampled actions are clipped to bounds before env.step, but the UNCLIPPED action is stored for the log-prob.
  - Invalid action masking: logits of invalid actions are set to −inf before the softmax.
  - Per-minibatch advantage normalization.
  - Entropy coefficient 0.01 (Atari) and 0.0 (MuJoCo).
  - Global grad-norm clipped to 0.5.
- Huang & Ontañón, "A Closer Look at Invalid Action Masking in Policy Gradient Algorithms" (FLAIRS 2022, arXiv 2006.14171). Masking yields valid policy gradients (a state-dependent differentiable transform) and scales with the number of invalid actions. Penalizing invalid actions with negative reward struggles and sometimes fails to find even a first reward. A masked agent can still behave usefully when the mask is removed. (abstract/search) — [arXiv](https://arxiv.org/abs/2006.14171)
- Andrychowicz et al., "What Matters in On-Policy RL? A Large-Scale Empirical Study" (ICLR 2021, arXiv 2006.05990): (fetched) — [ar5iv](https://ar5iv.labs.arxiv.org/html/2006.05990)
  - "Use softplus to transform network output into action standard deviation", adding a negative offset to lower the initial std. An initial std around 0.5 performed best on most tasks.
  - Use tanh for mapping samples into the bounded action space.
  - Initialize the last policy layer with ~100× smaller weights.
  - Regularizers, including entropy, gave no significant benefit except on HalfCheetah.
- Chou, Maturana, Scherer, "Improving Stochastic Policy Gradients... using the Beta Distribution" (ICML 2017). A Gaussian on bounded actions introduces bias. A Beta policy is bias-free and converged faster with higher scores than the Gaussian with TRPO/ACER on MuJoCo. (abstract/search) — [PMLR](https://proceedings.mlr.press/v70/chou17a.html)
- Tanh-squashed Gaussian log-likelihood correction: log π(a) = log N(u) − Σ log(1 − tanh²(uᵢ)). This is from SAC (Haarnoja et al. 2018), Appendix C. (not fetched) — [arXiv 1801.01290](https://arxiv.org/abs/1801.01290)

### Inferences (implementation recipe)
- Heads:
  - Continuous: steer ∈ [−1, 1] and gas ∈ [0, 1].
    - Option A, simplest and matching CleanRL: Gaussian with state-independent log σ (init ≈ log 0.5 to 0) and an env-side clip. Store the raw sample for the log-prob.
    - Option B: Beta(α, β), with α, β = 1 + softplus(·) mapped to the range.
    - Option C: tanh-squash with the Jacobian correction. Note there is no closed-form entropy; estimate it with −log π of the sample.
    - Clipping wastes probability mass at full lock or full throttle, which a racing agent uses often. Beta or tanh avoids that. Try A first and switch to Beta if steering saturates.
  - Discrete heads (one Categorical/Bernoulli each):
    - drift {0,1}
    - fire {0,1}, masked to 0 when powerup == none
    - nitro {0,1}, masked when energy == 0
    - brake {0,1}
    - rescue {0,1}: mask by default or leave out (see section 3)
- Joint quantities:
  - logp = logp_steer + logp_gas + Σ logp_discrete.
  - Ratio = exp(logp_new − logp_old) on the JOINT action, with a single clip.
  - Entropy bonus = Σ c_k H_k. Because the continuous Gaussian entropy is unbounded below and on a different scale from Bernoulli entropy (≤ ln 2), use separate coefficients: c_cont ≈ 0 to 1e-3 and c_disc ≈ 1e-3 to 1e-2. This is an inference; no source gives per-head values.
- Masked heads: when the mask forces a single action, that head contributes logp = 0 and entropy = 0, which is correct automatically with −inf logits. Store the mask in the rollout buffer so the PPO update recomputes log-probs under the same mask.
- Optional conditional structure: drift is only meaningful when steering hard. Independent heads conditioned on the same trunk features are usually sufficient (H-PPO style). Autoregressive heads (sampling the discrete head and feeding it into the continuous head) add complexity without clear evidence of benefit here.

### Gaps
- No primary source found that systematically compares per-head entropy weighting schemes in hybrid PPO.
- H-PPO's exact entropy handling was not verified from the full text.

## 3. Brake / nitro / rescue inclusion, and action repeat / frame skip

### Takeaway
Include brake, nitro and fire as Bernoulli heads with masking. Rescue is better handled by a scripted rule or by masking (it is rarely useful, and random exploration of it destroys progress). Use action repeat of roughly 2–4 physics steps. Andrychowicz et al. recommend trying frame skip, but it helps only in some settings.

### Cited Findings
- pystk2-gymnasium action space: (fetched) — [GitHub](https://github.com/bpiwowar/pystk2-gymnasium)
  - `acceleration` (float), `steer` (float, clipped to [−1, 1] in the example).
  - `brake`, `drift`, `fire`, `nitro`, `rescue` as 0/1 integer flags.
  - The observation space includes privileged-style fields: distance_down_track, center_path_distance, center_path, paths_start/end/width, karts_position (front-most first), items_position/type, powerup, attachment, energy, velocity, max_steer_angle, jumping, shield_time.
  - Wrappers: ConstantSizedObservations, PolarObservations, DiscreteActionsWrapper (accel 5 / steer 7 steps), FlattenerWrapper, OnlyContinuousActionsWrapper (zeroes all discrete actions).
  - No frame-skip wrapper is documented; `--render-sub-steps` only adds rendering sub-steps.
- pystk2 core Action: acceleration normalized 0..1; brake is boolean. (abstract/search) — [pystk2 docs](https://pystk2.readthedocs.io/en/stable/action.html)
- Andrychowicz et al.: "Try frame skip if possible"; it helped in 22 of 55 environment settings. (fetched) — [ar5iv](https://ar5iv.labs.arxiv.org/html/2006.05990)
- Atari convention (37 details): MaxAndSkipEnv repeats the action for 4 frames and sums rewards; FrameStack 4. (fetched) — [ICLR blog](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- RMA runs its base policy at 100 Hz on hardware, showing that different components can run at different rates. This is a robotics context and not directly transferable. (abstract/search) — [arXiv 2107.04034](https://arxiv.org/abs/2107.04034)

### Inferences
- Brake: in SuperTuxKart, brake also triggers reverse when stopped, and accel and brake together are contradictory. Options: (a) a Bernoulli brake head, masked or penalized when accel > 0.5; or (b) a single signed "longitudinal" continuous action in [−1, 1], with > 0 meaning accel and < 0 meaning brake. Option (b) removes the contradiction and is common in driving RL. It is my suggestion; I found no source.
- Nitro: Bernoulli, masked when energy == 0. Fire: Bernoulli, masked when powerup is empty.
- Rescue: random rescue presses teleport the kart back and waste time, which adds a lot of exploration noise. Recommendation: remove it from the policy and trigger it by rule, e.g. if progress < ε for T seconds or the kart is off-track/upside-down. Alternatively, keep the head but mask it unless a "stuck" detector fires. OnlyContinuousActionsWrapper shows the env designers also consider a continuous-only baseline.
- Curriculum: start with steer + gas (+ drift), then unlock nitro, fire and brake once lap completion is reliable. This keeps the early entropy small.
- Action repeat: k = 2–4 STK steps per decision (≈ 10–20 Hz if STK steps at ~1/20 s; verify `step_size` in pystk2). Stack 3–4 frames, or use an RNN, so velocity can be inferred from images. Also feed the numeric velocity, which the user already plans to do.

### Gaps
- Exact pystk2 physics step duration and default ticks per env.step were not verified in this session; check `pystk2.RaceConfig` / `step_size`.
- No source found on the SuperTuxKart-specific value of the brake, nitro or rescue actions for RL.

## 4. Recommended hyperparameters, normalization, encoder sharing

### Takeaway
Start from the CleanRL/Atari PPO defaults with continuous-control tweaks:
- γ ≈ 0.99 (0.995 if the reward is sparse progress over long laps), λ = 0.95 (0.9 per Andrychowicz)
- clip 0.1–0.2, at most 10 epochs, per-minibatch advantage normalization
- value-target normalization (running mean/std or PopArt), observation and privileged-state normalization
- reward scaling, and grad clipping at 0.5
Do NOT share the encoder between actor and critic when their inputs differ. Use separate networks, or let the critic consume the actor's features with stop-gradient plus the privileged vector.

### Cited Findings
- Defaults (37 details): (fetched) — [ICLR blog](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

  | Setting | Atari | MuJoCo |
  |---|---|---|
  | γ | 0.99 | 0.99 |
  | λ | 0.95 | 0.95 |
  | Clip ε | 0.1 | 0.2 |
  | Epochs | 4 | 10 |
  | Minibatches | 4 | 32 |
  | Learning rate | 2.5e-4, linearly annealed | 3e-4, linearly annealed |
  | Steps per rollout per env | 128 | 2048 |

  - Separate actor/critic networks by default for MuJoCo, where they "clearly win" in simpler environments.
  - Observation normalization with a running mean/var, clipped to [−10, 10].
  - Reward scaling by the std of the discounted return (no mean subtraction), clipped to [−10, 10].
  - Value-loss clipping implemented for fidelity, though it is not found to help.
- Andrychowicz et al.: (fetched) — [ar5iv](https://ar5iv.labs.arxiv.org/html/2006.05990)
  - "Use a wide value MLP (no layers shared with the policy)".
  - "Always use observation normalization".
  - "check if value function normalization improves performance" (large effect; it hurt on Walker2d).
  - GAE λ = 0.9, with no Huber loss and no PPO-style value clipping.
  - γ should be tuned per environment, starting at 0.99.
  - Clipping threshold: start at 0.25.
  - Per-minibatch advantage normalization "seems not to affect performance too much".
  - Recompute advantages once per data pass.
- MAPPO: value normalization stabilizes centralized-critic learning; ε < 0.2; at most 10 epochs on hard tasks; large batches. (fetched) — [ar5iv](https://ar5iv.labs.arxiv.org/html/2103.01955)
- PopArt (van Hasselt et al. 2016) adaptively normalizes value targets while preserving outputs. (not fetched) — [arXiv 1602.07714](https://arxiv.org/abs/1602.07714)
- GAE (Schulman et al. 2015). (not fetched) — [arXiv 1506.02438](https://arxiv.org/abs/1506.02438)
- Time-limit handling: bootstrap V(s_T) on truncation, not termination. (Pardo et al. 2018, not fetched) — [arXiv 1712.00378](https://arxiv.org/abs/1712.00378)

### Inferences
- γ for racing timescales: the effective horizon is 1/(1−γ) decisions.
  - With a 15 Hz decision rate, γ = 0.99 gives ≈ 100 steps ≈ 7 s look-ahead, which is enough to plan a corner.
  - γ = 0.995 gives ≈ 13 s, useful if the reward is sparse (rank at finish).
  - With dense progress reward (Δdistance_down_track), 0.99 is the right start.
- Encoder sharing: since the critic gets privileged inputs, a shared encoder lets value-loss gradients reshape the actor's visual features using information the actor cannot see. Andrychowicz and the 37-details defaults both favor separate networks for continuous control. Pragmatic option for vision (CNN cost):
  - critic_input = concat(actor_cnn(frame).detach(), velocity, privileged), or a small separate critic CNN.
  - This keeps the obs part of V(h, s) (section 1) without letting value gradients corrupt the actor's encoder.
  - An auxiliary loss, e.g. predicting privileged features from actor features, is a separate option that also speeds up representation learning (inference, uncited here).
- Suggested starting config:
  - 8–16 parallel envs, 256–512 steps each
  - 4 minibatches, 4–10 epochs, clip 0.1–0.2
  - lr 2.5e-4 to 3e-4 annealed
  - λ 0.95, γ 0.99
  - entropy: discrete 0.01, continuous ~0
  - value normalization ON, grad clip 0.5
  - images /255 at 84×84 to 128×128, frame stack 3–4
  - privileged-state running normalization
  - truncation bootstrapping
  - monitor approx_kl (target < ~0.02) and the clip fraction

### Gaps
- No racing-specific PPO hyperparameter study found.
- Andrychowicz found value normalization helps in some environments and hurts in others, so ablate it.

## 5. Teacher-student (privileged expert → DAgger vision policy) vs asymmetric actor-critic

### Takeaway
Teacher-student ("Learning by Cheating", Lee et al. 2020 ANYmal, RMA) usually converges faster and splits "learning to act" from "learning to see". It fails when the teacher's actions depend on information the student cannot infer (the imitation gap). Asymmetric AC optimizes the student's own objective directly, so it has no imitation gap, but it is slower from pixels. A strong hybrid uses both: (1) train a fast state-based PPO teacher whose actor uses only student-inferable state and whose critic is privileged; (2) DAgger-distill it into the vision policy; (3) optionally fine-tune with asymmetric PPO, using a BC/DAgger loss to anchor early training.

### Cited Findings
- Chen, Zhou, Koltun, Krähenbühl, "Learning by Cheating" (CoRL 2019, arXiv 1912.12294). A privileged agent drives with ground-truth layout and traffic positions, then supervises a vision-only student with on-policy supervision. It reached 100% success on the original CARLA benchmark, beat the prior state of the art on NoCrash, and cut infractions by about an order of magnitude. (abstract/search) — [PMLR](https://proceedings.mlr.press/v100/chen20a.html); [arXiv](https://arxiv.org/abs/1912.12294)
- Lee, Hwangbo, Wellhausen, Koltun, Hutter, "Learning Quadrupedal Locomotion over Challenging Terrain" (Science Robotics 2020) used a teacher with privileged terrain info distilled into a proprioceptive student. The teacher-student detail comes from a secondary summary; verify in the full text. (abstract/search) — [arXiv 2010.11251](https://arxiv.org/abs/2010.11251)
- RMA (Kumar, Fu, Pathak, Malik 2021) works in two phases. First, a base policy is trained with RL on an encoded privileged environment vector ("extrinsics"). Second, an adaptation module regresses those extrinsics from the state-action history with supervised learning. A-RMA notes that extrinsics are hard to estimate precisely and fine-tunes the base policy on the estimated values. (abstract/search) — [arXiv 2107.04034](https://arxiv.org/abs/2107.04034)
- Weihs et al., "Bridging the Imitation Gap by Adaptive Insubordination" (NeurIPS 2021). Imitating a privileged expert yields a student that "marginalizes out the privileged information", which can be suboptimal or even near-uniform (the "imitation gap"). ADVISOR adaptively weights IL vs RL losses per state and outperforms pure IL, pure RL and their sequential/parallel combinations. (abstract/search) — [arXiv](https://arxiv.org/abs/2007.12173); [NeurIPS](https://proceedings.neurips.cc/paper/2021/hash/9fc664916bce863561527f06a96f5ff3-Abstract.html)
- Song, Rohatgi, Singh, Bagnell, "To Distill or Decide? Understanding the Algorithmic Trade-off in Partially Observable RL" (NeurIPS 2025, arXiv 2510.03207). Distillation is more computationally efficient but has failure modes. The trade-off depends on the stochasticity of latent dynamics (approximate decodability vs belief contraction). "The optimal latent policy is not always the best latent policy to distill." (abstract/search) — [arXiv](https://arxiv.org/abs/2510.03207)
- Cai et al. (NeurIPS 2024): expert distillation is provably efficient under a deterministic filter condition. Vanilla asymmetric AC can be inefficient; a belief-weighted version is polynomial. (abstract/search) — [arXiv](https://arxiv.org/abs/2412.00985)
- Light-Loco-Parkour (2026) restricts the teacher actor to student-inferable inputs and gives the privileged info to the critic only, then distills. (abstract/search) — [arXiv](https://arxiv.org/pdf/2608.02653)
- DAgger, Ross, Gordon, Bagnell 2011: iteratively aggregate states visited by the learner, labeled by the expert, to avoid covariate shift. (not fetched) — [arXiv 1011.0686](https://arxiv.org/abs/1011.0686)

### Inferences
- In SuperTuxKart the latent dynamics are fairly deterministic and the track is mostly decodable from the frame + HUD/minimap. Per Song et al. and Cai et al. this favors distillation. The user's "rays/frames + velocity" student can largely infer the teacher's driving-relevant state: lateral offset, heading, upcoming curvature.
- Imitation-gap risks in STK: rivals behind the kart, items or bananas out of view, and exact progress on hidden track parts. The teacher should therefore NOT use rival positions or global geometry the student can't see. Restrict its actor to egocentric features: path points ahead, rivals within field of view, own item/energy. Keep full information in the teacher's critic (Light-Loco-Parkour pattern).
- When to choose which:
  - (a) Compute-limited and pixel rendering is the bottleneck → teacher-student. State-only rollouts can run without rendering, much faster; DAgger then needs far fewer rendered frames.
  - (b) Need the vision policy to exploit visual cues the teacher didn't use, or the gap is large → asymmetric PPO, or DAgger followed by asymmetric PPO fine-tuning (A-RMA-style fine-tuning, ADVISOR-style mixing).
- Practical DAgger loop: roll out the student, relabel each visited state with the teacher, and train with:
  - continuous heads: MSE on the mean, or NLL;
  - discrete heads: cross-entropy on the teacher's argmax/probabilities.
  - Mix β·teacher action early and decay it. Then initialize PPO from the distilled student and keep a decaying BC loss.

### Gaps
- No head-to-head benchmark of asymmetric PPO vs teacher-student in a racing game was found.
- Song et al.'s direction of the stochasticity effect was not verified from the full text (abstract only).
- The Lee et al. 2020 teacher-student specifics rest on a secondary summary.
