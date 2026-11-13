# Engineering a pixel-based PPO agent for SuperTuxKart: vision encoders, observation design, throughput and tooling

Scope note: about 18 search and fetch calls (Nov 2026). Primary sources read in full or in part: the GT Sophy Nature paper (full text extracted), the ICLR 2022 "37 PPO implementation details" blog, the pystk2 docs, the pystk2-gymnasium README, SB3 vec-env docs, and abstracts of Sample Factory, DrAC/RAD, Pinto et al., and Tsagkas et al. Details on Linesight, library APIs and versions were not confirmed from primary pages; those gaps are listed in each section.

## 1. Encoder, resolution, memory, auxiliary losses and augmentation for pixel PPO

### Takeaway
A small encoder trained end-to-end is the strongest default here: Nature-CNN or IMPALA-ResNet [16,32,32], about 64x64 to 96x96 input, a 2 to 4 frame stack, and DrAC-style random shift. Frozen pretrained encoders such as R3M, VC-1, DINOv2 and CLIP mainly save compute, and the evidence shows they often do not beat a learning-from-scratch baseline that uses augmentation. Predicting privileged state from pixels is a cheap auxiliary loss, and it fits well with an asymmetric critic.

### Cited Findings
- Standard Atari PPO preprocessing: 4-frame action repeat (max-pooled over the last two frames), grayscale, 84x84 resize, 4-frame stack, and pixels divided by 255. Without the /255 scaling, the first update produced a KL explosion. — [ICLR Blog Track: 37 PPO implementation details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- Nature-CNN as used in PPO: Conv(32, 8x8, s4) -> Conv(64, 4x4, s2) -> Conv(64, 3x3, s1) -> Linear(3136->512), with a trunk shared by policy and value. Orthogonal init uses gain sqrt(2) for hidden layers, 0.01 for the policy head and 1 for the value head. — [ICLR Blog Track](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- Procgen PPO uses the IMPALA CNN with depths [16,32,32] and a 256-d embedding, no LSTM. Settings: 256 steps per env, 8 minibatches, 3 epochs, gamma 0.999, lr 5e-4, clip 0.2, vf_coef 0.5, max_grad_norm 0.5, entropy 0.01, 25M timesteps on the "easy" distribution. — [ICLR Blog Track](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- Recurrent PPO: drop frame stacking (stack 1), reset LSTM state at episode boundaries, train on sequential chunks per sub-env, and store initial hidden states so that log-probs match the rollout. — [ICLR Blog Track](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- Continuous-action PPO details: state-independent log-std initialized to 0, separate actor and critic networks (generally better), clip actions only when passing them to the env and store the unclipped sample, running obs normalization clipped to [-10, 10], and reward scaling by the std of the discounted return. — [ICLR Blog Track](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- DrAC makes data augmentation sound for actor-critic methods, PPO included, by regularizing the policy and value to be invariant to augmentation rather than naively augmenting the PPO ratio. It improves Procgen test performance by about 40% over standard RL. UCB-DrAC automatically selects among 8 augmentations. — [Raileanu et al., arXiv 2006.12862](https://arxiv.org/pdf/2006.12862); code: [auto-drac](https://www.github.com/rraileanu/auto-drac)
- RAD found random crop to be the most effective augmentation "by a large margin", and DrAC also found crop best on DMControl. — [Laskin et al., RAD, arXiv 2004.14990](https://arxiv.org/pdf/2004.14990); [DrAC](https://arxiv.org/pdf/2006.12862)
- A learning-from-scratch baseline (shallow ConvNet plus augmentation) is competitive with frozen pretrained representations (PVR, MVP, R3M) and sometimes beats them, including with PPO on PixMC. The main advantage of the frozen encoders was compute cost. — [Hansen et al., "On Pre-Training for Visuo-Motor Control", arXiv 2212.05749](https://arxiv.org/pdf/2212.05749)
- MVP is the main on-policy (PPO) study of frozen pretrained encoders. — cited within [arXiv 2212.05749](https://arxiv.org/pdf/2212.05749)
- In one benchmark, policies with frozen pretrained encoders showed an average success-rate gap of 42% against end-to-end training in simulation. This was imitation-style training, not PPO. — [Wang et al., arXiv 2409.20248](https://arxiv.org/html/2409.20248v1)
- A comparison of frozen pretrained visual representations lists VC-1, iBOT, MAE, DINOv1, DINOv2, CLIP, ViT-B16 and R3M, with no single winner (for example VC-1 52.0%, CLIP 56.3%, DINOv2 43.3%, R3M 40.3% on the paper's aggregate metric; the metric was not verified). — [Tsagkas et al., arXiv 2502.03270](https://arxiv.org/pdf/2502.03270v2)
- GT Sophy did not use pixels at all. It used state features: 3D velocity, angular velocity, acceleration, tyre loads and slip angles, plus 60 course points per edge and centreline. Its MLPs were 4 layers of 2,048 units with ReLU (2048x2 for the July 2021 time-trial network, 2048x4 in other configurations). — [Wurman et al., Nature 602, 2022](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf)
- pystk2 renders color (HxWx3 uint8), depth (HxW float) and instance segmentation (HxW uint32; shift right by `ObjectType.object_type_shift` to get the object type) through `race.render_data`. — [pystk2 docs: setup](https://pystk2.readthedocs.io/en/latest/setup.html); [pystk2 data docs](https://pystk2.readthedocs.io/en/latest/data.html)

### Inferences
- Recommended baseline for SuperTuxKart: 96x96 (or 64x64 at first) RGB, with HUD cropped or masked. Color matters in STK because track edges, items and bananas are color-coded, so RGB is preferable to grayscale. Use IMPALA-CNN [16,32,32] -> 256, a 3 to 4 frame stack, and concatenate scalar velocity after the CNN. IMPALA-CNN is a safe upgrade over Nature-CNN based on its Procgen use; this is an extrapolation, not tested on STK.
- Use DrAC-style random shift (pad 4 to 6 px, then crop) with the regularization form rather than raw augmentation inside the PPO ratio. Avoid horizontal flips unless steering is also mirrored.
- Frozen encoders: DINOv2-S/14 or a CLIP ViT-B at 224px cost far more per frame than a small CNN. With a slow simulator, throughput is dominated by the sim, so inference cost is less critical. Still, the evidence above (mostly manipulation and imitation) does not show a clear win over a from-scratch CNN with augmentation. They are worth trying only as a second experiment, for example frozen DINOv2 features plus a small trainable head.
- Auxiliary losses: because privileged state is available in the sim, adding a supervised head that predicts it from the actor's features (track-relative angle, distance to centreline, curvature ahead, next-checkpoint vector, possibly depth and segmentation downsampled from `render_data`) is cheap. It likely speeds up encoder learning, by analogy with UNREAL- and SPR-style aux tasks. It stays deployment-legal because the head is dropped at test time. CURL and SPR add complexity with less clear benefit for on-policy PPO.
- Frame stack vs GRU: a 3 to 4 frame stack plus explicit velocity covers the main partial observability (motion). Use a GRU only if items or rivals need longer memory. Recurrent PPO is fragile, as the many LSTM details above show.

### Gaps
- No found paper benchmarks DINOv2, VC-1, R3M or CLIP frozen encoders under PPO on a racing or driving task. All comparisons found were manipulation or imitation.
- No source specifically quantifies a 64 vs 96 vs 128 resolution ablation for racing PPO.
- CURL and SPR results with PPO were not looked up in detail. SPR was designed for off-policy and data-efficient settings.

## 2. "Rays" observations: geometry raycasts vs pixel-derived features, and mixed observations

### Takeaway
The most successful racing agent, GT Sophy, used geometry-derived look-ahead points and "wall lidar" rather than pixels. In STK the same can be computed from pystk2 track paths (centreline segments, widths) or from depth and segmentation buffers. A mixed observation (low-res image + speed + a few geometry rays) is a sound design. Whether it counts as "deployment-legal" depends on whether rays come from pixels (legal) or the track map (privileged unless the map is considered player knowledge, as GT Sophy assumed).

### Cited Findings
- GT Sophy encoded the upcoming course as 60 equally spaced 3D points along each edge and the centreline, spanning about the next 6 s of travel (span scaled by velocity), in the agent's egocentric frame. It also used orientation relative to the centreline, wall and off-course contact flags, and the previous action values. — [Wurman et al., Nature 2022](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf)
- GT Sophy's feature ablation compared "Course points only", "Wall lidar, curvature" and "Projected position" variants against the baseline feature set. — [Nature 2022, Fig. ablation labels](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf)
- GT Sophy normalized each feature to a z-score using its known range, assuming a uniform distribution. — [Nature 2022](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf)
- pystk2-gymnasium exposes path geometry in the kart frame (z front, x left, y up): `paths_start`, `paths_end`, `paths_width`, `paths_distance`, `center_path`, `center_path_distance`, plus `front`, `velocity`, `distance_down_track`, items and other karts. — [pystk2-gymnasium README](https://github.com/bpiwowar/pystk2-gymnasium)
- pystk2 depth and instance-segmentation buffers are available alongside color. — [pystk2 docs](https://pystk2.readthedocs.io/en/latest/setup.html)

### Inferences
- **Geometry rays (from the track map):** for each of N angles in [-90 deg, +90 deg] (for example N = 9 to 17), cast a 2D ray in the kart frame and intersect it with the left and right track-edge polylines built from `paths_start`/`paths_end` +/- `paths_width/2` perpendicular to the segment. Normalize distances by a max range of about 50 to 100 m. This is cheap, exact and smooth. It is privileged by strict standards, but it plays the same role as Sophy's "static map" course points.
- **Pixel rays (deployment-legal):** from the depth buffer plus instance segmentation (road vs non-road), take K image columns of the lower half of the frame and, for each, the distance to the first non-road pixel. Alternatively, train a small segmentation or depth head and derive rays from it at deployment, supervised by the sim's buffers during training. This is the "rays from what a player sees" option.
- **Recommended actor observation (legal):** a low-res RGB stack, plus scalar speed (the HUD speedometer shows it, so it is player-visible), plus the previous action, plus optional pixel-derived rays. **Critic (privileged):** everything above plus geometry rays, centreline offset and angle, curvature look-ahead (Sophy style), distance down track, and item and rival states.
- Use asymmetric actor-critic: the critic sees full state and the actor sees images. The original paper applied this to DDPG but notes it works with any actor-critic. — [Pinto et al., arXiv 1710.06542](https://arxiv.org/pdf/1710.06542)
- A phased fallback: first train an actor on geometry rays and state to validate rewards and throughput quickly, then distill it into the pixel actor (DAgger or BC) or continue PPO from pixels with the same critic.

### Gaps
- No source quantified the gain of geometry rays vs pixels in STK specifically.
- The GT Sophy ablation numbers (lap-time deltas per feature set) were only visible as figure labels in the text extraction, not as numbers.

## 3. Throughput, vectorization, action repeat, and sample budgets of comparable racing agents

### Takeaway
Expect SuperTuxKart to be sim-bound at roughly hundreds of env-steps per second per machine at best. Use one STK process per env (pystk2 can only be initialized once per process), 8 to 16 subprocess envs (no more than the logical cores), action repeat of 2 to 4 (Sophy found no gain above 10 Hz), and the lowest graphics preset with `display=False`. Comparable agents needed millions to tens of millions of frames (Linesight forks: 2 to 30M), and GT Sophy used 10 PS4s for about 8 days.

### Cited Findings
- **pystk2 rendering:** presets `GraphicsConfig.ld()` (fastest), `sd()`, `hd()`. `display=False` keeps the GPU render path (`render_data`) without refreshing a window, which the docs recommend for pixel RL. `GraphicsConfig.none()` disables OpenGL entirely, giving no images. `pystk2.init()` must be called only once per process, and calling it twice crashes. — [pystk2 docs: setup](https://pystk2.readthedocs.io/en/latest/setup.html)
- `display_adapter` (GPU selection) is Linux-only. — [pystk2 docs](https://pystk2.readthedocs.io/en/latest/setup.html)
- pystk2-gymnasium notes "STK gymnasium uses one process", needs an `if __name__ == '__main__':` guard, and its README lists "no graphics information is available (i.e. pixmap)" as a limitation (the user's `STKPixelEnv` is therefore custom). Its race-server mode uses ZMQ with a thread pool (`--threads`, default half the cores). `--action-timeout` works only on Unix. — [pystk2-gymnasium README](https://github.com/bpiwowar/pystk2-gymnasium)
- **SB3 SubprocVecEnv:** you must wrap code in `if __name__ == "__main__":` when using spawn (the Windows default) or forkserver. Start method defaults to forkserver if available, else spawn. If the env is not IO-bound, the number of envs should not exceed the number of logical cores. — [SB3 vec_envs docs](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html); [SB3 SubprocVecEnv source](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/vec_env/subproc_vec_env.html)
- **Sample Factory (APPO):** up to about 130k FPS on one node with one GPU (VizDoom). DMLab reaches about 18k FPS with GPU rendering, and VizDoom reaches 45.4% of raw sim framerate vs 22.8% for rlpyt. — [Petrenko et al., arXiv 2006.11751](https://arxiv.org/abs/2006.11751). Independent Megaverse slides report APPO on DMLab at about 4,600 FPS with 12 CPUs and one RTX 3090. — [ICML 2021 slides](https://ICML.cc/media/icml-2021/Slides/10247.pdf)
- **GT Sophy:** acted at 10 Hz (100 ms) against a 60 Hz physics sim. The team "tested action frequencies from 5 Hz to 60 Hz and found no substantial performance gains" above 10 Hz. Time trial used 10 rollout workers (10 PS4s) for about 8 days, and racing used 21 workers for 7 to 12 days. Each worker node had about 2 vCPUs and 3.3 GB. The trainer ran on one V100 or half an A100 with about 8 vCPUs. Up to 20 cars per PS4 were simulated in parallel. Training reached the top 95% of humans "within a day or two", then ran 9 or more more days (over 45,000 driving hours). Seeds reached superhuman in 10 days typically, and 25 days at worst. Other figures: 150 s episodes, replay buffer of 10^7, minibatch 1,024, 6,000 gradient steps per epoch, and the algorithm was QR-SAC (off-policy). — [Wurman et al., Nature 2022](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf); [Sony AI blog](https://ai.sony/blog/blog-019/)
- **Linesight (Trackmania, IQN):** beat 10 of 12 official-campaign world records (May 2024), and the docs are at version 3.0.1. — [Linesight docs](https://linesight-rl.github.io/linesight/build/html/index.html); [GitHub](https://github.com/pb4git/linesight). A fork's FAQ estimates 2 to 5M frames (about 2 to 4 h with 4 collectors) for simple maps, 5 to 10M for medium and 15 to 30M for complex. It finds a "sweet spot" of 4 collectors on Linux vs 2 on Windows. — [TrackMania RL fork FAQ](https://artyomzemlyak.github.io/rulka/user_faq.html). A secondary report says an early version took 80 h at 9x game speed. — [Google Groups post (unverified)](https://groups.google.com/g/caumacomhams/c/hKYzyFIuKDI)
- **CarRacing-v2 PPO (SB3/rl-zoo):** the best community PPO models reach a mean of about 902 (igpaub) and about 883 (satcos), and many uploads are far lower or negative. The rl-zoo config uses a frame-skip of 2. — [HF DRL leaderboard data](https://huggingface.co/datasets/pkalkman/drlc-leaderboard-data/raw/87bb9441be6eb997f882eb8e2c42fb284a77176e/CarRacing-v2.csv); [pableitorr model card](https://huggingface.co/pableitorr/ppo-CarRacing-v2/blame/main/README.md)

### Inferences
- **Windows:** use `SubprocVecEnv(start_method="spawn")` or `gymnasium.vector.AsyncVectorEnv` with each env importing and initializing pystk2 inside the worker (inside the factory, never at module level). Each worker loads the game assets and an OpenGL context, so expect seconds of startup and several hundred MB of RAM per env (an estimate, not measured). On Linux, more workers per machine tend to fit, judging by the Linesight fork's Windows-vs-Linux difference (2 vs 4 collectors).
- **Sizing:** start with 8 envs, measure env-steps/s with a random policy, then scale to cores. A CPU-bound pixel PPO setup on 8 to 16 envs with STK at the ld preset plausibly yields about 100 to 600 env-steps/s. That is a guess, to be benchmarked; no published STK numbers were found. At 300 steps/s, 10M steps take about 9 hours.
- **Action repeat:** STK physics runs at a fixed internal step. Choose a decision rate around 10 to 15 Hz, following Sophy's 10 Hz result, which means repeat or skip k = 2 to 4 sim steps per action. This multiplies effective throughput by k at little cost.
- **Render resolution:** set `screen_width`/`screen_height` near the training resolution (for example 128x96, then resize to 96x96) rather than rendering 1280x720 and downsampling. This cuts GPU readback cost. That the ld preset plus small resolution is the biggest lever is an inference.
- **Sync vs async:** PPO is synchronous on-policy. With uneven step times (resets, loading) async collection helps (Sample Factory's APPO), but its custom-env integration and Windows support are less mature. Linux is recommended for Sample Factory or EnvPool. EnvPool only accelerates its built-in C++ envs, so it is not applicable to STK.
- **Budget expectation:** with racing comparables at 2 to 30M frames (Linesight fork) and Procgen PPO at 25M steps, plan for 5 to 20M env-steps for a single track, more for multiple tracks or items.

### Gaps
- No published steps-per-second benchmark for pystk2 or PySuperTuxKart (any version) was found. The user must measure it.
- The original Linesight observation and frame-rate details (believed to be a small grayscale image plus float features, with discrete actions) could not be confirmed from the primary docs in this session.
- AWS DeepRacer training-time numbers were not researched (out of call budget).
- Whether GPU rendering in multiple pystk2 processes on one Windows GPU scales linearly is unknown.

## 4. Library choice for PPO with hybrid actions and an asymmetric critic

### Takeaway
For a custom hybrid action space (continuous steer/accel plus discrete buttons such as drift, fire, nitro) and a critic that sees privileged state, a single-file CleanRL-style PPO is the most controllable option. SB3 works for Dict observations (MultiInputPolicy) but needs subclassing for hybrid distributions and asymmetric inputs. skrl and the Isaac-style ecosystem expose a separate critic "state" natively. RLlib, TorchRL and Sample Factory are heavier.

### Cited Findings
- The CleanRL-style reference implementation (`ppo_atari`, `ppo_procgen`, `ppo_continuous_action`, LSTM variants) is documented with exact defaults and the diagnostics `approx_kl` and `clipfrac`. — [ICLR Blog Track](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- Isaac Lab's RL-Games wrapper implements asymmetric actor-critic via a "critic" observation group that sets `num_states` and `state_space`, with observations given as a dict of "obs" and "states". — [Isaac Lab isaaclab_rl API](https://isaac-sim.github.io/IsaacLab/main/source/api/lab_rl/isaaclab_rl.html)
- skrl provides PPO as a model-free stochastic on-policy agent with state preprocessors (v0.10.2 docs). — [skrl PPO docs](https://skrl.readthedocs.io/en/0.10.2/modules/skrl.agents.ppo.html)
- TorchRL covers on- and off-policy model-free RL, PPO included. — [TorchRL paper, arXiv 2306.00577](https://arxiv.org/pdf/2306.00577)
- Sample Factory is designed for very high-throughput async PPO (APPO) on a single machine. — [arXiv 2006.11751](https://arxiv.org/abs/2006.11751)

### Inferences
- **CleanRL-style (recommended):** a hybrid action head is simple to write: a Normal (or tanh-squashed) distribution for [steer, accel/brake] plus independent Bernoulli or Categorical distributions for [drift, fire, nitro, rescue]. The total log-prob is the sum and the entropy is the sum, ideally with per-component entropy coefficients. The asymmetric critic is just a separate MLP reading the privileged vector from `info` or the obs dict. Everything is visible in one file, which helps debugging.
- **SB3:** MultiInputPolicy handles Dict obs, but SB3 PPO supports Box, Discrete, MultiDiscrete and MultiBinary actions, not native hybrid Tuple or Dict action spaces. The usual workaround is to encode discrete buttons as continuous thresholds or to write a custom policy and distribution. An asymmetric critic needs a custom `ActorCriticPolicy` with separate feature extractors (`share_features_extractor=False` exists, but both extractors still see the same obs dict, so you mask keys). This is feasible but fights the framework. (From general knowledge of the SB3 API; verify against the current SB3 2.x docs.)
- **skrl:** has separate policy and value model classes and (in recent 1.x versions) an env `state_space` concept used for asymmetric critics in Isaac setups. This was not verified in current docs this session.
- **RLlib (new API stack):** powerful but heavy on Windows, and Ray on Windows is less supported. Not recommended for a single-machine Windows project.
- **Sample Factory:** best raw throughput but Linux-oriented, with custom-env plus hybrid-action plus asymmetric-critic integration requiring encoder and model overrides. Worth it only if moving to Linux and becoming throughput-limited.

### Gaps
- Current (late 2026) version numbers for CleanRL, SB3, skrl, TorchRL, RLlib and Sample Factory were not verified in this session.
- Native hybrid-action and asymmetric-critic support in skrl, TorchRL and RLlib current releases was not confirmed from docs.

## 5. Stability tooling, diagnostics and failure modes in racing PPO

### Takeaway
Log `approx_kl` (healthy below about 0.02), `clipfrac`, explained variance, per-action-dimension entropy and std, action histograms, lap progress and off-track rate. Checkpoint plus periodic deterministic evaluation on fixed seeds. Most racing-specific failure modes (always-accelerate, steering entropy collapse, wall-hugging) come from reward design and the action parameterization.

### Cited Findings
- Diagnostics: `approx_kl` estimated as `((ratio-1) - logratio).mean()`. It should typically stay below about 0.02, and the ratio must equal 1 on the first minibatch (a sanity check). Also track `clipfrac`. — [ICLR Blog Track](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- Stability defaults: Adam eps 1e-5, linear LR annealing, global grad-norm clipping at 0.5, per-minibatch advantage normalization. Value-loss clipping may hurt (Andrychowicz et al.). — [ICLR Blog Track](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- GT Sophy's reward was progress, minus penalties for off-course, wall contact and loss of traction, plus collision penalties in racing. Removing the off-course penalty, or replacing progress with a "fixed step cost", were ablated as worse variants. — [Nature 2022](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf)
- GT Sophy combined throttle and brake into one [-1, 1] axis, because they are rarely used simultaneously, and used a squashed Gaussian with log-std bounds. — [Nature 2022](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf)
- 15 GT Sophy seeds all reached superhuman, but with 10 to 25 days of spread, so seed variance is large even at scale. — [Nature 2022](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf)
- pystk2-gymnasium's default reward combines track-distance progress, position among karts, and a finish bonus. — [pystk2-gymnasium (PyPI)](https://pypi.org/project/pystk2-gymnasium/)

### Inferences
- **Always-accelerate collapse:** accel is often trivially good early on. Follow Sophy and merge throttle and brake into one axis, use progress-based (not speed-based) reward so braking into corners pays off, and penalize off-track and wall contact. Monitor the accel/brake histogram.
- **Steering entropy collapse or bang-bang steering:** use a state-independent log-std with a lower bound (for example clamp log_std >= -2.5), a small entropy coefficient on the continuous part (0 to 0.005), and a separate, larger coefficient for discrete buttons. Log std per dimension. Optionally add a small steering-jerk penalty.
- **Wall-hugging or wall-riding:** a progress reward with no wall penalty lets the agent slide along walls. Add a contact penalty (privileged info is fine for the reward) and an off-centre or heading-error shaping term that is annealed later. Terminate or rescue after N steps stuck or going backwards.
- **Value-function issues:** with progress rewards and long episodes, use gamma 0.99 to 0.995 at 10 to 15 Hz (horizon of about 10 to 20 s), normalize rewards, and watch explained variance (it should rise toward 0.5 to 0.9). The asymmetric critic should give high explained variance early. If it does not, suspect reward bugs.
- **Tooling:** TensorBoard or W&B with scalars per update, a video every N updates from a dedicated eval env using the deterministic mean action, checkpoints every N updates plus "best eval lap time", saved obs-normalization statistics with the checkpoint, and fixed `torch`/`numpy`/env seeds per worker (seed + rank). Accept that STK rendering and physics may not be bit-exact across runs.
- **Curriculum:** start on one simple track without items or opponents, at short episode length, then broaden. This mirrors Sophy's scenario randomization (random start positions and speeds).

### Gaps
- No published practitioner write-up of PPO failure modes specifically in SuperTuxKart was found. The failure-mode remedies above are inferences from general PPO practice and the GT Sophy reward design.
- Explained-variance target ranges are a common heuristic and are not sourced.
