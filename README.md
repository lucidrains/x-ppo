<img src="./lunar.gif" width="400px"></img>

*1k steps*

## PPO

An implementation of PPO with recent random improvements

The phasic part has been removed, repository to be renamed. I do not think it does anything

## Install

```bash
$ pip install -r requirements.txt
```

You may need to install `swig`

```bash
$ apt install swig
```

## Usage

Training

```bash
$ python train.py
```

### actor loss

Clipped surrogate - Schulman et al.

```python
import torch
from x_ppo import ppo_actor_loss

action_log_probs = torch.randn(2, 8, requires_grad = True)
old_action_log_probs = torch.randn(2, 8)
advantages = torch.randn(2, 8)

loss = ppo_actor_loss(action_log_probs, old_action_log_probs, advantages)

loss.mean().backward()
```

`ppo_actor_loss`, `spo_actor_loss` and `hpo_actor_loss` all accept `return_stats = True` to also return the advantage z-score statistics, and `min_advantage_std` to return zero loss when the advantages are degenerate - DAPO dynamic sampling, [Yu et al.](https://arxiv.org/abs/2503.14476)

`hpo_actor_loss` defaults `min_advantage_std = 1e-6`; PPO and SPO default to `None`.

```python
loss, stats = ppo_actor_loss(
    action_log_probs,
    old_action_log_probs,
    advantages,
    normalize_advantages = True,
    return_stats = True,
    min_advantage_std = 1e-6
)

stats.mean, stats.var
```

### critic loss

Clipped value regression - Schulman et al.

```python
import torch
from x_ppo import clipped_value_loss_fn

values = torch.randn(2, 8, requires_grad = True)
old_values = torch.randn(2, 8)
returns = torch.randn(2, 8)

loss = clipped_value_loss_fn(values, old_values, returns)

loss.backward()
```

### advantages

Generalized advantage estimation - Schulman et al.

```python
from x_ppo import calc_gae

# rewards and values - (b n), masks - (b n)

returns, advantages = calc_gae(rewards, values, masks = masks, return_advantages = True)

actor_loss = ppo_actor_loss(action_log_probs, old_action_log_probs, advantages, mask = masks)
critic_loss = clipped_value_loss_fn(new_values, old_values, returns, mask = masks)
```

### multi-critic

One value network per reward group, each group's advantages normalized separately before combination - Mysore et al., Vijayan et al.

```python
from torch import tensor
from x_ppo import calc_gae, combine_grouped_advantages, ppo_actor_loss, clipped_value_loss_fn

# rewards and values - (b g n), one value head per reward group

gae = calc_gae(rewards, values, masks = masks, group_dim = 1, return_advantages = True)

# normalize each group, then combine with weights - (g)

weights = tensor([1., 0.5, 0.25])
advantages = combine_grouped_advantages(gae.advantages, weights = weights, group_dim = 1, mask = masks)

actor_loss = ppo_actor_loss(action_log_probs, old_action_log_probs, advantages, mask = masks)

# each value head regresses its own group's returns

critic_loss = clipped_value_loss_fn(new_values, values, gae.returns, mask = masks)

# or fold the combination into calc_gae

gae = calc_gae(rewards, values, masks = masks, weights = weights, group_dim = 1, return_advantages = True)
```

`spo_actor_loss` and `improved_value_clipping_fn` are drop-in alternatives to the actor and critic losses above, and `combine_grouped_advantages` accepts a `reduce` of `'sum'`, `'max'`, `'min'` or a custom callable

### dagger distillation

Distill any teacher into any student - one module, any action space, no environment required

```python
from x_ppo import DaggerDistiller

distiller = DaggerDistiller(student, teacher)   # state -> Distribution each

distill_loss, student_distr = distiller(states)

distill_loss.backward()
```

Composite action spaces (a pytree of heads), representation alignment against any layer of the teacher or student (repa), soft and deterministic objectives (`kl_loss`, `mse_loss`), and the full DAgger loop (roll the student out, label with the teacher, fit) are one argument each away

```python
from x_ppo import DaggerDistiller, kl_loss

distiller = DaggerDistiller(student, teacher,
    loss_fn = kl_loss,                          # soft distillation (default: hard-label nll)
    student_layer = 'trunk.2',                  # representation alignment, 1 - cosine
    teacher_layer = 'trunk.6',
    student_state_fn = lambda obs, info, step: obs['obs'],      # deployable fields
    teacher_obs_fn = lambda obs, info, step: obs['privileged'], # a privileged teacher
    env = env,
    optimizer = optimizer
)

distiller.pretrain(128, epochs = 4)   # teacher-rollout behavior-cloning warmup

for cycle in range(cycles):
    stats = distiller.step(collect_steps = 128, epochs = 2)
```

Pass `return_loss_breakdown = True` for the loss components - `LossBreakdown(loss, policy, entropy, alignment)` - on top of the total. In the loop, `kl_loss` and `mse_loss` re-evaluate the teacher on its buffered inputs at fit time, so keep the teacher in eval mode

## Citations

```bibtex
@article{Ross2011DAgger,
    title   = {A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning},
    author  = {Stephane Ross and Geoffrey J. Gordon and Drew Bagnell},
    journal = {ArXiv},
    year    = {2011},
    volume  = {abs/1011.0686},
    url     = {https://arxiv.org/abs/1011.0686}
}
```

```bibtex
@article{Yu2024REPA,
    title   = {Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think},
    author  = {Sihyun Yu and Sangkyung Kwak and Huiwon Jang and Jongheon Jeong and Jonathan Huang and Jinwoo Shin and Saining Xie},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2410.06940},
    url     = {https://arxiv.org/abs/2410.06940}
}
```

```bibtex
@article{Schulman2017ProximalPO,
    title   = {Proximal Policy Optimization Algorithms},
    author  = {John Schulman and Filip Wolski and Prafulla Dhariwal and Alec Radford and Oleg Klimov},
    journal = {ArXiv},
    year    = {2017},
    volume  = {abs/1707.06347},
    url     = {https://api.semanticscholar.org/CorpusID:28695052}
}
```

```bibtex
@article{Zhang2024ReLU2WD,
    title   = {ReLU2 Wins: Discovering Efficient Activation Functions for Sparse LLMs},
    author  = {Zhengyan Zhang and Yixin Song and Guanghui Yu and Xu Han and Yankai Lin and Chaojun Xiao and Chenyang Song and Zhiyuan Liu and Zeyu Mi and Maosong Sun},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2402.03804},
    url     = {https://api.semanticscholar.org/CorpusID:267499856}
}
```

```bibtex
@inproceedings{Lee2024SimBaSB,
    title  = {SimBa: Simplicity Bias for Scaling Up Parameters in Deep Reinforcement Learning},
    author = {Hojoon Lee and Dongyoon Hwang and Donghu Kim and Hyunseung Kim and Jun Jet Tai and Kaushik Subramanian and Peter R. Wurman and Jaegul Choo and Peter Stone and Takuma Seno},
    year   = {2024},
    url    = {https://api.semanticscholar.org/CorpusID:273346233}
}
```

```bibtex
@inproceedings{anonymous2024the,
    title   = {The Complexity Dynamics of Grokking},
    author  = {Anonymous},
    booktitle = {Submitted to The Thirteenth International Conference on Learning Representations},
    year    = {2024},
    url     = {https://openreview.net/forum?id=07N9jCfIE4},
    note    = {under review}
}
```

```bibtex
@article{Yang2020LearningLD,
    title   = {Learning Low-rank Deep Neural Networks via Singular Vector Orthogonality Regularization and Singular Value Sparsification},
    author  = {Huanrui Yang and Minxue Tang and Wei Wen and Feng Yan and Daniel Hu and Ang Li and Hai Helen Li and Yiran Chen},
    journal = {2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
    year    = {2020},
    pages   = {2899-2908},
    url     = {https://api.semanticscholar.org/CorpusID:213940794}
}
```

```bibtex
@article{Farebrother2024StopRT,
    title   = {Stop Regressing: Training Value Functions via Classification for Scalable Deep RL},
    author  = {Jesse Farebrother and Jordi Orbay and Quan Ho Vuong and Adrien Ali Taiga and Yevgen Chebotar and Ted Xiao and Alex Irpan and Sergey Levine and Pablo Samuel Castro and Aleksandra Faust and Aviral Kumar and Rishabh Agarwal},
    journal = {ArXiv},
    year   = {2024},
    volume = {abs/2403.03950},
    url    = {https://api.semanticscholar.org/CorpusID:268253088}
}
```

```bibtex
@article{Lee2024AnalysisClippedCritic,
    title   = {On Analysis of Clipped Critic Loss in Proximal Policy Gradient},
    author  = {Yongjin Lee, Moonyoung Chung},
    journal = {Authorea},
    year    = {2024}
}
```

```bibtex
@inproceedings{Felizardo2025ARL,
    title   = {A Reinforcement Learning Method for Environments with Stochastic Variables: Post-Decision Proximal Policy Optimization with Dual Critic Networks},
    author  = {Leonardo Kanashiro Felizardo and Edoardo Fadda and Paolo Brandimarte and Emilio Del-Moral-Hernandez and Mari'a Cristina Vasconcelos Nascimento},
    year    = {2025},
    url     = {https://api.semanticscholar.org/CorpusID:277621941}
}
```

```bibtex
@misc{xie2025simplepolicyoptimization,
    title   = {Simple Policy Optimization},
    author  = {Zhengpeng Xie and Qiang Zhang and Fan Yang and Marco Hutter and Renjing Xu},
    year    = {2025},
    eprint  = {2401.16025},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url = {https://arxiv.org/abs/2401.16025},
}
```

```bibtex
@inproceedings{anonymous2025flow,
    title   = {Flow Policy Gradients for Legged Robots},
    author  = {Anonymous},
    booktitle = {Submitted to The Fourteenth International Conference on Learning Representations},
    year    = {2025},
    url     = {https://openreview.net/forum?id=BA6n0nmagi},
    note    = {under review}
}
```

```bibtex
@inproceedings{Seitzer2022Pitfalls,
    title   = {On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks},
    author  = {Maximilian Seitzer and Abdul-Saboor Sheikh and Georg Martius},
    booktitle = {International Conference on Learning Representations},
    year    = {2022},
    url     = {https://openreview.net/forum?id=9O_xF9y7A-}
}
```

```bibtex
@inproceedings{Wang2025EvolutionaryPO,
    title   = {Evolutionary Policy Optimization},
    author  = {Jianren Wang and Yifan Su and Abhinav Gupta and Deepak Pathak},
    booktitle = {Advances in Neural Information Processing Systems},
    year    = {2025},
    url     = {https://arxiv.org/abs/2503.19037}
}
```

```bibtex
@misc{osband2026delightfulpolicygradient,
    title   = {Delightful Policy Gradient},
    author  = {Ian Osband},
    year    = {2026},
    eprint  = {2603.14608},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2603.14608},
}
```

```bibtex
@misc{eysenbach2018diversityneedlearningskills,
    title   = {Diversity is All You Need: Learning Skills without a Reward Function},
    author  = {Benjamin Eysenbach and Abhishek Gupta and Julian Ibarz and Sergey Levine},
    year    = {2018},
    eprint  = {1802.06070},
    archivePrefix = {arXiv},
    primaryClass = {cs.AI},
    url     = {https://arxiv.org/abs/1802.06070},
}
```

```bibtex
@misc{mustafaoglu2025evolutionarypolicyoptimization,
    title   = {Evolutionary Policy Optimization},
    author  = {Zelal Su "Lain" Mustafaoglu and Keshav Pingali and Risto Miikkulainen},
    year    = {2025},
    eprint  = {2504.12568},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2504.12568},
}
```

```bibtex
@misc{han2026firefrobeniusisometryreinitializationbalancing,
    title   = {FIRE: Frobenius-Isometry Reinitialization for Balancing the Stability-Plasticity Tradeoff},
    author  = {Isaac Han and Sangyeon Park and Seungwon Oh and Donghu Kim and Hojoon Lee and Kyung-Joong Kim},
    year    = {2026},
    eprint  = {2602.08040},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2602.08040},
}
```

```bibtex
@misc{tian2026chunkingcritic,
    title   = {Chunking the Critic: A Transformer-based Soft Actor-Critic with N-Step Returns},
    author  = {Dong Tian and Onur Celik and Gerhard Neumann},
    year    = {2026},
    eprint  = {2503.03660},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2503.03660},
}
```

```bibtex
@misc{hahn2025action,
    title   = {Action Chunking Proximal Policy Optimization for Universal Robotic Dexterous Grasping},
    author  = {Sanghyun Hahn and Jonghyun Choi},
    year    = {2025},
    url     = {https://openreview.net/forum?id=WFQnqY1c39}
}
```

```bibtex
@misc{shin2026adaptiveactionchunkingmultichunk,
    title   = {Adaptive Action Chunking via Multi-Chunk Q Value Estimation},
    author  = {Yongjae Shin and Jongseong Chae and Seongmin Kim and Jongeui Park and Youngchul Sung},
    year    = {2026},
    eprint  = {2605.10044},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2605.10044},
}
```

```bibtex
@inproceedings{ye2020mastering,
    title   = {Mastering Complex Control in MOBA Games with Deep Reinforcement Learning},
    author  = {Deheng Ye and Zhao Liu and Mingfei Sun and Bei Shi and Peilin Zhao and Hao Wu and Hongsheng Yu and Shaojie Yang and Xipeng Wu and Qingwei Guo and Qiaobo Chen and Yinyuting Yin and Hao Zhang and Tengfei Shi and Liang Wang and Qiang Fu and Wei Yang and Lanxiao Huang},
    booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
    volume  = {34},
    number  = {04},
    pages   = {6672--6679},
    year    = {2020},
    url     = {https://arxiv.org/abs/1912.09729}
}
```

```bibtex
@inproceedings{sharma2020dynamics,
    title   = {Dynamics-Aware Unsupervised Discovery of Skills},
    author  = {Archit Sharma and Shixiang Gu and Sergey Levine and Vikash Kumar and Karol Hausman},
    booktitle = {International Conference on Learning Representations},
    year    = {2020},
    url     = {https://arxiv.org/abs/1907.01657}
}
```

```bibtex
@inproceedings{mysore2022multicritic,
    title   = {Multi-Critic Actor Learning: Teaching RL Policies to Act with Style},
    author  = {Siddharth Mysore and George Cheng and Yunqi Zhao and Kate Saenko and Meng Wu},
    booktitle = {International Conference on Learning Representations},
    year    = {2022},
    url     = {https://openreview.net/forum?id=rJvY_5OzoI}
}
```

```bibtex
@article{vijayan2025multicritic,
    title   = {Multi-critic Learning for Whole-body End-effector Twist Tracking},
    author  = {Aravind Elanjimattathil Vijayan and Andrei Cramariuc and Mattia Risiglione and Christian Gehring and Marco Hutter},
    journal = {arXiv preprint arXiv:2507.08656},
    year    = {2025}
}
```

```bibtex
@article{huang2025host,
    title   = {Learning Humanoid Standing-up Control across Diverse Postures},
    author  = {Huang, Tao and Ren, Junli and Wang, Huayi and Wang, Zirui and Ben, Qingwei and Wen, Muning and Chen, Xiao and Li, Jianan and Pang, Jiangmiao},
    journal = {arXiv preprint arXiv:2502.08378},
    year    = {2025}
}
```

```bibtex
@inproceedings{schaul2015universal,
    title   = {Universal Value Function Approximators},
    author  = {Tom Schaul and Daniel Horgan and Karol Gregor and David Silver},
    booktitle = {International Conference on Machine Learning},
    year    = {2015},
    url     = {https://proceedings.mlr.press/v37/schaul15.html}
}
```

```bibtex
@article{farebrother2026jumpy,
    title   = {Compositional Planning with Jumpy World Models},
    author  = {Jesse Farebrother and Matteo Pirotta and Andrea Tirinzoni and Marc G. Bellemare and Alessandro Lazaric and Ahmed Touati},
    journal = {arXiv preprint arXiv:2602.19634},
    year    = {2026}
}
```

```bibtex
@article{kaddour2026target,
    title   = {Target Policy Optimization},
    author  = {Jean Kaddour},
    journal = {arXiv preprint arXiv:2604.06159},
    year    = {2026},
    url     = {https://arxiv.org/abs/2604.06159}
}
```
