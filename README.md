# PHA: Part-wise Heterogeneous Agents with Reusable Policy Priors for Motion Synthesis
[Website](TODO) | [Technical Paper](TODO) | [Videos](TODO)

![](./images/teaser.png)

Official implementation of the paper: "PHA: Part-wise Heterogeneous Agents with Reusable Policy Priors for Motion Synthesis" for SCA 2025.

---

## Installation

Download the Isaac Gym Preview 4 release from the [website](https://developer.nvidia.com/isaac-gym), then follow the installation instructions in the documentation.

Ensure that Isaac Gym works on your system by running one of the examples from the `python/examples` directory, like `joint_monkey.py`. Follow troubleshooting steps described in the Isaac Gym Preview 4 install instructions if you have any trouble running the samples.

Once Isaac Gym is installed and samples work within your current python environment, install the dependencies for this repo:

```bash
pip install -e .
```

---

## Training
Our method consists of two stages: skill training and task training. In the first stage, we train body-part agents to learn skills such as grabbing and climbing. In the second stage, we train multi-agent full-body humanoids to perform complex tasks like rope climbing, monkey bars, and rock bouldering.

## Skill Training

In this first training stage, our hands body-part agents learn the following skills:

### Hand Bar Grab

To train:

```bash
python train.py task=HumanoidRightHandGrabBar headless=True
```

To see the results of a pre-trained model:

```bash
python train.py task=HumanoidRightHandGrabBar headless=True test=True num_envs=4 checkpoint=./pretrained_models/hand_bar_grab.pth
```

### Hand Rock Grab

To train:

```bash
python train.py task=HumanoidRightHandGrab headless=True
```

To see the results of a pre-trained model:

```bash
python train.py task=HumanoidRightHandGrab headless=True test=True num_envs=4 checkpoint=./pretrained_models/hand_rock_grab.pth
```

## Task Training

In the second training stage, our multi-agent full-bodies learn the following tasks:

### Rope Climbing task

#### PHA with Reusable Policy Priors

To train:

```bash
python train.py task=HumanoidRopeClimbingHARLPMP3SetsIP headless=True
```

To see the results of a pre-trained model:

```bash
python train.py task=HumanoidRopeClimbingHARLPMP3SetsIP headless=True test=True num_envs=4 checkpoint=./pretrained_models/rope_climbing_pha_with_policy_prior.pth
```

#### PHA without Reusable Policy Priors

To train:

```bash
python train.py task=HumanoidRopeClimbingHARLPMP3SetsIP headless=True
```

To see the results of a pre-trained model:

```bash
python train.py task=HumanoidRopeClimbingHARLPMP3SetsIP headless=True test=True num_envs=4 checkpoint=./pretrained_models/rope_climbing_pha.pth
```

#### Part-wise Motion Priors (PMP) [[Bae et al., 2023]](https://dl.acm.org/doi/10.1145/3588432.3591487)

To train:

```bash
python train.py task=HumanoidRopeClimbingPMP3SetsIP headless=True
```

To see the results of a pre-trained model:

```bash
python train.py task=HumanoidRopeClimbingPMP3SetsIP headless=True test=True num_envs=4 checkpoint=./pretrained_models/rope_climbing_pmp.pth
```

### Monkey Bars task

#### PHA with Reusable Policy Priors

To train:

```bash
python train.py task=HumanoidMonkeyBarsHARLPMP4SetsIP headless=True
```

To see the results of a pre-trained model:

```bash
python train.py task=HumanoidMonkeyBarsHARLPMP4SetsIP headless=True test=True num_envs=4 checkpoint=./pretrained_models/monkey_bars_pha_with_policy_prior.pth
```

#### PHA without Reusable Policy Priors

To train:

```bash
python train.py task=HumanoidMonkeyBarsHARLPMP4SetsIP headless=True
```

To see the results of a pre-trained model:

```bash
python train.py task=HumanoidMonkeyBarsHARLPMP4SetsIP headless=True test=True num_envs=4 checkpoint=./pretrained_models/monkey_bars_pha.pth
```

#### Part-wise Motion Priors (PMP) [[Bae et al., 2023]](https://dl.acm.org/doi/10.1145/3588432.3591487)

To train:

```bash
python train.py task=HumanoidMonkeyBarsPMP4SetsIP headless=True
```

To see the results of a pre-trained model:

```bash
python train.py task=HumanoidMonkeyBarsPMP4SetsIP headless=True test=True num_envs=4 checkpoint=./pretrained_models/monkey_bars_pmp.pth
```

### Rock Bouldering task

#### PHA with Reusable Policy Priors

To train:

```bash
python train.py task=HumanoidBoulderingWallHARLPMP4SetsIP headless=True
```

To see the results of a pre-trained model:

```bash
python train.py task=HumanoidBoulderingWallHARLPMP4SetsIP headless=True test=True num_envs=4 checkpoint=./pretrained_models/bouldering_wall_pha_with_policy_prior.pth
```

#### PHA without Reusable Policy Priors

To train:

```bash
python train.py task=HumanoidBoulderingWallHARLPMP4SetsIP headless=True
```

To see the results of a pre-trained model:

```bash
python train.py task=HumanoidBoulderingWallHARLPMP4SetsIP headless=True test=True num_envs=4 checkpoint=./pretrained_models/bouldering_wall_pha.pth
```

#### Part-wise Motion Priors (PMP) [[Bae et al., 2023]](https://dl.acm.org/doi/10.1145/3588432.3591487)

To train:

```bash
python train.py task=HumanoidBoulderingWallHARLPMP4SetsIP headless=True
```

To see the results of a pre-trained model:

```bash
python train.py task=HumanoidBoulderingWallPMP4SetsIP headless=True test=True num_envs=4 checkpoint=./pretrained_models/bouldering_wall_pmp.pth
```

---

## Citing

Please cite this work as:
```
@misc{carranza2025pha,
    author={Luis Carranza and Oscar Argudo and Carlos Andujar},
    year={2025},
    title={PHA: Part-wise Heterogeneous Agents with Reusable Policy Priors for Motion Synthesis}, 
    journal={Proc. ACM Comput. Graph. Interact. Tech.}
    url = {https://doi.org/10.1145/3747870},
    doi = {10.1145/3747870},
}
```

Also consider citing these prior works that helped contribute to this project:
```
@inproceedings{bae2023pmp,
    author = {Bae, Jinseok and Won, Jungdam and Lim, Donggeun and Min, Cheol-Hui and Kim, Young Min},
    title = {PMP: Learning to Physically Interact with Environments using Part-wise Motion Priors},
    year = {2023},
    isbn = {9798400701597},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3588432.3591487},
    doi = {10.1145/3588432.3591487},
    booktitle = {ACM SIGGRAPH 2023 Conference Proceedings},
    articleno = {64},
    numpages = {10},
    location = {Los Angeles, CA, USA},
    series = {SIGGRAPH '23}
}

@article{zhong2024harl,
    author  = {Yifan Zhong and Jakub Grudzien Kuba and Xidong Feng and Siyi Hu and Jiaming Ji and Yaodong Yang},
    title   = {Heterogeneous-Agent Reinforcement Learning},
    journal = {Journal of Machine Learning Research},
    year    = {2024},
    volume  = {25},
    number  = {32},
    pages   = {1--67},
    url     = {http://jmlr.org/papers/v25/23-0488.html}
}
```