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

## Pretrained Models

Download the contents of [`pretrained_models`](https://huggingface.co/locoxsoco/pha/tree/main) from our Hugging Face model page and move them to the `isaacgymenvs/` directory.

---

## Hand Demonstration Data

To run the task training stage, you can use the hand motion reference from [`PMP hand demonstration data`](https://drive.google.com/file/d/1h-FYRUoiSnaBExxLJx-ngBprk_26wS6c/view) [[Bae et al., 2023]](https://dl.acm.org/doi/10.1145/3588432.3591487). After downloading, place the file in `assets/amp/motions/`.

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
python train.py task=HumanoidRightHandGrabBar test=True num_envs=4 checkpoint=pretrained_models/skill_training/HumanoidRightHandGrabBar_2025-04-02_14-07-41_12000.pth
```

### Hand Rock Grab

To train:

```bash
python train.py task=HumanoidRightHandGrab headless=True
```

To see the results of a pre-trained model:

```bash
python train.py task=HumanoidRightHandGrab test=True num_envs=4 checkpoint=pretrained_models/skill_training/HumanoidRightHandGrab_2025-03-23_13-44-42_8600.pth
```

## Task Training

In the second training stage, our multi-agent full-bodies learn the following tasks:

### Rope Climbing task

#### PHA with Reusable Policy Priors

To train:

```bash
python train.py task=HumanoidRopeClimbingPHA3SetsIP headless=True
```

To see the results of a pre-trained model:

```bash
python train.py task=HumanoidRopeClimbingPHA3SetsIP test=True num_envs=4 checkpoint=pretrained_models/task_training/rope_climbing/HumanoidRopeClimbingHARLPMP3SetsIP_2025-03-26_07-06-43_5000.pth
```

#### Part-wise Motion Priors (PMP) [[Bae et al., 2023]](https://dl.acm.org/doi/10.1145/3588432.3591487)

To train:

```bash
python train.py task=HumanoidRopeClimbingPMP3SetsIP headless=True
```

To see the results of a pre-trained model:

```bash
python train.py task=HumanoidRopeClimbingPMP3SetsIP test=True num_envs=4 checkpoint=pretrained_models/task_training/rope_climbing/HumanoidRopeClimbingPMP3SetsIP_2025-04-02_08-45-26_5000.pth
```

### Monkey Bars task

#### PHA with Reusable Policy Priors

To train:

```bash
python train.py task=HumanoidMonkeyBarsPHA4SetsIP headless=True
```

To see the results of a pre-trained model:

```bash
python train.py task=HumanoidMonkeyBarsPHA4SetsIP test=True num_envs=4 checkpoint=pretrained_models/task_training/monkey_bars/HumanoidMonkeyBarsHARLPMP4SetsIP_2025-03-25_19-08-59_12000.pth
```

#### Part-wise Motion Priors (PMP) [[Bae et al., 2023]](https://dl.acm.org/doi/10.1145/3588432.3591487)

To train:

```bash
python train.py task=HumanoidMonkeyBarsPMP4SetsIP headless=True
```

To see the results of a pre-trained model:

```bash
python train.py task=HumanoidMonkeyBarsPMP4SetsIP test=True num_envs=4 checkpoint=pretrained_models/task_training/monkey_bars/HumanoidMonkeyBarsPMP4SetsIP_2025-03-25_18-52-57_12000.pth
```

### Rock Bouldering task

#### PHA with Reusable Policy Priors

To train:

```bash
python train.py task=HumanoidRockBoulderingPHA4SetsIP headless=True
```

To see the results of a pre-trained model:

```bash
python train.py task=HumanoidRockBoulderingPHA4SetsIP test=True num_envs=4 checkpoint=pretrained_models/task_training/rock_bouldering/HumanoidBoulderingHARLPMP4SetsIP_2025-03-23_18-20-31_8000.pth
```

#### Part-wise Motion Priors (PMP) [[Bae et al., 2023]](https://dl.acm.org/doi/10.1145/3588432.3591487)

To train:

```bash
python train.py task=HumanoidRockBoulderingPHA4SetsIP headless=True
```

To see the results of a pre-trained model:

```bash
python train.py task=HumanoidRockBoulderingPMP4SetsIP test=True num_envs=4 checkpoint=pretrained_models/task_training/rock_bouldering/HumanoidBoulderingPMP4SetsIP_2025-03-22_09-53-22_8000.pth
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