# EvoTorch Baseline for Myosuite Challenge

This repository contains resources implementing an [EvoTorch](https://evotorch.ai) baseline for the [Myosuite Challenge](https://sites.google.com/view/myochallenge).

## Table of Contents

* [Installation](#installation)
* [Training](#training)
* [Pre-trained Policy](#pre-trained-policy)
* [Visualization](#visualization)
* [Submission](#submission)
* [Authors](#authors)

## Installation

The easiest way to use these resources is to have [conda](https://docs.conda.io/en/latest/miniconda.html) installed. Then you can run:

```bash
git clone https://github.com/nnaisense/evotorch-myosuite-starter
cd evotorch-myosuite-starter
conda env create -f env.yml
conda activate myosuite-challenge
jupyter notebook
```

## Training

To begin training, open `train.ipynb` and step through the existing code blocks. This will train a neural net controller for the Baoding challenge using the PGPE algorithm with the ClipUp optimizer. On a 60 CPU core machine, this will take about 96 hours. The network has a slightly custom architecture that you can find in `policy.py`. From here you can easily start customizing the policy, reward function, learning algorithm or optimizer.

When we ran this notebook, we obtained the training curve below:

![Training curve for boading](boading_train_plot.png?raw=true)

## Pre-trained Policy
This repository includes a pre-trained policy in the file `agent/policies/learned_policy_boading.pkl` that obtains a score of 0.62 on the leaderboard. To visualize the behavior of this policy, see the next section.

## Visualization

To visualize, you need a policy `learned_policy_boading.pkl` in `agent/policies`. We've already provided a pre-trained agent, but by following the instructions for training above, this will now contain your new trained agent. Simply open `visualize.ipynb` and step through the existing code blocks to visualize the learned behaviours and get estimations of performance metrics.

## Submission

To submit, follow the instructions at https://github.com/ET-BE/myoChallengeEval. We've provided a modified `agent/` folder which you can copy into the submission directory, overriding any existing files that it replaces. Then you should be able to follow the submission instructoins as normal and submit your learned agent!

While following the instructions for submission, we recommend that you create a new `conda` environment to avoid mixing dependencies:

```bash
conda deactivate
conda create -n myosuite-submit
conda activate myosuite-submit
...
```

## Contact Us

If you have any questions or need any help, feel free to join our [slack](https://join.slack.com/t/evotorch/shared_invite/zt-1gglttpsz-N6K60U~9av_~tfF6tkN7CA)!

## Authors

- [**Timothy Atkinson**](mailto:timothy@nnaisense.com)
- [**Nihat Engin Toklu**](mailto:engin@nnaisense.com)
