# Pytorch Official

The Pytorch official models are a collection of example models that use Pytorch's high-level APIs. They are intended to be well-maintained, tested, and kept up to date with the latest Pytorch API. They should also be reasonably optimized for fast performance while still being easy to read.

These models are used as end-to-end tests, ensuring that the models run with the same speed and performance with each new Pytorch build.

## Pytorch releases
The master branch of the models are **in development**, and they target the [nightly binaries](https://github.com/pytorch/pytorch#installation) built from the [master branch of pytorch](https://github.com/pytorch/pytorch/tree/master). We aim to keep them backwards compatible with the latest release when possible (currently pytorch 1.5), but we cannot always guarantee compatibility.

**Stable versions** of the official models targeting releases of Pytorch are available as tagged branches or [downloadable releases](https://github.com/pytorch/models/releases). Model repository version numbers match the target pytorch release, such that [branch r1.4.0](https://github.com/pytorch/models/tree/r1.4.0) and [release v1.4.0](https://github.com/pytorch/models/releases/tag/v1.4.0) are compatible with [pytorch v1.4.0](https://github.com/pytorch/pytorch/releases/tag/v1.4.0).

If you are on a version of pytorch earlier than 0.4, please [update your installation](https://www.pytorch.org/install/).

## Requirements
Please follow the below steps before running models in this repo:


1. pytorch [nightly binaries](https://github.com/pytorch/pytorch#installation)

2. Install dependencies:
   ```
   pip3 install --user -r official/requirements.txt
   ```
   or
   ```
   pip install --user -r official/requirements.txt
   ```


To make Official Models easier to use, we are planning to create a pip installable Official Models package. This is being tracked in [#917](https://github.com/pytorch/models/issues/917).


## Available models

**NOTE:** Please make sure to follow the steps in the [Requirements](#requirements) section.

* [boosted_trees](boosted_trees): A Gradient Boosted Trees model to classify higgs boson process from HIGGS Data Set.
* [mnist](mnist): A basic model to classify digits from the MNIST dataset.
* [resnet](resnet): A deep residual network that can be used to classify both CIFAR-10 and ImageNet's dataset of 1000 classes.
* [transformer](transformer): A transformer model to translate the WMT English to German dataset.
* [wide_deep](wide_deep): A model that combines a wide model and deep network to classify census income data.
* More models to come!

If you would like to make any fixes or improvements to the models, please [submit a pull request](https://github.com/pytorch/models/compare).

## New Models

The team is actively working to add new models to the repository. Every model should follow the following guidelines, to uphold the
our objectives of readable, usable, and maintainable code.

**General guidelines**
* Code should be well documented and tested.
* Runnable from a blank environment with relative ease.
* Trainable on: single GPU/CPU (baseline), multiple GPUs, TPU
* Compatible with Python 2 and 3 (using [six](https://pythonhosted.org/six/) when necessary)
* Conform to [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)

**Implementation guidelines**

These guidelines exist so the model implementations are consistent for better readability and maintainability.

* Use [common utility functions](utils)
* Export SavedModel at the end of training.
* Consistent flags and flag-parsing library ([read more here](utils/flags/guidelines.md))
* Produce benchmarks and logs ([read more here](utils/logs/guidelines.md))