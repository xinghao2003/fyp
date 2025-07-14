Title: Efficient Optimization Algorithms — Optuna 4.4.0 documentation

URL Source: http://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html

Markdown Content:
Note

[Go to the end](http://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html#sphx-glr-download-tutorial-10-key-features-003-efficient-optimization-algorithms-py) to download the full example code.

Optuna enables efficient hyperparameter optimization by adopting state-of-the-art algorithms for sampling hyperparameters and pruning efficiently unpromising trials.

Sampling Algorithms[](http://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html#sampling-algorithms "Link to this heading")
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Samplers basically continually narrow down the search space using the records of suggested parameter values and evaluated objective values, leading to an optimal search space which giving off parameters leading to better objective values. More detailed explanation of how samplers suggest parameters is in [`BaseSampler`](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.BaseSampler.html#optuna.samplers.BaseSampler "optuna.samplers.BaseSampler").

Optuna provides the following sampling algorithms:

*   Grid Search implemented in [`GridSampler`](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.GridSampler.html#optuna.samplers.GridSampler "optuna.samplers.GridSampler")

*   Random Search implemented in [`RandomSampler`](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.RandomSampler.html#optuna.samplers.RandomSampler "optuna.samplers.RandomSampler")

*   Tree-structured Parzen Estimator algorithm implemented in [`TPESampler`](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html#optuna.samplers.TPESampler "optuna.samplers.TPESampler")

*   CMA-ES based algorithm implemented in [`CmaEsSampler`](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.CmaEsSampler.html#optuna.samplers.CmaEsSampler "optuna.samplers.CmaEsSampler")

*   Gaussian process-based algorithm implemented in [`GPSampler`](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.GPSampler.html#optuna.samplers.GPSampler "optuna.samplers.GPSampler")

*   Algorithm to enable partial fixed parameters implemented in [`PartialFixedSampler`](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.PartialFixedSampler.html#optuna.samplers.PartialFixedSampler "optuna.samplers.PartialFixedSampler")

*   Nondominated Sorting Genetic Algorithm II implemented in [`NSGAIISampler`](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.NSGAIISampler.html#optuna.samplers.NSGAIISampler "optuna.samplers.NSGAIISampler")

*   A Quasi Monte Carlo sampling algorithm implemented in [`QMCSampler`](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.QMCSampler.html#optuna.samplers.QMCSampler "optuna.samplers.QMCSampler")

The default sampler is [`TPESampler`](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html#optuna.samplers.TPESampler "optuna.samplers.TPESampler").

Switching Samplers[](http://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html#switching-samplers "Link to this heading")
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import optuna

By default, Optuna uses [`TPESampler`](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html#optuna.samplers.TPESampler "optuna.samplers.TPESampler") as follows.

study = optuna.create_study()
print(f"Sampler is {study.sampler. __class__ . __name__ }")

Sampler is TPESampler

If you want to use different samplers for example [`RandomSampler`](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.RandomSampler.html#optuna.samplers.RandomSampler "optuna.samplers.RandomSampler") and [`CmaEsSampler`](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.CmaEsSampler.html#optuna.samplers.CmaEsSampler "optuna.samplers.CmaEsSampler"),

Sampler is RandomSampler
Sampler is CmaEsSampler

Pruning Algorithms[](http://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html#pruning-algorithms "Link to this heading")
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

`Pruners` automatically stop unpromising trials at the early stages of the training (a.k.a., automated early-stopping). Currently [`pruners`](https://optuna.readthedocs.io/en/stable/reference/pruners.html#module-optuna.pruners "optuna.pruners") module is expected to be used only for single-objective optimization.

Optuna provides the following pruning algorithms:

*   Median pruning algorithm implemented in [`MedianPruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html#optuna.pruners.MedianPruner "optuna.pruners.MedianPruner")

*   Non-pruning algorithm implemented in [`NopPruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.NopPruner.html#optuna.pruners.NopPruner "optuna.pruners.NopPruner")

*   Algorithm to operate pruner with tolerance implemented in [`PatientPruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.PatientPruner.html#optuna.pruners.PatientPruner "optuna.pruners.PatientPruner")

*   Algorithm to prune specified percentile of trials implemented in [`PercentilePruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.PercentilePruner.html#optuna.pruners.PercentilePruner "optuna.pruners.PercentilePruner")

*   Asynchronous Successive Halving algorithm implemented in [`SuccessiveHalvingPruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.SuccessiveHalvingPruner.html#optuna.pruners.SuccessiveHalvingPruner "optuna.pruners.SuccessiveHalvingPruner")

*   Hyperband algorithm implemented in [`HyperbandPruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html#optuna.pruners.HyperbandPruner "optuna.pruners.HyperbandPruner")

*   Threshold pruning algorithm implemented in [`ThresholdPruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.ThresholdPruner.html#optuna.pruners.ThresholdPruner "optuna.pruners.ThresholdPruner")

*   A pruning algorithm based on [Wilcoxon signed-rank test](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test) implemented in [`WilcoxonPruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.WilcoxonPruner.html#optuna.pruners.WilcoxonPruner "optuna.pruners.WilcoxonPruner")

We use [`MedianPruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html#optuna.pruners.MedianPruner "optuna.pruners.MedianPruner") in most examples, though basically it is outperformed by [`SuccessiveHalvingPruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.SuccessiveHalvingPruner.html#optuna.pruners.SuccessiveHalvingPruner "optuna.pruners.SuccessiveHalvingPruner") and [`HyperbandPruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html#optuna.pruners.HyperbandPruner "optuna.pruners.HyperbandPruner") as in [this benchmark result](https://github.com/optuna/optuna/wiki/Benchmarks-with-Kurobako).

Activating Pruners[](http://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html#activating-pruners "Link to this heading")
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

To turn on the pruning feature, you need to call [`report()`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.report "optuna.trial.Trial.report") and [`should_prune()`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.should_prune "optuna.trial.Trial.should_prune") after each step of the iterative training. [`report()`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.report "optuna.trial.Trial.report") periodically monitors the intermediate objective values. [`should_prune()`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.should_prune "optuna.trial.Trial.should_prune") decides termination of the trial that does not meet a predefined condition.

We would recommend using integration modules for major machine learning frameworks. Exclusive list is [`integration`](https://optuna.readthedocs.io/en/stable/reference/integration.html#module-optuna.integration "optuna.integration") and usecases are available in [optuna-examples](https://github.com/optuna/optuna-examples/).

import logging
import sys

import sklearn.datasets
import sklearn.linear_model
import sklearn.model_selection

def objective(trial):
    iris = [sklearn.datasets.load_iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris "sklearn.datasets.load_iris")()
    classes = list(set(iris.target))
    train_x, valid_x, train_y, valid_y = [sklearn.model_selection.train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split "sklearn.model_selection.train_test_split")(
        iris.data, iris.target, test_size=0.25, random_state=0
    )

    alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
    clf = [sklearn.linear_model.SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier "sklearn.linear_model.SGDClassifier")(alpha=alpha)

    for step in range(100):
        clf.partial_fit(train_x, train_y, classes=classes)

        # Report intermediate objective value.
        intermediate_value = 1.0 - clf.score(valid_x, valid_y)
        trial.report(intermediate_value, step)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise [optuna.TrialPruned](https://docs.python.org/3/library/exceptions.html#Exception "builtins.Exception")()

    return 1.0 - clf.score(valid_x, valid_y)

Set up the median stopping rule as the pruning condition.

A new study created in memory with name: no-name-38415a45-0f57-4938-80e1-e3760fab16c5
Trial 0 finished with value: 0.3157894736842105 and parameters: {'alpha': 0.02969746941842484}. Best is trial 0 with value: 0.3157894736842105.
Trial 1 finished with value: 0.3157894736842105 and parameters: {'alpha': 0.026648616235176992}. Best is trial 0 with value: 0.3157894736842105.
Trial 2 finished with value: 0.368421052631579 and parameters: {'alpha': 3.402575841449272e-05}. Best is trial 0 with value: 0.3157894736842105.
Trial 3 finished with value: 0.4736842105263158 and parameters: {'alpha': 1.3582425084613152e-05}. Best is trial 0 with value: 0.3157894736842105.
Trial 4 finished with value: 0.42105263157894735 and parameters: {'alpha': 1.68696597165952e-05}. Best is trial 0 with value: 0.3157894736842105.
Trial 5 pruned.
Trial 6 pruned.
Trial 7 pruned.
Trial 8 pruned.
Trial 9 pruned.
Trial 10 pruned.
Trial 11 pruned.
Trial 12 pruned.
Trial 13 pruned.
Trial 14 pruned.
Trial 15 pruned.
Trial 16 pruned.
Trial 17 pruned.
Trial 18 pruned.
Trial 19 pruned.

As you can see, several trials were pruned (stopped) before they finished all of the iterations. The format of message is `"Trial <Trial Number> pruned."`.

Which Sampler and Pruner Should be Used?[](http://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html#which-sampler-and-pruner-should-be-used "Link to this heading")
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

From the benchmark results which are available at [optuna/optuna - wiki “Benchmarks with Kurobako”](https://github.com/optuna/optuna/wiki/Benchmarks-with-Kurobako), at least for not deep learning tasks, we would say that

*   For [`RandomSampler`](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.RandomSampler.html#optuna.samplers.RandomSampler "optuna.samplers.RandomSampler"), [`MedianPruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html#optuna.pruners.MedianPruner "optuna.pruners.MedianPruner") is the best.

*   For [`TPESampler`](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html#optuna.samplers.TPESampler "optuna.samplers.TPESampler"), [`HyperbandPruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html#optuna.pruners.HyperbandPruner "optuna.pruners.HyperbandPruner") is the best.

However, note that the benchmark is not deep learning. For deep learning tasks, consult the below table. This table is from the [Ozaki et al., Hyperparameter Optimization Methods: Overview and Characteristics, in IEICE Trans, Vol.J103-D No.9 pp.615-631, 2020](https://doi.org/10.14923/transinfj.2019JDR0003) paper, which is written in Japanese.

| Parallel Compute Resource | Categorical/Conditional Hyperparameters | Recommended Algorithms |
| --- | --- | --- |
| Limited | No | TPE. GP-EI if search space is low-dimensional and continuous. |
| Yes | TPE. GP-EI if search space is low-dimensional and continuous |
| Sufficient | No | CMA-ES, Random Search |
| Yes | Random Search or Genetic Algorithm |

Integration Modules for Pruning[](http://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html#integration-modules-for-pruning "Link to this heading")
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

To implement pruning mechanism in much simpler forms, Optuna provides integration modules for the following libraries.

For the complete list of Optuna’s integration modules, see [`integration`](https://optuna.readthedocs.io/en/stable/reference/integration.html#module-optuna.integration "optuna.integration").

For example, [LightGBMPruningCallback](https://optuna-integration.readthedocs.io/en/stable/reference/generated/optuna_integration.LightGBMPruningCallback.html) introduces pruning without directly changing the logic of training iteration. (See also [example](https://github.com/optuna/optuna-examples/blob/main/lightgbm/lightgbm_integration.py) for the entire script.)

import optuna.integration

pruning_callback = optuna.integration.LightGBMPruningCallback(trial, 'validation-error')
gbm = lgb.train(param, dtrain, valid_sets=[dvalid], callbacks=[pruning_callback])

**Total running time of the script:** (0 minutes 0.879 seconds)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io/)