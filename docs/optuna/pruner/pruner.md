Title: optuna.pruners — Optuna 4.4.0 documentation

URL Source: http://optuna.readthedocs.io/en/stable/reference/pruners.html

Markdown Content:
The [`pruners`](http://optuna.readthedocs.io/en/stable/reference/pruners.html#module-optuna.pruners "optuna.pruners") module defines a [`BasePruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.BasePruner.html#optuna.pruners.BasePruner "optuna.pruners.BasePruner") class characterized by an abstract [`prune()`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.BasePruner.html#optuna.pruners.BasePruner.prune "optuna.pruners.BasePruner.prune") method, which, for a given trial and its associated study, returns a boolean value representing whether the trial should be pruned. This determination is made based on stored intermediate values of the objective function, as previously reported for the trial using [`optuna.trial.Trial.report()`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.report "optuna.trial.Trial.report"). The remaining classes in this module represent child classes, inheriting from [`BasePruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.BasePruner.html#optuna.pruners.BasePruner "optuna.pruners.BasePruner"), which implement different pruning strategies.

Warning

Currently [`pruners`](http://optuna.readthedocs.io/en/stable/reference/pruners.html#module-optuna.pruners "optuna.pruners") module is expected to be used only for single-objective optimization.

See also

[User-Defined Pruner](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/006_user_defined_pruner.html#user-defined-pruner) tutorial could be helpful if you want to implement your own pruner classes.

[`BasePruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.BasePruner.html#optuna.pruners.BasePruner "optuna.pruners.BasePruner")Base class for pruners.
[`MedianPruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html#optuna.pruners.MedianPruner "optuna.pruners.MedianPruner")Pruner using the median stopping rule.
[`NopPruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.NopPruner.html#optuna.pruners.NopPruner "optuna.pruners.NopPruner")Pruner which never prunes trials.
[`PatientPruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.PatientPruner.html#optuna.pruners.PatientPruner "optuna.pruners.PatientPruner")Pruner which wraps another pruner with tolerance.
[`PercentilePruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.PercentilePruner.html#optuna.pruners.PercentilePruner "optuna.pruners.PercentilePruner")Pruner to keep the specified percentile of the trials.
[`SuccessiveHalvingPruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.SuccessiveHalvingPruner.html#optuna.pruners.SuccessiveHalvingPruner "optuna.pruners.SuccessiveHalvingPruner")Pruner using Asynchronous Successive Halving Algorithm.
[`HyperbandPruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html#optuna.pruners.HyperbandPruner "optuna.pruners.HyperbandPruner")Pruner using Hyperband.
[`ThresholdPruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.ThresholdPruner.html#optuna.pruners.ThresholdPruner "optuna.pruners.ThresholdPruner")Pruner to detect outlying metrics of the trials.
[`WilcoxonPruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.WilcoxonPruner.html#optuna.pruners.WilcoxonPruner "optuna.pruners.WilcoxonPruner")Pruner based on the [Wilcoxon signed-rank test](https://en.wikipedia.org/w/index.php?title=Wilcoxon_signed-rank_test&oldid=1195011212).