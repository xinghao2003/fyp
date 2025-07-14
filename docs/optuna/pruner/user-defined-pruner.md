Title: User-Defined Pruner — Optuna 4.4.0 documentation

URL Source: http://optuna.readthedocs.io/en/stable/tutorial/20_recipes/006_user_defined_pruner.html

Markdown Content:
Note

[Go to the end](http://optuna.readthedocs.io/en/stable/tutorial/20_recipes/006_user_defined_pruner.html#sphx-glr-download-tutorial-20-recipes-006-user-defined-pruner-py) to download the full example code.

In [`optuna.pruners`](https://optuna.readthedocs.io/en/stable/reference/pruners.html#module-optuna.pruners "optuna.pruners"), we described how an objective function can optionally include calls to a pruning feature which allows Optuna to terminate an optimization trial when intermediate results do not appear promising. In this document, we describe how to implement your own pruner, i.e., a custom strategy for determining when to stop a trial.

Overview of Pruning Interface[](http://optuna.readthedocs.io/en/stable/tutorial/20_recipes/006_user_defined_pruner.html#overview-of-pruning-interface "Link to this heading")
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The [`create_study()`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html#optuna.study.create_study "optuna.study.create_study") constructor takes, as an optional argument, a pruner inheriting from [`BasePruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.BasePruner.html#optuna.pruners.BasePruner "optuna.pruners.BasePruner"). The pruner should implement the abstract method [`prune()`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.BasePruner.html#optuna.pruners.BasePruner.prune "optuna.pruners.BasePruner.prune"), which takes arguments for the associated [`Study`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study "optuna.study.Study") and [`Trial`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial "optuna.trial.Trial") and returns a boolean value: [`True`](https://docs.python.org/3/library/constants.html#True "(in Python v3.13)") if the trial should be pruned and [`False`](https://docs.python.org/3/library/constants.html#False "(in Python v3.13)") otherwise. Using the Study and Trial objects, you can access all other trials through the [`get_trials()`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.get_trials "optuna.study.Study.get_trials") method and, and from a trial, its reported intermediate values through the [`intermediate_values()`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.FrozenTrial.html#optuna.trial.FrozenTrial.intermediate_values "optuna.trial.FrozenTrial.intermediate_values") (a dictionary which maps an integer `step` to a float value).

You can refer to the source code of the built-in Optuna pruners as templates for building your own. In this document, for illustration, we describe the construction and usage of a simple (but aggressive) pruner which prunes trials that are in last place compared to completed trials at the same step.

Note

Please refer to the documentation of [`BasePruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.BasePruner.html#optuna.pruners.BasePruner "optuna.pruners.BasePruner") or, for example, [`ThresholdPruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.ThresholdPruner.html#optuna.pruners.ThresholdPruner "optuna.pruners.ThresholdPruner") or [`PercentilePruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.PercentilePruner.html#optuna.pruners.PercentilePruner "optuna.pruners.PercentilePruner") for more robust examples of pruner implementation, including error checking and complex pruner-internal logic.

An Example: Implementing `LastPlacePruner`[](http://optuna.readthedocs.io/en/stable/tutorial/20_recipes/006_user_defined_pruner.html#an-example-implementing-lastplacepruner "Link to this heading")
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

We aim to optimize the `loss` and `alpha` hyperparameters for a stochastic gradient descent classifier (`SGDClassifier`) run on the sklearn iris dataset. We implement a pruner which terminates a trial at a certain step if it is in last place compared to completed trials at the same step. We begin considering pruning after a “warmup” of 1 training step and 5 completed trials. For demonstration purposes, we [`print()`](https://docs.python.org/3/library/functions.html#print "(in Python v3.13)") a diagnostic message from `prune` when it is about to return [`True`](https://docs.python.org/3/library/constants.html#True "(in Python v3.13)") (indicating pruning).

It may be important to note that the `SGDClassifier` score, as it is evaluated on a holdout set, decreases with enough training steps due to overfitting. This means that a trial could be pruned even if it had a favorable (high) value on a previous training set. After pruning, Optuna will take the intermediate value last reported as the value of the trial.

import numpy as np
from sklearn.datasets import [load_iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris "sklearn.datasets.load_iris")
from sklearn.model_selection import [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split "sklearn.model_selection.train_test_split")
from sklearn.linear_model import [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier "sklearn.linear_model.SGDClassifier")

import optuna
from optuna.pruners import [BasePruner](https://docs.python.org/3/library/abc.html#abc.ABC "abc.ABC")
from optuna.trial._state import TrialState

class LastPlacePruner([BasePruner](https://docs.python.org/3/library/abc.html#abc.ABC "abc.ABC")):
    def  __init__ (self, warmup_steps, warmup_trials):
        self._warmup_steps = warmup_steps
        self._warmup_trials = warmup_trials

    def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:
        # Get the latest score reported from this trial
        step = trial.last_step

        if step:  # trial.last_step == None when no scores have been reported yet
            this_score = trial.intermediate_values[step]

            # Get scores from other trials in the study reported at the same step
            completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
            other_scores = [
                t.intermediate_values[step]
                for t in completed_trials
                if step in t.intermediate_values
            ]
            other_scores = sorted(other_scores)

            # Prune if this trial at this step has a lower value than all completed trials
            # at the same step. Note that steps will begin numbering at 0 in the objective
            # function definition below.
            if step >= self._warmup_steps and len(other_scores) > self._warmup_trials:
                if this_score < other_scores[0]:
                    print(f"prune() True: Trial {trial.number}, Step {step}, Score {this_score}")
                    return True

        return False

Lastly, let’s confirm the implementation is correct with the simple hyperparameter optimization.

def objective(trial):
    iris = [load_iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris "sklearn.datasets.load_iris")()
    classes = [np.unique](https://numpy.org/doc/stable/reference/generated/numpy.unique.html#numpy.unique "numpy.unique")(iris.target)
    X_train, X_valid, y_train, y_valid = [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split "sklearn.model_selection.train_test_split")(
        iris.data, iris.target, train_size=100, test_size=50, random_state=0
    )

    loss = trial.suggest_categorical("loss", ["hinge", "log_loss", "perceptron"])
    alpha = trial.suggest_float("alpha", 0.00001, 0.001, log=True)
    clf = [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier "sklearn.linear_model.SGDClassifier")(loss=loss, alpha=alpha, random_state=0)
    score = 0

    for step in range(0, 5):
        clf.partial_fit(X_train, y_train, classes=classes)
        score = clf.score(X_valid, y_valid)

        trial.report(score, step)

        if trial.should_prune():
            raise [optuna.TrialPruned](https://docs.python.org/3/library/exceptions.html#Exception "builtins.Exception")()

    return score

pruner = [LastPlacePruner](https://docs.python.org/3/library/abc.html#abc.ABC "abc.ABC")(warmup_steps=1, warmup_trials=5)
study = optuna.create_study(direction="maximize", pruner=pruner)
study.optimize(objective, n_trials=50)

prune() True: Trial 7, Step 1, Score 0.68
prune() True: Trial 11, Step 1, Score 0.6
prune() True: Trial 12, Step 2, Score 0.62
prune() True: Trial 13, Step 1, Score 0.54
prune() True: Trial 15, Step 2, Score 0.64
prune() True: Trial 16, Step 1, Score 0.64
prune() True: Trial 17, Step 1, Score 0.7
prune() True: Trial 25, Step 1, Score 0.66
prune() True: Trial 27, Step 2, Score 0.64
prune() True: Trial 30, Step 4, Score 0.64
prune() True: Trial 31, Step 4, Score 0.68
prune() True: Trial 33, Step 4, Score 0.68
prune() True: Trial 34, Step 2, Score 0.64
prune() True: Trial 35, Step 4, Score 0.68
prune() True: Trial 36, Step 4, Score 0.7
prune() True: Trial 45, Step 1, Score 0.68
prune() True: Trial 47, Step 2, Score 0.64
prune() True: Trial 49, Step 1, Score 0.62

**Total running time of the script:** (0 minutes 0.471 seconds)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io/)