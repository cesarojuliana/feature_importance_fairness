# Measuring fairness from feature importance

This repository contains an empirical comparison between feature importance and fairness measures,
by examining measures before and after bias removal technique. We use reweighing to remove bias and
SHAP (SHapley Additive exPlanations) to measure feature importance.

We calculate the following measures to evaluate fairness: disparate impact, equality of opportunity and disparate impact.

SHAP is a technique that produces local explanations for classifier predictions. The technique can be applied to any machine learning model. For more details you can see SHAP [paper](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions) and [github](https://github.com/slundberg/shap) implementation in Python.

We use AIF-360 library to apply reweighing and to calculate disparate impact and equality of opportunity metrics. For a description of library you can see [paper](https://arxiv.org/abs/1810.01943) description, and the implementation in Python can be found [here](https://github.com/IBM/AIF360).
