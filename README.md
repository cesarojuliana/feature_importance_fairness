# Measuring fairness from feature importance

This repository contains an empirical comparision between feature importance and fairness measures,
by examining measures before and after bias removal technique. We use reweighing to remove bias and
SHAP (SHapley Additive exPlanations) to measure feature importance.

SHAP is a technique that produces local explanations for classifier predictions. The technique can be applied to any machine learning model. For more details you can see SHAP [paper](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions) and [github](https://github.com/slundberg/shap) implementation in Python.
