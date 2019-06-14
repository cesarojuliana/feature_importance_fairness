# Measuring fairness through feature importance

This repository contains an empirical comparison between feature importance and fairness measures,
by examining measures with and without bias removal technique. We use reweighing to remove bias and
SHAP (SHapley Additive exPlanations) to measure feature importance.

We calculate the following measures to evaluate fairness: disparate impact, equality of opportunity and consitency. 

We use four datasets for the study: Adult, German, Default and COMPAS, these datasets are often discussed with respect to fairness. We evaluate the results with three models: Random Forest, Gradient Boosting and Logistic Regression. The datasets used are available in folder `data`, and the results obtained are in folder `results`. 

In `utils.py` are implemented the functions used to calculate fairness measures, and to evaluate feature importance result.

**SHAP** is a technique that produces local explanations for classifier predictions. The technique can be applied to any machine learning model. For more details you can see SHAP [paper](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions) and [github](https://github.com/slundberg/shap) implementation in Python.

**Reweighing** is a echinique to reduce bias from model that assigns weights to the points in the training dataset. Lower  weights  are  assigned  to  instances  that  theprivileged class favors. For more details you can see reweighing [paper](https://link.springer.com/article/10.1007/s10115-011-0463-8).

We use AIF-360 library to apply reweighing and to calculate disparate impact and equality of opportunity metrics. For a description of library you can see the [paper](https://arxiv.org/abs/1810.01943), and the implementation in Python can be found [here](https://github.com/IBM/AIF360).
