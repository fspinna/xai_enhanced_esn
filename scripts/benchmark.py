import json
import random
from dataclasses import asdict

import awkward as ak
import numpy as np
import torch
from captum.attr import GradientShap, InputXGradient
from downtime import load_dataset
from downtime.conversion import awkward_to_sktime, awkward_to_tslearn, sktime_to_awkward
from hydra_zen import just, make_config, store, zen
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from xai_esn.esn_classifier import DeepReservoirClassifier
from xai_esn.gradient import Gradient

from classes import (
    DatasetString,
    ExperimentParams,
    ModelParams,
    add_noise,
    instance_wise_weighted_average,
)
from constants import DATASETS
from evaluation import score_classification_predict

data_store = store(group="dataset")
for dataset in DATASETS:
    data_store(DatasetString, dataset=dataset, name=dataset)


experiment_store = store(group="experiment_params")
experiment_store(
    ExperimentParams,
    multiply_by_inputs=True,
    noise_scale=0.1,
    noise_ratio=1.0,
    name="default",
)
for i in np.arange(0.1, 1.1, 0.1):
    experiment_store(
        ExperimentParams,
        multiply_by_inputs=True,
        noise_scale=0.1,
        noise_ratio=float(i),
        name=f"noise_ratio_{round(i,1)}__noise_scale_0.1",
    )


explainer_store = store(group="explainer")
explainer_store(just(GradientShap), name="gradient_shap")
explainer_store(just(InputXGradient), name="input_x_gradient")
explainer_store(just(Gradient), name="gradient")


model_store = store(group="clf")
model_store(
    MLPClassifier,
    hidden_layer_sizes=(),
    activation="logistic",
    solver="lbfgs",
    max_iter=1000,
    name="default",
    populate_full_signature=True,
)


esn_store = store(group="esn")
esn_store(
    ModelParams,
    input_size=1,
    tot_units=100,
    input_scaling=1,
    inter_scaling=1,
    spectral_radius=0.99,
    leaky=0.01,
    connectivity_recurrent=100,
    connectivity_input=100,
    connectivity_inter=10,
    name="default",
    populate_full_signature=True,
)

store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"dataset": "Wine"},
            {"esn": "default"},
            {"experiment_params": "default"},
            {"clf": "default"},
            {"explainer": "input_x_gradient"},
        ],
        dataset=None,
        esn=None,
        experiment_params=None,
        clf=None,
        explainer=None,
    ),
    name="config",
)


def benchmark(dataset, esn, experiment_params, clf, explainer):
    # set seed for reproducibility
    torch.manual_seed(experiment_params.random_seed)
    random.seed(experiment_params.random_seed)
    np.random.seed(experiment_params.random_seed)

    scores = dict()

    # load data
    d = load_dataset(name=dataset.dataset)
    X_train, y_train, X_test, y_test = d()
    X_train, train_mask = add_noise(
        np.swapaxes(np.array(X_train), 1, 2),
        noise_ratio=experiment_params.noise_ratio,
        noise_scale=experiment_params.noise_scale,
    )
    X_test, test_mask = add_noise(
        np.swapaxes(np.array(X_test), 1, 2),
        noise_ratio=experiment_params.noise_ratio,
        noise_scale=experiment_params.noise_scale,
    )
    X_train = ak.Array(np.swapaxes(X_train, 1, 2))
    X_test = ak.Array(np.swapaxes(X_test, 1, 2))
    X_train_sktime, X_test_sktime = awkward_to_sktime(X_train), awkward_to_sktime(
        X_test
    )
    scaler = TabularToSeriesAdaptor(StandardScaler(), fit_in_transform=True)
    X_train = awkward_to_tslearn(
        sktime_to_awkward(scaler.fit_transform(X_train_sktime))
    )
    X_test = awkward_to_tslearn(sktime_to_awkward(scaler.transform(X_test_sktime)))

    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)

    model = DeepReservoirClassifier(**asdict(esn))
    model.build(2)

    X_train_sequences = (
        model.forward_rnn(X_train_torch, return_sequences=True)
        .detach()
        .numpy()
        .astype(np.float_)
    )
    X_test_sequences = (
        model.forward_rnn(X_test_torch, return_sequences=True)
        .detach()
        .numpy()
        .astype(np.float_)
    )

    clf.fit(X_train_sequences.mean(axis=1), y_train)
    w = np.swapaxes(clf.coefs_[0], 0, 1)
    b = clf.intercepts_[0]

    # set readout weights to the model
    model.output.weight.data = torch.tensor(w, dtype=torch.float)
    model.output.bias.data = torch.tensor(b, dtype=torch.float)

    ## average hidden state
    y_pred = clf.predict(X_test_sequences.mean(axis=1))
    scores_ = score_classification_predict(y_test, y_pred)
    for key, value in scores_.items():
        scores[f"avg_{key}"] = value
    scores = {**scores, **scores_}

    ## last hidden state
    clf.fit(X_train_sequences[:, -1, :], y_train)
    y_pred = clf.predict(X_test_sequences[:, -1, :])
    scores_ = score_classification_predict(y_test, y_pred)
    for key, value in scores_.items():
        scores[f"last_{key}"] = value
    scores = {**scores, **scores_}

    ## random weights
    random_weights_train, _ = instance_wise_weighted_average(
        np.random.rand(*X_train_sequences.shape), X_train_sequences
    )
    random_weights_test, _ = instance_wise_weighted_average(
        np.random.rand(*X_test_sequences.shape), X_test_sequences
    )
    clf.fit(random_weights_train, y_train)
    y_pred = clf.predict(random_weights_test)
    scores_ = score_classification_predict(y_test, y_pred)
    for key, value in scores_.items():
        scores[f"random_{key}"] = value
    scores = {**scores, **scores_}

    # XAI
    if explainer is GradientShap:
        exp = explainer(model, multiply_by_inputs=experiment_params.multiply_by_inputs)
        shap_values_train = (
            exp.attribute(X_train_torch, baselines=X_train_torch)
            .detach()
            .numpy()
            .astype(np.float_)
        )
        shap_values_test = (
            exp.attribute(X_test_torch, baselines=X_train_torch)
            .detach()
            .numpy()
            .astype(np.float_)
        )
    else:
        exp = explainer(model)
        shap_values_train = (
            exp.attribute(X_train_torch).detach().numpy().astype(np.float_)
        )
        shap_values_test = (
            exp.attribute(X_test_torch).detach().numpy().astype(np.float_)
        )

    weighted_average_train, norm_importance_train = instance_wise_weighted_average(
        shap_values_train, X_train_sequences
    )
    weighted_average_test, norm_importance_test = instance_wise_weighted_average(
        shap_values_test, X_test_sequences
    )

    clf.fit(weighted_average_train, y_train)
    y_pred = clf.predict(weighted_average_test)
    scores_ = score_classification_predict(y_test, y_pred)
    for key, value in scores_.items():
        scores[f"wavg_{key}"] = value
    scores = {**scores, **scores_}

    with open("scores.json", "w") as file:
        json.dump(scores, file)


if __name__ == "__main__":
    store.add_to_hydra_store()
    zen(benchmark).hydra_main(
        config_name="config", version_base="1.1", config_path=None
    )
