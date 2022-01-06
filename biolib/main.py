import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch

from graphnn import data, model_uncertainty


def get_model(args, **kwargs):
    net = model_uncertainty.SchnetModel(
        num_interactions=args.num_interactions,
        hidden_state_size=args.node_size,
        cutoff=args.cutoff,
        update_edges=args.update_edges,
        normalize_atomwise=args.atomwise_normalization,
        scale_transform=args.scale_transform,
        **kwargs
    )
    return net


def compute_ensemble_predictions(samples, model_index=None):
    mean_samples = np.array([s["mean"] for s in samples])
    var_samples = np.array([s["var"] for s in samples])
    mean = mean_samples.mean(axis=0)
    var_epistemic = mean_samples.var(axis=0)
    var_aleatoric = var_samples.mean(axis=0)
    var = var_epistemic + var_aleatoric
    predictions = {
        "mean": mean,
        "var": var,
        "var_epistemic": var_epistemic,
        "var_aleatoric": var_aleatoric
    }
    return predictions


def predict(dataset, model_path, model_args, device):
    # load model
    state_dict = torch.load(model_path, map_location=device)
    model = get_model(model_args)
    model.to(device)
    model.load_state_dict(state_dict["model"])
    # prepare data loader
    data_loader = torch.utils.data.DataLoader(dataset, 32,
        collate_fn=data.CollateAtomsdata(pin_memory=device.type == "cuda")
    )
    # predict in batches
    means = []
    vars = []
    for batch in data_loader:
        device_batch = {
            k: v.to(device=device, non_blocking=True) for k, v in batch.items()
        }
        with torch.no_grad():
            mean, var = model(device_batch)
            means.append(mean.detach().cpu().numpy().squeeze())
            vars.append(var.detach().cpu().numpy().squeeze())
    return {"mean": np.concatenate(means), "var": np.concatenate(vars)}



def ensemble_predict(dataset, model_paths, model_args, device):
    prediction_samples = []
    for model_path in model_paths:
        prediction_samples.append(
            predict(dataset, model_path, model_args, device)
        )
    # ensemple approximation
    predictions = compute_ensemble_predictions(prediction_samples)
    # return
    return predictions, prediction_samples


def compute_calibrated_predictions(calibration_model, predictions):
    calibrated_var = calibration_model.predict(predictions["var"])
    s = calibrated_var / predictions["var"]
    res = {
        "mean (eV)": predictions["mean"],
        "var (eV^2)": calibrated_var,
        "var_epistemic (eV^2)": s * predictions["var_epistemic"],
        "var_aleatoric (eV^2)": s * predictions["var_aleatoric"],
        "scaling_factor": s,
    }
    assert np.abs(
        res["var (eV^2)"] -
        (res["var_epistemic (eV^2)"] + res["var_aleatoric (eV^2)"])
    ).max() < 1e-6
    return res


def main(dataset_path):

    # load model args
    models_dir = Path("./models")
    with open(models_dir / "arguments.json", "r") as f:
        model_args = argparse.Namespace(**json.load(f))

    # load model paths
    model_paths = list(models_dir.glob("*.pth"))
    assert len(model_paths) > 0, "No models found."

    # load input dataset
    dataset = data.AseDbData(
        dataset_path,
        data.TransformRowToGraph(cutoff=model_args.cutoff),
    )
    dataset = data.BufferData(dataset)
    assert len(dataset) > 0, "Dataset appears to be empty."

    # compute predictions
    device = torch.device("cpu")
    predictions, prediction_samples = ensemble_predict(
        dataset, model_paths, model_args, device
    )

    # calibrate predictions
    calibration_model = pickle.load(open(models_dir / "calibration_model.pkl", "rb"))
    calibrated_predictions = compute_calibrated_predictions(
        calibration_model, predictions
    )

    # output results
    results = pd.DataFrame(calibrated_predictions)
    pd.set_option('display.width', None)
    pd.set_option('max_columns', None)  # show all columns
    pd.set_option("max_rows", None)  # show all rows
    pd.set_option("precision", 8)
    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help="Path to input ASE database.")
    args = parser.parse_args()
    main(args.dataset)
