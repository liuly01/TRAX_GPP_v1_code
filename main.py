# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn import preprocessing
from statsmodels.tsa.seasonal import STL
from torch.utils.data import TensorDataset, DataLoader
from hyperopt import hp

from models import LSTMMain
from data_utils import (
    decide_dev_test_and_folds,
    get_rolling_window_multistep,
    combine_sites_windows,
    build_dev_train_val_for_component,
    build_yoy_datasets_for_trend,
)
from train_utils import (
    run_training,
    optimize_component_hyperparams,
)

# ------------------------ Paths & basic config ------------------------ #

BASE_DIR = r""
INPUT_DATA_DIR = r""

MODEL_PATHS = {
    "trend_model": os.path.join(BASE_DIR, "trend_model.pth"),
    "season_model": os.path.join(BASE_DIR, "season_model.pth"),
    "resid_model": os.path.join(BASE_DIR, "resid_model.pth"),
}

RESULT_PATHS = {
    "combined": os.path.join(BASE_DIR, "ALL_SITES_combined_test.csv"),
}

FEATURE_COLUMNS = [
]
TARGET_COLUMN = "GPP_DT_VUT_REF"

SEASONAL_PERIOD = 12    # monthly data
INPUT_LENGTH = 12       # LSTM window length (months)
OUTPUT_LENGTH = 1       # predict 1-month GPP
LAG_YOY = 12            # YoY lag for trend IADC
LAMBDA_YOY_DEFAULT = 1.0
HYP_EVALS = 50
FINAL_EPOCHS = 300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------ Data I/O & RSTL ------------------------ #

def read_csv(path):
    data0 = []
    data = []
    site_names = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.endswith(".csv"):
                continue
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path)

            selected_cols = ["time", TARGET_COLUMN] + FEATURE_COLUMNS
            df = df[selected_cols]

            target = df[[TARGET_COLUMN]].values
            features = df[FEATURE_COLUMNS].values

            combined = np.hstack([features, target])
            data0.append(df.values)
            data.append(combined)

            site_name = os.path.splitext(file)[0]
            site_names.append(site_name)

    return data0, data, site_names


def compute_rstl_for_all_sites(data):
    trend_data_sites = []
    season_data_sites = []
    resid_data_sites = []

    for combined in data:
        trend_combined = np.zeros_like(combined)
        season_combined = np.zeros_like(combined)
        resid_combined = np.zeros_like(combined)

        for col in range(combined.shape[1]):
            series = combined[:, col]
            stl = STL(series, period=SEASONAL_PERIOD, robust=True)
            result = stl.fit()
            trend_combined[:, col] = result.trend
            season_combined[:, col] = result.seasonal
            resid_combined[:, col] = result.resid

        trend_data_sites.append(trend_combined)
        season_data_sites.append(season_combined)
        resid_data_sites.append(resid_combined)

    return trend_data_sites, season_data_sites, resid_data_sites


def scale_all_sites(trend_data_sites, season_data_sites, resid_data_sites):
    feature_scaler_trend = preprocessing.MinMaxScaler()
    target_scaler_trend = preprocessing.MinMaxScaler()
    feature_scaler_season = preprocessing.MinMaxScaler()
    target_scaler_season = preprocessing.MinMaxScaler()
    feature_scaler_resid = preprocessing.MinMaxScaler()
    target_scaler_resid = preprocessing.MinMaxScaler()

    trend_data_vs = np.vstack(trend_data_sites)
    season_data_vs = np.vstack(season_data_sites)
    resid_data_vs = np.vstack(resid_data_sites)

    trend_features_data = trend_data_vs[:, :-1]
    trend_target_data = trend_data_vs[:, -1].reshape(-1, 1)
    season_features_data = season_data_vs[:, :-1]
    season_target_data = season_data_vs[:, -1].reshape(-1, 1)
    resid_features_data = resid_data_vs[:, :-1]
    resid_target_data = resid_data_vs[:, -1].reshape(-1, 1)

    scaled_trend_features = feature_scaler_trend.fit_transform(trend_features_data)
    scaled_trend_target = target_scaler_trend.fit_transform(trend_target_data)
    scaled_season_features = feature_scaler_season.fit_transform(season_features_data)
    scaled_season_target = target_scaler_season.fit_transform(season_target_data)
    scaled_resid_features = feature_scaler_resid.fit_transform(resid_features_data)
    scaled_resid_target = target_scaler_resid.fit_transform(resid_target_data)

    scaled_trend_data = np.hstack([scaled_trend_features, scaled_trend_target])
    scaled_season_data = np.hstack([scaled_season_features, scaled_season_target])
    scaled_resid_data = np.hstack([scaled_resid_features, scaled_resid_target])

    scaled_trend_sites = []
    scaled_season_sites = []
    scaled_resid_sites = []

    start_idx = 0
    for arr in trend_data_sites:
        end_idx = start_idx + arr.shape[0]
        scaled_trend_sites.append(scaled_trend_data[start_idx:end_idx, :])
        start_idx = end_idx

    start_idx = 0
    for arr in season_data_sites:
        end_idx = start_idx + arr.shape[0]
        scaled_season_sites.append(scaled_season_data[start_idx:end_idx, :])
        start_idx = end_idx

    start_idx = 0
    for arr in resid_data_sites:
        end_idx = start_idx + arr.shape[0]
        scaled_resid_sites.append(scaled_resid_data[start_idx:end_idx, :])
        start_idx = end_idx

    return (
        scaled_trend_sites,
        scaled_season_sites,
        scaled_resid_sites,
        feature_scaler_trend,
        target_scaler_trend,
        feature_scaler_season,
        target_scaler_season,
        feature_scaler_resid,
        target_scaler_resid,
    )


# ------------------------ Integrated Gradients ------------------------ #

def compute_integrated_gradients_importance(model, X, feature_names, out_csv, baseline=None, steps=50):
    original_training_mode = model.training
    model.eval()
    if hasattr(model, "dropout"):
        model.dropout.train(False)

    dev = next(model.parameters()).device
    X = X.to(dev)
    N, num_features, seq_len = X.shape

    if baseline is None:
        baseline = torch.zeros_like(X, device=dev)
    else:
        baseline = baseline.to(dev)

    scaled_inputs = []
    for k in range(1, steps + 1):
        alpha = float(k) / float(steps)
        scaled = baseline + alpha * (X - baseline)
        scaled.requires_grad_(True)
        scaled_inputs.append(scaled)

    integrated_gradients = torch.zeros_like(X, device=dev)

    for scaled in scaled_inputs:
        preds = model(scaled.permute(0, 2, 1)).squeeze(-1)
        out = preds.mean()
        grads = torch.autograd.grad(out, scaled, retain_graph=False, create_graph=False)[0]
        integrated_gradients += grads

    integrated_gradients = (X - baseline) * integrated_gradients / float(steps)

    # Aggregate attribution over time and samples: [num_features]
    attr = integrated_gradients.abs().sum(dim=(0, 2))  # sum over N and seq_len
    attr_np = attr.detach().cpu().numpy()
    total = np.sum(attr_np) + 1e-12
    percentage = (attr_np / total) * 100.0

    df_importance = pd.DataFrame({
        "Feature": feature_names,
        "IntegratedGradients(%)": percentage,
    })
    df_importance.sort_values("IntegratedGradients(%)", ascending=False, inplace=True)
    df_importance.to_csv(out_csv, index=False)
    print(f"Integrated Gradients feature importance saved: {out_csv}")

    model.train(original_training_mode)
    if hasattr(model, "dropout"):
        model.dropout.train(original_training_mode)


# ------------------------ Main training & evaluation ------------------------ #

def main():
    torch.set_num_threads(4)

    print("Reading site CSVs...")
    data0, data, site_names = read_csv(INPUT_DATA_DIR)
    num_sites = len(site_names)
    print(f"Total sites: {num_sites}")

    print("Running robust STL decomposition (trend / season / resid)...")
    trend_data_sites, season_data_sites, resid_data_sites = compute_rstl_for_all_sites(data)

    print("Scaling all sites with global MinMax...")
    (
        scaled_trend_sites,
        scaled_season_sites,
        scaled_resid_sites,
        feature_scaler_trend,
        target_scaler_trend,
        feature_scaler_season,
        target_scaler_season,
        feature_scaler_resid,
        target_scaler_resid,
    ) = scale_all_sites(trend_data_sites, season_data_sites, resid_data_sites)

    print("Building transposed data arrays...")
    trend_labels_sites = [arr[:, -1] for arr in scaled_trend_sites]
    season_labels_sites = [arr[:, -1] for arr in scaled_season_sites]
    resid_labels_sites = [arr[:, -1] for arr in scaled_resid_sites]

    labels_trend_transposed = [arr.reshape(1, -1) for arr in trend_labels_sites]
    labels_season_transposed = [arr.reshape(1, -1) for arr in season_labels_sites]
    labels_resid_transposed = [arr.reshape(1, -1) for arr in resid_labels_sites]

    data_trend_transposed = [arr.T for arr in scaled_trend_sites]   # [F+1, T]
    data_season_transposed = [arr.T for arr in scaled_season_sites]
    data_resid_transposed = [arr.T for arr in scaled_resid_sites]

    print("Deciding Dev / Final Test / folds for each site...")
    site_infos = []
    for i in range(len(data_trend_transposed)):
        L = data_trend_transposed[i].shape[1]
        info = decide_dev_test_and_folds(L, INPUT_LENGTH)
        site_infos.append(info)

    # Hyperparameter spaces
    common_space = {
        "hidden_size": hp.quniform("hidden_size", 32, 512, 32),
        "learning_rate": hp.loguniform("learning_rate", np.log(1e-5), np.log(1e-2)),
        "num_layers": hp.choice("num_layers", [1, 2, 3]),
        "p": hp.uniform("p", 0.1, 0.5),
        "weight_decay": hp.loguniform("weight_decay", np.log(1e-5), np.log(1e-2)),
        "batch_size": hp.quniform("batch_size", 8, 64, 2),
    }
    trend_space = {
        "hidden_size": hp.quniform("hidden_size", 32, 512, 32),
        "learning_rate": hp.loguniform("learning_rate", np.log(1e-5), np.log(1e-2)),
        "num_layers": hp.choice("num_layers", [1, 2, 3]),
        "p": hp.uniform("p", 0.05, 0.3),
        "weight_decay": hp.loguniform("weight_decay", np.log(1e-6), np.log(1e-2)),
        "batch_size": hp.quniform("batch_size", 8, 64, 2),
    }

    seeds = []

    print("Hyperparameter optimization for TREND component (expanding-window CV)...")
    best_params_trend, trials_trend = optimize_component_hyperparams(
        data_transposed_list=data_trend_transposed,
        labels_transposed_list=labels_trend_transposed,
        site_infos=site_infos,
        space=trend_space,
        seeds=seeds,
        window_len=INPUT_LENGTH,
        out_len=OUTPUT_LENGTH,
        device=device,
        max_evals=HYP_EVALS,
    )
    print("Best hyperparameters for TREND:", best_params_trend)

    print("Hyperparameter optimization for SEASON component (expanding-window CV)...")
    best_params_season, trials_season = optimize_component_hyperparams(
        data_transposed_list=data_season_transposed,
        labels_transposed_list=labels_season_transposed,
        site_infos=site_infos,
        space=common_space,
        seeds=seeds,
        window_len=INPUT_LENGTH,
        out_len=OUTPUT_LENGTH,
        device=device,
        max_evals=HYP_EVALS,
    )
    print("Best hyperparameters for SEASON:", best_params_season)

    print("Hyperparameter optimization for RESID component (expanding-window CV)...")
    best_params_resid, trials_resid = optimize_component_hyperparams(
        data_transposed_list=data_resid_transposed,
        labels_transposed_list=labels_resid_transposed,
        site_infos=site_infos,
        space=common_space,
        seeds=seeds,
        window_len=INPUT_LENGTH,
        out_len=OUTPUT_LENGTH,
        device=device,
        max_evals=HYP_EVALS,
    )
    print("Best hyperparameters for RESID:", best_params_resid)

    # ------------------------ Build Dev train/val datasets ------------------------ #

    print("Building Dev train/val datasets (SEASON & RESID)...")
    season_dev_train_ds, season_dev_val_ds = build_dev_train_val_for_component(
        data_transposed_list=data_season_transposed,
        labels_transposed_list=labels_season_transposed,
        site_infos=site_infos,
        window_len=INPUT_LENGTH,
        out_len=OUTPUT_LENGTH,
    )
    resid_dev_train_ds, resid_dev_val_ds = build_dev_train_val_for_component(
        data_transposed_list=data_resid_transposed,
        labels_transposed_list=labels_resid_transposed,
        site_infos=site_infos,
        window_len=INPUT_LENGTH,
        out_len=OUTPUT_LENGTH,
    )

    print("Building Dev train/val datasets and YoY datasets (TREND)...")
    trend_train_feats_per_site = []
    trend_train_labels_per_site = []
    trend_val_feats_per_site = []
    trend_val_labels_per_site = []

    for i, info in enumerate(site_infos):
        if info.get("test_only", False):
            continue
        dev_len = info["dev_len"]
        val_len = info["val_len"]

        feats_full = data_trend_transposed[i]
        labs_full = labels_trend_transposed[i]

        feats_dev = feats_full[:, :dev_len]
        labs_dev = labs_full[:, :dev_len]

        if dev_len <= INPUT_LENGTH + val_len:
            continue

        train_end = dev_len - val_len

        tr_feat = feats_dev[:, :train_end]
        tr_lab = labs_dev[:, :train_end]
        v_feat = feats_dev[:, train_end:dev_len]
        v_lab = labs_dev[:, train_end:dev_len]

        trend_train_feats_per_site.append(tr_feat)
        trend_train_labels_per_site.append(tr_lab)
        trend_val_feats_per_site.append(v_feat)
        trend_val_labels_per_site.append(v_lab)

    # Rolling windows per site for trend
    trend_train_xy_list, trend_train_y_list = get_rolling_window_multistep(
        OUTPUT_LENGTH,
        start_idx=0,
        window_len=INPUT_LENGTH,
        feats_per_site=trend_train_feats_per_site,
        labels_per_site=trend_train_labels_per_site,
    )
    trend_val_xy_list, trend_val_y_list = get_rolling_window_multistep(
        OUTPUT_LENGTH,
        start_idx=0,
        window_len=INPUT_LENGTH,
        feats_per_site=trend_val_feats_per_site,
        labels_per_site=trend_val_labels_per_site,
    )

    trend_train_xy, trend_train_y = combine_sites_windows(trend_train_xy_list, trend_train_y_list)
    trend_val_xy, trend_val_y = combine_sites_windows(trend_val_xy_list, trend_val_y_list)

    trend_train_xy_tensor = torch.from_numpy(trend_train_xy).to(torch.float32)
    trend_train_y_tensor = torch.from_numpy(trend_train_y).to(torch.float32).squeeze(1)
    trend_val_xy_tensor = torch.from_numpy(trend_val_xy).to(torch.float32)
    trend_val_y_tensor = torch.from_numpy(trend_val_y).to(torch.float32).squeeze(1)

    trend_dev_train_ds = TensorDataset(trend_train_xy_tensor, trend_train_y_tensor)
    trend_dev_val_ds = TensorDataset(trend_val_xy_tensor, trend_val_y_tensor)

    # YoY datasets for trend IADC
    trend_yoy_train_ds, trend_yoy_val_ds = build_yoy_datasets_for_trend(
        train_xy_list=trend_train_xy_list,
        train_y_list=trend_train_y_list,
        val_xy_list=trend_val_xy_list,
        val_y_list=trend_val_y_list,
        lag=LAG_YOY,
    )

    yoy_train_loader = DataLoader(
        dataset=trend_yoy_train_ds,
        batch_size=int(best_params_trend["batch_size"]),
        shuffle=True,
    ) if trend_yoy_train_ds is not None else None

    yoy_val_loader = DataLoader(
        dataset=trend_yoy_val_ds,
        batch_size=int(best_params_trend["batch_size"]),
        shuffle=False,
    ) if trend_yoy_val_ds is not None else None

    # ------------------------ Final training on Dev ------------------------ #

    print("Training final TREND model on Dev (with IADC)...")
    trend_model = LSTMMain(
        input_size=trend_train_xy_tensor.shape[1],
        output_len=OUTPUT_LENGTH,
        lstm_hidden=int(best_params_trend["hidden_size"]),
        num_layers=int(best_params_trend["num_layers"]),
        batch_size=int(best_params_trend["batch_size"]),
        p=float(best_params_trend["p"]),
        device=device,
    ).to(device)

    trend_train_result = run_training(
        model=trend_model,
        train_dataset=trend_dev_train_ds,
        val_dataset=trend_dev_val_ds,
        batch_size=int(best_params_trend["batch_size"]),
        learning_rate=float(best_params_trend["learning_rate"]),
        weight_decay=float(best_params_trend["weight_decay"]),
        device=device,
        epochs=FINAL_EPOCHS,
        early_stop_patience=30,
        yoy_loader=yoy_train_loader,
        lambda_yoy=LAMBDA_YOY_DEFAULT if trend_yoy_train_ds is not None else 0.0,
        yoy_val_loader=yoy_val_loader,
    )

    trend_checkpoint = {
        "model_state_dict": trend_model.state_dict(),
        "hyperparameters": best_params_trend,
    }
    torch.save(trend_checkpoint, MODEL_PATHS["trend_model"])
    joblib.dump(feature_scaler_trend, os.path.join(BASE_DIR, "trend_feature_scaler.joblib"))
    joblib.dump(target_scaler_trend, os.path.join(BASE_DIR, "trend_target_scaler.joblib"))
    print("TREND model and scalers saved.")

    print("Training final SEASON model on Dev...")
    season_model = LSTMMain(
        input_size=season_dev_train_ds[0][0].shape[1],
        output_len=OUTPUT_LENGTH,
        lstm_hidden=int(best_params_season["hidden_size"]),
        num_layers=int(best_params_season["num_layers"]),
        batch_size=int(best_params_season["batch_size"]),
        p=float(best_params_season["p"]),
        device=device,
    ).to(device)

    season_train_result = run_training(
        model=season_model,
        train_dataset=season_dev_train_ds,
        val_dataset=season_dev_val_ds,
        batch_size=int(best_params_season["batch_size"]),
        learning_rate=float(best_params_season["learning_rate"]),
        weight_decay=float(best_params_season["weight_decay"]),
        device=device,
        epochs=FINAL_EPOCHS,
        early_stop_patience=30,
        yoy_loader=None,
        lambda_yoy=0.0,
        yoy_val_loader=None,
    )

    season_checkpoint = {
        "model_state_dict": season_model.state_dict(),
        "hyperparameters": best_params_season,
    }
    torch.save(season_checkpoint, MODEL_PATHS["season_model"])
    joblib.dump(feature_scaler_season, os.path.join(BASE_DIR, "season_feature_scaler.joblib"))
    joblib.dump(target_scaler_season, os.path.join(BASE_DIR, "season_target_scaler.joblib"))
    print("SEASON model and scalers saved.")

    print("Training final RESID model on Dev...")
    resid_model = LSTMMain(
        input_size=resid_dev_train_ds[0][0].shape[1],
        output_len=OUTPUT_LENGTH,
        lstm_hidden=int(best_params_resid["hidden_size"]),
        num_layers=int(best_params_resid["num_layers"]),
        batch_size=int(best_params_resid["batch_size"]),
        p=float(best_params_resid["p"]),
        device=device,
    ).to(device)

    resid_train_result = run_training(
        model=resid_model,
        train_dataset=resid_dev_train_ds,
        val_dataset=resid_dev_val_ds,
        batch_size=int(best_params_resid["batch_size"]),
        learning_rate=float(best_params_resid["learning_rate"]),
        weight_decay=float(best_params_resid["weight_decay"]),
        device=device,
        epochs=FINAL_EPOCHS,
        early_stop_patience=30,
        yoy_loader=None,
        lambda_yoy=0.0,
        yoy_val_loader=None,
    )

    resid_checkpoint = {
        "model_state_dict": resid_model.state_dict(),
        "hyperparameters": best_params_resid,
    }
    torch.save(resid_checkpoint, MODEL_PATHS["resid_model"])
    joblib.dump(feature_scaler_resid, os.path.join(BASE_DIR, "resid_feature_scaler.joblib"))
    joblib.dump(target_scaler_resid, os.path.join(BASE_DIR, "resid_target_scaler.joblib"))
    print("RESID model and scalers saved.")

    # ------------------------ Final Test prediction & evaluation ------------------------ #

    print("\nPredicting on Final Test and saving site-wise results...")

    all_sites_dfs = []
    overall_true_list = []
    overall_pred_list = []

    finaltest_xy_trend_all = []
    finaltest_xy_season_all = []
    finaltest_xy_resid_all = []

    for i, site_name in enumerate(site_names):
        info = site_infos[i]
        L = info["L"]
        dev_len = info["dev_len"]
        final_test_len = info["final_test_len"]
        test_start = dev_len
        test_end = L

        first_label_idx = max(test_start, INPUT_LENGTH)
        if first_label_idx >= test_end:
            print(f"[Skip] Site {site_name}: not enough length for Final Test windows.")
            continue

        true_values = data0[i][first_label_idx:test_end, 1].astype(float)

        def build_test_windows_for_component(data_list, label_list):
            feats_full = data_list[i]   # [F+1, T]
            labs_full = label_list[i]   # [1, T]

            left = max(0, test_start - INPUT_LENGTH)
            right = test_end
            feats_seg = feats_full[:, left:right]
            labs_seg = labs_full[:, left:right]

            xy_list, y_list = get_rolling_window_multistep(
                OUTPUT_LENGTH,
                start_idx=0,
                window_len=INPUT_LENGTH,
                feats_per_site=[feats_seg],
                labels_per_site=[labs_seg],
            )
            if len(xy_list) == 0:
                return None, None, None

            xy, y = combine_sites_windows(xy_list, y_list)
            # xy: [N, num_features, window_len], labels time index = left + INPUT_LENGTH .. right-1
            first_label_idx_series = left + INPUT_LENGTH
            skip = max(0, test_start - first_label_idx_series)

            if skip >= xy.shape[0]:
                return None, None, None

            xy = xy[skip:, :, :]
            if y is not None:
                y = y[skip:, :, :]
            label_start_idx = first_label_idx_series + skip
            return xy, y, label_start_idx

        test_xy_trend, _, label_start_idx_trend = build_test_windows_for_component(
            data_trend_transposed, labels_trend_transposed
        )
        test_xy_season, _, label_start_idx_season = build_test_windows_for_component(
            data_season_transposed, labels_season_transposed
        )
        test_xy_resid, _, label_start_idx_resid = build_test_windows_for_component(
            data_resid_transposed, labels_resid_transposed
        )

        if (
            test_xy_trend is None
            or test_xy_season is None
            or test_xy_resid is None
        ):
            print(f"[Skip] Site {site_name}: empty Final Test windows.")
            continue

        assert (
            label_start_idx_trend == first_label_idx
        ), f"Label alignment mismatch for site {site_name} (trend)."

        test_xy_trend_tensor = torch.from_numpy(test_xy_trend).to(torch.float32).to(device)
        test_xy_season_tensor = torch.from_numpy(test_xy_season).to(torch.float32).to(device)
        test_xy_resid_tensor = torch.from_numpy(test_xy_resid).to(torch.float32).to(device)

        finaltest_xy_trend_all.append(test_xy_trend)
        finaltest_xy_season_all.append(test_xy_season)
        finaltest_xy_resid_all.append(test_xy_resid)

        trend_model.eval()
        season_model.eval()
        resid_model.eval()

        with torch.no_grad():
            pred_trend_scaled = trend_model(test_xy_trend_tensor.permute(0, 2, 1)).cpu().numpy()
            pred_season_scaled = season_model(test_xy_season_tensor.permute(0, 2, 1)).cpu().numpy()
            pred_resid_scaled = resid_model(test_xy_resid_tensor.permute(0, 2, 1)).cpu().numpy()

        pred_trend_actual = target_scaler_trend.inverse_transform(pred_trend_scaled)
        pred_season_actual = target_scaler_season.inverse_transform(pred_season_scaled)
        pred_resid_actual = target_scaler_resid.inverse_transform(pred_resid_scaled)

        pred_gpp_simple = (
            pred_trend_actual + pred_season_actual + pred_resid_actual
        ).squeeze()
        pred_gpp_simple = np.clip(pred_gpp_simple, 0, None)

        assert len(pred_gpp_simple) == len(true_values), (
            f"Alignment failed for site {site_name}: "
            f"pred={len(pred_gpp_simple)}, true={len(true_values)}, "
            f"test_start={test_start}, first_label_idx={first_label_idx}"
        )

        df_site = pd.DataFrame({
            "True": true_values,
            "Predicted_Simple": pred_gpp_simple,
        })
        site_result_path = os.path.join(BASE_DIR, f"{site_name}_test_results.csv")
        df_site.to_csv(site_result_path, index=False)
        print(f"Site {site_name} Final Test results saved to: {site_result_path}")

        all_sites_dfs.append(df_site)
        overall_true_list.append(true_values)
        overall_pred_list.append(pred_gpp_simple)

    if len(all_sites_dfs) > 0:
        combined_df_noOLS = pd.concat(all_sites_dfs, axis=0, ignore_index=True)
        combined_results_path_noOLS = RESULT_PATHS["combined_noOLS"]
        combined_df_noOLS.to_csv(combined_results_path_noOLS, index=False)
        print(f"Combined Final Test results saved to: {combined_results_path_noOLS}")
    else:
        print("No valid Final Test results were produced.")

    # ------------------------ Integrated Gradients on Final Test ------------------------ #

    if len(finaltest_xy_trend_all) > 0:
        finaltest_xy_trend = np.concatenate(finaltest_xy_trend_all, axis=0)
        finaltest_xy_season = np.concatenate(finaltest_xy_season_all, axis=0)
        finaltest_xy_resid = np.concatenate(finaltest_xy_resid_all, axis=0)

        test_xy_trend_tensor_all = torch.from_numpy(finaltest_xy_trend).to(torch.float32)
        test_xy_season_tensor_all = torch.from_numpy(finaltest_xy_season).to(torch.float32)
        test_xy_resid_tensor_all = torch.from_numpy(finaltest_xy_resid).to(torch.float32)

        ig_trend_csv = os.path.join(BASE_DIR, "trend_feature_importance.csv")
        ig_season_csv = os.path.join(BASE_DIR, "season_feature_importance.csv")
        ig_resid_csv = os.path.join(BASE_DIR, "resid_feature_importance.csv")

        compute_integrated_gradients_importance(
            model=trend_model,
            X=test_xy_trend_tensor_all,
            feature_names=FEATURE_COLUMNS,
            out_csv=ig_trend_csv,
            baseline=None,
            steps=50,
        )
        compute_integrated_gradients_importance(
            model=season_model,
            X=test_xy_season_tensor_all,
            feature_names=FEATURE_COLUMNS,
            out_csv=ig_season_csv,
            baseline=None,
            steps=50,
        )
        compute_integrated_gradients_importance(
            model=resid_model,
            X=test_xy_resid_tensor_all,
            feature_names=FEATURE_COLUMNS,
            out_csv=ig_resid_csv,
            baseline=None,
            steps=50,
        )

    print("MISSION COMPLETE!")


if __name__ == "__main__":
    main()
