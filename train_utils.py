import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from hyperopt import fmin, tpe, Trials, STATUS_OK, space_eval

from models import LSTMMain
from data_utils import build_fold_datasets_for_component


def run_training(
    model,
    train_dataset,
    val_dataset,
    batch_size,
    learning_rate,
    weight_decay,
    device,
    epochs=300,
    early_stop_patience=30,
    yoy_loader=None,
    lambda_yoy=0.0,
    yoy_val_loader=None,
):
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-5,
    )

    mse_loss = nn.MSELoss(reduction="mean")
    smooth_l1 = nn.SmoothL1Loss(reduction="mean")

    val_best = float("inf")
    best_state = None
    train_loss_history = []
    val_loss_history = []
    early_stop_counter = 0

    if yoy_loader is not None and lambda_yoy > 0:
        yoy_iter = iter(yoy_loader)
    else:
        yoy_iter = None

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        for feature_batch, label_batch in train_loader:
            feature_batch = feature_batch.to(device).permute(0, 2, 1)
            label_batch = label_batch.to(device)

            pred = model(feature_batch)
            loss_mse = mse_loss(pred, label_batch)
            loss = loss_mse

            if yoy_iter is not None:
                try:
                    yoy_X_t, yoy_X_t12, yoy_y_t, yoy_y_t12 = next(yoy_iter)
                except StopIteration:
                    yoy_iter = iter(yoy_loader)
                    yoy_X_t, yoy_X_t12, yoy_y_t, yoy_y_t12 = next(yoy_iter)

                yoy_X_t = yoy_X_t.to(device).permute(0, 2, 1)
                yoy_X_t12 = yoy_X_t12.to(device).permute(0, 2, 1)
                yoy_y_t = yoy_y_t.to(device)
                yoy_y_t12 = yoy_y_t12.to(device)

                pred_t = model(yoy_X_t)
                pred_t12 = model(yoy_X_t12)

                delta_pred = pred_t - pred_t12
                delta_true = yoy_y_t - yoy_y_t12
                loss_yoy = smooth_l1(delta_pred, delta_true)

                loss = loss_mse + lambda_yoy * loss_yoy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_steps += 1

        avg_train_loss = train_loss_sum / max(1, train_steps)
        train_loss_history.append(avg_train_loss)
        scheduler.step()

        model.eval()
        val_loss_sum = 0.0
        val_steps = 0

        with torch.no_grad():
            for feature_batch, label_batch in val_loader:
                feature_batch = feature_batch.to(device).permute(0, 2, 1)
                label_batch = label_batch.to(device)

                pred = model(feature_batch)
                loss_mse = mse_loss(pred, label_batch)
                loss = loss_mse

                if yoy_val_loader is not None and lambda_yoy > 0:
                    yoy_val_sum = 0.0
                    yoy_val_steps = 0

                    for vy_X_t, vy_X_t12, vy_y_t, vy_y_t12 in yoy_val_loader:
                        vy_X_t = vy_X_t.to(device).permute(0, 2, 1)
                        vy_X_t12 = vy_X_t12.to(device).permute(0, 2, 1)
                        vy_y_t = vy_y_t.to(device)
                        vy_y_t12 = vy_y_t12.to(device)

                        pred_t = model(vy_X_t)
                        pred_t12 = model(vy_X_t12)

                        delta_pred = pred_t - pred_t12
                        delta_true = vy_y_t - vy_y_t12
                        yoy_l = smooth_l1(delta_pred, delta_true)

                        yoy_val_sum += yoy_l.item()
                        yoy_val_steps += 1

                    if yoy_val_steps > 0:
                        loss = loss_mse + lambda_yoy * (yoy_val_sum / yoy_val_steps)
                    else:
                        loss = loss_mse

                val_loss_sum += loss.item()
                val_steps += 1

        avg_val_loss = val_loss_sum / max(1, val_steps)
        val_loss_history.append(avg_val_loss)

        if avg_val_loss < val_best:
            val_best = avg_val_loss
            best_state = model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "loss": float(val_best),
        "status": STATUS_OK,
        "model_state": best_state,
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
    }


def create_model_from_params(
    input_size,
    output_len,
    params,
    device,
):
    hidden_size = int(params["hidden_size"])
    num_layers = int(params["num_layers"])
    batch_size = int(params["batch_size"])
    p = float(params["p"])

    model = LSTMMain(
        input_size=input_size,
        output_len=output_len,
        lstm_hidden=hidden_size,
        num_layers=num_layers,
        batch_size=batch_size,
        p=p,
        device=device,
    ).to(device)

    return model


def cv_objective_for_component(
    params,
    data_transposed_list,
    labels_transposed_list,
    site_infos,
    seeds,
    window_len,
    out_len,
    device,
):
    K_global = max(
        [info.get("K", 0) for info in site_infos if not info.get("test_only", False)]
        + [0]
    )
    if K_global == 0:
        return {"loss": 1e9, "status": STATUS_OK}

    batch_size = int(params["batch_size"])
    learning_rate = float(params["learning_rate"])
    weight_decay = float(params["weight_decay"])

    fold_losses = []
    valid_fold_count = 0

    for fold_idx in range(K_global):
        ds = build_fold_datasets_for_component(
            data_transposed_list=data_transposed_list,
            labels_transposed_list=labels_transposed_list,
            site_infos=site_infos,
            fold_index=fold_idx,
            window_len=window_len,
            out_len=out_len,
        )
        if ds is None or ds[0] is None:
            continue

        train_dataset, val_dataset = ds
        seed_losses = []

        input_size = train_dataset[0][0].shape[1]

        for s in seeds:
            torch.manual_seed(int(s))
            np.random.seed(int(s))

            model = create_model_from_params(
                input_size=input_size,
                output_len=out_len,
                params=params,
                device=device,
            )

            result = run_training(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                device=device,
                epochs=300,
                early_stop_patience=30,
                yoy_loader=None,
                lambda_yoy=0.0,
                yoy_val_loader=None,
            )
            seed_losses.append(result["loss"])

        if len(seed_losses) > 0:
            fold_losses.append(min(seed_losses))
            valid_fold_count += 1

    if valid_fold_count == 0:
        return {"loss": 1e9, "status": STATUS_OK}

    mean_loss = float(np.mean(fold_losses))
    return {"loss": mean_loss, "status": STATUS_OK}


def optimize_component_hyperparams(
    data_transposed_list,
    labels_transposed_list,
    site_infos,
    space,
    seeds,
    window_len,
    out_len,
    device,
    max_evals=200,
):
    trials = Trials()

    def _objective(params):
        return cv_objective_for_component(
            params=params,
            data_transposed_list=data_transposed_list,
            labels_transposed_list=labels_transposed_list,
            site_infos=site_infos,
            seeds=seeds,
            window_len=window_len,
            out_len=out_len,
            device=device,
        )

    best = fmin(
        fn=_objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
    )
    best_params = space_eval(space, best)
    return best_params, trials
