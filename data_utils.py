import numpy as np
import torch
from torch.utils.data import TensorDataset


def decide_dev_test_and_folds(L, window_len):
    final_test_len = int(round(0.2 * L))
    final_test_len = max(6, min(24, final_test_len))

    if L - final_test_len < window_len + 3:
        final_test_len = max(6, L - (window_len + 3))

    dev_len = L - final_test_len

    if dev_len < window_len + 3:
        return {
            "dev_len": dev_len,
            "final_test_len": final_test_len,
            "test_only": True,
            "L": L,
        }

    for vl in [12, 6, 3]:
        if dev_len >= window_len + vl:
            val_len = vl
            break
    else:
        return {
            "dev_len": dev_len,
            "final_test_len": final_test_len,
            "test_only": True,
            "L": L,
        }

    step = val_len
    K_max_possible = (dev_len - window_len) // step
    if K_max_possible <= 0:
        return {
            "dev_len": dev_len,
            "final_test_len": final_test_len,
            "test_only": True,
            "L": L,
        }

    K_target = min(5, max(2, K_max_possible)) if K_max_possible >= 2 else 1
    init_train_len = dev_len - K_target * val_len

    while init_train_len < window_len and K_target > 0:
        K_target -= 1
        init_train_len = dev_len - K_target * val_len

    if K_target <= 0 or init_train_len < window_len:
        return {
            "dev_len": dev_len,
            "final_test_len": final_test_len,
            "test_only": True,
            "L": L,
        }

    fold_train_ends = []
    for j in range(1, K_target + 1):
        train_end = init_train_len + (j - 1) * step
        fold_train_ends.append(train_end)

    return {
        "dev_len": dev_len,
        "final_test_len": final_test_len,
        "val_len": val_len,
        "step": step,
        "init_train_len": init_train_len,
        "K": K_target,
        "fold_train_ends": fold_train_ends,
        "test_only": False,
        "L": L,
    }


def get_rolling_window_multistep(out_len,
                                 start_idx,
                                 window_len,
                                 feats_per_site,
                                 labels_per_site):
    feature_list = []
    label_list = []

    for feats, labs in zip(feats_per_site, labels_per_site):
        if feats is None or labs is None:
            continue

        T = feats.shape[1]
        if T <= window_len + out_len + start_idx:
            continue

        site_xy = []
        site_y = []

        for end in range(window_len + out_len + start_idx, T + 1):
            s = end - window_len - out_len
            e = end - out_len

            x_window = feats[:-1, s:e]          # exclude target column
            y_window = labs[:, e - 1:e]         # last month as label

            site_xy.append(x_window.T)          # [window_len, features]
            site_y.append(y_window.T)           # [1, 1]

        if len(site_xy) == 0:
            continue

        site_xy_arr = np.stack(site_xy, axis=0)
        site_y_arr = np.stack(site_y, axis=0)
        feature_list.append(site_xy_arr)
        label_list.append(site_y_arr)

    return feature_list, label_list


def combine_sites_windows(feature_list, label_list):
    if len(feature_list) == 0 or len(label_list) == 0:
        return None, None

    xy_combined = np.concatenate(feature_list, axis=0)
    y_combined = np.concatenate(label_list, axis=0)
    return xy_combined, y_combined


def build_fold_datasets_for_component(
    data_transposed_list,
    labels_transposed_list,
    site_infos,
    fold_index,
    window_len,
    out_len,
):
    train_feats_per_site = []
    train_labels_per_site = []
    val_feats_per_site = []
    val_labels_per_site = []

    any_site = False

    for i, info in enumerate(site_infos):
        if info.get("test_only", False):
            continue

        K = info.get("K", 0)
        if fold_index >= K:
            continue

        dev_len = info["dev_len"]
        val_len = info["val_len"]

        feats_full = data_transposed_list[i]
        labs_full = labels_transposed_list[i]

        feats = feats_full[:, :dev_len]
        labs = labs_full[:, :dev_len]

        train_end = info["fold_train_ends"][fold_index]
        if train_end < window_len:
            continue

        tr_feat = feats[:, :train_end]
        tr_lab = labs[:, :train_end]
        train_feats_per_site.append(tr_feat)
        train_labels_per_site.append(tr_lab)

        val_left = max(0, train_end - window_len)
        val_right = min(train_end + val_len, dev_len)
        if val_right - val_left < window_len + out_len:
            continue

        v_feat = feats[:, val_left:val_right]
        v_lab = labs[:, val_left:val_right]
        val_feats_per_site.append(v_feat)
        val_labels_per_site.append(v_lab)

        any_site = True

    if not any_site:
        return None, None

    train_xy_list, train_y_list = get_rolling_window_multistep(
        out_len,
        start_idx=0,
        window_len=window_len,
        feats_per_site=train_feats_per_site,
        labels_per_site=train_labels_per_site,
    )
    val_xy_list, val_y_list = get_rolling_window_multistep(
        out_len,
        start_idx=0,
        window_len=window_len,
        feats_per_site=val_feats_per_site,
        labels_per_site=val_labels_per_site,
    )

    if len(train_xy_list) == 0 or len(val_xy_list) == 0:
        return None, None

    train_xy, train_y = combine_sites_windows(train_xy_list, train_y_list)
    val_xy, val_y = combine_sites_windows(val_xy_list, val_y_list)

    if train_xy is None or val_xy is None:
        return None, None

    train_xy_tensor = torch.from_numpy(train_xy).to(torch.float32)
    train_y_tensor = torch.from_numpy(train_y).to(torch.float32).squeeze(1)
    val_xy_tensor = torch.from_numpy(val_xy).to(torch.float32)
    val_y_tensor = torch.from_numpy(val_y).to(torch.float32).squeeze(1)

    train_dataset = TensorDataset(train_xy_tensor, train_y_tensor)
    val_dataset = TensorDataset(val_xy_tensor, val_y_tensor)
    return train_dataset, val_dataset


def build_dev_train_val_for_component(
    data_transposed_list,
    labels_transposed_list,
    site_infos,
    window_len,
    out_len,
):
    train_feats_per_site = []
    train_labels_per_site = []
    val_feats_per_site = []
    val_labels_per_site = []

    for i, info in enumerate(site_infos):
        if info.get("test_only", False):
            continue

        dev_len = info["dev_len"]
        val_len = info["val_len"]

        if dev_len <= window_len + val_len:
            continue

        feats_full = data_transposed_list[i]
        labs_full = labels_transposed_list[i]

        feats = feats_full[:, :dev_len]
        labs = labs_full[:, :dev_len]

        train_end = dev_len - val_len

        tr_feat = feats[:, :train_end]
        tr_lab = labs[:, :train_end]
        v_feat = feats[:, train_end:dev_len]
        v_lab = labs[:, train_end:dev_len]

        train_feats_per_site.append(tr_feat)
        train_labels_per_site.append(tr_lab)
        val_feats_per_site.append(v_feat)
        val_labels_per_site.append(v_lab)

    train_xy_list, train_y_list = get_rolling_window_multistep(
        out_len,
        start_idx=0,
        window_len=window_len,
        feats_per_site=train_feats_per_site,
        labels_per_site=train_labels_per_site,
    )
    val_xy_list, val_y_list = get_rolling_window_multistep(
        out_len,
        start_idx=0,
        window_len=window_len,
        feats_per_site=val_feats_per_site,
        labels_per_site=val_labels_per_site,
    )

    train_xy, train_y = combine_sites_windows(train_xy_list, train_y_list)
    val_xy, val_y = combine_sites_windows(val_xy_list, val_y_list)

    train_xy_tensor = torch.from_numpy(train_xy).to(torch.float32)
    train_y_tensor = torch.from_numpy(train_y).to(torch.float32).squeeze(1)
    val_xy_tensor = torch.from_numpy(val_xy).to(torch.float32)
    val_y_tensor = torch.from_numpy(val_y).to(torch.float32).squeeze(1)

    train_dataset = TensorDataset(train_xy_tensor, train_y_tensor)
    val_dataset = TensorDataset(val_xy_tensor, val_y_tensor)

    return train_dataset, val_dataset


def _concat_yoy_pairs_single_site(xy_arr, y_arr, lag):
    if xy_arr is None or y_arr is None:
        return None

    if y_arr.ndim == 3:
        y_arr_flat = y_arr.squeeze(2)
    else:
        y_arr_flat = y_arr

    N = xy_arr.shape[0]
    if N <= lag:
        return None

    X_t_list = []
    X_t12_list = []
    y_t_list = []
    y_t12_list = []

    for i in range(lag, N):
        X_t_list.append(xy_arr[i])
        X_t12_list.append(xy_arr[i - lag])
        y_t_list.append(y_arr_flat[i])
        y_t12_list.append(y_arr_flat[i - lag])

    if len(X_t_list) == 0:
        return None

    X_t = np.stack(X_t_list, axis=0)
    X_t12 = np.stack(X_t12_list, axis=0)
    y_t = np.stack(y_t_list, axis=0)
    y_t12 = np.stack(y_t12_list, axis=0)

    return X_t, X_t12, y_t, y_t12


def build_yoy_datasets_for_trend(
    train_xy_list,
    train_y_list,
    val_xy_list,
    val_y_list,
    lag=12,
):
    X_t_all = []
    X_t12_all = []
    y_t_all = []
    y_t12_all = []

    for xy_arr, y_arr in zip(train_xy_list, train_y_list):
        result = _concat_yoy_pairs_single_site(xy_arr, y_arr, lag)
        if result is None:
            continue
        X_t, X_t12, y_t, y_t12 = result
        X_t_all.append(X_t)
        X_t12_all.append(X_t12)
        y_t_all.append(y_t)
        y_t12_all.append(y_t12)

    if len(X_t_all) > 0:
        X_t_train = np.concatenate(X_t_all, axis=0)
        X_t12_train = np.concatenate(X_t12_all, axis=0)
        y_t_train = np.concatenate(y_t_all, axis=0)
        y_t12_train = np.concatenate(y_t12_all, axis=0)

        X_t_train_tensor = torch.from_numpy(X_t_train).to(torch.float32)
        X_t12_train_tensor = torch.from_numpy(X_t12_train).to(torch.float32)
        y_t_train_tensor = torch.from_numpy(y_t_train).to(torch.float32)
        y_t12_train_tensor = torch.from_numpy(y_t12_train).to(torch.float32)

        yoy_train_dataset = TensorDataset(
            X_t_train_tensor,
            X_t12_train_tensor,
            y_t_train_tensor,
            y_t12_train_tensor,
        )
    else:
        yoy_train_dataset = None

    Xv_t_all = []
    Xv_t12_all = []
    yv_t_all = []
    yv_t12_all = []

    for xy_arr, y_arr in zip(val_xy_list, val_y_list):
        result = _concat_yoy_pairs_single_site(xy_arr, y_arr, lag)
        if result is None:
            continue
        X_t, X_t12, y_t, y_t12 = result
        Xv_t_all.append(X_t)
        Xv_t12_all.append(X_t12)
        yv_t_all.append(y_t)
        yv_t12_all.append(y_t12)

    if len(Xv_t_all) > 0:
        X_t_val = np.concatenate(Xv_t_all, axis=0)
        X_t12_val = np.concatenate(Xv_t12_all, axis=0)
        y_t_val = np.concatenate(yv_t_all, axis=0)
        y_t12_val = np.concatenate(yv_t12_all, axis=0)

        X_t_val_tensor = torch.from_numpy(X_t_val).to(torch.float32)
        X_t12_val_tensor = torch.from_numpy(X_t12_val).to(torch.float32)
        y_t_val_tensor = torch.from_numpy(y_t_val).to(torch.float32)
        y_t12_val_tensor = torch.from_numpy(y_t12_val).to(torch.float32)

        yoy_val_dataset = TensorDataset(
            X_t_val_tensor,
            X_t12_val_tensor,
            y_t_val_tensor,
            y_t12_val_tensor,
        )
    else:
        yoy_val_dataset = None

    return yoy_train_dataset, yoy_val_dataset
