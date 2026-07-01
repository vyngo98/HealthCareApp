from sleep_core.ppg import *
import json
# from cache import analysis_cache



def load_study(subject_id: str):
    print("LOAD_STUDY CALLED")
    if os.path.exists(os.path.join(TEMP_FOLDER, f"{subject_id}_ppg.parquet")) and os.path.exists(os.path.join(TEMP_FOLDER, f"{subject_id}_acc.parquet")):
        return f"This study has been already loaded and processed, please skip this step and go to the next step."
    ppg_files = glob.glob(os.path.join(MAIN_FOLDER, subject_id, 'ppg_*.bdf'))
    acc_files = glob.glob(os.path.join(MAIN_FOLDER, subject_id, 'acc_*.bdf'.format(subject_id)))

    ppg_files.sort()
    acc_files.sort()

    _, first_ppg_start_time, _ = read_signals(ppg_files[0], 'ppg')
    _, _, last_ppg_end_time = read_signals(ppg_files[-1], 'ppg')

    total_ppg = np.ones((int((last_ppg_end_time - first_ppg_start_time) * FS_PPG), PPG_N_CHANNELS)) * -1
    total_ppg_timestamp = np.arange(0, int((
                        last_ppg_end_time - first_ppg_start_time) * FS_PPG)) / FS_PPG + first_ppg_start_time

    for ppg_file in ppg_files:
        ppg_data, ppg_start_time, ppg_end_time = read_signals(ppg_file, 'ppg')
        if ppg_data is None:
            continue

        timestamp_ind = np.flatnonzero(
            (total_ppg_timestamp >= ppg_start_time) & (total_ppg_timestamp < ppg_end_time))

        total_ppg[timestamp_ind] = ppg_data

    _, first_acc_start_time, _ = read_signals(acc_files[0], 'acc')
    _, _, last_acc_end_time = read_signals(acc_files[-1], 'acc')

    total_acc = np.empty((int((last_ppg_end_time - first_ppg_start_time) * FS_ACC), ACC_N_CHANNELS))
    total_acc[:] = np.nan

    total_acc_timestamp = np.arange(0, int((
                                                   last_ppg_end_time - first_ppg_start_time) * FS_ACC)) / FS_ACC + first_ppg_start_time

    for acc_file in acc_files:
        acc_data, acc_start_time, acc_end_time = read_signals(acc_file, 'acc')
        if acc_data is None:
            continue
        timestamp_ind = np.flatnonzero(
            (total_acc_timestamp >= acc_start_time) & (total_acc_timestamp < acc_end_time))
        if len(timestamp_ind) == 0:
            continue
        acc_timestamp = np.arange(len(acc_data)) / FS_ACC + acc_start_time
        acc_mask = (acc_timestamp >= total_acc_timestamp[timestamp_ind][0]) & \
                   (acc_timestamp < total_acc_timestamp[timestamp_ind][-1] + 1 / FS_ACC)

        acc_data_subset = acc_data[acc_mask]

        total_acc[timestamp_ind] = acc_data_subset

    total_ppg = np.array(total_ppg)
    total_acc = np.array(total_acc)

    df = pd.DataFrame({
        "timestamp": total_ppg_timestamp,
        "ppg_red": total_ppg[:, 0],
        "ppg_ir": total_ppg[:, 1],
    })

    df_acc = pd.DataFrame({
        "timestamp": total_acc_timestamp,
        "acc_x": total_acc[:, 0],
        "acc_y": total_acc[:, 1],
        "acc_z": total_acc[:, 2]
    })

    df.to_parquet(os.path.join(TEMP_FOLDER, f"{subject_id}_ppg.parquet"))
    df_acc.to_parquet(os.path.join(TEMP_FOLDER, f"{subject_id}_acc.parquet"))

    return {
    "subject_id": subject_id,
    "ppg_file_path": os.path.join(TEMP_FOLDER, f"{subject_id}_ppg.parquet"),
    "acc_file_path": os.path.join(TEMP_FOLDER, f"{subject_id}_acc.parquet"),
    "ppg_sampling_rate": 128,
    "acc_sampling_rate": 25,
    "duration_hours": round(len(total_ppg) / FS_PPG / 3600, 2),
    "acc_channels": [
        "timestamp",
        "ppg_red",
        "ppg_ir",
        "acc_x",
        "acc_y",
        "acc_z"
    ],
    "ppg_channels": [
        "timestamp",
        "acc_x",
        "acc_y",
        "acc_z"
    ]
}


# load_study('2026-05-20-BaoLuu-4mm-10mA')

predictor = SleepStagePredictor()
def predict_sleep_stage(subject_id: str):
    print("PREDICT_SLEEP_STAGE CALLED")
    return predictor.get_sleep_stage(subject_id)

# predict_sleep_stage("2026-05-20-BaoLuu-4mm-10mA")

def compute_sleep_metrics(subject_id: str):
    print("COMPUTE_SLEEP_METRICS CALLED")

    hypnogram_file = os.path.join(TEMP_FOLDER, f"{subject_id}_hypnogram.parquet")
    if not os.path.exists(hypnogram_file):
        return "Study isn't processed to have hypnogram_file yet, we need to run sleep_stage_agent first."

    df = pd.read_parquet(hypnogram_file)
    filtered_pred = df['sleep_stage']
    percent_rem_pred = percent_rem(filtered_pred, rem_label=1)
    percent_wake_pred = percent_rem(filtered_pred, rem_label=0)
    percent_nrem_pred = percent_rem(filtered_pred, rem_label=2)

    ratio_pred = rem_nrem_ratio(filtered_pred, rem_label=1, nrem_label=2)

    latency_pred = sleep_latency(filtered_pred, epoch_sec=1, min_sleep_epochs=600)

    tst = compute_tst(filtered_pred)

    sleep_efficiency = compute_sleep_efficiency(filtered_pred)

    metrics = {
        "total_sleep_time": tst,
        "rem_percent": round(percent_rem_pred, 2),
        "wake_percent": round(percent_wake_pred, 2),
        "nrem_percent": round(percent_nrem_pred, 2),
        "rem_nrem_ratio": round(ratio_pred, 2),
        "sleep_onset_latency_in_minutes": round(latency_pred, 2),
        "sleep_efficiency": round(sleep_efficiency, 2),
    }

    with open(hypnogram_file.replace('hypnogram.parquet', 'metrics.json'), "w") as file:
        json.dump(metrics, file)

    # if subject_id not in analysis_cache:
    #     analysis_cache[subject_id] = {}
    #
    # analysis_cache[subject_id]["metrics"] = metrics
    # analysis_cache[subject_id]["hypnogram_file"] = hypnogram_file

    return  {
                "metrics_path":
                    hypnogram_file.replace('hypnogram.parquet', 'metrics.json'),}

# print(compute_sleep_metrics('2026-05-20-BaoLuu-4mm-10mA'))

def load_metrics(subject_id: str):
    metrics_path = os.path.join(TEMP_FOLDER, f"{subject_id}_metrics.json")
    if not os.path.exists(metrics_path):
        return "Study isn't processed to have file metrics.json yet, we need to handoff sleep_metrics_agent first."
    with open(metrics_path) as f:
        metrics = json.load(f)
    print(metrics)
    return json.dumps(
        metrics,
        indent=2
    )


def get_data_quality(subject_id: str) -> str:
    print("GET_DATA_QUALITY CALLED")
    acc_file_path = os.path.join(TEMP_FOLDER, f"{subject_id}_acc.parquet")
    ppg_file_path = os.path.join(TEMP_FOLDER, f"{subject_id}_ppg.parquet")
    if not os.path.exists(acc_file_path):
        return "Study isn't processed to have acc_file_path and ppg_file_path yet, we need to run load_study() first."
    df = pd.read_parquet(acc_file_path)
    acc_z = df['acc_z']
    acc_y = df['acc_y']
    acc_x = df['acc_x']
    total_acc = np.array([acc_x, acc_y, acc_z]).T

    df_ppg = pd.read_parquet(ppg_file_path)
    ppg_red = df_ppg['ppg_red']
    ppg_ir = df_ppg['ppg_ir']
    total_ppg = np.array([ppg_red, ppg_ir]).T

    # Create a mask for rows where all channels are NOT -1
    valid_mask = ~(np.all(total_acc == -1, axis=1))

    if np.any(valid_mask):  # make sure there is at least one valid row
        first_idx = np.argmax(valid_mask)  # first True
        last_idx = len(valid_mask) - 1 - np.argmax(valid_mask[::-1])  # last True
        total_duration_acc = (last_idx - first_idx)/FS_ACC
    else:
        total_duration_acc = 0

    ppg_valid_mask = ~(np.all(total_ppg == -1, axis=1))
    if np.any(ppg_valid_mask):  # make sure there is at least one valid row
        last_idx = len(ppg_valid_mask) - 1 - np.argmax(ppg_valid_mask[::-1])  # last True
        total_duration_ppg = last_idx/FS_PPG
    else:
        total_duration_ppg = 0

    # Check quality of PPG signal
    worn_mask = np.ones(len(total_ppg))
    ppg_good_quality = 0

    if len(total_ppg[:, 0]) >= 4*FS_PPG:
        red_segment = segment(total_ppg[:, 0], len(total_ppg[:, 0]), 4*FS_PPG, 4*FS_PPG)
        std_red = np.std(red_segment, axis=1)
        bad_quality = len(np.flatnonzero(std_red <= 10)) * 4  # in second
        ppg_good_quality = len(total_ppg)/FS_PPG - bad_quality
        leadoff_ind = np.flatnonzero(std_red <= 10)
        for ind in leadoff_ind:
            worn_mask[int(ind * 4 * FS_PPG): int((ind + 1) * 4 * FS_PPG)] = 0
    else:
        std_red = np.std(total_ppg[:, 0])
        if std_red > 10:
            ppg_good_quality += len(total_ppg)/FS_PPG
        else:
            worn_mask = np.zeros(len(total_ppg))

    missed_ind = np.flatnonzero(total_ppg[:, 0] == -1)
    worn_mask[missed_ind] = -1

    good_ppg_percent = round(100 * ppg_good_quality / total_duration_ppg) if total_duration_ppg > 0 else 0

    len_acc_epoch = len(np.flatnonzero(total_acc[:, 0] != -1))
    len_ppg_epoch = len(np.flatnonzero(total_ppg[:, 0] != -1))

    acc_percent = round(100 * len_acc_epoch / FS_ACC / total_duration_acc) if total_duration_acc > 0 else 0
    ppg_percent = round(100 * len_ppg_epoch/ FS_PPG / total_duration_ppg) if total_duration_ppg > 0 else 0

    summary = {"total_duration_acc": total_duration_acc,
               "total_duration_ppg": total_duration_ppg,
               "accelerometer_data_availability_percent": acc_percent,
               "ppg_data_availability_percent": ppg_percent,
               "device_worn_ppg_percent": good_ppg_percent}
    # if subject_id not in analysis_cache:
    #     analysis_cache[subject_id] = {}
    #
    # analysis_cache[subject_id]["signal_quality"] = summary

    with open(os.path.join(TEMP_FOLDER, f"{subject_id}_data_quality.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file)

    return json.dumps(summary, indent=2)
