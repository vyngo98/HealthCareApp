import numpy as np
from sleep_core.define import *
import mne
import glob
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from scipy.signal import butter, filtfilt, resample
import neurokit2 as nk

from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt


def read_signals(fname, expected_channels='acc'):
    if expected_channels == 'acc':
        expected_channels = 3
        target_fs = 25
    elif expected_channels == 'ppg':
        expected_channels = 2
        target_fs = 128
    try:
        if fname[-3:] == 'edf':
            raw = mne.io.read_raw_edf(fname, preload=True)
        else:
            raw = mne.io.read_raw_bdf(fname, preload=True)
        if len(raw.ch_names) < expected_channels:
            raise ValueError(
                f"Expected at least {expected_channels} channels, but got {len(raw.ch_names)}.")
        raw = raw.pick_channels(raw.ch_names[:expected_channels])
        fs = raw.info['sfreq']
        data, _ = raw[:]
        data = data.T
        scale = target_fs / fs
        if scale != 1:
            n = round(len(data) * scale)
            data_resample = resample(data, n, axis=0)
        else:
            data_resample = data
        meas_date = raw.info.get('meas_date')
        if meas_date is not None:
            if isinstance(meas_date, tuple):
                start_time = meas_date[0] + meas_date[1] * 1e-6
            elif hasattr(meas_date, 'timestamp'):
                start_time = meas_date.timestamp()
            else:
                start_time = None
        else:
            start_time = None
        if start_time is not None:
            duration_sec = data_resample.shape[0] / target_fs
            end_time = start_time + duration_sec
            return data_resample, int(start_time), int(end_time)
    except Exception as e:
        print('error', f"Unexpected error while reading {fname}: {e}")
        return None, None, None

def segment(data, max_time, sub_window_size, stride_size):
    sub_windows = np.arange(sub_window_size)[
                      None, :] + np.arange(0, max_time, stride_size)[:, None]

    row, col = np.where(sub_windows >= max_time)
    uniq_row = len(np.unique(row))

    if uniq_row > 0 and row[0] > 0:
        sub_windows = sub_windows[:-uniq_row, :]

    return data[sub_windows]

def load_study(subject_id):
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
        "acc_x": total_acc[:, 0],
        "acc_y": total_acc[:, 1],
        "acc_z": total_acc[:, 2]
    })

    df.to_parquet(os.path.join(TEMP_FOLDER, f"{subject_id}_ppg_acc.parquet"))
    return {
    "subject_id": subject_id,
    "file_path": os.path.join(TEMP_FOLDER, f"{subject_id}_ppg_acc.parquet"),
    "ppg_sampling_rate": 128,
    "acc_sampling_rate": 25,
    "duration_hours": round(len(total_ppg) / FS_PPG / 3600, 2),
    "channels": [
        "timestamp",
        "ppg_red",
        "ppg_ir",
        "acc_x",
        "acc_y",
        "acc_z"
    ]
}


class SleepStagePredictor:
    def __init__(self):
        self.n_epochs = 250
        self.n_classes = 3
        self.stride_size = int(30*60)


    @staticmethod
    def create_local_mask(seq_len, window_size=5):
        mask = tf.ones((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                if abs(i - j) > window_size:
                    mask = tf.tensor_scatter_nd_update(
                        mask, [[i, j]], [0.0]
                    )
        return mask

    def build_sleep_model_multiscale(self,
            epoch_len=256,
            n_epochs=250,
            n_channels=2,
            d_model=128,
            training=False
    ):
        class ConcatLayer(layers.Layer):
            def call(self, inputs, axis):
                # The TensorFlow function is used inside the call method
                return tf.concat(inputs, axis=axis)

        class TCNBlock(layers.Layer):
            def __init__(self, filters, kernel_size=3, dilation_rate=1, dropout=0.1):
                super().__init__()
                self.conv1 = layers.Conv1D(
                    filters, kernel_size,
                    padding="causal",
                    dilation_rate=dilation_rate,
                    activation="relu"
                )
                self.bn1 = layers.BatchNormalization()

                self.conv2 = layers.Conv1D(
                    filters, kernel_size,
                    padding="causal",
                    dilation_rate=dilation_rate,
                    activation="relu"
                )
                self.bn2 = layers.BatchNormalization()

                self.dropout = layers.Dropout(dropout)

                self.downsample = layers.Conv1D(filters, 1, padding="same")

            def call(self, x, training=None):
                residual = x

                x = self.conv1(x)
                x = self.bn1(x, training=training)

                x = self.conv2(x)
                x = self.bn2(x, training=training)

                x = self.dropout(x, training=training)

                # match dimension
                if residual.shape[-1] != x.shape[-1]:
                    residual = self.downsample(residual)

                return tf.nn.relu(x + residual)

        class PositionalEncoding2(layers.Layer):
            def __init__(self, d_model):
                super().__init__()
                self.d_model = d_model

            def call(self, x):
                seq_len = tf.shape(x)[1]
                pos = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
                i = tf.range(self.d_model, dtype=tf.float32)[tf.newaxis, :]

                angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / self.d_model)
                angle_rads = pos * angle_rates

                sines = tf.sin(angle_rads[:, 0::2])
                cosines = tf.cos(angle_rads[:, 1::2])

                pos_encoding = tf.concat([sines, cosines], axis=-1)
                pos_encoding = pos_encoding[tf.newaxis, ...]

                return x + pos_encoding

        class TransformerBlockMS(layers.Layer):
            def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
                super().__init__()
                self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
                self.ffn = tf.keras.Sequential([
                    layers.Dense(ff_dim, activation="relu"),
                    layers.Dense(d_model),
                ])
                self.norm1 = layers.LayerNormalization()
                self.norm2 = layers.LayerNormalization()
                self.dropout1 = layers.Dropout(dropout)
                self.dropout2 = layers.Dropout(dropout)

            def call(self, x, training=None, mask=None):
                attn_output = self.att(x, x, attention_mask=mask)
                attn_output = self.dropout1(attn_output, training=training)
                x1 = self.norm1(x + attn_output)

                ffn_output = self.ffn(x1)
                ffn_output = self.dropout2(ffn_output, training=training)
                return self.norm2(x1 + ffn_output)

        input_hr = tf.keras.Input(shape=(n_epochs, epoch_len, 1))
        input_activity = tf.keras.Input(shape=(n_epochs, epoch_len, 1))

        x = ConcatLayer()([input_hr, input_activity], axis=-1)

        # =========================
        # 1. TCN encoder (local signal)
        # =========================
        x = layers.Lambda(lambda t: tf.reshape(t, (-1, epoch_len, n_channels)))(x)

        for d in [1, 2, 4, 8]:
            x = TCNBlock(64, kernel_size=5, dilation_rate=d)(x)

        x = layers.Conv1D(d_model, 1, activation="relu")(x)
        x = layers.GlobalAveragePooling1D()(x)

        # x = layers.Lambda(lambda t: tf.reshape(t, (-1, epoch_len, d_model)))(x)  # (B, T, D)
        x = layers.Lambda(lambda t: tf.reshape(t, (-1, n_epochs, d_model)))(x)
        x = PositionalEncoding2(d_model)(x)

        # =========================
        # 2. SHORT CONTEXT BRANCH
        # =========================
        short_x = x

        seq_len = n_epochs
        local_mask = self.create_local_mask(seq_len, window_size=5)

        for _ in range(2):
            short_x = TransformerBlockMS(
                d_model, num_heads=2, ff_dim=128
            )(short_x, mask=local_mask, training=training)

        # =========================
        # 3. LONG CONTEXT BRANCH
        # =========================
        long_x = x

        for _ in range(4):
            long_x = TransformerBlockMS(
                d_model, num_heads=4, ff_dim=256
            )(long_x)

        # =========================
        # 4. FUSION
        # =========================
        x = ConcatLayer()([short_x, long_x], axis=-1)
        # x = FlatAttentionFusion(d_model)(short_x, long_x)

        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.2)(x)

        # =========================
        # 5. CLASSIFIER
        # =========================
        outputs = layers.Dense(3, activation="softmax")(x)

        model = tf.keras.Model((input_hr, input_activity), outputs)

        return model

    def load_model(self):
        model = self.build_sleep_model_multiscale(
            epoch_len=256,
            n_epochs=250,
            n_channels=2,
            d_model=128,
            training=False
        )
        model.load_weights(SLEEPSTAGE_CKPT)
        return model

    @staticmethod
    def check_data_valid_(ppg_segment, acc_segment, start_valid_group, stop_valid_group):
        # acc_valid_mask = ~(np.all(acc_segment == -1, axis=1))
        acc_valid_mask = ~(np.all(np.isnan(acc_segment), axis=1))
        if np.any(acc_valid_mask):  # make sure there is at least one valid row
            last_idx = len(acc_valid_mask) - 1 - np.argmax(acc_valid_mask[::-1])  # last True
            total_duration_acc = last_idx / FS_ACC
        else:
            total_duration_acc = 0

        if total_duration_acc < THR_DURATION_DATA:
            print(
                'warning', 'ACC duration is insufficient for sleep stage analysis.')
            return -1, -1, -1, -1, -1, -1, -1

        ppg_valid_mask = ~(np.all(ppg_segment == -1, axis=1))
        if np.any(ppg_valid_mask):  # make sure there is at least one valid row
            last_idx = len(ppg_valid_mask) - 1 - np.argmax(ppg_valid_mask[::-1])  # last True
            total_duration_ppg = last_idx / FS_PPG
        else:
            total_duration_ppg = 0

        # Check quality of PPG signal
        worn_mask = np.ones(len(ppg_segment))
        start_worn_mask = np.arange(start_valid_group, stop_valid_group)
        ppg_good_quality = 0

        if len(ppg_segment[:, 0]) >= 4 * FS_PPG:
            red_segment = segment(ppg_segment[:, 0], len(ppg_segment[:, 0]), 4 * FS_PPG, 4 * FS_PPG)
            std_red = np.std(red_segment, axis=1)
            # bad_quality = len(np.flatnonzero(std_red <= 20)) * 4  # in second
            bad_quality = len(np.flatnonzero(std_red <= 10)) * 4  # in second
            ppg_good_quality = len(ppg_segment) / FS_PPG - bad_quality
            # leadoff_ind = np.flatnonzero(std_red <= 20)
            leadoff_ind = np.flatnonzero(std_red <= 10)
            for ind in leadoff_ind:
                worn_mask[int(ind * 4 * FS_PPG): int((ind + 1) * 4 * FS_PPG)] = 0
        else:
            std_red = np.std(ppg_segment[:, 0])
            # if std_red > 20:
            if std_red > 10:
                ppg_good_quality += len(ppg_segment) / FS_PPG
            else:
                worn_mask = np.zeros(len(ppg_segment))

        missed_ind = np.flatnonzero(ppg_segment[:, 0] == -1)
        worn_mask[missed_ind] = -1

        good_ppg_percent = round(100 * ppg_good_quality / total_duration_ppg) if total_duration_ppg > 0 else 0

        if total_duration_ppg < THR_DURATION_DATA:
            print(
                'warning', 'PPG duration is insufficient for sleep stage analysis.')
            return -1, -1, start_worn_mask, worn_mask, total_duration_ppg, -1, -1

        # len_acc_epoch = len(np.flatnonzero(acc_segment[:, 0] != -1))
        len_ppg_epoch = len(np.flatnonzero(ppg_segment[:, 0] != -1))

        # acc_percent = round(100 * len_acc_epoch / FS_ACC / total_duration_acc) if total_duration_acc > 0 else 0
        ppg_percent = round(100 * len_ppg_epoch / FS_PPG / total_duration_ppg) if total_duration_ppg > 0 else 0


        if ppg_percent < THR_PERCENT:
            print('warning',
                  'The actual percentage of PPG data available is {}%, which is insufficient for sleep analysis.'.format(
                      ppg_percent))
            return -1, -1, start_worn_mask, worn_mask, total_duration_ppg, ppg_percent, -1

        if (good_ppg_percent < THR_PERCENT):
            print('warning',
                  'The percentage of good quality PPG data available is {}%, which is insufficient for sleep analysis.'.format(
                      good_ppg_percent))
            return -1, -1, start_worn_mask, worn_mask, total_duration_ppg, ppg_percent, good_ppg_percent

        return [], [], -1, -1, -1, -1, -1

    @staticmethod
    def zero_crossing_slope(signal, slope_th):

        crossings = 0
        slope = []
        for i in range(1, len(signal)):

            if signal[i] * signal[i - 1] < 0:
                slope.append(abs(signal[i] - signal[i - 1]))

                if abs(signal[i] - signal[i - 1]) > slope_th:
                    crossings += 1

        return crossings, slope

    def compute_activity_counts(self, acc_z, fs, epoch_sec=30, dreamt=False):
        epoch_samples = int(fs * epoch_sec)
        n_epochs = len(acc_z) // epoch_samples

        counts = []
        slopes = []
        slope_th = max(np.percentile(np.abs(np.diff(acc_z)), 98), 100) if not dreamt else np.percentile(
            np.abs(np.diff(acc_z)), 98)  # v3
        # slope_th = np.percentile(np.abs(np.diff(acc_z)), 98) # v1 +v2

        for i in range(n_epochs):
            seg = acc_z[i * epoch_samples:(i + 1) * epoch_samples]
            count, slope = self.zero_crossing_slope(seg, slope_th=slope_th)
            counts.append(count)
            slopes.extend(slope)

        return np.array(counts), np.array(slopes), slope_th

    @staticmethod
    def bandpass_filter(signal, fs, low=0.5, high=11, order=4):
        nyq = 0.5 * fs
        low /= nyq
        high /= nyq

        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal)

    @staticmethod
    def pad_ppg_for_last_segment(data, sub_window_size, desired_real_len):
        pad_len = max(0, sub_window_size - desired_real_len)

        # đảm bảo dtype hỗ trợ NaN
        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(float)

        padding = np.full((pad_len, data.shape[1]), np.nan)

        data_padded = np.concatenate([data, padding])
        return data_padded, pad_len

    @staticmethod
    def get_strong_wake_mask(
            wake_ihr,
            wake_act,
            act_percentile=80,
            hr_std_percentile=70,
            hr_peak_percentile=70,
            window_sec=60,
            fs_hr=2
    ):

        wake_ihr = np.array(wake_ihr)
        wake_act = np.array(wake_act)

        # -------------------------
        # 1. activity threshold
        # -------------------------
        act_th = np.percentile(
            wake_act[wake_act > 0],
            act_percentile
        ) if np.any(wake_act > 0) else 0

        high_act = wake_act > act_th

        return high_act

    @staticmethod
    def fill_padding_from_wake(pool, target_len):
        """
        Fill padding bằng wake physiology
        """
        if len(pool) == 0:
            return np.zeros(target_len)

        repeat_times = int(np.ceil(target_len / len(pool)))

        filled = np.tile(pool, repeat_times)

        return filled[:target_len]

    @staticmethod
    def smooth_ihr(ihr):
        ihr_clean = medfilt(ihr, 5)
        ihr_smooth = gaussian_filter1d(ihr_clean, sigma=2)
        return ihr_smooth

    @staticmethod
    def normalize_with_padding(ihr_smooth):
        ihr_signals = ihr_smooth.copy().astype(float)

        # mask phần không phải padding
        valid_mask = ihr_signals != 0

        # normalize chỉ trên valid region
        valid_values = ihr_signals[valid_mask]

        vmin = np.min(valid_values)
        vmax = np.max(valid_values)

        ihr_signals[valid_mask] = (
                (valid_values - vmin) / (vmax - vmin + 1e-8)
        )

        # giữ nguyên padding = 0
        ihr_signals[~valid_mask] = 0
        return ihr_signals

    @staticmethod
    def kurtosis_ppg(ppg):
        """
        Kurtosis theo đúng công thức trong paper
        ppg: 1D numpy array (PPG segment)
        """
        ppg = np.asarray(ppg)
        mu = np.mean(ppg)
        sigma = np.std(ppg, ddof=0)  # population std

        if sigma == 0:
            return 0.0

        kurt = np.mean(((ppg - mu) / sigma) ** 4)
        return kurt

    def compute_hr_ppg(self, ppg, fs=128, hr_min=40, hr_max=180, window_size=4, stride_size=0.5):
        red_filtered_signal = self.bandpass_filter(ppg[:, 0], fs, 0.5, 5)
        ppg_segments = segment(red_filtered_signal, len(red_filtered_signal), int(window_size * fs),
                               int(stride_size * fs))
        ppg_raw_segments = segment(ppg, len(ppg), int(window_size * fs), int(stride_size * fs))

        hr_values = np.empty(len(ppg_segments))
        hr_values[:] = 0

        # rr_values = np.empty(len(ppg_segments))
        # rr_values[:] = 0

        for i, ppg_seg in enumerate(ppg_segments):
            ppg_raw_seg = ppg_raw_segments[i]
            kurt = self.kurtosis_ppg(ppg_seg)
            if kurt > 3:
                continue

            ppg_seg = (ppg_seg - np.mean(ppg_seg)) / (np.std(ppg_seg) + 1e-6)

            troughs_nk_dict = nk.ppg_findpeaks(ppg_seg, sampling_rate=fs)

            troughs_ = troughs_nk_dict['PPG_Peaks']

            invert_segment = -ppg_seg
            peaks_nk_dict = nk.ppg_findpeaks(invert_segment, sampling_rate=fs)
            peaks_ = peaks_nk_dict['PPG_Peaks']

            valid_trough_mask = np.ones(len(troughs_))
            for n in range(len(troughs_) - 1):
                tmp_peaks = np.flatnonzero((peaks_ > troughs_[n]) & (peaks_ < troughs_[n + 1]))
                if len(tmp_peaks) == 0:
                    valid_trough_mask[n] = 0
            peaks = troughs_[np.flatnonzero(valid_trough_mask)]

            if len(peaks) < 2:
                continue

            # Abnormal HR
            rr_intervals = np.diff(peaks)
            ind_abnormal_rr = np.flatnonzero((rr_intervals > RR_MAX_THR) | (rr_intervals < RR_MIN_THR)) + 1
            valid_beats_index = np.delete(np.arange(len(peaks)), ind_abnormal_rr)
            group_valid_beats = np.split(valid_beats_index, np.flatnonzero(np.diff(valid_beats_index) != 1) + 1)

            final_rr_intervals = []
            for group in group_valid_beats:
                final_rr_intervals.extend(np.diff(peaks[group]))

            final_rr_intervals = np.array(final_rr_intervals) / fs
            hr = 60.0 / np.mean(final_rr_intervals)
            if hr_min <= hr <= hr_max:
                hr_values[i] = hr
            else:
                continue

        return hr_values

    def get_ihr_activity_count(self, total_acc, total_ppg, acc_timestamp, timestamp):
        # interpolate Nan value in acc_z
        acc_z = total_acc[:, 2]
        acc_z_ind = np.flatnonzero(~np.isnan(acc_z))
        if len(acc_z_ind) > 0:
            acc_z = np.interp(np.arange(0, len(acc_z)), acc_z_ind, acc_z[acc_z_ind])

        acc_filt = self.bandpass_filter(acc_z, fs=FS_ACC, order=5)
        activity_count_cole_long, slopes, slope_thr = self.compute_activity_counts(acc_filt, fs=FS_ACC, epoch_sec=1)

        x_timestamp = np.arange(len(activity_count_cole_long) * 2) / 2 + timestamp[0]

        activity_count_cole_long30, slopes, slope_thr = self.compute_activity_counts(acc_filt, fs=FS_ACC, epoch_sec=30)
        activity_count_cole_long30_timestamp = np.arange(len(activity_count_cole_long30)) * 30 + timestamp[0] + 30
        activity_count_cole_long30 = np.interp(x_timestamp, activity_count_cole_long30_timestamp,
                                               activity_count_cole_long30)

        activity_count_cole_long30 = (activity_count_cole_long30 - np.min(activity_count_cole_long30)) / (
                np.max(activity_count_cole_long30) - np.min(activity_count_cole_long30))

        # valid_ppg_padded, pad_len = self.pad_ppg_for_last_segment(total_ppg,
        #                                                      sub_window_size=int((self.n_epochs * 30 + 49 * 2) * 128),
        #                                                      desired_real_len=int(self.stride_size * 128))
        # valid_ppg_segs = segment(
        #     valid_ppg_padded,
        #     len(valid_ppg_padded),
        #     sub_window_size=int((self.n_epochs * 30 + 49 * 2) * 128),
        #     stride_size=int(self.stride_size * 128)
        # )

        valid_ppg_segs = segment(
            total_ppg,
            len(total_ppg),
            sub_window_size=int((self.n_epochs * 30 + 49 * 2) * 128),
            stride_size=int(self.stride_size * 128)
        )

        # activity_count_cole_long30 = np.concatenate([activity_count_cole_long30, np.zeros(pad_len // 128 * 2)])
        x_timestamp = np.arange(len(activity_count_cole_long30)) / 2 + x_timestamp[0]

        valid_ppg_timestamp = np.arange(len(total_ppg)) / 128 + timestamp[0]
        valid_ppg_timestamp_segs = segment(valid_ppg_timestamp, len(valid_ppg_timestamp),
                                           sub_window_size=int((self.n_epochs * 30 + 49 * 2) * 128),
                                           stride_size=int(self.stride_size * 128))

        total_ihr_org, total_ihr_segs = [], []
        # total_label = []
        total_activity_counts_cole_long_segs = []

        seg_indices = []
        start_time_sleep = []
        # wake_ihr_pool, wake_act_pool = [], []
        for n, valid_ppg_seg in enumerate(valid_ppg_segs):
            valid_acc_seg = total_acc[np.flatnonzero((acc_timestamp >= (valid_ppg_timestamp_segs[n][0]))
                                                     & (acc_timestamp <= valid_ppg_timestamp_segs[n][-1]))]

            (ind_ppg_get, ind_acc_get, start_worn_mask, worn_mask,
             total_duration_ppg,
             ppg_percent, good_ppg_percent) = self.check_data_valid_(valid_ppg_seg, valid_acc_seg,
                                                                valid_ppg_timestamp_segs[n][0],
                                                                valid_ppg_timestamp_segs[n][-1])

            if not isinstance(ind_ppg_get, int):
                seg_indices.append(n)
                start_time_sleep.append(valid_ppg_timestamp_segs[n][0])
                is_missing = (valid_ppg_seg[:, 0] == -1) | (np.isnan(valid_ppg_seg[:, 0]))
                change_points = np.flatnonzero(np.diff(is_missing, prepend=is_missing[0], append=~is_missing[-1]))
                ppg_groups = np.split(valid_ppg_seg, change_points)
                ppg_groups_timestamp = np.split(valid_ppg_timestamp_segs[n], change_points)
                ihr_signals = []
                # labels = []

                last_ppg_groups_timestamp = 0
                for j, ppg_group in enumerate(ppg_groups):
                    if len(ppg_group) == 0:
                        continue

                    if np.all(np.isnan(ppg_group)):
                        ihr_signals.extend(np.ones(int(len(ppg_group) / FS_PPG) * FS_HR) * 0)


                    elif np.all(ppg_group == -1):
                        ihr_signals.extend(np.ones(int(len(ppg_group) / FS_PPG) * FS_HR) * -1)

                    elif len(ppg_group) / FS_PPG < PPG_SEG_TIME:
                        ihr_signals.extend(np.ones(int(len(ppg_group) / FS_PPG) * FS_HR) * -1)

                    else:
                        start_ppg_timestamp = ppg_groups_timestamp[j][0]

                        ppg_group = ppg_group[np.flatnonzero(ppg_groups_timestamp[j] >= start_ppg_timestamp)]
                        ppg_timestamp_group = ppg_groups_timestamp[j][
                            np.flatnonzero(ppg_groups_timestamp[j] >= start_ppg_timestamp)]

                        # calculate IHR
                        ihr = self.compute_hr_ppg(ppg_group, FS_PPG, window_size=4, stride_size=0.5)

                        n_pad = int(len(ppg_group) / FS_PPG * FS_HR) - len(ihr)
                        ihr = np.pad(ihr, (n_pad, 0), mode='constant', constant_values=0)

                        # interpolate ihr
                        ihr_ind = np.flatnonzero(ihr != 0)
                        if len(ihr_ind) > 0:
                            ihr = np.interp(np.arange(0, len(ihr)), ihr_ind, ihr[ihr_ind])

                        ihr_signals.extend(ihr)

                ihr_signals = np.array(ihr_signals)
                n_pad = (30 * 249 + 128) * FS_HR - len(ihr_signals)
                if n_pad > 0:
                    ihr_signals = np.concatenate([ihr_signals, np.ones(n_pad) * -1])

                # labels.extend(cont_sleep_stage[np.flatnonzero((sleep_stage_timestamp > last_ppg_groups_timestamp) & (
                #                     sleep_stage_timestamp <= last_ppg_groups_timestamp + 30))])
                # labels = np.array(labels)
                # labels_2hz = np.repeat(labels, 2)
                # labels = labels[49:]

                # interpolate nan in labels
                # labels_ind = np.flatnonzero(~np.isnan(labels))
                # if len(labels_ind) < len(labels):
                #     labels_df = pd.Series(labels)
                #     label_filled = labels_df.ffill().bfill()
                #     labels = np.array(label_filled)

                # interpolate -1 values in HR and Movement
                ihr_signals_ind = np.flatnonzero(ihr_signals != -1)
                if len(ihr_signals_ind) > 0:
                    ihr_signals = np.interp(np.arange(0, len(ihr_signals)), ihr_signals_ind,
                                            ihr_signals[ihr_signals_ind])

                valid_activity_count_30_seg = activity_count_cole_long30[
                    np.flatnonzero((x_timestamp >= (valid_ppg_timestamp_segs[n][0]))
                                   & (x_timestamp <= valid_ppg_timestamp_segs[n][-1]))]

                ihr_signals_org = ihr_signals.copy()
                # Fill IHR and Activity count in padded region by IHR and Activity count in Wake stage
                # pad_len_in_seg = len(np.flatnonzero(ihr_signals == 0))
                # valid_wake_ihr = ihr_signals[labels_2hz == 0]
                # valid_wake_ihr_ind = np.flatnonzero(valid_wake_ihr != 0)
                # valid_wake_ihr = valid_wake_ihr[valid_wake_ihr_ind]
                # valid_wake_act = valid_activity_count_30_seg[labels_2hz == 0][valid_wake_ihr_ind]

                # select strong wake only
                # strong_mask = self.get_strong_wake_mask(
                #     valid_wake_ihr,
                #     valid_wake_act)

                # wake_ihr_pool.extend(valid_wake_ihr[strong_mask])
                # wake_act_pool.extend(valid_wake_act[strong_mask])

                # if pad_len_in_seg > 0:
                #     fake_wake_ihr = self.fill_padding_from_wake(
                #         wake_ihr_pool,
                #         pad_len_in_seg
                #     )
                #     pad_ind = np.flatnonzero(ihr_signals == 0)
                #     ihr_signals[pad_ind] = fake_wake_ihr
                #     fake_wake_act = self.fill_padding_from_wake(
                #         wake_act_pool,
                #         pad_len_in_seg
                #     )
                #     valid_activity_count_30_seg[pad_ind] = fake_wake_act
                # valid_activity_count_30_seg = ((valid_activity_count_30_seg - np.min(valid_activity_count_30_seg)) /
                #                                (np.max(valid_activity_count_30_seg) - np.min(valid_activity_count_30_seg)))

                ihr_smooth = self.smooth_ihr(ihr_signals)

                ihr_signals = self.normalize_with_padding(ihr_smooth)

                ihr_segs = segment(ihr_signals, len(ihr_signals), int(128 * FS_HR), int(30 * FS_HR))

                # label_segs = segment(labels, len(labels), 30, 30)

                # label_segs = np.array([np.unique(label_segs[count]) for count in range(len(label_segs))]).flatten()

                # label_segs = label_segs[: len(ihr_segs)]

                activity_counts_cole_long30_segs = segment(valid_activity_count_30_seg,
                                                           len(valid_activity_count_30_seg),
                                                           int(128 * FS_HR),
                                                           int(30 * FS_HR))

                # if self.n_classes == 3:
                #     label_segs[label_segs == 3] = 2

                total_ihr_org.append(ihr_signals_org)
                total_ihr_segs.append(ihr_segs)
                # total_label.append(label_segs)
                total_activity_counts_cole_long_segs.append(activity_counts_cole_long30_segs)

        total_ihr_org = np.array(total_ihr_org)
        total_ihr_segs = np.array(total_ihr_segs)
        # total_label = np.array(total_label)
        total_activity_counts_cole_long_segs = np.array(total_activity_counts_cole_long_segs)

        clean_ppg_segments = np.expand_dims(total_ihr_segs, axis=-1)
        clean_activity_counts_long_segs = np.expand_dims(total_activity_counts_cole_long_segs, axis=-1)
        return (clean_ppg_segments, clean_activity_counts_long_segs, start_time_sleep, seg_indices,
                total_ihr_org)

    @staticmethod
    def get_continuous_signal_stride(seg_signal, stride, fs):
        step = int(stride*fs)

        # Start with the first segment
        continuous_signal = seg_signal[0][:step].tolist()

        # Append non-overlapping half of each segment
        for i in range(1, len(seg_signal)):
            continuous_signal.extend(seg_signal[i][:step])

        # Append the full last segment
        continuous_signal.extend(seg_signal[-1][step:])
        continuous_signal = np.array(continuous_signal)
        return continuous_signal

    @staticmethod
    def reconstruct_continuous_signal(seg_signal, seg_indices, stride, fs):
        step = int(stride * fs)
        L = seg_signal.shape[1]

        # Tính độ dài signal cuối
        max_idx = seg_indices[-1]
        total_len = max_idx * step + L

        # init với NaN (để biết chỗ missing)
        continuous = np.full(total_len, np.nan)

        for seg, idx in zip(seg_signal, seg_indices):
            start = idx * step
            end = start + L

            # fill vào timeline
            continuous[start:end] = seg

        return continuous

    @staticmethod
    def create_weight(seg_len):
        x = np.linspace(-1, 1, seg_len)
        sigma = 0.5
        w = np.exp(-0.5 * (x / sigma) ** 2)
        return w

    def merge_predictions_weighted_reconstruct(self,
            all_preds,  # list of (T, C)
            seg_indices,  # (k,) index trong n segment gốc
            stride_sec=1800,
            seg_len_sec=7500,
            n_classes=3,
            epoch_sec=30,
    ):
        seg_len = seg_len_sec // epoch_sec
        stride = stride_sec // epoch_sec

        # tổng độ dài theo timeline thật
        max_idx = seg_indices[-1]
        total_len = max_idx * stride + seg_len

        # weight (center > edge)
        weight = self.create_weight(seg_len)  # shape (T,)

        votes = np.zeros((total_len, n_classes))
        weight_sum = np.zeros(total_len)

        for pred, idx in zip(all_preds, seg_indices):
            start = idx * stride
            end = start + seg_len

            # vectorized (nhanh hơn loop t)
            votes[start:end] += weight[:, None] * pred
            weight_sum[start:end] += weight

        # normalize
        valid_mask = weight_sum > 0
        votes[valid_mask] /= weight_sum[valid_mask, None]

        # chỗ missing (không có segment nào cover)
        votes[~valid_mask] = np.nan

        final_pred = np.argmax(votes, axis=-1)

        return final_pred

    @staticmethod
    def get_segments(labels):
        """
        Return:
            [(start, end, label)]
        end is exclusive
        """
        segments = []

        start = 0
        current = labels[0]

        for i in range(1, len(labels)):

            if labels[i] != current:
                segments.append((start, i, current))
                start = i
                current = labels[i]

        segments.append((start, len(labels), current))

        return segments

    def post_process_sleep_labels(self,
            labels,
            epoch_sec=30,

            # thresholds
            min_wake_sec=60,
            min_rem_sec=90,

            # hole filling
            max_gap_sec=60,

            # wake inside rem
            wake_inside_rem_sec=120,
    ):
        """
        wake = 0
        rem = 1
        nrem = 2
        """

        labels = labels.copy()

        min_wake_epochs = max(1, min_wake_sec // epoch_sec)
        min_rem_epochs = max(1, min_rem_sec // epoch_sec)
        max_gap_epochs = max(1, max_gap_sec // epoch_sec)
        wake_inside_rem_epochs = max(1, wake_inside_rem_sec // epoch_sec)

        changed = True

        while changed:

            changed = False

            segs = self.get_segments(labels)

            # =====================================================
            # 1. Remove very short wake
            # =====================================================

            for start, end, cls in segs:

                length = end - start

                if cls == 0 and length <= min_wake_epochs:

                    prev_cls = labels[start - 1] if start > 0 else None
                    next_cls = labels[end] if end < len(labels) else None

                    if prev_cls == next_cls and prev_cls is not None:
                        labels[start:end] = prev_cls
                        changed = True

            segs = self.get_segments(labels)

            # =====================================================
            # 2. Remove very short REM
            # =====================================================

            for start, end, cls in segs:

                length = end - start

                if cls == 1 and length <= min_rem_epochs:
                    labels[start:end] = 2
                    changed = True

            segs = self.get_segments(labels)

            # =====================================================
            # 3. Fill short NREM gaps inside REM
            # REM - NREM(short) - REM
            # =====================================================

            for i in range(1, len(segs) - 1):

                prev_seg = segs[i - 1]
                cur_seg = segs[i]
                next_seg = segs[i + 1]

                s, e, cls = cur_seg
                length = e - s

                if (
                        cls == 2 and
                        length <= max_gap_epochs and
                        prev_seg[2] == 1 and
                        next_seg[2] == 1
                ):
                    labels[s:e] = 1
                    changed = True

            segs = self.get_segments(labels)

            # =====================================================
            # 4. Wake inside long REM -> REM
            # =====================================================

            for i in range(1, len(segs) - 1):

                prev_seg = segs[i - 1]
                cur_seg = segs[i]
                next_seg = segs[i + 1]

                s, e, cls = cur_seg
                length = e - s

                if (
                        cls == 0 and
                        length <= wake_inside_rem_epochs and
                        prev_seg[2] == 1 and
                        next_seg[2] == 1
                ):
                    labels[s:e] = 1
                    changed = True

            segs = self.get_segments(labels)

            # =====================================================
            # 5. Merge tiny isolated NREM between same class
            # =====================================================

            for start, end, cls in segs:

                length = end - start

                if length <= 1:

                    prev_cls = labels[start - 1] if start > 0 else None
                    next_cls = labels[end] if end < len(labels) else None

                    if (
                            prev_cls == next_cls and
                            prev_cls is not None
                    ):
                        labels[start:end] = prev_cls
                        changed = True

        return labels

    @staticmethod
    def trailing_neg_length(arr, value):
        arr = np.asarray(arr)

        # find indices where value != 0
        non_neg_idx = np.where(arr != value)[0]

        if len(non_neg_idx) == 0:
            return len(arr)  # all zeros

        last_neg_zero = non_neg_idx[-1]
        return len(arr) - last_neg_zero - 1

    def get_sleep_stage(self, subject_id):
        acc_file_path = os.path.join(TEMP_FOLDER, f"{subject_id}_acc.parquet")
        ppg_file_path = os.path.join(TEMP_FOLDER, f"{subject_id}_ppg.parquet")
        if not os.path.exists(acc_file_path):
            return "Study isn't processed to have acc_file_path and ppg_file_path yet, we need to run load_study() first."
        if os.path.exists(os.path.join(TEMP_FOLDER, f"{subject_id}_hypnogram.parquet")):
            sleep_stage_status = pd.read_parquet(os.path.join(TEMP_FOLDER, f"{subject_id}_hypnogram.parquet"))['sleep_stage']
            return {
                "hypnogram_path":
                    f"{ppg_file_path.replace('ppg', 'hypnogram')}",

                "summary":
                    {
                        "duration_hours": round(len(sleep_stage_status) / 3600, 2),

                        "wake_percent": round(
                            len(np.flatnonzero(sleep_stage_status == 0)) / len(sleep_stage_status) * 100, 2),
                        "rem_percent": round(
                            len(np.flatnonzero(sleep_stage_status == 1)) / len(sleep_stage_status) * 100, 2),
                        "nrem_percent": round(
                            len(np.flatnonzero(sleep_stage_status == 2)) / len(sleep_stage_status) * 100, 2),
                    }
            }

        df = pd.read_parquet(acc_file_path)
        acc_timestamp = df['timestamp']
        acc_z = df['acc_z']
        acc_y = df['acc_y']
        acc_x = df['acc_x']
        total_acc = np.array([acc_x, acc_y, acc_z]).T

        df_ppg = pd.read_parquet(ppg_file_path)
        timestamp = df_ppg['timestamp']
        ppg_red = df_ppg['ppg_red']
        ppg_ir = df_ppg['ppg_ir']
        total_ppg = np.array([ppg_red, ppg_ir]).T

        ihr_segments, activity_count, start_time_sleep, seg_indices, ihr_signals_org = self.get_ihr_activity_count(total_acc, total_ppg, acc_timestamp, timestamp)
        model = self.load_model()

        # if len(np.flatnonzero(ihr_signals)) == 0:
        #     sleep_stage_status = []
        #     return sleep_stage_status, ihr_signals

        y_predict = model.predict((ihr_segments, activity_count))

        ihr_segments_squeeze = np.squeeze(ihr_segments)
        cont_ihr_segs = []
        for n in range(len(ihr_segments)):
            ihr_seg = ihr_segments_squeeze[n]
            cont_ihr_seg = self.get_continuous_signal_stride(ihr_seg, stride=30, fs=FS_HR)
            cont_ihr_segs.append(cont_ihr_seg)

        # cont_ihr_segs = np.array(cont_ihr_segs)
        # cont_ihr = get_continuous_signal_stride(cont_ihr_segs, stride=int(30 * 60), fs=FS_HR)
        # cont_ihr = self.reconstruct_continuous_signal(cont_ihr_segs, seg_indices, stride=int(30 * 60), fs=FS_HR)

        cont_ihr_org = self.reconstruct_continuous_signal(ihr_signals_org, seg_indices, stride=int(30 * 60), fs=FS_HR)

        final_prediction = self.merge_predictions_weighted_reconstruct(y_predict, seg_indices, stride_sec=int(30 * 60),
                                                                  seg_len_sec=7500, n_classes=3, epoch_sec=30)

        new_filtered_pred = self.post_process_sleep_labels(final_prediction, epoch_sec=30,

                                                      # thresholds
                                                      min_wake_sec=30,
                                                      min_rem_sec=90,

                                                      # hole filling
                                                      max_gap_sec=330,

                                                      # wake inside rem
                                                      wake_inside_rem_sec=120, )

        # find long 0 ihr duration in the end
        pad_len = self.trailing_neg_length(cont_ihr_org, value=0)
        orginal_len = len(new_filtered_pred) - pad_len
        # Remove padding prediction
        new_filtered_pred = new_filtered_pred[: orginal_len]

        sleep_stage_status = np.repeat(new_filtered_pred, 30)
        sleepstage_timestamp = np.arange(len(sleep_stage_status)) + start_time_sleep[0]
        hypnogram_df = pd.DataFrame({
            "timestamp": sleepstage_timestamp,
            "sleep_stage": sleep_stage_status
        })

        hypnogram_df.to_parquet(
            f"{ppg_file_path.replace('ppg', 'hypnogram')}"
        )

        return {
                    "hypnogram_path":
                        f"{ppg_file_path.replace('ppg', 'hypnogram')}",

                    "summary":
                    {
                        "duration_hours": round(len(sleep_stage_status)/3600, 2),

                        "wake_percent": round(len(np.flatnonzero(sleep_stage_status == 0))/len(sleep_stage_status)*100, 2),
                        "rem_percent":  round(len(np.flatnonzero(sleep_stage_status == 1))/len(sleep_stage_status)*100, 2),
                        "nrem_percent":  round(len(np.flatnonzero(sleep_stage_status == 2))/len(sleep_stage_status)*100, 2),
                    }
                }


def percent_rem(y, rem_label=1):  # giả sử REM=2
    total_sleep = len(y)
    rem_time = np.sum(y == rem_label)
    return 100 * rem_time / (total_sleep + 1e-8)

def rem_nrem_ratio(y, rem_label=2, nrem_label=1):
    rem = np.sum(y == rem_label)
    nrem = np.sum(y == nrem_label)
    return rem / (nrem + 1e-8)

# def count_rem_episodes(y, rem_label=2):
#     rem_mask = (y == rem_label).astype(int)
#     transitions = np.diff(rem_mask, prepend=0)
#     return np.sum(transitions == 1)

def find_sleep_onset(y, min_sleep_epochs=20):  # 20 * 30s = 10 phút
    sleep_mask = y != 0  # Wake = 0

    for i in range(len(y) - min_sleep_epochs):
        if np.sum(sleep_mask[i:i+min_sleep_epochs]) == min_sleep_epochs:
            return i
    return None


def sleep_latency(y, epoch_sec=30, min_sleep_epochs=600):
    onset = find_sleep_onset(y, min_sleep_epochs)
    if onset is None:
        return np.nan
    return onset * epoch_sec / 60

def compute_tst(y):
    sleep = (y != 0)
    return np.sum(sleep) / 60  # minutes

def compute_sleep_efficiency(y):
    tst = np.sum(y != 0)

    tib = len(y)

    return tst / tib * 100
