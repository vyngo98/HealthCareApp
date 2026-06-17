import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MAIN_FOLDER = "/mnt/DataSet2/Projects/POC_Sleep_study/hsat_data/sleep_study/sleep_test"
AGENTS_FOLDER = "/mnt/DataSet2/Projects/POC_Sleep_study/Agents"
TEMP_FOLDER = '/mnt/DataSet2/Projects/POC_Sleep_study/Agents/temp_folder'
SLEEPSTAGE_CKPT = '/mnt/DataSet2/SleepStage_checkpoint/ppg_acc_3classes_norm_hsat_improvedcoattention_250_Cate_v18r/best_loss/0040.keras'


PPG_N_CHANNELS = 2
ACC_N_CHANNELS = 3

SAMPLING_RATE = 250
NOISE_DURATION_THRESHOLD = 5
LEN_EPOCH = 256
NUMBER_OF_SEGMENT = 299
NUM_CLASSES = 2
MIDDLE_HR_SEGMENT = 98

IN_LENGTH_ACTIVITY = 125
FS_ACC = 25
ACC_CHANNELS = 3
ACC_SEG_TIME = 5  # second
ACC_BATCHSIZE = 3


# THRESHOLD
THR_DURATION_DATA = int(2 * 60 * 60)  # 2h
THR_PERCENT = 70  # 70 %
THR_AWAKE = 10*60  # 10 mins


ROOT_TIMEZONE = 7
TIMEZONE = 7
HOUR_START_DAY = 8

NUM_THREADS = 5

CONFIG_S3_BUCKET = os.environ.get('S3_BUCKET', 'btcy-test-s3-data-lake')


SLEEP_MODEL_PATH = 'app/services/ai_service/tflite_model/sleep_stage.tflite'

ADC_GAIN = 26.164

PADDING_THRES = 1  # second

HR_MAX_THR = 200
HR_MIN_THR = 25
RR_MAX_PERCENT = 0.3
RR_MIN_PERCENT = 0.3
rrMinKamathTHR = 0.5
rrMaxKamathTHR = 1.3

NPY_TABLE = {'samples': 1, 'symbols': 2, 'events': 3, 'sv_events': 4}

FS_PPG = 128
PPG_CHANNELS = 2
SECOND_IN_MINUTE = 60
RR_MIN_THR = SECOND_IN_MINUTE * FS_PPG / HR_MAX_THR
RR_MAX_THR = SECOND_IN_MINUTE * FS_PPG / HR_MIN_THR

CHECKPOINT_ACTIVITY = 'app/services/ai_service/tflite_model/activities_segment_5s_bz3_o_lye_stand_walk.tflite'
CHECKPOINT_APNEA = "app/services/ai_service/tflite_model/spo2_apnea_model2.tflite"
CHECKPOINT_APNEA_LGBM = "app/services/ai_service/tflite_model/LGBM_MODEL_best_acc.pkl"

PPG_SEG_TIME = 4
PPG_STRIDE = 0.5

FS_HR = 2
FS_SPO2 = 1
PPG_SPO2_STRIDE = 1

# Test local
CHECKPOINT_APNEA_LGBM = "/home/dongbui-5070ti/Workspace/itr-bioring-backend-sleep_study_poc/app/services/ai_service/tflite_model/LGBM_MODEL_best_acc.pkl"
DEFAULT_TF_SERVER_NAME = '192.168.27.159'
DEFAULT_TF_SERVER_PORT = 7000


MODEL_CONFIG = {
    'spo2': {
        'shape': [60, 1, 1],
        'name': 'spo2',
        'signature_name': 'serving_default',
        'input_name': 'serving_default_input_layer',
        'output_name': 'StatefulPartitionedCall_1'
    },
    'acc': {
        'shape': [3, 125, 3],
        'name': 'acc',
        'signature_name': 'serving_default',
        'input_name': 'serving_default_keras_tensor_14',
        'output_name': 'StatefulPartitionedCall_1'
    },
    'sleep_stage': {
        'shape': [299, 256, 1],
        'name': 'sleep_stage',
        'signature_name': 'serving_default',
        'input_name': 'segment',
        'output_name': 'Identity'
    }

}