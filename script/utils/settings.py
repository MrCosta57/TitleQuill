
import os

from titlequill.utils.io_ import load_json


# Local settings file
ROOT_DIR          = os.path.abspath(os.path.join(__file__, "..", "..",".."))
LOCAL_SETTINGS_FP = os.path.abspath(os.path.join(ROOT_DIR, "local_settings.json"))
LOCAL_SETTINGS    = load_json(LOCAL_SETTINGS_FP)

# Local paths
DATASET_DIR      = LOCAL_SETTINGS["dataset_dir"]
DATASET_TSV_FILE = LOCAL_SETTINGS["dataset_tsv_fp"]
