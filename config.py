import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get('DTMGC_DATA_ROOT', ROOT / 'data'))
SAVED_ROOT = Path(os.environ.get('DTMGC_SAVED_ROOT', ROOT / 'saved'))

PATH_TO_FEATURES = {
    'CMUMOSI': DATA_ROOT / 'CMUMOSI' / 'features',
    'IEMOCAPFour': DATA_ROOT / 'IEMOCAP' / 'features',
    'IEMOCAPSix': DATA_ROOT / 'IEMOCAP' / 'features',
}
PATH_TO_LABEL = {
    'CMUMOSI': DATA_ROOT / 'CMUMOSI' / 'CMUMOSI_features_raw_2way.pkl',
    'IEMOCAPFour': DATA_ROOT / 'IEMOCAP' / 'IEMOCAP_features_raw_4way.pkl',
    'IEMOCAPSix': DATA_ROOT / 'IEMOCAP' / 'IEMOCAP_features_raw_6way.pkl',
}
LOG_DIR = SAVED_ROOT / 'log'
