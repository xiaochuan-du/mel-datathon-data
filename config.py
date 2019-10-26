from pathlib import Path
import os
import socket

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    # HOST_NAME = socket.getfqdn(socket.gethostname())
    # HOST_IP = socket.gethostbyname(HOST_NAME)

    # s3 setting
    AWS_REGION_NAME = 'us-east-1'
    S3_BUCKET_NAME = 'default'
    S3_PREFIX = 'dfs'

    # DIR
    DATA_ROOT = None
    MODEL_ROOT = None

    # Debug setting
    DEBUG = True

    # train monitor
    ML_URI = None
    ML_ARTIFACTS = None

    # proj config
    PROJ_TITLE = os.getenv(
        'PROJ_TITLE',
        'MEL_DATATHRON'
    )
    PROJ_DESC = os.getenv(
        'PROJ_DESC',
        'Melbourne Datathon: http://www.datasciencemelbourne.com/datathon/'
    )
    PROJ_VER = os.getenv(
        'PROJ_VER',
        '0.0.1'
    )

    # DIR config
    STATIC_DIR = os.getenv(
        'PROJ_VER',
        f'{basedir}/static'
    )
    BASE_URL = os.getenv(
        'BASE_URL',
        '/api/v1'
    )
    STATIC_URL = f'{BASE_URL}/static'

    # MODEL SETTING
    ML_PARMS = dict(
        bs=512,
    )
    @staticmethod
    def init_app(app):
        pass

class OnPremiseWorker(Config):
    DATA_ROOT = Path(f'/data/ecg/data')
    MODEL_ROOT = Path(f'/data/ecg/models')
    ML_URI = 'mysql://healsci:HealsciAWS1@pathology.cc2wi7jayqzb.rds.cn-north-1.amazonaws.com.cn:3306/anmlflow'
    DEBUG = False
    ML_ARTIFACTS = 's3://hs-ai/an-mlflow/ecg'

class DebugWorker(Config):
    DATA_ROOT = Path(f'{basedir}/data')
    MODEL_ROOT = Path(f'{basedir}/models')
    ML_URI = 'mysql://root:bigdata@35.247.29.81:3306/mlflow-db-v1'
    ML_ARTIFACTS = 's3://hs-ai/an-mlflow/ecg'

config_cls = {
    'default': DebugWorker,
    'prem': OnPremiseWorker,
    'debug': DebugWorker,
}
