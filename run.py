import matplotlib
# matplotlib.use('Agg')
import click
import sys
import random
import os
from loguru import logger
import luigi
import subprocess
from mlflow.cli import _run_server
import mlflow
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize, dummy_minimize
from src.models.train_model import train_m
from config import config_cls


config = config_cls[os.getenv('ENV', 'default')]
mlflow.set_tracking_uri(config.ML_URI)


log_config = {
    "handlers": [
        {"sink": sys.stdout, "level": "INFO"},
    ],
}
logger.configure(**log_config)

@click.group()
def cli():
    pass


@cli.command()
@click.option(
    '--win_len', type=int, default=10,)
@click.option(
    '--expname', type=click.Choice(['one', 'tune']) ,default='tune')
@click.option(
    '--n_calls', type=int, default=1 ,)
def tune(win_len, expname, n_calls):
    try:
        mlflow.create_experiment(expname, config.ML_ARTIFACTS)
    except Exception:
        pass
    finally:
        mlflow.set_experiment(expname)
    space  = [
        Integer(0, 4, name='sizes_idx'),
        Integer(0, 2, name='layers_idx'),
        Categorical(['sigres', 'conv'], name='conv_mode')]
    @use_named_args(space)
    def objective(**params):
        call_params = dict(
            # call params
                params
            )
        mpath, acc, f1 = train_m(
            **call_params
            )
        score = -1 * f1
        with mlflow.start_run():
            mlflow.log_params(
                {key:str(call_params[key]) for key in call_params.keys()})
            mlflow.log_metric('sig_f1', f1)
            mlflow.log_metric('sig_accuracy', acc)
        return score
    
    if n_calls < 10:
        logger.info(f'dummy_minimize {n_calls}')
        res_gp = dummy_minimize(
            objective, space, n_calls=n_calls)
    else:
        logger.info(f'gp_minimize {n_calls}')
        res_gp = gp_minimize(
            objective, space, n_calls=n_calls)
    
    logger.info(f'Best f1 score is {res_gp.fun}')


@cli.command()
@click.option(
    '--win_len', type=int, default=10,)
def train(win_len, ):
    call_params = config.ML_PARMS
    mpath, acc, f1 = train_m(
        **call_params
    )
    logger.info(f'Best f1 score is {f1}, acc is {acc}, model is in {mpath})')

@cli.command()
@click.option(
    '--seed', type=int, default=2018,)
def build_data(seed):
    pass


@cli.command()
@click.option("--host", "-h", metavar="HOST", default="127.0.0.1",
              help="The network address to listen on (default: 127.0.0.1). "
                   "Use 0.0.0.0 to bind to all addresses if you want to access the tracking "
                   "server from other machines.")
@click.option("--port", "-p", default=5000,
              help="The port to listen on (default: 5000).")
def server(host, port):
    import uvicorn
    from src.deployment.app import app
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == '__main__':
    cli()
