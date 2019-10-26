import matplotlib
# matplotlib.use('Agg')
import click
import sys
import random
import os
from loguru import logger
# import luigi
import subprocess
from src.models.train_model import train_m
from config import config_cls


config = config_cls[os.getenv('ENV', 'default')]

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
