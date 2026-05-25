import os, argparse, torch
from pmc.config import Configurator
import sys
import functools
sys.path.append('..')
import time

parser = argparse.ArgumentParser(description='Autonomous Diffusion Model (ADM)')
parser.add_argument(
    "--config", "-c", 
    type=str, 
    help="Path to config file"
)


parser.add_argument(
    '--name',
    type=str,
    default='dummy',
    help='Experiment name. If --optuna is given, this is the study name and the experiment '
)

def main():

    # parse arguments
    args = parser.parse_args()
    # configurate and save configuration file
    cc = Configurator(args)
    os.makedirs(cc.cfg.exp_dir, exist_ok=True)
    with open(f'{cc.cfg.exp_dir}/config.yaml', 'w') as f:
        f.write(str(cc.cfg))

    # regular run
    exp, model, dataloader, callbacks = cc.init_all()
    exp(model, dataloader, callbacks)


if __name__ == '__main__':
    main()
