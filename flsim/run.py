import argparse
import config
import logging
import os
import server
from datetime import datetime
import time


# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='./config.json',
                    help='Federated learning configuration file.')
parser.add_argument('-l', '--log', type=str, default='INFO',
                    help='Log messages level.')

args = parser.parse_args()

# Set logging
logging.basicConfig(
    format='[%(levelname)s][%(asctime)s]: %(message)s', level=getattr(logging, args.log.upper()), datefmt='%H:%M:%S')


def main():
    """Run a federated learning simulation."""

    # Read configuration file
    fl_config = config.Config(args.config)
    # ---- expose raw config + dp at top level (don't touch fl.dp) ----
    import json
    try:
        with open(args.config, "r") as fh:
            raw = json.load(fh)
        setattr(fl_config, "raw", raw)
        dp_block = raw.get("dp") or raw.get("federated_learning", {}).get("dp")
        if dp_block is not None:
            setattr(fl_config, "dp", dp_block)
    except Exception as e:
        logging.debug(f"DP config not attached: {e}")
# -----------------------------------------------------------------


    # Initialize server
    fl_server = {
        "basic": server.Server(fl_config),
        "accavg": server.AccAvgServer(fl_config),
        "directed": server.DirectedServer(fl_config),
        "kcenter": server.KCenterServer(fl_config),
        "kmeans": server.KMeansServer(fl_config),
        "magavg": server.MagAvgServer(fl_config),
        # "dqn": server.DQNServer(fl_config), # DQN server disabled
        # "dqntrain": server.DQNTrainServer(fl_config), # DQN server disabled
        "sync": server.SyncServer(fl_config),
        "async": server.AsyncServer(fl_config),
    }[fl_config.server]
    fl_server.boot()

    # Run federated learning
    fl_server.run()

    # Save and plot accuracy-time curve
    if fl_config.server == "sync" or fl_config.server == "async":
        d_str = datetime.now().strftime("%m-%d-%H-%M-%S")
        network_type = fl_config.network.type
        total_clients = str(fl_config.clients.total)
        per_round = str(fl_config.clients.per_round)

        fl_server.records.save_record('{}_{}_{}_{}outOf{}.csv'.format(
            fl_config.server, d_str, network_type, per_round, total_clients
        ))
        fl_server.records.plot_record('{}_{}_{}_{}outOf{}.png'.format(
            fl_config.server, d_str, network_type, per_round, total_clients
        ))

    # Delete global model
    #os.remove(fl_config.paths.model + '/global')


if __name__ == "__main__":
    st = time.time()
    main()
    elapsed = time.time() - st
    logging.info('The program takes {} s'.format(
        time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))
    ))
