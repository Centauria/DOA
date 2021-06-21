# -*- coding: utf-8 -*-
import argparse

import zmq
from tqdm.auto import tqdm

from dataset import GenDOA

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='server', description='Dataset server.')
    parser.add_argument('--port', '-p', default=2021, type=int)
    parser.add_argument('--dataset-path', '-d', required=True, help='Specify path to GenDOA dataset')
    parser.add_argument('--loss-type', '-l', choices=['cartesian', 'polar', 'xpolar'], default='xpolar',
                        help='Choose loss representation')
    args = parser.parse_args()

    loss_type = args.loss_type
    dataset_path = args.dataset_path

    train_data = GenDOA(dataset_path, split='train', loss_type=loss_type)
    val_data = GenDOA(dataset_path, split='test', loss_type=loss_type)

    context = zmq.Context.instance()
    sock = context.socket(zmq.REP)
    sock.bind(f'tcp://*:{args.port}')
    print(f'Bind port {args.port}')

    try:
        total_transport = 0
        with tqdm() as bar:
            while True:
                message = sock.recv_pyobj()
                split, index = message
                if split == 'train':
                    data = train_data
                elif split == 'test':
                    data = val_data
                else:
                    raise ValueError(f'Split {split} is not valid')
                sock.send_pyobj(data[index])
                total_transport += 1
                bar.set_description(f'Total transport: {total_transport}, Last data: {split}[{index}]')
                bar.update(1)
    except KeyboardInterrupt:
        print('Bye')
