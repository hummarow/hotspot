import argparse

def get_argparser():
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument('--data_path', default='./data/',
                        help='Dataset path')

    # Model
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Model learning rate')
    return parser

