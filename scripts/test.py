"""
Classificator test script
"""

import argparse
from classificator.classify import Classificator

def main(args):
    # Run the classificator with the given configuration file
    clf = Classificator(config_loc=args.config_loc)
    clf.choose_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run classification')
    parser.add_argument('-c', '--config',
                        dest='config_loc',
                        default='scripts/config.json',
                        help='input configuration location')
    args = parser.parse_args()
    main(args)

