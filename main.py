from recommend import *
import argparse
import numpy as np
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('-uid', '--user_id', type=int, required=True)
    parser.add_argument('--top_n', type=int, default=10)
    parser.add_argument('--normalize', action='store_true', default=False)
    parser.add_argument('--binarize', action='store_true', default=False)
    parser.add_argument('--include_known', action='store_true', default=False)
    parser.add_argument('--random_state', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    print("Loading csv file to pandas.DataFrame")
    data_df = pd.read_csv(args.csv_path)
    
    if args.model == 'nmf-cf':
        recommend_by_NMFBaseCF(args, data_df)
    else:
        print(f'no such model [{args.model}]')

        
if __name__ == '__main__':
    main()