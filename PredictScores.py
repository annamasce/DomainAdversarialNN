import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

from ImportDataset import split_even_odd, input_variables

def main():
    input_path = '/afs/cern.ch/work/a/amascell/HNLanalysis/hnl-coffea/results/df_231024_Run2_2018_noTimeSel.pkl'  
    output_path = '../dataframes/' 
    model_tags = {
        # Model trained on even event numbers
        'even': '2023-11-14T112242',
        # Model trained on odd event numbers
        'odd': '2023-11-14T114256'
    }

    models = {}
    for key in model_tags.keys():
        models[key] = tf.keras.models.load_model(os.path.join('data', model_tags[key], 'model/best'))
        print(models[key].summary())

    # Import signal and background samples
    with open(input_path, 'rb') as f:
        df_dict = pickle.load(f)

    df_dict_withNNscore = {}
    for ds in df_dict.keys():
        print(f'Predicting scores for sample {ds}')
        df = df_dict[ds]
        # Transform the data as training input
        x_val = np.array(df[input_variables], dtype=np.float32)
        
        # Compute the scores for both even and odd models
        scores_odd = np.array(models['odd'].predict(x_val)[0].flatten())
        print(scores_odd)
        scores_even = np.array(models['even'].predict(x_val)[0].flatten())

        # Use odd scores for even event numbers and even scores for odd event numbers
        mask_even, mask_odd = split_even_odd(df)
        final_scores = np.where(mask_even, scores_odd, scores_even)
        df['NN_score'] = final_scores
        # apply bjet cut
        df = df[df['n_bjets']==0]
        df_dict_withNNscore[ds] = df
    
    plt.hist(df_dict_withNNscore['Data_C']['NN_score'], range=[0.02, 1], bins=10, density=True, alpha=0.7)
    plt.hist(df_dict_withNNscore['Data_A']['NN_score'], range=[0.02, 1], bins=10, density=True, alpha=0.7)
    plt.yscale('log')
    plt.savefig('test.pdf')

    # Save dataframes with NN score 
    with open(output_path + 'df_231024_Run2_2018_noTimeSel_withNN_231114.pkl', 'wb') as f:
        pickle.dump(df_dict, f)

if __name__ == '__main__':
    main()