import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle

input_variables = ['ch_iso_dsa_1', 'ch_iso_dsa_2',
       'cos_oa', 'm', 'dr', 'dphi_1', 'dphi_2',
       'dimuon_pt', 'dimuon_pt_eff', 'dimuon_fit_pt', 'dimuon_fit_pt_eff',
       'dxy', 'dxyz', 'dxy_sig',
       'pt_dsa_1', 'eta_dsa_1', 'phi_dsa_1', 'pt_dsa_2', 'eta_dsa_2',
       'phi_dsa_2', 'lead_muon_eta', 'lead_muon_phi']

def split_even_odd(data):
    evt_nbr = data['event'].to_numpy()
    even_mask = np.where(evt_nbr % 2 == 0, True, False)
    odd_mask = np.where(evt_nbr % 2 == 1, True, False)

    return even_mask, odd_mask

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str, help='path to input pkl file') 
    parser.add_argument('--output', required=False, default='data', type=str, help='path to output directory')
    args = parser.parse_args()

    # Import signal and background samples
    with open(f'{args.input}', 'rb') as f:
        df_dict = pickle.load(f)
    df_bkg_sideband = df_dict['Data_C']
    # apply bjet cut
    df_bkg_sideband = df_bkg_sideband[df_bkg_sideband['n_bjets']==0]
    print(len(df_bkg_sideband))
    bkg_sum = len(df_bkg_sideband)
    df_bkg_sigReg = df_dict['Data_A']
    # apply bjet cut
    df_bkg_sigReg = df_bkg_sigReg[df_bkg_sigReg['n_bjets']==0]
    print(len(df_bkg_sigReg))
    bkg_sum_sigReg = len(df_bkg_sigReg)
    df_sig = pd.concat([df_dict['HNL1'], df_dict['HNL2'], df_dict['HNL3'], df_dict['HNL4'], df_dict['HNL5'], df_dict['HNL6']], ignore_index=True)
    # apply bjet cut
    df_sig = df_sig[df_sig['n_bjets']==0]
    print(len(df_sig))
    sig_sum = len(df_sig)

    # Add target column (ground truth)
    target_column_name = 'target'
    # 0 for background in sideband
    df_bkg_sideband[target_column_name] = np.zeros(len(df_bkg_sideband), dtype=np.float32)
    # 2 for background in signal region
    df_bkg_sigReg[target_column_name] = 2 * np.ones(len(df_bkg_sigReg), dtype=np.float32)
    # 1 for signal
    df_sig[target_column_name] = np.ones(len(df_sig), dtype=np.float32)

    # Add sample weights column (to balance datasets)
    weight_column_name = 'sample_weight'
    # 1 for background in sideband
    df_bkg_sideband[weight_column_name] = np.ones(len(df_bkg_sideband), dtype=np.float32)
    # sumBkg/sumBkg_sigReg for data in signal region
    df_bkg_sigReg[weight_column_name] = np.ones(len(df_bkg_sigReg), dtype=np.float32) * np.array(bkg_sum/bkg_sum_sigReg, dtype=np.float32)
    # sumBkg/sumSig for signal
    df_sig[weight_column_name] = np.ones(len(df_sig), dtype=np.float32) * np.array(bkg_sum/sig_sum, dtype=np.float32)

    # Put all data toghether and split by even and odd event numbers
    df_all = pd.concat([df_bkg_sideband, df_bkg_sigReg, df_sig], ignore_index=True)
    even_mask, odd_mask = split_even_odd(df_all)
    train_samples_dict = {'even_v2': df_all[even_mask], 'odd_v2': df_all[odd_mask]}

    for key, df in train_samples_dict.items():
        data = np.array(df[[target_column_name, weight_column_name] + input_variables], dtype=np.float32)
        # Shuffling in place
        np.random.shuffle(data)
        # First column corresponds to target
        y = data[:, 0]
        # Second column corresponds to sample weights
        w = data[:, 1]
        # All other columns correspond to input features
        x = data[:, 2:]
        dataset = tf.data.Dataset.from_tensor_slices((x, y, w))
        for element in dataset.take(1):
            print(element)
        output_path = os.path.join(args.output, key)
        dataset.save(output_path, compression='GZIP')