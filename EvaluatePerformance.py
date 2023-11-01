import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import mplhep
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

plt.style.use(mplhep.style.CMS)

def plot_train_history(model_path, output_path):
    # Load csv file with training history
    csv_path = os.path.join(model_path, 'training_log.csv')
    df_history = pd.read_csv(csv_path)

    with PdfPages(os.path.join(output_path, 'NN_train_history.pdf')) as pdf:
        # Plot class. loss and adversarial loss function of train and val samples during the training 
        epochs = np.array(df_history['epoch'])
        class_loss_train = np.array(df_history['class_loss'])
        class_loss_val = np.array(df_history['val_class_loss'])
        adv_loss_train = np.array(df_history['adv_loss'])
        adv_loss_val = np.array(df_history['val_adv_loss'])
        fig = plt.figure()
        plt.plot(epochs, class_loss_train, label='Class. loss - train sample', color='purple', linestyle='-')
        plt.plot(epochs, class_loss_val, label='Class. loss - validation sample', color='purple', linestyle='--')
        plt.plot(epochs, adv_loss_train, label='Adv. loss - train sample', color='orange', linestyle='-')
        plt.plot(epochs, adv_loss_val, label='Adv. loss - validation sample', color='orange', linestyle='--')
        plt.legend()
        plt.ylabel('Log Loss')
        plt.xlabel('epochs')
        plt.yscale('log')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Plot class. accuracy function of train and val samples during the training 
        class_acc_train = np.array(df_history['class_accuracy'])
        class_acc_val = np.array(df_history['val_class_accuracy'])
        fig = plt.figure()
        plt.plot(epochs, class_acc_train, label='Train sample', color='purple', linestyle='-')
        plt.plot(epochs, class_acc_val, label='Validation sample', color='purple', linestyle='--')
        plt.legend()
        plt.ylabel('Area under the ROC')
        plt.xlabel('epochs')
        plt.yscale('log')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Plot adv. accuracy function of train and val samples during the training 
        adv_acc_train = np.array(df_history['adv_accuracy'])
        adv_acc_val = np.array(df_history['val_adv_accuracy'])
        fig = plt.figure()
        plt.plot(epochs, adv_acc_train, label='Train sample', color='orange', linestyle='-')
        plt.plot(epochs, adv_acc_val, label='Validation sample', color='orange', linestyle='--')
        plt.ylim(0.19, 0.81)
        plt.legend()
        plt.ylabel('Accuracy')
        plt.xlabel('epochs')
        plt.tight_layout()
        pdf.savefig()
        plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_tag', required=True, type=str)
    # parser.add_argument('--class-size', required=True, type=int)
    # parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--dataset-val', required=False, default='data/val', type=str)
    args = parser.parse_args()

    model_tag = args.model_tag
    model_path = f'data/{model_tag}/model/'

    # Create directory to save output plots
    output_path = f'plots/{model_tag}/'
    os.makedirs(output_path, exist_ok=True)

    # Load the trained model
    model = tf.keras.models.load_model(os.path.join(model_path, 'best'))
    # Check its architecture
    print(model.summary())

    # Load the validation dataset
    dataset_val = tf.data.Dataset.load(args.dataset_val, compression='GZIP')
    x_val, y_val = tuple(zip(*dataset_val))
    x_val = np.array(x_val)
    y_val = np.array(y_val)

    # Make predictions (NN score) on the validation set using the model
    val_predictions = np.array(model.predict(x_val)[0])

    # Plot the NN score on signal and background distribution
    plt.hist(val_predictions[y_val==0], color='blue', range=[0, 1], bins=100, density=True, alpha=0.7, label='Data in sideband')
    plt.hist(val_predictions[y_val==1], color='red', range=[0, 1], bins=100, density=True, alpha=0.7, label='Signal')
    plt.yscale('log')
    plt.xlabel('NN score')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'NN_score_val.pdf'))
    plt.close()

    # Plot training history of the model
    plot_train_history(model_path, output_path)
