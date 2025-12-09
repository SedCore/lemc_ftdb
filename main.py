# Learned-Weight Monte Carlo FT-DropBlock (LEMC-FTDB) with per-model temperature scaling
# Attribution to: Pierre Sedi N.
# https://github.com/SedCore/lemc_ftdb
# Original paper: Nzakuna, Pierre Sedi and Gallo, Vincenzo and CarratÙ, Marco and Paciello, Vincenzo and Pietrosanto, Antonio and Lay-Ekuakille, Aimé
#   “Learned-Weight Ensemble Monte Carlo DropBlock for Uncertainty Estimation and EEG Classification,”
#   in IEEE Open Journal of Instrumentation and Measurement,
#   doi: 10.1109/OJIM.2025.3638922
# 
import os
import time
import torch
import argparse

from skorch.helper import SliceDataset
from braindecode.util import set_random_seeds
from sklearn.metrics import cohen_kappa_score
from utils import SuppressPrint, load_dataset

import numpy as np
import pandas as pd
from collections import Counter
import random
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import entropy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 18,
    'font.family':'sans-serif',
    'font.sans-serif':'FreeSans',
    'axes.titlesize': 20,
    'axes.titleweight': 'bold',
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'text.color': 'black',
    'axes.labelcolor': 'black',
    #'axes.edgecolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black'})
 
from sklearn.metrics import log_loss, roc_auc_score, roc_curve, auc
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K


import warnings
warnings.filterwarnings("ignore", message="The structure of `inputs` doesn't match")

cuda = torch.cuda.is_available() # Check if GPU is available. If not, use CPU.
device = 'cuda' if cuda else 'cpu'

# Random seed to make results reproducible
def set_seeds(seed):
    set_random_seeds(seed=seed, cuda=cuda) # Set random seed to be able to reproduce results
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)

set_seeds(seed=42)  # Set a fixed seed for reproducibility

# Handle parameters from the terminal CLI
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='EEGNetv4', choices=['EEGNetv4','EEGITNet'], help='Select the model: EEGNetv4 or EEGITNet.')
parser.add_argument('--prob', type=float, default=0.5, choices=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], help='Define the overall drop probability: 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9')
parser.add_argument('--block', type=int, default=15, choices=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], help='Define an integer block size value for FT-DropBlock.')
parser.add_argument('--subject', type=int, default=10, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], help='Select the subject ID (1-9) or 10 to load all.')
parser.add_argument('--u', type=int, default=1, choices=[1,2])
args = parser.parse_args()

prob = args.prob; block_size = args.block; mod = args.model; sub = None if args.subject == 10 else args.subject; u = args.u

print('Subject: ', sub)
print('Selected model: ', mod)
print('Drop probability: ', prob)
print('Block size: ', block_size)

# Load the dataset
print("Loading the dataset BCI Competition IV 2a...")
with SuppressPrint(): windows_dataset, n_channels, n_times, n_classes, sfreq = load_dataset("BCICIV_2a", subject_id=sub)

print("Number of EEG electrode channels = ", n_channels)
print("Number of time points per window (number of samples per second) = ", n_times)
print("Frequency = ", sfreq)

# Split the dataset per subject, to get Train & Test subdatasets for each subject
subjects_windows_dataset = windows_dataset.split('subject')
n_subjects = len(subjects_windows_dataset.items())

# Ensembling parameters
M = 15 # Number of models to ensemble
T = 200 # Number of forward passes to average the predictions
print(f"Number of models in the ensemble: {M}, Number of forward passes: {T}")

# timing statistics
train_times = {name: [] for name in ['MCD','MCSD','MC-FTDB','E-Dropout','EMCD','LEMC-FTDB']}
inf_times   = {name: [] for name in train_times}

eps = 1e-8
lr = 1e-3
batchsize = 64
callbacks = []
learned_temperatures = []

# Model evaluation
accs = []
ks = []

# Calculate and print uncertainty metrics
def calculate_uncertainty_metrics(predictions, test_y, flat_scaled=None, w_learned=None):
    learned= True if flat_scaled is not None and w_learned is not None else False

    # Compute the predictive mean probability for each sample
    predictive_mean = np.einsum('nmk,nm->nk', flat_scaled, w_learned) if learned else np.mean(predictions, axis=0)
    pred_labels = np.argmax(predictive_mean, axis=1) 
    accuracy = np.mean(pred_labels == test_y)
    kappa = cohen_kappa_score(test_y, pred_labels)
    print(f"Overall accuracy: {accuracy*100:.2f}% , Kappa: {kappa:.4f}")

    # 1) Predictive entropy H[ p̄ ]
    H_pred = entropy(predictive_mean.T, base=2)   # returns shape (n_samples,)
    # 2) Expected entropy E_t[ H( p^(t) ) ]
    # Flatten all_preds -> shape (n_samples, M*T, n_classes)
    flat = flat_scaled if learned else predictions.transpose(1,0,2)  # (n_samples, M*T, n_classes)
    H_each = entropy(flat, base=2, axis=2)    # (n_samples, M*T)
    E_H = np.sum(w_learned * H_each, axis=1) if learned else np.mean(H_each, axis=1)  # (n_samples,) --- weighted or unweighted average
    # (3) Mutual information = H_pred - E_H
    MI = H_pred - E_H
    # (4) Predictive variance (averaged over classes)
    variance = np.einsum('nmk,nm->nk', (flat_scaled - predictive_mean[:,None,:])**2, w_learned).mean(axis=1) if learned else np.mean(np.var(flat, axis=1), axis=1)
    # (5) Expected Calibration Error (ECE)
    n_bins = 10
    logits = np.log(predictive_mean + eps)
    ece = tfp.stats.expected_calibration_error(num_bins=n_bins, logits=logits, labels_true=test_y.astype(np.int32)).numpy()
    # (6) Maximum calibration error (MCE)
    # We use the same confidences as for ECE / reliability diagram
    conf = np.max(predictive_mean, axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(conf, bins) - 1
    mce = 0.0
    for i in range(n_bins):
        mask = (bin_ids == i)
        if np.any(mask):
            # empirical accuracy in this bin
            acc_bin  = np.mean(pred_labels[mask] == test_y[mask])
            # mean confidence in this bin
            conf_bin = np.mean(conf[mask])
            mce = max(mce, abs(acc_bin - conf_bin))
    # 7) NLL
    nll = log_loss(test_y, predictive_mean)  # monolithic, same as NLL
    # Flatten for Brier: reshape to (N, K) and one-hot encode y_test
    A, B = predictive_mean.shape
    test_y_onehot = np.eye(B)[test_y]
    # 8) Brier Score (for multi‐class)
    brier = np.mean(np.sum((predictive_mean - test_y_onehot)**2, axis=1))
    # 9) Error‐detection AUC (using entropy)
    unc = entropy(predictive_mean.T, base=2)           # (N,)
    correct = (np.argmax(predictive_mean,1) == test_y)
    auc_err = roc_auc_score(~correct, unc)             # ~correct marks misclassifications
    # 10) FPR@95%TPR
    fpr, tpr, thresholds = roc_curve(~correct, unc)
    fpr95 = fpr[np.where(tpr >= 0.95)[0][0]] if np.any(tpr>=0.95) else np.nan
    # 11) Confidence
    conf = np.max(predictive_mean, axis=1)

    print(f"Mean predictive entropy:  {np.mean(H_pred):.4f}")
    print(f"Mean expected entropy:    {np.mean(E_H):.4f}")
    print(f"Mean mutual information:  {np.mean(MI):.4f}")
    print(f"Mean predictive variance: {np.mean(variance):.4f}")
    print(f"ECE:     {ece:.4f}")
    print(f"MCE:     {mce:.4f}")
    print(f"NLL:     {nll:.4f}")
    print(f"Brier:   {brier:.4f}")
    print(f"Err-AUC: {auc_err:.4f}")
    print(f"FPR@95%TPR: {fpr95:.4f}")
    print(f"Mean confidence: {np.mean(conf):.4f}")

    metrics = {
        'accuracy': accuracy,
        'kappa': kappa,
        'predictive_mean': predictive_mean,
        'preds': pred_labels,
        'H_each': H_each,
        'H_pred': H_pred,
        'E_H': E_H,
        'MI': MI,
        'variance': variance,
        'ece': ece,
        'mce': mce,
        'nll': nll,
        'brier': brier,
        'auc_err': auc_err,
        'fpr95': fpr95,
        'confidence': conf
    }
    return metrics


def montecarlo(subject, drop, model, train_x, test_x, trainy_oh, test_y, epochs):
    K.clear_session() # Clear the Keras session for each model
    model_loaded = None
    train_time_mc = 0
    inf_time_mc = 0
    save_path = f'save/{model.name}/S{subject}_MC_{drop}_c{block_size}.keras'

    if not os.path.exists(save_path):
        train_time_mc = time.time()
        model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=lr))
        model.fit(train_x, trainy_oh, batch_size=batchsize, epochs=epochs, callbacks=callbacks, verbose=0)
        model.save(save_path)
        train_time_mc = time.time() - train_time_mc
        print(f"Trained model in {train_time_mc:.2f} seconds")

    # Work with the saved model to ensure reproducible results
    if drop == 'FTDropBlock2D':
        from models.ftdropblock import FTDropBlock2D
        model_loaded = tf.keras.models.load_model(save_path, custom_objects={'FTDropBlock2D': FTDropBlock2D}, safe_mode=False)
    elif drop == 'SpatialDropout2D' or drop == 'Dropout':
        model_loaded = tf.keras.models.load_model(save_path, safe_mode=False)
    print(f"Loaded the saved model: S{subject}_MC_{drop}.")

    all_preds_mc = []  # will collect (1*T, n_samples, n_classes)
    inf_time_mc = time.time()
    for _ in range(T):
        # DropBlock active by setting training=True
        preds = model_loaded(test_x, training=True).numpy()
        all_preds_mc.append(preds)
    inf_time_mc = time.time() - inf_time_mc
    print(f"Total Forward pass time: {inf_time_mc:.2f} seconds")
    all_preds_mc = np.array(all_preds_mc)  # shape = (1*T, n_samples, n_classes)

    metrics_mc = calculate_uncertainty_metrics(all_preds_mc, test_y)
    return metrics_mc, train_time_mc, inf_time_mc


def ensemble_montecarlo(subject, drop, train_x, test_x, trainy_oh, test_y, epochs, mc=True, trainmodels=True, dropout=True):
    ensemble = []
    all_preds_emc = []  # will collect (M*T, n_samples, n_classes)
    model_loaded = None

    # Training
    train_time_emc = 0
    for i in range(M):
        set_seeds(42 + i) # Set different seed for each model
        save_path = f'save/{mod}/S{subject}_EMC_{drop}_m{i+1}.keras' if dropout else f'save/{mod}/S{subject}_EMC_{drop}_m{i+1}_c{block_size}.keras'
        if trainmodels and not os.path.exists(save_path):
            train_time_emc = time.time() if i == 0 else train_time_emc
            if mod == 'EEGNetv4':
                model = EEGNetv4(n_classes, Chans=n_channels, Samples=n_times, dropoutRate=prob, F1=8, kernLength=32, D=2, F2=2*8, dropType=drop, block=block_size)
            elif mod == 'EEGITNet':
                model = EEGITNet(out_class=n_classes, drop_rate=prob, dropType=drop, blocksize=block_size)
            start_time = time.time()
            model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=lr))
            model.fit(train_x, trainy_oh, batch_size=batchsize, epochs=epochs, callbacks=callbacks, verbose=0)
            print(f"Trained model {i+1}/{M} in {(time.time() - start_time):.2f} seconds")
            model.save(save_path)
            
    if trainmodels:
        train_time_emc = time.time() - train_time_emc
        print(f"Ensemble training took {train_time_emc:.2f} seconds")    
    
    for i in range(M):
        #K.clear_session() # Clear the Keras session for each model
        set_seeds(42 + i)
        if dropout:
            model_loaded = tf.keras.models.load_model(f'save/{mod}/S{subject}_EMC_{drop}_m{i+1}.keras', safe_mode=False)
        else:
            from models.ftdropblock import FTDropBlock2D
            model_loaded = tf.keras.models.load_model(f'save/{mod}/S{subject}_EMC_{drop}_m{i+1}_c{block_size}.keras', custom_objects={'FTDropBlock2D': FTDropBlock2D}, safe_mode=False)
        print(f"Loaded model {i+1}/{M}")
        ensemble.append(model_loaded)
    
    
    set_seeds(42) # Reset seed
    
    # Inference
    inf_time_emc = time.time()
    g=0
    for m in ensemble:
        g+=1
        a = T if mc else 1
        fwd_pass_time = []
        for _ in range(a):
            start_time = time.time()
            # DropBlock active by setting training=True
            preds = m(test_x, training=True).numpy() if mc else m(test_x, training=False).numpy()
            all_preds_emc.append(preds)
            fwd_pass_time.append(time.time() - start_time)
        print(f"Model {g}/{M} Forward pass time: {np.sum(fwd_pass_time):.2f} seconds")
    inf_time_emc = time.time() - inf_time_emc
    print(f"Total Inference time: {inf_time_emc:.2f} seconds")
    all_preds_emc = np.array(all_preds_emc)  # shape = (M*T, n_samples, n_classes)
    
    metrics_emc = calculate_uncertainty_metrics(all_preds_emc, test_y)
    return metrics_emc, all_preds_emc, train_time_emc, inf_time_emc


def ensemble_montecarlo_inference_per_model_temp(ensemble_preds, test_y, use_temps=True, use_weights=True):
    global learned_temperatures
    set_seeds(seed=42) # Set a fixed seed for reproducibility
    # -----------------------------
    # 0) Setup & Per-Model Scaling
    # -----------------------------
    flat = ensemble_preds.transpose(1,0,2)
    flat_logits = np.log(flat + eps) #np.array of shape (N, M*T, K) where N = number of samples, M*T = number of model passes, K = number of classes.
    N, MT, K = flat_logits.shape # (N, M*T, K)

    # Reshape logits and learn per-model temperatures
    logits_rs = tf.constant(flat_logits.reshape(N, M, T, K), dtype=tf.float32)

    if use_temps:
        T_model = tf.Variable(tf.ones([M], dtype=tf.float32), trainable=True)
        opt_T = Adam(learning_rate=1e-3)
        y_tf = tf.constant(test_y, dtype=tf.int32)

        train_time_lemc = time.time()

        for step in range(500):
            with tf.GradientTape() as tape:
                scaled = logits_rs / T_model[None, :, None, None]
                probs_rs = tf.nn.softmax(scaled, axis=-1)
                flat_scaled = tf.reshape(probs_rs, (N, M*T, K))
                pred_mean = tf.reduce_mean(flat_scaled, axis=1)
                loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_tf, pred_mean, from_logits=False))
            grads = tape.gradient(loss, [T_model])
            opt_T.apply_gradients(zip(grads, [T_model]))
            T_model.assign(tf.clip_by_value(T_model, 0.01, 10.0))
        train_time_lemc = time.time() - train_time_lemc
        print("Learned per-model temperatures:", T_model.numpy())
        learned_temperatures = T_model.numpy()

        # Final per-model scaled probabilities
        probs_rs = tf.nn.softmax(logits_rs / T_model[None, :, None, None], axis=-1).numpy()
    else:
        probs_rs = tf.nn.softmax(logits_rs, axis=-1).numpy()
        train_time_lemc = 0.0
    
    flat_scaled = probs_rs.reshape(N, M*T, K) # (N, M*T, K) for all variants

    if use_weights:
        # ---------------------------------------------
        # 1) Learned Weighting Network with Composite Loss
        # ---------------------------------------------
        # Prepare per-pass features
        conf_each = np.max(flat_scaled, axis=2)               # (N, M*T)
        H_each    = entropy(flat_scaled, base=2, axis=2)      # (N, M*T)
        X_feats   = np.stack([conf_each, H_each], axis=2)     # (N, M*T, 2)

        P_tf = tf.constant(flat_scaled, dtype=tf.float32)     # (N, M*T, K)
        y_tf = tf.constant(test_y, dtype=tf.int32)            # (N,)

        # Define MLP g_phi
        inp = tf.keras.Input(shape=(2,))
        x   = tf.keras.layers.Dense(16, activation="relu")(inp)
        x   = tf.keras.layers.Dense(8,  activation="relu")(x)
        w_out = tf.keras.layers.Dense(1, activation="softplus")(x)
        weight_model = tf.keras.Model(inputs=inp, outputs=w_out)

        opt_w = Adam(1e-3)
        epochs    = 200

        train_time2 = time.time()

        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                feats_flat   = tf.reshape(X_feats, (-1, 2))           # (N*M*T, 2)
                raw_w_flat   = tf.squeeze(weight_model(feats_flat),1) # (N*M*T,)
                raw_w        = tf.reshape(raw_w_flat, (N, M*T))       # (N, M*T)
                w_norm       = raw_w / tf.reduce_sum(raw_w, axis=1, keepdims=True)

                # Weighted predictive mean
                p_weighted = tf.einsum("nij,ni->nj", P_tf, w_norm)    # (N, K)

                # NLL term
                nll = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_tf, p_weighted, from_logits=False))

                # MI term
                H_pred    = -tf.reduce_sum(p_weighted * tf.math.log(p_weighted+eps), axis=1)
                H_each_tf = -tf.reduce_sum(P_tf * tf.math.log(P_tf+eps), axis=2)
                E_H_tf    = tf.reduce_sum(w_norm * H_each_tf, axis=1)
                mi        = tf.reduce_mean(H_pred - E_H_tf)

                # Composite loss
                loss = nll

            grads = tape.gradient(loss, weight_model.trainable_weights)
            opt_w.apply_gradients(zip(grads, weight_model.trainable_weights))
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} — Loss={loss.numpy():.4f}, NLL={nll.numpy():.4f}, MI={mi.numpy():.4f}")

        # Compute learned weights
        feats_flat = tf.reshape(X_feats, (-1, 2))
        raw_w_flat= tf.squeeze(weight_model(feats_flat),1)
        raw_w     = tf.reshape(raw_w_flat, (N, M*T))
        w_learned = raw_w / tf.reduce_sum(raw_w, axis=1, keepdims=True)
        w_np      = w_learned.numpy()
    else:
        w_np = np.full((N, M*T), 1.0/(M*T), dtype=np.float32) # Uniform weights (no learned aggregation)
        train_time2 = 0.0

    final_time_lemc = time.time() - train_time2 + train_time_lemc

    metrics_lwemcdb = calculate_uncertainty_metrics(ensemble_preds, test_y, flat_scaled=flat_scaled, w_learned=w_np)
    return flat_scaled, w_np, ensemble_preds, metrics_lwemcdb, final_time_lemc

class ResetCovarianceCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        """Resets covariance matrix at the beginning of the epoch."""
        if epoch > 0:
            # Find the layer by name or type
            for layer in self.model.layers:
                if hasattr(layer, "reset_covariance_matrix"):
                    layer.reset_covariance_matrix()

def sngp(subject, train_x, test_x, trainy_oh, test_y, epochs, method='SNGP'):
    #K.clear_session() # Clear the Keras session for each model
    set_seeds(seed=42) # Set a fixed seed for reproducibility
    model = None
    train_time = 0
    inf_time = 0
    save_path = f'save/{mod}/S{subject}_{method}_c{block_size}.keras'
    
    if not os.path.exists(save_path):
        print(f"Training a new {method} model for subject {subject}...")
        
        if mod == 'EEGNetv4':
            from models.sngp_models import EEGNetv4 as EEGNetv4_sngp
            model = EEGNetv4_sngp(n_classes, Chans=n_channels, Samples=n_times, dropoutRate=prob, num_inducing=(512*3))
        elif mod == 'EEGITNet':
            from models.sngp_models import EEGITNet as EEGITNet_sngp
            model = EEGITNet_sngp(out_class=n_classes, Chans=n_channels, Samples=n_times, drop_rate=prob, num_inducing=512)

        # Print model output shape
        print(f"{method} model output shape: {model.output_shape}")
        model.compile(loss=[tf.keras.losses.CategoricalCrossentropy(from_logits=True), None], optimizer=Adam(learning_rate=lr))
        
        callbacks = [ResetCovarianceCallback()] # Reset covariance at each epoch start
        train_time = time.time()
        model.fit(train_x, trainy_oh, batch_size=batchsize, epochs=epochs, callbacks=callbacks, verbose=0)
        model.save(save_path)
        print(f"Trained {method} model in {(time.time() - train_time):.2f} seconds")
    else:
        from models.spectral_normalization import SpectralNormalization
        from models.gaussian_process import RandomFeatureGaussianProcess
        model = tf.keras.models.load_model(save_path, custom_objects={'SpectralNormalization': SpectralNormalization, 'RandomFeatureGaussianProcess': RandomFeatureGaussianProcess}, safe_mode=False) # Work with the saved model to ensure reproducible results
        print(f"Loaded the saved {method} model: S{subject}_{method}.")

    inf_time = time.time()
    #preds = model(test_x, training=False).numpy() # shape (N,K)
    logits, cov_mat = model(test_x, training=False)
    preds = tf.nn.softmax(logits, axis=-1).numpy() # shape (N,K)
    print(f"{method} Inference time: {(time.time() - inf_time):.2f} seconds")

    variance = tf.linalg.diag_part(cov_mat).numpy() # Covariance matrix shape (N,N) -> predictive variance shape (N,)
    print(f"Logits shape: {logits.shape}, Covariance matrix shape: {cov_mat.shape}, Variance shape: {variance.shape}, Mean variance: {np.mean(variance):.4f}, Mean Std. dev. from the variance: {np.sqrt(np.mean(variance)):.4f}")

    # Sample T=200 times from the Gaussian distribution of logits for each test sample and convert to probabilities with softmax
    print("Sampling from the Gaussian distribution of logits...")
    samples = np.random.randn(T, logits.shape[0], n_classes) # Sample from Gaussian distribution: shape (T, N, K)
    logits_samples = logits.numpy() + np.sqrt(variance)[:, None] * samples
    logits_samples = logits_samples.astype(np.float32)
    probs_samples = tf.nn.softmax(logits_samples, axis=-1).numpy()
    print(f"Sampled probs shape: {probs_samples.shape}")


    metrics = calculate_uncertainty_metrics(probs_samples, test_y)
    return metrics, train_time, inf_time


def get_duq_backbone(full_model, dense_units=100, norm_rate=0.25, length_scale=0.5, centroid_dims=100, trainable_centroids=True):
    from models.rbf_layers import RBFClassifier, add_l2_regularization
    
    # Use Functional API to handle complex architectures
    input_shape = full_model.input_shape[1:]  # Remove batch dimension
    x_input = tf.keras.layers.Input(shape=input_shape)
    
    # Pass through the existing backbone
    features = full_model(x_input)
    
    # Add the new Dense layer
    x = tf.keras.layers.Dense(units=dense_units, activation='relu', kernel_constraint=tf.keras.constraints.MaxNorm(norm_rate), kernel_regularizer=tf.keras.regularizers.l2(1e-1), name='duq_dense')(features)
    
    # Add RBF classifier
    rbf_output = RBFClassifier(num_classes=n_classes, length_scale=length_scale, centroid_dims=centroid_dims, trainable_centroids=trainable_centroids)(x)
    
    # Create the complete DUQ model
    duq_model = tf.keras.Model(inputs=x_input, outputs=rbf_output, name=full_model.name + "_duq")
    
    duq_model.compile(loss='binary_crossentropy', metrics=["categorical_accuracy"], optimizer=Adam(learning_rate=1e-2))
    add_l2_regularization(duq_model, l2_strength=1e-2)
    
    return duq_model


def duq(subject, train_x, test_x, trainy_oh, test_y, epochs, method='DUQ'):
    set_seeds(seed=42) # Set a fixed seed for reproducibility
    train_time = 0
    inf_time = 0
    save_path = f'save/{mod}/S{subject}_{method}_c{block_size}.keras'
    callbacks = []#tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)]
    
    if not os.path.exists(save_path):
        print(f"Training a new {method} model for subject {subject}...")

        if mod == 'EEGNetv4':
            from models.duq_models import EEGNetv4_DUQ
            backbone = EEGNetv4_DUQ(Chans=n_channels, Samples=n_times, F1=8, kernLength=32, D=2, F2=16)

        elif mod == 'EEGITNet':
            from models.duq_models import EEGITNet_DUQ
            backbone = EEGITNet_DUQ(Chans=n_channels, Samples=n_times)

        duq_model = get_duq_backbone(backbone)
        
        train_time = time.time()
        duq_model.fit(train_x, trainy_oh, batch_size=batchsize, epochs=epochs, callbacks=callbacks, verbose=0)
        train_time = time.time() - train_time
        duq_model.save(save_path)
        print(f"Trained {method} model in {train_time:.2f} seconds")
    else:
        from models.rbf_layers import RBFClassifier
        duq_model = tf.keras.models.load_model(save_path, custom_objects={'RBFClassifier': RBFClassifier}, compile=False, safe_mode=False)
        print(f"Loaded the saved {method} model: S{subject}_{method}.")

    inf_time = time.time()
    preds = duq_model.predict(test_x, verbose=0) # shape (N,n_classes)
    inf_time = time.time() - inf_time
    print(f"{method} Inference time: {inf_time:.2f} seconds")

    metrics = calculate_uncertainty_metrics(preds[None,...], test_y) # Reuse your metrics function: wrap probs as (1, N, K)
    return metrics, train_time, inf_time

def eval_subset(m, t, N, K, preds, test_y):
    global M, T
    sub = preds[:m, :t].reshape(m*t, N, K)  # back to (m*t, N, K)
    M = m
    T = t
    
    _, _, _, out, _ = ensemble_montecarlo_inference_per_model_temp(sub, test_y, use_temps=True, use_weights=True)
    return out

M_list = [1, 5, 10, 15]
T_list = [1, 10, 50, 100, 150, 200]


# Evaluate the model for each subject in the dataset
for subject, windows_dataset in subjects_windows_dataset.items():
    if int(subject) != sub: continue
    print("---------------------------------------------------------------------------------------------------------------------------------")
    print(f"Subject {subject}")
    print("---------------------------------------------------------------------------------------------------------------------------------\n")
    train_dataset = windows_dataset.split('session')['0train']
    test_dataset = windows_dataset.split('session')['1test'] 
    train_X = np.array([x for x in SliceDataset(train_dataset, idx=0)]) #print(train_X.shape) = (288,22,1125) batch,eeg_ch,times
    train_y = np.array([y for y in SliceDataset(train_dataset, idx=1)]) #print(train_y.shape) = (288,)
    
    test_X = np.array([x for x in SliceDataset(test_dataset, idx=0)])#
    test_y = np.array([y for y in SliceDataset(test_dataset, idx=1)])
    
    #### FOR TENSORFLOW KERAS
    train_X_keras = np.expand_dims(train_X, axis=1) # batch,1,eeg_channels,times (N,C,H,W) Channels first
    test_X_keras = np.expand_dims(test_X, axis=1) # batch,1,eeg_channels,times (N,C,H,W) Channels first
    trainy_oh = to_categorical(train_y, num_classes=4)
    testy_oh = to_categorical(test_y, num_classes=4)

    keras_model = None
    epochs = 500

    if mod=='EEGNetv4':
        from models.eegnet import EEGNetv4
        keras_model_dropout = EEGNetv4(n_classes, Chans=n_channels, Samples=n_times, dropoutRate=prob, F1=8, kernLength=32, D=2, F2=2*8, dropType='Dropout')
        keras_model_spdropout = EEGNetv4(n_classes, Chans=n_channels, Samples=n_times, dropoutRate=prob, F1=8, kernLength=32, D=2, F2=2*8, dropType='SpatialDropout2D')
        keras_model_ftdropblock = EEGNetv4(n_classes, Chans=n_channels, Samples=n_times, dropoutRate=prob, F1=8, kernLength=32, D=2, F2=2*8, dropType='FTDropBlock2D', block=block_size)
    
    elif mod=='EEGITNet':
        from models.eegitnet import EEGITNet
        keras_model_dropout = EEGITNet(out_class=n_classes, drop_rate=prob, dropType='Dropout')
        keras_model_spdropout = EEGITNet(out_class=n_classes, drop_rate=prob, dropType='SpatialDropout2D')
        keras_model_ftdropblock = EEGITNet(out_class=n_classes, drop_rate=prob, dropType='FTDropBlock2D', blocksize=block_size)
        train_X_keras = train_X_keras.transpose(0,2,3,1) # to match the channels last format
        test_X_keras = test_X_keras.transpose(0,2,3,1) # to match the channels last format

    
    #################################################
        
    results = {}
    
    print("\n===== Single Model Monte Carlo Dropout =====")
    results['MCD'], t_train, t_inf = montecarlo(subject, 'Dropout', keras_model_dropout, train_X_keras, test_X_keras, trainy_oh, test_y, epochs)
    train_times['MCD'].append(t_train); inf_times['MCD'].append(t_inf)
    np.save(f'save/{mod}/S{subject}_MCD.npy', results['MCD'])

    print("\n===== Single Model Monte Carlo Spatial Dropout =====")
    results['MCSD'], t_train, t_inf = montecarlo(subject, 'SpatialDropout2D', keras_model_spdropout, train_X_keras, test_X_keras, trainy_oh, test_y, epochs)
    train_times['MCSD'].append(t_train); inf_times['MCSD'].append(t_inf)
    np.save(f'save/{mod}/S{subject}_MCSD.npy', results['MCSD'])

    print("\n===== Single Model Monte Carlo FT-DropBlock =====")
    results['MC-FTDB'], t_train, t_inf = montecarlo(subject, 'FTDropBlock2D', keras_model_ftdropblock, train_X_keras, test_X_keras, trainy_oh, test_y, epochs)
    train_times['MC-FTDB'].append(t_train); inf_times['MC-FTDB'].append(t_inf)
    np.save(f'save/{mod}/S{subject}_MC-FTDB.npy', results['MC-FTDB'])

    print("\n===== Unweighted Deep Ensemble dropout =====")
    results['E-Dropout'], _, t_train, t_inf = ensemble_montecarlo(subject, 'Dropout', train_X_keras, test_X_keras, trainy_oh, test_y, epochs, mc=False, trainmodels=True, dropout=True)
    train_times['E-Dropout'].append(t_train); inf_times['E-Dropout'].append(t_inf)
    np.save(f'save/{mod}/S{subject}_E-Dropout.npy', results['E-Dropout'])

    print("\n===== Unweighted Ensemble Monte Carlo Dropout =====")
    results['EMCD'], _, t_train, t_inf = ensemble_montecarlo(subject, 'Dropout', train_X_keras, test_X_keras, trainy_oh, test_y, epochs, mc=True, trainmodels=False, dropout=True)
    train_times['EMCD'].append(t_train); inf_times['EMCD'].append(t_inf)
    np.save(f'save/{mod}/S{subject}_EMCD.npy', results['EMCD'])

    print("\n===== Deterministic Uncertainty Quantification =====")
    results['DUQ'], train_times['DUQ'], inf_times['DUQ'] = duq(subject, train_X_keras, test_X_keras, trainy_oh, test_y, epochs)

    print("\n===== Spectral-normalized Neural Gaussian Process =====")
    results['SNGP'], train_times['SNGP'], inf_times['SNGP'] = sngp(subject, train_X_keras, test_X_keras, trainy_oh, test_y, epochs, method='SNGP')

    print("\n===== Unweighted Ensemble MC FT-DropBlock =====")
    res, ensemble_ftdropblock_preds, t_train, t_inf = ensemble_montecarlo(subject, 'FTDropBlock2D', train_X_keras, test_X_keras, trainy_oh, test_y, epochs, mc=True, trainmodels=True, dropout=False)

    print("\n===== Learned-Weighted Ensemble MC DropBlock with Per-Model Temperature =====")
    flat_s, w_learned, lw_pred, results['LEMC-FTDB'], final_time_lemc = ensemble_montecarlo_inference_per_model_temp(ensemble_ftdropblock_preds, test_y, use_temps=True, use_weights=True)
    train_times['LEMC-FTDB'].append(t_train + final_time_lemc); inf_times['LEMC-FTDB'].append(t_inf)
    np.save(f'save/{mod}/S{subject}_LEMC-FTDB.npy', results['LEMC-FTDB'])
    

    #################################################

    np.save(f'save/{mod}/S{subject}_results.npy', results)#_c{block_size}
    np.save(f'save/{mod}/S{subject}_train_times.npy', train_times)#_c{block_size}
    np.save(f'save/{mod}/S{subject}_inf_times.npy', inf_times)#_c{block_size}
    
    #################################################
    
    N = len(test_y)
    modname = 'EEGNet' if mod == 'EEGNetv4' else 'EEG-ITNet'
    method_colors = {'MCD':'orange', 'MCSD':'red', 'MC-FTDB': 'blue', 'E-Dropout':'cyan', 'EMCD': 'brown', 'DUQ':'purple', 'SNGP':'pink', 'LEMC-FTDB': 'green'}

    # 1) Reliability diagram
    plt.figure()
    n_bins = 10
    bins = np.linspace(0.0, 1.0, n_bins+1)
    for name, m in results.items():
        conf = m['confidence']
        correct = (m['preds'] == test_y)
        bin_ids = np.digitize(conf, bins) - 1

        bin_conf, bin_acc = [], []
        for i in range(n_bins):
            mask = bin_ids==i
            if np.any(mask):
                bin_conf.append(conf[mask].mean())
                bin_acc.append(correct[mask].mean())
            else:
                bin_conf.append(np.nan)
                bin_acc.append(np.nan)

        color = method_colors.get(name, None)  # Use default if not found
        plt.plot(bin_conf, bin_acc, marker='o', label=name, color=color)

    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel('Mean Predicted Confidence')
    plt.ylabel('Empirical Accuracy')
    plt.title(f'Reliability Diagram - S{subject}, {modname}')
    plt.legend(loc='best', fontsize='small', ncol=2)
    plt.tight_layout()
    plt.savefig(f'figs/{mod}/S{subject}_reliability_c{block_size}.png')
    plt.close()

    # 2) ROC Curve for Error‐Detection
    plt.figure()
    for name, m in results.items():
        H_pred = m['H_pred']
        correct = (m['preds'] == test_y)
        fpr, tpr, _ = roc_curve(~correct, H_pred)
        roc_auc = auc(fpr, tpr)
        color = method_colors.get(name, None)  # Use default if not found
        plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.2f})', color=color)

    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Error‐Detection ROC Curve - S{subject}, {modname}')
    plt.legend(loc='lower right', fontsize='small')
    plt.tight_layout()
    plt.savefig(f'figs/{mod}/S{subject}_roc_curve_c{block_size}.png')
    plt.close()

    # 3) Risk–Coverage Curve
    plt.figure()
    N = len(test_y)
    for name, m in results.items():
        H_pred = m['H_pred']
        preds  = m['preds']
        order = np.argsort(H_pred)  # low entropy first
        risks, covers = [], []
        for k in range(1, N+1):
            idx = order[:k]
            risk = np.mean(preds[idx] != test_y[idx])
            cover = k / N
            risks.append(risk)
            covers.append(cover)
        color = method_colors.get(name, None)  # Use default if not found
        plt.plot(covers, risks, marker='.', label=name, color=color)

    plt.xlabel('Coverage')
    plt.ylabel('Error Rate')
    plt.title(f'Risk–Coverage Curve - S{subject}, {modname}')
    plt.legend(loc='best', fontsize='small', ncol=2)
    plt.tight_layout()
    plt.savefig(f'figs/{mod}/S{subject}_risk_coverage_c{block_size}.png')
    plt.close()

    # 4) Entropy Boxplot by Outcome
    plt.figure(figsize=(len(results)*2, 7))
    data, labels = [], []
    for name, m in results.items():
        H_pred  = m['H_pred']
        correct = (m['preds'] == test_y)
        data.append(H_pred[correct])
        labels.append(f'{name}\ncorrect')
        data.append(H_pred[~correct])
        labels.append(f'{name}\nincorrect')
    
    box_colors = ['black', 'red', 'black', 'red', 'black', 'red', 'black', 'red', 'black', 'red', 'black', 'red', 'black', 'red']  # Alternate for correct/incorrect
    bp = plt.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor('white')
        patch.set_edgecolor(color)           # Set the outline color
        patch.set_linewidth(2)
    plt.ylabel('Predictive Entropy (bits)')
    plt.title(f'Entropy by Outcome (Correct vs Incorrect) - Subject {subject}, {modname}')
    plt.xticks(fontsize=18, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'figs/{mod}/S{subject}_entropy_boxplot_c{block_size}.png')
    plt.close()