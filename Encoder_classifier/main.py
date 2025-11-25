#only because i don't wnat ot break the other main.py
import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)

from time import time
from classifier import ClassifierBinary
from modelEncoderDecoderAndvancedV3VAE import MIEOVAE
from itertools import product
from tqdm import tqdm
import torch
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utilsData import dataset_loader_full, set_gpu, set_cpu, load_past_results_and_models, is_intel_xeon
import json
from sklearn.metrics import classification_report
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
from skorch.callbacks import EarlyStopping

######################################################
#                                                    #
#                                                    #
#                    _______________                 #
#                   /      RIP      \                #
#                  /                 \               #
#                 |     Here Lies     |              #
#                 |       Intel       |              #
#                 |        Xeon       |              #
#                 |    2024 - 2024    |              #
#                 |___________________|              #
#                      |         |                   #
#                      |         |                   #
#                                                    #
######################################################

# CLASSIFIER PARAMETERS
param_grid = {
        'optimizer__lr' : [0.0005, 0.001, 0.005, 0.01],
        'optimizer__weight_decay' : [0.01e-5, 0.05e-5, 0.1e-5, 0.5e-5],
        'max_epochs' : [50],
        'batch_size' : [100, 300]
    }
# ENCODER PARAMETERS
EN_binary_loss_weight = [ 0.001, 0.01, 0.5, None]# very important
EN_batch_size = [300]
EN_learning_rate = [0.0015, 0.003]
EN_plot = False
EN_embedding_perc_list = [2, 3, 0.8, 0.5]
EN_kl = [0, 0.1, 0.5, 1.0, 1.5]
EN_num_epochs = [250]
EN_masked_percentage_list = [0, 0.2, 0.35, 0.5]
EN_patience = [10]

device = set_cpu()

# LOAD DATASET
#TODO: load dataset
dict = None
if dict is None:
    raise NotImplementedError("Load your dataset here and prepare training and validation sets.")
extended_tr_data = dict['tr_unlabled']
tr_data = dict['tr_data']
tr_out = dict['tr_out']
val_data = dict['val_data']
val_out = dict['val_out']
binary_clumns = dict['bin_col']
#count the pos number
posCount = tr_out.sum()
negCount = tr_out.shape[0] - posCount
posWeight = negCount/posCount
print(f'Positive count: {posCount}')
print(f'Negative count: {negCount}')
print(f'Positive weight: {posWeight}')
print(f'Shape of tr_out: {tr_out.shape}')

# json      models in directory     validated models
results,    existing_models,        validated_models = load_past_results_and_models()

begin = time()
combinations = list(product(EN_binary_loss_weight, EN_batch_size, EN_learning_rate, EN_embedding_perc_list, EN_kl, EN_num_epochs, EN_masked_percentage_list, EN_patience))
combinations.insert(0, (None,)*len(combinations[0]))
print(f'Number of combinations: {len(combinations)}')
for comb in tqdm(combinations, desc="Processing combinations", colour="green"):
    en_bin_loss_w, en_bs, en_lr, en_emb_perc, en_kl, en_num_ep, en_masked_perc, en_pt = comb
    torch.manual_seed(42)

    encoder_string = f'encoder_{en_bin_loss_w}_{en_bs}_{en_lr}_{en_emb_perc}_{en_kl}_{en_num_ep}_{en_masked_perc}_{en_pt}'

    ################################################################################################
    # TRAIN ENCODER ################################################################################
    ################################################################################################
    if comb != (None,)*len(comb):
        if encoder_string in validated_models:
            print(f'{encoder_string} already validated')
            continue
        # create, train and save encoder
        #TODO: create encoder defining the structure properly (hidden dims and latent dim)
        latent_dim = None
        hidden_dims = None
        if latent_dim is None or hidden_dims is None:
            raise NotImplementedError("Define the hidden dimensions and latent dimension for the MIEOVAE model.")
        encoder = MIEOVAE(
            input_dim=tr_data.shape[1], 
            binary=binary_clumns,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims
        )
        # check if encoder exists
        if encoder_string + '.pth' in existing_models:
            # load encoder
            encoder:MIEOVAE = torch.load('./Encoder_classifier/gridResults/Models/' + encoder_string + '.pth', weights_only=False)
            #encoder.load_state_dict(torch.load('./Encoder_classifier/gridResults/Models/' + encoder_string + '.pth', weights_only=True))
        else:
            optimizer = torch.optim.Adam(encoder.parameters(), lr=en_lr)
            encoder.fit(
                tr=extended_tr_data,
                vl=val_data,
                optim=optimizer,
                bs=en_bs,
                bw=en_bin_loss_w,
                kl_beta=en_kl,
                ep=en_num_ep,
                mask_perc=en_masked_perc,
                es=en_pt,
                pedantic=True
            )
            encoder.saveModel(f'./Encoder_classifier/gridResults/Models/{encoder_string}.pth')
            existing_models.append(encoder_string + '.pth')

        encoder.freeze()
        encoded_tr_data = encoder.encode(tr_data)
        val_data_encoded = encoder.encode(val_data)
        embedding_dim = encoder.latent_dim
    else:
        # TODO: check this part when the data is available
        raise NotImplementedError("Check this part when the data is available.")
        encoded_tr_data = tr_data.clone().detach()
        val_data_encoded = val_data.clone().detach()
        embedding_dim = tr_data.shape[1]

    ################################################################################################
    # TRAIN CLASSIFIER #############################################################################
    ################################################################################################


    model = NeuralNetClassifier(
        module = ClassifierBinary,
        module__inputSize = embedding_dim,
        optimizer = torch.optim.Adam,
        device = device,
        criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([posWeight])),#TODO: this will be a hyperparameter (if they pay us enough)
        verbose=0,
        callbacks=[('early_stopping', EarlyStopping(patience=10))]
    )

    grid = GridSearchCV(estimator=model, 
                               param_grid=param_grid, 
                               n_jobs=-1,
                               verbose=0,
                               scoring='balanced_accuracy',
                               cv=4,
                               #random_state=42, //For halving search
                               )
    
    grid_result = grid.fit(encoded_tr_data, tr_out)
    y_pred = grid.predict(val_data_encoded)
    report = classification_report(val_out, y_pred, output_dict=True)

    results.append({
        'encoder_string': encoder_string,
        'encoder': {
            'binary_loss_weight': en_bin_loss_w,
            'batch_size': en_bs,
            'lr': en_lr,
            'emb_perc': en_emb_perc,
            'kl': en_kl,
            'num_ep': en_num_ep,
            'masked_perc': en_masked_perc,
            'pt': en_pt
        },
        'classifier': grid.best_params_,
        'results': report
    })
    with open('./Encoder_classifier/gridResults/results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    
end = time()
print(f'using device: {device}')
tot_time = end - begin
print(f'Total time: {tot_time//60}m {tot_time%60}s')
#print(results)

