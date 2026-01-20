import argparse
import os.path
from pathlib import Path

def main(args):
    import json, time, os, sys, glob
    import shutil
    import warnings
    import numpy as np
    import torch
    from torch import optim
    from torch.utils.data import DataLoader
    import queue
    import copy
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    import os.path
    import subprocess
    from concurrent.futures import ProcessPoolExecutor    
    from utils import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset, StructureDataset, StructureLoader
    from model_utils import featurize, loss_smoothed, loss_nll, get_std_opt, ProteinMPNN, loss_ordinal, loss_emd, loss_rce

    from blosum_distance import BLOSUM60_DISTANCE, CROSS_ENTROPY_DISTANCE, RANDOM_DISTANCE, BLOSUM60_DISTANCE_NP, CROSS_ENTROPY_DISTANCE_NP, RANDOM_DISTANCE_NP, make_softlabels, make_distance_embedding, make_all_distances
    scaler = torch.cuda.amp.GradScaler()
     
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(f"Running on {device}.")
    base_folder = time.strftime(args.path_for_outputs, time.localtime())

    if base_folder[-1] != '/':
        base_folder += '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolders = ['model_weights']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)

    PATH = args.previous_checkpoint

    logfile = base_folder + 'log.txt'
    if not PATH:
        with open(logfile, 'w') as f:
            f.write('Epoch\tTrain\tValidation\n')

    data_path = args.path_for_training_data
    params = {
        "LIST"    : f"{data_path}/list.csv", 
        "VAL"     : f"{data_path}/valid_clusters.txt",
        "TEST"    : f"{data_path}/test_clusters.txt",
        "DIR"     : f"{data_path}",
        "DATCUT"  : "2030-Jan-01",
        "RESCUT"  : args.rescut, #resolution cutoff for PDBs
        "HOMO"    : 0.70 #min seq.id. to detect homo chains
    }


    LOAD_PARAM = {'batch_size': 1,
                  'shuffle': True,
                  'pin_memory':False,
                  'num_workers': 4}

   
    if args.debug:
        args.num_examples_per_epoch = 50
        args.max_protein_length = 1000
        args.batch_size = 1000

    train, valid, test = build_training_clusters(params, args.debug)
    train_set = PDB_dataset(list(train.keys()), loader_pdb, train, params)
    train_loader = torch.utils.data.DataLoader(train_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    valid_set = PDB_dataset(list(valid.keys()), loader_pdb, valid, params)
    valid_loader = torch.utils.data.DataLoader(valid_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)

    model = ProteinMPNN(node_features=args.hidden_dim, 
                        edge_features=args.hidden_dim, 
                        hidden_dim=args.hidden_dim, 
                        num_encoder_layers=args.num_encoder_layers, 
                        num_decoder_layers=args.num_encoder_layers, 
                        k_neighbors=args.num_neighbors, 
                        dropout=args.dropout, 
                        augment_eps=args.backbone_noise)
    model.to(device)


    if PATH:
        checkpoint = torch.load(PATH)
        total_step = checkpoint['step'] #write total_step from the checkpoint
        epoch = checkpoint['epoch'] #write epoch from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        total_step = 0
        epoch = 0

    #optimizer = get_std_opt(model.parameters(), args.hidden_dim, total_step)
    #return NoamOpt(
    #    d_model, 2, 4000, torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9), step
    #)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08) # PyTorch default

    if PATH:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    matrix_name = args.matrix
    beta = float(args.beta) # changed to float for mll
    if matrix_name == "pretraining":
        distance_embedding = None
    elif matrix_name == "crossentropy":
        softlabels = make_softlabels(CROSS_ENTROPY_DISTANCE_NP, beta=int(beta), proba=False, weight=float(args.omega)).to(device)
        distance_embedding = make_distance_embedding(CROSS_ENTROPY_DISTANCE, beta).to(device)
        if args.type == "apl":
            all_distances = make_all_distances(CROSS_ENTROPY_DISTANCE, beta).to(device)
    elif matrix_name == "blosum":
        softlabels = make_softlabels(BLOSUM60_DISTANCE_NP, beta=int(beta), proba=False, weight=float(args.omega)).to(device)
        distance_embedding = make_distance_embedding(BLOSUM60_DISTANCE, beta).to(device)
        if args.type == "apl":
            all_distances = make_all_distances(BLOSUM60_DISTANCE, beta).to(device)
    elif matrix_name == "blosum_probability":
        distance_embedding = make_distance_embedding(BLOSUM60_DISTANCE, beta, proba=True).to(device)
        softlabels = make_softlabels(BLOSUM60_DISTANCE_NP, beta=int(beta), proba=True, weight=float(args.omega)).to(device)        
        if args.type == "apl":
            all_distances = make_all_distances(BLOSUM60_DISTANCE, beta, proba=True).to(device)
    else:
        softlabels = make_softlabels(RANDOM_DISTANCE_NP, beta=int(beta), proba=False, weight=float(args.omega) ).to(device)
        distance_embedding = make_distance_embedding(RANDOM_DISTANCE, beta).to(device)
        if args.type == "apl":
            all_distances = make_all_distances(RANDOM_DISTANCE, beta).to(device)
    with ProcessPoolExecutor(max_workers=12) as executor:
        q = queue.Queue(maxsize=3)
        p = queue.Queue(maxsize=3)
        for i in range(3):
            q.put_nowait(executor.submit(get_pdbs, train_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
            p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
        pdb_dict_train = q.get().result()
        pdb_dict_valid = p.get().result()
       
        dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length) 
        dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
        
        loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
        loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
        reload_c = 0 
        for e in range(args.num_epochs):
            t0 = time.time()
            e = epoch + e
            model.train()
            train_sum, train_weights = 0., 0.
            train_acc = 0.
            if e % args.reload_data_every_n_epochs == 0:
                if reload_c != 0:
                    pdb_dict_train = q.get().result()
                    dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length)
                    loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
                    pdb_dict_valid = p.get().result()
                    dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
                    loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
                    q.put_nowait(executor.submit(get_pdbs, train_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
                    p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
                reload_c += 1
            for _, batch in enumerate(loader_train):
                start_batch = time.time()
                X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                elapsed_featurize = time.time() - start_batch
                optimizer.zero_grad()
                mask_for_loss = mask*chain_M
                
                if args.mixed_precision:
                    with torch.cuda.amp.autocast():
                        log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                        #_, loss_av_smoothed = loss_ordinal(S, log_probs, mask_for_loss, distance=None)
                        #if args.matrix != "pretraining" and args.lambd !=0 :
                        #    if args.ordinal:
                        #        _, loss_av_emd, abs_emd = loss_ordinal(S, log_probs, mask, distance=distance_embedding)
                        #    else:
                        #        _, loss_av_emd, abs_emd = loss_emd(S, log_probs, mask, args.omega, args.mu, distance_embedding)
                        #    scaling = (torch.abs(loss_av_smoothed)/(args.lambd*abs_emd+1e-10)).detach() 
                        #    print(f"scalin {scaling}")
                        #    #scaling = min(1, scaling)
                        #    comb_loss = loss_av_smoothed+scaling*loss_av_emd
                        if args.matrix == "pretraining":
                            _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss, soft_labels=None)
                            comb_loss = loss_av_smoothed
                        elif args.type == "ordinal":
                            _, comb_loss, abs_emd = loss_ordinal(S, log_probs, mask, distance=distance_embedding)
                        elif args.type == "emd":
                            _, comb_loss, _ = loss_emd(S, log_probs, mask, args.omega, args.mu, distance_embedding)
                        elif args.type == "apl":
                            _, comb_loss = loss_rce(S, log_probs, mask, distance=distance_embedding, alpha=float(args.alpha), beta=float(args.mu), all_distances=all_distances)
                        else:
                            _, comb_loss = loss_smoothed(S, log_probs, mask_for_loss, soft_labels=softlabels)
                    #["pretraining", "apl", "smoothing", "emd", "ordinal"]
                    scaler.scale(comb_loss).backward()
                    print(f"comb_loss: {comb_loss.item()}")
                    if args.gradient_norm > 0.0:
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                        #_, loss_av_smoothed = loss_ordinal(S, log_probs, mask_for_loss, distance=None)
                        #if args.matrix != "pretraining" and args.lambd !=0 :
                        #    if args.ordinal:
                        #        _, loss_av_emd, abs_emd = loss_ordinal(S, log_probs, mask, distance=distance_embedding)
                        #    else:
                        #        _, loss_av_emd, abs_emd = loss_emd(S, log_probs, mask, args.omega, args.mu, distance_embedding)
                        #    scaling = (torch.abs(loss_av_smoothed)/(args.lambd*abs_emd+1e-10)).detach() 
                        #    print(f"scalin {scaling}")
                        #    #scaling = min(1, scaling)
                        #    comb_loss = loss_av_smoothed+scaling*loss_av_emd
                    if args.matrix == "pretraining":
                        _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss, soft_labels=None)
                        comb_loss = loss_av_smoothed
                    elif args.type == "ordinal":
                        _, comb_loss, abs_emd = loss_ordinal(S, log_probs, mask, distance=distance_embedding)
                    elif args.type == "emd":
                        _, comb_loss, _ = loss_emd(S, log_probs, mask, args.omega, args.mu, distance_embedding)
                    elif args.type == "apl":
                        _, comb_loss = loss_rce(S, log_probs, mask, distance=distance_embedding, alpha=float(args.alpha), beta=beta, all_distances=all_distances)
                    else:
                        _, comb_loss = loss_smoothed(S, log_probs, mask_for_loss, soft_labels=softlabels)
                    scaler.scale(comb_loss).backward()
                    #print(f"comb_loss: {comb_loss.item()}")
                    if args.gradient_norm > 0.0:
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                    optimizer.step()
                
                loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
            
                train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                train_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                train_weights += torch.sum(mask_for_loss).cpu().data.numpy()

                total_step += 1

            model.eval()
            with torch.no_grad():
                validation_sum, validation_weights = 0., 0.
                validation_acc = 0.
                for _, batch in enumerate(loader_valid):
                    X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                    mask_for_loss = mask*chain_M
                    loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
                    
                    validation_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                    validation_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                    validation_weights += torch.sum(mask_for_loss).cpu().data.numpy()
            
            train_loss = train_sum / train_weights
            train_accuracy = train_acc / train_weights
            train_perplexity = np.exp(train_loss)
            validation_loss = validation_sum / validation_weights
            validation_accuracy = validation_acc / validation_weights
            validation_perplexity = np.exp(validation_loss)
            
            train_perplexity_ = np.format_float_positional(np.float32(train_perplexity), unique=False, precision=3)     
            validation_perplexity_ = np.format_float_positional(np.float32(validation_perplexity), unique=False, precision=3)
            train_accuracy_ = np.format_float_positional(np.float32(train_accuracy), unique=False, precision=3)
            validation_accuracy_ = np.format_float_positional(np.float32(validation_accuracy), unique=False, precision=3)
    
            t1 = time.time()
            dt = np.format_float_positional(np.float32(t1-t0), unique=False, precision=1) 
            with open(logfile, 'a') as f:
                f.write(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}\n')
            print(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}')
            
            checkpoint_filename_last = base_folder+'model_weights/epoch_last.pt'.format(e+1, total_step)
            torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, checkpoint_filename_last)

            if (e+1) % args.save_model_every_n_epochs == 0:
                checkpoint_filename = base_folder+'model_weights/epoch{}_step{}.pt'.format(e+1, total_step)
                torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise, 
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, checkpoint_filename)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--path_for_training_data", type=str, default="my_path/pdb_2021aug02", help="path for loading training data") 
    argparser.add_argument("--path_for_outputs", type=str, default="./exp_020", help="path for logs and model weights")
    argparser.add_argument("--previous_checkpoint", type=str, default="", help="path for previous model weights, e.g. file.pt")
    argparser.add_argument("--num_epochs", type=int, default=200, help="number of epochs to train for")
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="save model weights every n epochs")
    argparser.add_argument("--reload_data_every_n_epochs", type=int, default=2, help="reload training data every n epochs")
    argparser.add_argument("--num_examples_per_epoch", type=int, default=1000000, help="number of training example to load for one epoch")
    argparser.add_argument("--batch_size", type=int, default=10000, help="number of tokens for one batch")
    argparser.add_argument("--max_protein_length", type=int, default=10000, help="maximum length of the protein complext")
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers") 
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=48, help="number of neighbors for the sparse graph")   
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.2, help="amount of noise added to backbone during training")   
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--debug", type=bool, default=False, help="minimal data loading for debugging")
    argparser.add_argument("--gradient_norm", type=float, default=-1.0, help="clip gradient norm, set to negative to omit clipping")
    argparser.add_argument("--mixed_precision", type=bool, default=True, help="train with mixed precision")
    argparser.add_argument("--type", type=str, choices=["pretraining", "apl", "smoothing", "emd", "ordinal"], help="Type of loss")
    argparser.add_argument("--alpha", type=float, default=0, help="Alpha for APL")
    argparser.add_argument("--beta", type=float, default=1, help="Beta for soft-labels")
    argparser.add_argument("--omega", type=float, default=0.1, help="Omega for EMD or epsilon for noise")
    argparser.add_argument("--mu", type=float, default=0.25, help="Mu for EMD or beta for APL passive component")
    argparser.add_argument("--lambd", type=float, default=3.5, help="Ratio between normal loss and EMD/OLL")
    argparser.add_argument("--matrix", type=str, choices=["pretraining","crossentropy", "blosum", "random", "blosum_probability"], default="pretraining",help="Type of matrix to use. Options: pretraining, crossentropy, blosum, random (default: pretraining)")
    argparser.add_argument("--lr", type=float, default=1e-4, help="Learning rate, default 1e-4")
    args = argparser.parse_args()
    print(f"type {args.type}")
    print(f"Alpha {args.alpha}")
    print(f"Beta {args.beta}")
    print(f"Omega {args.omega}")
    print(f"Mu {args.mu}")
    print(f"Lambda {args.lambd}")
    print(f"Matrix {args.matrix}")
    print(f"Learning rate {args.lr}")
    main(args)   
