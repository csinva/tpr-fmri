
import logging
from random import shuffle
import math

import torch
import torch.nn as nn
from torch import optim

import config
from evaluation import *
from models import RoleLearner

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Functions for training models

# Training procedure:
#   - Evaluate on 'valid_set' after every 'eval_every' batches
#   - If the loss has not improved for 'patience' evals, then
#     decay the learning rate by dividing it in half, and reload
#     the model from its saved weights (which were saved at the model's
#     point of best performance so far)
#   - If there have been more than 'learning_rate_decay_patience'
#     of these decays, then halt training
#   - Otherwise, training halts after 'n_epochs' epochs
#   - By default the loss is used to determine early stopping, but you
#     can instead use accuracy by setting 'stopping_criterion' to "acc"
#   - Any time you improve on 'stopping_criterion', save the model weights
#     at weights/model_name.weights
# Options for 'task_type':
#   - seq2seq: Sequence to sequence task, with negative log likelihood loss
#     over the sequence of output tokens
#   - encoding: Encoding an input sequence, and trying to minimize the mean
#     squared error between the encoding and a reference encoding
def train(model=None, training_set=None, valid_set=None, n_epochs=100, patience=5, learning_rate_decay_patience=0, stopping_criterion="loss", eval_every=1000, model_name=None, lr=0.001, task_type="seq2seq", no_eos=False, get_padding_loss=False, weight_dir="weights/", save_encoder_only=False, pad_trg_left=False, **kwargs):

    # Put model in training mode
    model.train()
    if hasattr(model, 'role_assigner'):
        model.role_assigner.encoding_model.eval()

    # For determining when to early stop
    best_valid_acc = -1
    best_valid_loss = math.inf
    checkpoints_since_improved = 0
    decays = 0
    done = False

    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    if task_type == "seq2seq":
        if get_padding_loss:
            criterion = nn.NLLLoss()
        else:
            # ignore_index is to ignore the zeroes that were added as padding
            # to handle variable-length sequences within the batch
            criterion = nn.NLLLoss(ignore_index=config.PAD_TOKEN)
    elif task_type == "encoding" or task_type == "encoding_seq":
        criterion = nn.MSELoss()

    # Loop over batches
    index = 0
    done = False

    for epoch_index in range(n_epochs):
        if done:
            break

        training_set.reset()

        for batch in training_set:
            index += 1

            if task_type == "seq2seq":
                # Get the model's outputs
                preds, log_probs = model(batch, tf_ratio=1.0)

                # Determine what the correct output sequence should be
                if no_eos:
                    correct_output = torch.LongTensor(batch["target_seq"]).to(device=device)
                else:
                    correct_output = torch.LongTensor(add_eos(batch["target_seq"], batch["target_lengths"], pad_trg_left=pad_trg_left)).to(device=device)

                model_output = log_probs.transpose(0,1).transpose(1,2)
        
            elif task_type == "encoding":
                _, model_output = model(batch)
                correct_output = batch["encoding"]

            elif task_type == "encoding_seq":
                model_output, _ = model(batch)
                correct_output = batch["encoding"]
       

            # Backpropagate
            optimizer.zero_grad()
            loss = criterion(model_output, correct_output)
            if hasattr(model, 'role_predictions') and model.role_predictions is not None:
                role_predictions = model.role_predictions.clone()
                # Zero out the padded role vectors
                lengths = torch.Tensor(batch['target_lengths'])
                mask = lengths[:, None] <= torch.arange(role_predictions.shape[1])
                role_predictions[mask, :] = 0
                batch_one_hot_loss, batch_l2_loss, batch_unique_loss = RoleLearner.get_regularization_loss(role_predictions, softmax_roles=kwargs['softmax_roles'])
                loss += kwargs['role_learning_hyper'] * (batch_one_hot_loss + batch_l2_loss + batch_unique_loss)

            loss.backward()
            optimizer.step()

            # Evaluate on the validation set
            if index % eval_every == 0:

                # Get the loss and accuracy on the validation set
                valid_acc, valid_loss = evaluate(model, valid_set, task_type=task_type, criterion=criterion, no_eos=no_eos, tf_ratio=1.0, pad_trg_left=pad_trg_left, role_learning_hyper=kwargs['role_learning_hyper'], softmax_roles=kwargs['softmax_roles'])

                # Return to training mode
                model.train()
                if hasattr(model, 'role_assigner'):
                    model.role_assigner.encoding_model.eval()

                logging.info("Epoch: " + str(epoch_index) +"\tValid acc: " + str(valid_acc) + "\tValid loss: " + str(valid_loss))
                # Determine whether to early stop; save model weights
                # if the model has improved since last saving
                if stopping_criterion == "acc":

                    if valid_acc > best_valid_acc:
                        best_valid_acc = valid_acc
                        checkpoints_since_improved = 0
                        if save_encoder_only:
                            torch.save(model.encoder.state_dict(), weight_dir + model_name + ".weights")
                        else:
                            torch.save(model.state_dict(), weight_dir + model_name + ".weights")
                    else:
                        checkpoints_since_improved += 1


                elif stopping_criterion == "loss":
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        checkpoints_since_improved = 0
                        if save_encoder_only:
                            torch.save(model.encoder.state_dict(), weight_dir + model_name + ".weights")
                        else:
                            torch.save(model.state_dict(), weight_dir + model_name + ".weights")
                    else:
                        checkpoints_since_improved += 1

                if checkpoints_since_improved >= patience:
                    decays += 1
                    if decays > learning_rate_decay_patience:
                        done = True
                        break

                    lr = lr / 2.0
                    logging.info("REDUCING LEARNING RATE TO " + str(lr))

                    # Return to best saved model
                    if save_encoder_only:
                        model.encoder.load_state_dict(torch.load(weight_dir + model_name + ".weights"))
                    else:
                        model.load_state_dict(torch.load(weight_dir + model_name + ".weights"))

                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    checkpoints_since_improved = 0

