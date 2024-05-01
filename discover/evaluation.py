
import logging
import math
from training import *
from models import *

from torch.nn import functional as F
from models import RoleLearner

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Add an EOS token at the end of each sequence in a padded batch
def add_eos(target_seqs, target_lengths, pad_trg_left=False):
    if pad_trg_left:
        new_target_seqs = [[config.PAD_TOKEN] + seq for seq in target_seqs]

        for index, length in enumerate(target_lengths):
            new_target_seqs[index][-1*(length+1)] = config.EOS_TOKEN
    else:
        new_target_seqs = [seq + [config.PAD_TOKEN] for seq in target_seqs]

        for index, length in enumerate(target_lengths):
            new_target_seqs[index][length] = config.EOS_TOKEN

    return new_target_seqs
        
# Trim a sequence at the first occurrence of the EOS token
# Otherwise trim at first occurrence of padding
# Otherwise don't trim
def trim_at_eos(seq, pad_trg_left=False):
    if pad_trg_left:
        seq = seq[::-1]
        if config.EOS_TOKEN in seq:
            seq = seq[:seq.index(config.EOS_TOKEN) + 1]
        elif config.PAD_TOKEN in seq:
            seq = seq[:seq.index(config.PAD_TOKEN)]
        
        return seq[::-1]
    else:
        if config.EOS_TOKEN in seq:
            return seq[:seq.index(config.EOS_TOKEN) + 1]
        elif config.PAD_TOKEN in seq:
            return seq[:seq.index(config.PAD_TOKEN)] 
        else:
            return seq


# Function for evaluation of seq2seq models and DISCOVER models
# If target_model is not none, this returns the proportion of examples
# for which model gives the same output as target_model
def evaluate(model, dataset, task_type="seq2seq", criterion=None, no_eos=False, tf_ratio=0.0, target_model=None, prnt=False, pad_trg_left=False, softmax_roles=True, role_learning_hyper=1):
    with torch.no_grad():
        model.eval()
        if target_model is not None:
            target_model.eval()

        count_correct = 0
        total = 0

        total_loss = 0
        if hasattr(model, 'role_predictions'):
            # Start at 1 to avoid dividing by zero
            num_elements = 1
            num_elements_above_98 = 0
            num_elements_above_90 = 0
        for batch in dataset:

            if task_type == "seq2seq":
                # Determine the correct output
                if target_model is not None:
                    correct_output, correct_log_probs = target_model(batch, tf_ratio=tf_ratio)
                elif no_eos:
                    correct_output = batch["target_seq"]
                else:
                    correct_output = add_eos(batch["target_seq"], batch["target_lengths"], pad_trg_left=pad_trg_left)

                # Get predictions on batch
                preds, log_probs = model(batch, tf_ratio=tf_ratio)

                # Check if each output in the batch is correct
                for index, (pred, correct) in enumerate(zip(preds, correct_output)):

                    if prnt:
                        print("correct", correct)
                        print("pred", pred)

                    if no_eos:
                        if pad_trg_left:
                            correct = correct[::-1]
                            pred = pred[::-1]

                        correct = correct[:batch["target_lengths"][index]]
                        pred = pred[:batch["target_lengths"][index]]

                        if pad_trg_left:
                            correct = correct[::-1]
                            pred = pred[::-1]
                    else:

                        correct = trim_at_eos(correct, pad_trg_left=pad_trg_left)
                        pred = trim_at_eos(pred, pad_trg_left=pad_trg_left)

                    if prnt:
                        print("correct", correct)
                        print("pred", pred)
                        print("")

                    if pred == correct:
                        count_correct += 1
                    total += 1

                model_output = log_probs.transpose(0,1).transpose(1,2)
                correct_output = torch.LongTensor(correct_output).to(device=device)

            elif task_type == "encoding":
                _, model_output = model(batch)
                correct_output = batch["encoding"].to(device=device)
                total += len(correct_output)

            elif task_type == "encoding_seq":
                model_output, _ = model(batch)
                correct_output = batch["encoding"].to(device=device)
                total += len(correct_output)

            regularization_loss = 0
            if hasattr(model, 'role_predictions') and model.role_predictions is not None:
                role_predictions = model.role_predictions
                # Zero out the padded role vectors
                lengths = torch.Tensor(batch['target_lengths'])
                mask = lengths[:, None] <= torch.arange(role_predictions.shape[1])
                role_predictions[mask, :] = 0
                batch_one_hot_loss, batch_l2_loss, batch_unique_loss = RoleLearner.get_regularization_loss(role_predictions, softmax_roles=softmax_roles)
                regularization_loss = role_learning_hyper * (batch_one_hot_loss + batch_l2_loss + batch_unique_loss)
                max_role_attention = torch.max(role_predictions, dim=2)[0]
                num_elements_above_98 += torch.sum(max_role_attention > .98)
                num_elements_above_90 += torch.sum(max_role_attention > .9)
                num_elements += torch.sum(lengths)

            if criterion is not None:
                # Get the loss for the batch
                # Multiplied by the batch size because the criterion internally
                # averages over the batch size
                loss = criterion(model_output, correct_output)*len(correct_output)

                total_loss += loss.item()

            total_loss += regularization_loss

        # Average the loss and accuracy
        avg_loss = total_loss / total
        acc = count_correct*1.0 / total

        if hasattr(model, 'role_predictions'):
            logging.info("Role assignment above .9: {:.3f}".format(float(num_elements_above_90 / num_elements)))
            logging.info("Role assignment above .98: {:.3f}".format(float(num_elements_above_98 / num_elements)))

        # Reset so that the dataset can be used again
        dataset.reset()

        return acc, avg_loss


# Not supported for when the encoding is a sequence of vectors (i.e., in Transformers)
def nearest_neighbor(model, dataset, all_batch_list):

    all_batches = []
    all_seqs = []
    for batch_list in all_batch_list:
        batch_list.reset()
        for batch in batch_list:
            all_batches.append(batch)
            for seq in batch["input_seq"]:
                all_seqs.append(seq)

    model.eval()

    count_correct = 0
    count_total = 0

    all_neighbor_embs = [x["encoding"] for x in all_batches]
    all_neighbor_matrix = torch.cat(all_neighbor_embs, 1).detach()

    dataset.reset()


    for index, batch in enumerate(dataset):

        _, model_output = model(batch)
        
        correct_seqs = batch["input_seq"]

        dists = -1*torch.sum(((model_output.transpose(0,1).detach() - all_neighbor_matrix)**2), dim=2)
        topv, topi = dists.topk(1)

        for index_in_batch, elt in enumerate(topi):
            correct_seq = correct_seqs[index_in_batch]
            nearest_neighbor = all_seqs[elt.item()]

            count_total += 1

            if nearest_neighbor == correct_seq:
                count_correct += 1

    acc = count_correct*1.0/count_total

    return acc


def one_minus_tvd(model, target_decoder, dataset, target_encoder=None, bert_mask=False, bert_indices=False, no_eos=False, pad_trg_left=False, encoding_is_seq=False):
    model.eval()
    target_decoder.eval()

    count_total = 0
    total_tvd = 0

    if no_eos:
        eos_modifier = 0
    else:
        eos_modifier = 1

    for index, batch in enumerate(dataset):
        max_length = max(batch["target_lengths"])

        if bert_mask or bert_indices:
            target_output, target_hidden = target_encoder(batch)
            target_output = target_output.detach()
            pred_output = target_output.detach().clone()
            _, pred_hidden = model(batch)

            target_probs = torch.exp(target_decoder(batch, target_output, target_hidden))

            for sentence_index, input_seq in enumerate(batch["input_seq"]):
                if bert_mask:
                    mask_index = input_seq.index(target_encoder.mask_id)
                else:
                    # +1 because of the CLS token
                    mask_index = batch["word_index"][sentence_index] + 1
                pred_output[sentence_index][mask_index] = pred_hidden[0][sentence_index]

            pred_probs = torch.exp(target_decoder(batch, pred_output, pred_hidden))

            for seq_index, seq in enumerate(pred_probs):
                if bert_mask:
                    mask_index = batch["input_seq"][seq_index].index(target_encoder.mask_id)
                else:
                    # +1 because of the CLS token
                    mask_index = batch["word_index"][seq_index] + 1

                tvd = 0.5*torch.sum(torch.abs(pred_probs[seq_index][mask_index].detach() - target_probs[seq_index][mask_index].detach()))
                total_tvd += tvd
                count_total += 1

        else:
            target_encoding = batch["encoding"].detach()
            pred_output, pred_hidden = model(batch) 
        
            pred_probs = torch.exp(target_decoder(batch, pred_output, pred_hidden, tf_ratio=1.0))

            if encoding_is_seq:
                target_probs = torch.exp(target_decoder(batch, target_encoding, None, tf_ratio=1.0))
            else:
                target_probs = torch.exp(target_decoder(batch, None, target_encoding, tf_ratio=1.0))

            for seq_position, elts_at_position in enumerate(pred_probs):
                for seq_index, _ in enumerate(elts_at_position):

                    # We don't count padding
                    if (not pad_trg_left and batch["target_lengths"][seq_index] + eos_modifier > seq_position) or (pad_trg_left and max_length - batch["target_lengths"][seq_index] <= seq_position):
                        tvd = 0.5*torch.sum(torch.abs(pred_probs[seq_position][seq_index].detach() - target_probs[seq_position][seq_index].detach()))
                        total_tvd += tvd
                        count_total += 1

    dataset.reset()

    tvd_average = (total_tvd*1.0/count_total).item()

    return 1 - tvd_average



def explained_variance(model, dataset, encoding_is_seq=False):
    model.eval()

    first = True
    count = 0
    for batch in dataset:
        
        correct_output = batch["encoding"].to(device=device).transpose(0,1)
        count += correct_output.shape[0]

        if first:
            total = torch.sum(correct_output.detach(), dim=0)
            first = False
        else:
            total = total + torch.sum(correct_output.detach(), dim=0)

    mean = total / count

    dataset.reset()

    first = True
    for batch in dataset:
        
        error_tot = torch.sum(torch.pow(batch["encoding"].to(device=device).transpose(0,1).detach() - mean, 2), dim=0)

        if encoding_is_seq:
            model_output, _ = model(batch)
        else:
            _, model_output = model(batch)
        
        error_reg = torch.sum(torch.pow(batch["encoding"].to(device=device).transpose(0,1).detach() - model_output.transpose(0,1).detach(), 2), dim=0)
        if first:
            sse_tot = error_tot
            sse_reg = error_reg
            first = False
        else:
            sse_tot = sse_tot + error_tot
            sse_reg = sse_reg + error_reg


    dataset.reset()

    r2 = 1 - torch.mean(sse_reg) / torch.mean(sse_tot)

    return r2.item()


