import tqdm
from transformers import AdamW
import torch
import torch.nn as nn
import numpy as np
import os


class Trainer:

    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.loss_function = nn.BCEWithLogitsLoss()
        self.optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    def __dcg_at_k(self, r, k, method=1):
        r = np.asfarray(r)[:k]
        if r.size:
            if method == 0:
                return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
            elif method == 1:
                return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        return 0.

    def __ndcg_at_k(self, r, k, method=1):
        dcg_max = self.__dcg_at_k(sorted(r, reverse=True), k, method)
        if not dcg_max:
            return 0.
        return self.__dcg_at_k(r, k, method) / dcg_max

    def train(self, train_users, val_users, train_dl, val_dl):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_epochs = self.config['num_epochs']
        for epoch in range(num_epochs):
            self.model.train()
            epoch_precision, epoch_recall, epoch_ndcg = [], [], []
            epoch_loss = 0
            train_users_dict = {user: [[], [], []] for user in train_users}
            val_users_dict = {user: [[], [], []] for user in val_users}
            #     print(train_users_dict)
            for batch in tqdm(train_dl):
                user_ids_raw = batch[0]
                item_ids_raw = batch[1]
                user_input_ids = batch[2]['input_ids'].squeeze(1).to(device)
                user_attention_mask = batch[2]['attention_mask'].to(device)
                item_input_ids = batch[3]['input_ids'].squeeze(1).to(device)
                item_attention_mask = batch[3]['attention_mask'].to(device)
                ratings = batch[4].to(device)
                image = batch[6].to(device)
                # true_ratings = batch[5].to(device)

                # scores = model(user_ids_raw, item_ids_raw, user_input_ids, user_attention_mask, item_input_ids, item_attention_mask, True)
                scores = self.model(user_ids_raw, item_ids_raw, user_input_ids, user_attention_mask, item_input_ids,
                                    item_attention_mask, image, True)

                loss = self.loss_function(scores, ratings)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()

                predictions = torch.sigmoid(scores)
                predictions = (predictions >= 0.5) * 1
                for i, user in enumerate(user_ids_raw):
                    train_users_dict[user][0].append(scores[i].item())
                    train_users_dict[user][1].append(predictions[i].cpu())
                    train_users_dict[user][2].append(ratings[i].cpu())
                # precision, recall = calculate_precision_recall(predictions, ratings)
            for user in tqdm(train_users_dict):
                scores = train_users_dict[user][0]
                predictions = train_users_dict[user][1]
                ratings = train_users_dict[user][2]
                if len(scores) and np.sum(ratings, axis=0):
                    if len(scores) < 5:
                        k = len(scores)
                    else:
                        k = 5

                    top_k_indices = np.argpartition(scores, -k, axis=0)[-k:]
                    # Get predicted and true labels for top k indices
                    top_k_preds = np.take_along_axis(np.array(predictions), top_k_indices, axis=0)
                    top_k_true = np.take_along_axis(np.array(ratings), top_k_indices, axis=0)

                    # Calculate true positives
                    true_positives = np.sum(top_k_preds * top_k_true, axis=0)

                    # Calculate precision and recall
                    precision = true_positives / k
                    recall = true_positives / np.sum(ratings, axis=0)
                    epoch_precision.append(precision)
                    epoch_recall.append(recall)
                    epoch_ndcg.append(self.__ndcg_at_k(top_k_true, k))
            average_epoch_precision = np.mean(epoch_precision)
            average_epoch_recall = np.mean(epoch_recall)
            average_epoch_ndcg = np.mean(epoch_ndcg)

            self.model.eval()
            val_loss = 0
            val_precision = []
            val_recall = []
            val_ndcg = []
            with torch.no_grad():
                for batch in tqdm(val_dl):
                    user_ids_raw = batch[0]
                    item_ids_raw = batch[1]
                    user_input_ids = batch[2]['input_ids'].squeeze(1).to(device)
                    user_attention_mask = batch[2]['attention_mask'].to(device)
                    item_input_ids = batch[3]['input_ids'].squeeze(1).to(device)
                    item_attention_mask = batch[3]['attention_mask'].to(device)
                    ratings = batch[4].to(device)
                    image = batch[6].to(device)

                    # scores = model(user_ids_raw, item_ids_raw, user_input_ids, user_attention_mask, item_input_ids, item_attention_mask, False)
                    scores = self.model(user_ids_raw, item_ids_raw, user_input_ids, user_attention_mask, item_input_ids,
                                        item_attention_mask, image, False)
                    loss = self.loss_function(scores, ratings)
                    val_loss += loss.item()

                    predictions = torch.sigmoid(scores)
                    predictions = (predictions >= 0.5) * 1

                    for i, user in enumerate(user_ids_raw):
                        val_users_dict[user][0].append(scores[i].item())
                        val_users_dict[user][1].append(predictions[i].cpu())
                        val_users_dict[user][2].append(ratings[i].cpu())
                    # precision, recall = calculate_precision_recall(predictions, ratings)

                for user in tqdm(val_users_dict):

                    scores = val_users_dict[user][0]
                    predictions = val_users_dict[user][1]
                    ratings = val_users_dict[user][2]
                    if len(scores) and np.sum(ratings, axis=0):
                        if len(scores) < 5:
                            k = len(scores)
                        else:
                            k = 5
                        top_k_indices = np.argpartition(scores, -k, axis=0)[-k:]
                        # Get predicted and true labels for top k indices
                        top_k_preds = np.take_along_axis(np.array(predictions), top_k_indices, axis=0)
                        top_k_true = np.take_along_axis(np.array(ratings), top_k_indices, axis=0)

                        # Calculate true positives
                        true_positives = np.sum(top_k_preds * top_k_true, axis=0)

                        # Calculate precision and recall
                        precision = true_positives / k
                        recall = true_positives / np.sum(ratings, axis=0)
                        val_precision.append(precision)
                        val_recall.append(recall)
                        val_ndcg.append(self.__ndcg_at_k(top_k_true, k))
                average_val_precision = np.mean(val_precision)
                average_val_recall = np.mean(val_recall)
                average_val_ndcg = np.mean(val_ndcg)
            print(f'Saving model for epoch {epoch}')
            model_path = config['save_path'] + '/models/' + f'Epoch{epoch}_model_state_dict.pth'

            if not os.path.exists('/'.join(model_path.split('/')[:-1])):
                os.makedirs('/'.join(model_path.split('/')[:-1]))
                print(f"Directory created: {'/'.join(model_path.split('/')[:-1])}")
            else:
                print(f"Directory already exists: {'/'.join(model_path.split('/')[:-1])}")

            torch.save(self.model.state_dict(), model_path)

            print(
                f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss / len(train_dl)}, Validation Loss: {val_loss / len(val_dl)}, Training Precision: {average_epoch_precision}, Training Recall: {average_epoch_recall}, Training NDCG: {average_epoch_ndcg}, Val Precision: {average_val_precision}, Val Recall: {average_val_recall}, Val NDCG: {average_val_ndcg}')

    def test(self, infer_users, test_dl):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        infer_users_dict = {user: [[], [], []] for user in infer_users}
        infer_loss = 0
        val_loss = 0
        infer_precision = []
        infer_recall = []
        with torch.no_grad():
            for batch in tqdm(test_dl):
                user_ids_raw = batch[0]
                item_ids_raw = batch[1]
                user_input_ids = batch[2]['input_ids'].squeeze(1).to(device)
                user_attention_mask = batch[2]['attention_mask'].to(device)
                item_input_ids = batch[3]['input_ids'].squeeze(1).to(device)
                item_attention_mask = batch[3]['attention_mask'].to(device)
                ratings = batch[4].to(device)
                image = batch[6].to(device)

                # scores = model(user_ids_raw, item_ids_raw, user_input_ids, user_attention_mask, item_input_ids, item_attention_mask, False)
                scores = self.model(user_ids_raw, item_ids_raw, user_input_ids, user_attention_mask, item_input_ids,
                                    item_attention_mask, image, False)
                loss = self.loss_function(scores, ratings)
                val_loss += loss.item()

                predictions = torch.sigmoid(scores)
                predictions = (predictions >= 0.5) * 1

                for i, user in enumerate(user_ids_raw):
                    infer_users_dict[user][0].append(scores[i].item())
                    infer_users_dict[user][1].append(predictions[i].cpu())
                    infer_users_dict[user][2].append(ratings[i].cpu())

            for i, user in tqdm(enumerate(infer_users_dict)):
                scores = infer_users_dict[user][0]
                predictions = infer_users_dict[user][1]
                ratings = infer_users_dict[user][2]
                if len(scores) and np.sum(ratings, axis=0):
                    if len(scores) < 5:
                        k = len(scores)
                    else:
                        k = 5
                    top_k_indices = np.argpartition(scores, -k, axis=0)[-k:]
                    # Get predicted and true labels for top k indices
                    top_k_preds = np.take_along_axis(np.array(predictions), top_k_indices, axis=0)
                    top_k_true = np.take_along_axis(np.array(ratings), top_k_indices, axis=0)
                    # print(top_k_preds, top_k_true)

                    # Calculate true positives
                    true_positives = np.sum(top_k_preds * top_k_true, axis=0)

                    # Calculate precision and recall
                    precision = true_positives / k
                    recall = true_positives / np.sum(ratings, axis=0)
                    infer_precision.append(precision)
                    infer_recall.append(recall)
            average_infer_precision = np.mean(infer_precision)
            average_infer_recall = np.mean(infer_recall)
            print(f'Training Precision: {average_infer_precision}, Training Recall: {average_infer_recall}')
