import torch.nn as nn
import torch
from transformers import BertModel
from transformers import CLIPModel

LATENT_DIM = 200


class BertWithCustomHead(nn.Module):
    def __init__(self, output_dim=200, fine_tune_last_n_layers=1, use_clip=True):
        super(BertWithCustomHead, self).__init__()
        self.bert = BertModel.from_pretrained('bert_model')
        self.hidden_size = self.bert.config.hidden_size
        self.use_clip = use_clip

        for name, param in self.bert.named_parameters():
            if name.startswith('encoder.layer') and int(name.split('.')[2]) < (
                    self.bert.config.num_hidden_layers - fine_tune_last_n_layers):
                param.requires_grad = False

        if self.use_clip:
            self.CLIPmodel = CLIPModel.from_pretrained('clip_model')
            self.hidden_size += self.CLIPmodel.config.text_config.hidden_size

        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_size, LATENT_DIM),
            nn.ReLU(),
            nn.Linear(LATENT_DIM, LATENT_DIM),
            nn.ReLU(),
            nn.Linear(LATENT_DIM, output_dim),
            nn.ReLU()
        )

    def forward(self, input_ids, attention_mask, image_embedding=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        if self.use_clip:
            pooled_output = torch.cat((pooled_output, image_embedding.squeeze(dim=1)), dim=1)
        custom_output = self.feed_forward(pooled_output)
        return custom_output


class TwoTowerModel(nn.Module):
    def __init__(self, user_embeddings_dict=None, item_embeddings_dict=None, output_dim=200, use_clip=False):
        super(TwoTowerModel, self).__init__()
        self.user_tower = BertWithCustomHead(output_dim=output_dim, use_clip=use_clip)
        self.item_tower = BertWithCustomHead(output_dim=output_dim, use_clip=use_clip)

        if user_embeddings_dict is not None:
            self.user_embeddings_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in
                                         user_embeddings_dict.items()}
        if item_embeddings_dict is not None:
            self.item_embeddings_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in
                                         item_embeddings_dict.items()}

    def forward(self, user_raw_ids, item_raw_ids, user_input_ids, user_attention_mask, item_input_ids,
                item_attention_mask, image_embedding=None, train=True):
        user_repr = self.user_tower(user_input_ids, user_attention_mask, image_embedding)
        item_repr = self.item_tower(item_input_ids, item_attention_mask, image_embedding)

        if train == True:
            user_additional_embeddings = torch.stack(
                [self.user_embeddings_dict[id].to(user_input_ids.device) for id in user_raw_ids])
            item_additional_embeddings = torch.stack(
                [self.item_embeddings_dict[id].to(item_input_ids.device) for id in item_raw_ids])

            user_repr = torch.cat((user_repr, user_additional_embeddings), dim=1)
            item_repr = torch.cat((item_repr, item_additional_embeddings), dim=1)

        score = torch.sum(user_repr * item_repr, dim=1)
        return score
