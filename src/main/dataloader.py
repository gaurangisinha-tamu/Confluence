import pandas as pd
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO
import requests
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


class DataSetLoader:

    def __init__(self, config):
        self.config = config

    def load_data(self):
        tokenizer = AutoTokenizer.from_pretrained('bert_tokenizer')
        train_ds = ReviewsDataset(f"{self.config['save_path']}/{self.config['source_domain']}_user_profiles.csv",
                                  f"{self.config['save_path']}/{self.config['source_domain']}_item_profiles.csv",
                                  f"{self.config['save_path']}/{self.config['source_domain']}_train_user_item_ratings.csv",
                                  tokenizer)
        val_ds = ReviewsDataset(f"{self.config['save_path']}/{self.config['source_domain']}_user_profiles.csv",
                                f"{self.config['save_path']}/{self.config['source_domain']}_item_profiles.csv",
                                f"{self.config['save_path']}/{self.config['source_domain']}_val_user_item_ratings.csv",
                                tokenizer)
        test_ds = ReviewsDataset(f"{self.config['save_path']}/{self.config['source_domain']}_user_profiles.csv",
                                 f"{self.config['save_path']}/{self.config['target_domain']}_item_profiles.csv",
                                 f"{self.config['save_path']}/{self.config['target_domain']}_user_item_ratings.csv",
                                 tokenizer)
        train_dl = DataLoader(train_ds, batch_size=self.config['batch_size'], shuffle=True, drop_last=True)
        val_dl = DataLoader(val_ds, batch_size=self.config['batch_size'], shuffle=False, drop_last=True)
        test_dl = DataLoader(test_ds, batch_size=self.config['batch_size'], shuffle=False, drop_last=True)
        return train_dl, val_dl, test_dl


class ReviewsDataset(Dataset):

    @staticmethod
    def load_user_profiles(user_profile_filepath):
        user_profiles = pd.read_csv(user_profile_filepath)
        return dict(zip(user_profiles['user_id'], user_profiles['keyphrases']))

    @staticmethod
    def load_item_profiles(item_profile_filepath):
        item_profiles = pd.read_csv(item_profile_filepath)
        return dict(zip(item_profiles['item_id'], zip(item_profiles['keyphrases'], item_profiles['image_links'])))

    def __init__(self, user_profile_filepath, item_profile_filepath, ratings_filepath, tokenizer):
        self.user_profiles = ReviewsDataset.load_user_profiles(user_profile_filepath)
        self.item_profiles = ReviewsDataset.load_item_profiles(item_profile_filepath)
        self.ratings = pd.read_csv(ratings_filepath)
        self.tokenizer = tokenizer
        self.CLIPmodel = CLIPModel.from_pretrained('clip_model')
        self.CLIPprocessor = CLIPProcessor.from_pretrained('clip_processor')
        self.CLIPhidden_size = self.CLIPmodel.config.text_config.hidden_size

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        rating = self.ratings.iloc[idx]['rating']
        true_rating = self.ratings.iloc[idx]['true_rating']
        user_profile = self.user_profiles[self.ratings.iloc[idx]['user_id']]
        item_profile = self.item_profiles[self.ratings.iloc[idx]['item_id']][0]
        image_link = self.item_profiles[self.ratings.iloc[idx]['item_id']][1]
        if pd.isna(image_link):
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            try:
                image = np.array(Image.open(BytesIO(requests.get(image_link).content)))
                if len(image.shape) == 2:
                    image = np.stack((image,) * 3, axis=-1)
            except Exception as e:
                print(f'Failed to load image because of error : {e}')
                image = np.zeros((224, 224, 3), dtype=np.uint8)
        CLIPinput = self.CLIPprocessor(images=image, return_tensors="pt")
        CLIPoutput = self.CLIPmodel.get_image_features(**CLIPinput)
        user_profile_tokens = self.tokenizer(text=user_profile,
                                             padding="max_length",
                                             max_length=128,
                                             add_special_tokens=True,
                                             return_attention_mask=True,
                                             truncation=True,
                                             return_tensors="pt")

        item_profile_tokens = self.tokenizer(text=item_profile,
                                             padding="max_length",
                                             max_length=128,
                                             add_special_tokens=True,
                                             return_attention_mask=True,
                                             truncation=True,
                                             return_tensors="pt")

        return self.ratings.iloc[idx]['user_id'], self.ratings.iloc[idx][
            'item_id'], user_profile_tokens, item_profile_tokens, rating, true_rating, CLIPoutput
