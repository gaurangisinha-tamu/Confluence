from transformers import (
    Text2TextGenerationPipeline,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from datasets import load_dataset
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split


class KeyPhraseGenerationPipeline(Text2TextGenerationPipeline):
    def __init__(self, model, keyphrase_sep_token=";", *args, **kwargs):
        super().__init__(
            model=AutoModelForSeq2SeqLM.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )
        self.keyphrase_sep_token = keyphrase_sep_token

    def postprocess(self, model_outputs):
        results = super().postprocess(
            model_outputs=model_outputs
        )
        return [[keyphrase.strip() for keyphrase in result.get("generated_text").split(self.keyphrase_sep_token) if
                 keyphrase != ""] for result in results]


class DataGeneration:
    def __init__(self, source_domain_name, target_domain_name, save_path, train_distribution_coeff=4,
                 test_distribution_coeff=4, model_name="ml6team/keyphrase-generation-t5-small-inspec"):
        self.source_domain_name = source_domain_name
        self.target_domain_name = target_domain_name
        self.save_path = save_path
        self.train_distribution_coeff = train_distribution_coeff
        self.test_distribution_coeff = test_distribution_coeff
        self.model_name = model_name
        self.generator = KeyPhraseGenerationPipeline(model=model_name)

    def load_and_filter_data(self):
        source_reviews = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{self.source_domain_name}",
                                      trust_remote_code=True)
        source_meta = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{self.source_domain_name}",
                                   trust_remote_code=True)
        target_reviews = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{self.target_domain_name}",
                                      trust_remote_code=True)
        target_meta = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{self.target_domain_name}",
                                   trust_remote_code=True)

        common_users = list(
            set(source_reviews["full"][:]['user_id']).intersection(set(target_reviews["full"][:]['user_id'])))

        # Create dataframes
        source_reviews_df = source_reviews["full"].to_pandas()
        self.source_meta_df = source_meta["full"].to_pandas()
        target_reviews_df = target_reviews["full"].to_pandas()
        self.target_meta_df = target_meta["full"].to_pandas()

        # Filter dataframe
        common_users_source_df = source_reviews_df[source_reviews_df['user_id'].isin(common_users)]
        temp = common_users_source_df['user_id'].value_counts()
        self.source_filtered_data = source_reviews_df[source_reviews_df['user_id'].isin(temp[temp >= 3].index)]
        self.source_filtered_data = self.source_filtered_data[self.source_filtered_data['text'] != '']
        print(f'Unique common users: {len(self.source_filtered_data["user_id"].unique())}')

        self.target_filtered_data = target_reviews_df[
            target_reviews_df['user_id'].isin(self.source_filtered_data['user_id'].unique())]
        # temp = common_users_target_df['user_id'].value_counts()
        # self.target_filtered_data = target_reviews_df[target_reviews_df['user_id'].isin(temp[temp >= 3].index)]
        # self.target_filtered_data = self.target_filtered_data[self.target_filtered_data['text'] != '']

    def generate_user_item_ratings_csv(self, target=False):
        # user-item ratings
        if target:
            filtered_data = self.target_filtered_data
            name = self.target_domain_name
            datasetmeta_df = self.target_meta_df
            coeff = self.test_distribution_coeff
        else:
            filtered_data = self.source_filtered_data
            name = self.source_domain_name
            datasetmeta_df = self.source_meta_df
            coeff = self.train_distribution_coeff

        filtered_data['true_rating'] = filtered_data['rating']
        filtered_data.loc[filtered_data['rating'] <= 3, 'rating'] = 0

        filtered_data.loc[filtered_data['rating'] != 0, 'rating'] = 1

        positive_samples = filtered_data[filtered_data['rating'] == 1][
            ['user_id', 'parent_asin', 'rating', 'true_rating']]
        negative_samples = filtered_data[filtered_data['rating'] == 0][
            ['user_id', 'parent_asin', 'rating', 'true_rating']]

        positive_samples_ind = list(
            zip(positive_samples['user_id'], positive_samples['parent_asin'], positive_samples['rating'],
                positive_samples['true_rating']))
        negative_samples_ind = list(
            zip(negative_samples['user_id'], negative_samples['parent_asin'], negative_samples['rating'],
                negative_samples['true_rating']))
        total_samples = np.array(positive_samples_ind + negative_samples_ind)

        num_positive_samples = len(positive_samples)
        num_negative_samples = len(negative_samples)

        print(
            f'Positive samples: {num_positive_samples}, Negative samples: {num_negative_samples}, Total samples: {len(total_samples)}')

        tobe_sampled = coeff * num_positive_samples - num_negative_samples
        unrated_items = list(set(datasetmeta_df['parent_asin']) - set(filtered_data['parent_asin']))
        all_users = filtered_data['user_id'].tolist()
        all_items = datasetmeta_df['parent_asin'].tolist()

        while tobe_sampled > 0:
            sampled_neagtive_item = random.choices(all_items, k=tobe_sampled)
            sampled_neagtive_user = random.choices(all_users, k=tobe_sampled)
            sampled_items = set(list(zip(sampled_neagtive_item, sampled_neagtive_user)))
            total_itemset = set(total_samples[:, :2].flatten())
            if sampled_items & total_itemset:
                tobe_added = sampled_items - total_itemset
                tobe_sampled = len(sampled_items & total_itemset)
                total_samples = np.vstack((total_samples, np.array([(x, y, 0, -1) for x, y in tobe_added])))
            else:
                total_samples = np.vstack((total_samples, np.array(list(
                    zip(sampled_neagtive_user, sampled_neagtive_item, [0] * len(sampled_neagtive_user),
                        [-1] * len(sampled_neagtive_user))))))
                tobe_sampled = 0
        print(total_samples.shape)
        total_samples = list(total_samples)
        # If miraculously the negative samples are more,
        # else: bombastic side eye

        self.useritem_df = pd.DataFrame(total_samples, columns=['user_id', 'item_id', 'rating', 'true_rating'])
        self.useritem_df.to_csv(f'{self.save_path}/{name}_user_item_ratings.csv', index=False)

    def generate_item_profile(self, target=False):
        if target:
            filtered_data = self.target_filtered_data
            name = self.target_domain_name
            datasetmeta_df = self.target_meta_df
            coeff = self.test_distribution_coeff
            self.useritem_df = pd.read_csv(f'{self.save_path}/{self.target_domain_name}_user_item_ratings.csv')
        else:
            filtered_data = self.source_filtered_data
            name = self.source_domain_name
            datasetmeta_df = self.source_meta_df
            coeff = self.train_distribution_coeff
            self.useritem_df = pd.read_csv(f'{self.save_path}/{name}_user_item_ratings.csv')
        image_links = []
        unique_items = self.useritem_df['item_id'].unique()  # unique is not really necessary, but just to be sure
        all_items_keyphrases = []
        for item in unique_items:
            titles = datasetmeta_df[datasetmeta_df['parent_asin'] == item]['title']
            features = datasetmeta_df[datasetmeta_df['parent_asin'] == item]['features']
            descriptions = datasetmeta_df[datasetmeta_df['parent_asin'] == item]['description']
            categories = datasetmeta_df[datasetmeta_df['parent_asin'] == item]['categories']
            details = datasetmeta_df[datasetmeta_df['parent_asin'] == item]['details']
            # print(datasetmeta_df[datasetmeta_df['parent_asin'] == item]['images'])
            if not len(datasetmeta_df[datasetmeta_df['parent_asin'] == item]['images'].tolist()):
                print('oh no', item, (item in datasetmeta_df['parent_asin']))
                print(datasetmeta_df[datasetmeta_df['parent_asin'] == item]['images'].tolist())
                image_links.append('')
            elif 'large' in datasetmeta_df[datasetmeta_df['parent_asin'] == item]['images'].tolist()[0]:
                if len(datasetmeta_df[datasetmeta_df['parent_asin'] == item]['images'].tolist()[0]['large']):
                    image_links.append(
                        datasetmeta_df[datasetmeta_df['parent_asin'] == item]['images'].tolist()[0]['large'][0])
                else:
                    image_links.append('')
            else:
                image_links.append('')
            item_keyphrases = set()
            for title, feature, description, category, detail in zip(titles, features, descriptions, categories,
                                                                     details):
                keyphrases = []
                keyphrases.append(title)
                if feature.size:
                    keyphrases.append(feature[0])
                if description.size:
                    keyphrases.append(description[0])
                if category.size:
                    keyphrases.append(category[0])
                if detail:
                    keyphrases.append(','.join(detail[1:-1].split(', ')))
                item_keyphrases.update([','.join(keyphrases)])

            all_items_keyphrases.append(list(item_keyphrases))

        item_profile_df = pd.DataFrame(
            {'item_id': unique_items, 'keyphrases': all_items_keyphrases, 'image_links': image_links})
        item_profile_df.to_csv(f'{self.save_path}/{name}_item_profiles.csv', index=False)

    def generate_user_profile(self, target=False):
        if target:
            filtered_data = self.target_filtered_data
            name = self.target_domain_name
            datasetmeta_df = self.target_meta_df
            coeff = self.test_distribution_coeff
        else:
            filtered_data = self.source_filtered_data
            name = self.source_domain_name
            datasetmeta_df = self.source_meta_df
            coeff = self.train_distribution_coeff
        unique_users = filtered_data['user_id'].unique()
        all_users_keyphrases = []
        for user in unique_users:
            titles = filtered_data[filtered_data['user_id'] == user]['title']
            reviews = filtered_data[filtered_data['user_id'] == user]['text']
            user_keyphrases = set()
            for title, review in zip(titles, reviews):
                keyphrases = self.generator(title + '. ' + review)
                user_keyphrases.update(keyphrases[0])
            all_users_keyphrases.append(list(user_keyphrases))
        user_profile_df = pd.DataFrame({'user_id': unique_users, 'keyphrases': all_users_keyphrases})
        user_profile_df.to_csv(f'{self.save_path}/{name}_user_profiles.csv', index=False)

    def generate_user_profile_parallel(self, user_chunk):
        print('New worker')
        user_keyphrases = []
        for user in user_chunk:
            print(user)
            titles = self.source_filtered_data[self.source_filtered_data['user_id'] == user]['title']
            reviews = self.source_filtered_data[self.source_filtered_data['user_id'] == user]['text']
            user_keyphrase_set = set()
            for title, review in zip(titles, reviews):
                keyphrases = self.generator(title + '. ' + review, max_new_tokens=50)
                user_keyphrase_set.update(keyphrases[0])
            user_keyphrases.append(list(user_keyphrase_set))
        return user_keyphrases

    def train_val_test_split(self, stratified=True, split=0.8):
        useritem_df = pd.read_csv(f'{self.save_path}/{self.source_domain_name}_user_item_ratings.csv')
        useritem_df = useritem_df.sample(frac=1).reset_index(drop=True)
        X = useritem_df[['user_id', 'item_id', 'true_rating']]
        y = useritem_df['rating']

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

        train_useritem_df = X_train.copy()
        train_useritem_df['rating'] = y_train

        val_useritem_df = X_val.copy()
        val_useritem_df['rating'] = y_val
        # train_useritem_df = useritem_df.sample(int(split*len(useritem_df)))
        # val_useritem_df = useritem_df.sample(int((1 - split)*len(useritem_df)))

        train_useritem_df.to_csv(f'{self.save_path}/{self.source_domain_name}_train_user_item_ratings.csv', index=False)
        val_useritem_df.to_csv(f'{self.save_path}/{self.source_domain_name}_val_user_item_ratings.csv', index=False)

    def generation_pipeline(self):
        print('Loading and Filtering Data...')
        self.load_and_filter_data()
        print('Generating source user-item csv...')
        self.generate_user_item_ratings_csv()
        print('Generating target user-item csv...')
        self.generate_user_item_ratings_csv(target=True)
        print('Generating source user profile csv...')
        self.generate_user_profile()
        # print('Generating source user profile csv...')
        # self.generate_user_profile(target = True)
        print('Generating source item profile csv...')
        self.generate_item_profile()
        print('Generating target item profile csv...')
        self.generate_item_profile(target=True)
        print('Splitting ratings into train-val...')
        self.train_val_test_split(split=0.8)
        print('Done!')
