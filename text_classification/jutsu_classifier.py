import torch
import huggingface_hub
import pandas as pd
from .cleaner import Cleaner
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset

class JutsuClassifier:
    def __init__(self,model_path,data_path=None,
                 text_column_name='text',label_column_name='jutsu',
                 model_name = "distilbert/distilbert-base-uncased",
                 test_size=0.2,num_labels=3,
                 huggingface_token = None):
        self.model_path = model_path
        self.data_path = data_path
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        self.model_name = model_name
        self.test_size = test_size
        self.num_labels = num_labels
        self.huggingface_token = huggingface_token
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.huggingface_token is not None:
            huggingface_hub.login(self.huggingface_token)

        self.tokenizer = self.load_tokenizer()

        if not huggingface_hub.repo_exists(self.model_path):
            #check if data path is provided
            if data_path is None:
                raise ValueError("Data path is required to train the model since the model path doesnt exist in huggingface")

            train_data,test_data = self.load_data(self.data_path)

    def simplify_jutsu(self,jutsu):
        if "Genjutsu" in jutsu:
            return "Genjutsu"
        if "Ninjutsu" in jutsu:
            return "Ninjutsu"
        if "Taijutsu" in jutsu:
            return "Taijutsu"

    def preprocess_function(self,tokenizer, examples):
        return tokenizer(examples['text_cleaned'], padding="max_length", truncation=True)


    def load_data(self,data_path):
        df = pd.read_csv(data_path , lines=True)
        df['jutsu_type_simplified'] = df['jutsu_type'].apply(self.simplify_jutsu)
        df['text'] = df['jutsu_name'] + ". " + df['jutsu_description']
        df['jutsu'] = df['jutsu_type_simplified']
        df = df[['text', 'jutsu']]
        df = df.dropna()

        #Clean the data
        cleaner = Cleaner()
        df['text_cleaned'] = df[self.text_column_name].apply(cleaner.clean)

        #Encode the labels
        le = preprocessing.LabelEncoder()
        le.fit(df[self.label_column_name].tolist())

        # One line for loop that detect the dict
        label_dict = {index: label_name for index, label_name in enumerate(le.__dict__['classes_'].tolist())}
        df['label'] = le.transform(df[self.label_column_name].tolist())

        #Train test split
        test_size = 0.2
        df_train, df_test = train_test_split(df, test_size=test_size,
                                             stratify=df['label'])

        # Convert Pandas to a hugging face Dataset
        train_dataset = Dataset.from_pandas(df_train)
        test_dataset = Dataset.from_pandas(df_test)

        # Tokenize the dataset
        tokenized_train = train_dataset.map(lambda examples: self.preprocess_function(self.tokenizer, examples), batched=True)
        tokenized_test = test_dataset.map(lambda examples: self.preprocess_function(self.tokenizer, examples), batched=True)

        return tokenized_train, tokenized_test

    def load_tokenizer(self):
        if huggingface_hub.repo_exists(self.model_path):
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return tokenizer
