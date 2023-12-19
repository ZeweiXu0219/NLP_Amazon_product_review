import os
import pickle
import re
import time

import enchant
import nltk
import numpy as np
# nltk.download('stopwords')
from nltk.corpus import stopwords
import torch
from transformers import BertTokenizer, BertModel
sw = stopwords.words('english')
d = enchant.Dict("en_US")

class Embedding(object):
    def __init__(self, embedding_size, sentence_length,  bert_model_path, data_dir):
        self.sentence_length = sentence_length
        self.embedding_size = embedding_size
        self.bert_model_dir = bert_model_path
        self.data_dir = data_dir

    def sentence_embedding(self):
        tokenizer = BertTokenizer.from_pretrained(self.bert_model_dir)
        input_ids = tokenizer.encode('hello world bert!')
        ids = torch.LongTensor(input_ids)
        print('ids =', ids)
        text = tokenizer.convert_ids_to_tokens(input_ids)
        print('converted beck to text:', text)
        # %%

        model = BertModel.from_pretrained(self.bert_model_dir, output_hidden_states=True)
        # Set the device to GPU (cuda) if available, otherwise stick with CPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        ids = ids.to(device)

        model.eval()
        print(ids.size())
        # unsqueeze IDs to get batch size of 1 as added dimension
        granola_ids = ids.unsqueeze(0)
        print(granola_ids.size())

        out = model(input_ids=granola_ids)  # tuple

        hidden_states = out[2]
        print("last hidden state:", out[0].shape)  # torch.Size([1, 6, 768])
        print("pooler_output of classification token:", out[1].shape)  # [1,768] cls
        print("all hidden_states:", len(out[2]))

        # %%

        for i, each_layer in enumerate(hidden_states):
            print('layer=', i, each_layer)

        # %%

        sentence_embedding = torch.mean(hidden_states[-1], dim=1).squeeze()
        print(sentence_embedding)
        print(sentence_embedding.size())

        # %% 有理论支撑，最后四层的结果concat起来再取平均可以让sentence embedding效果达到最好
        # 但是在做不同的项目时可以试一下不同的层数（最后一层，最后四层，全部等），和不同的池化策略（平均，最大，最小等）

        # get last four layers
        last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]
        # cast layers to a tuple and concatenate over the last dimension
        cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)
        print(cat_hidden_states.size())

        # take the mean of the concatenated vector over the token dimension
        cat_sentence_embedding = torch.mean(cat_hidden_states, dim=1).squeeze()
        print(cat_sentence_embedding)
        print(cat_sentence_embedding.size())

    def word_embedding(self):
        tokenizer = BertTokenizer.from_pretrained(self.bert_model_dir)
        text = "This is an example sentence."
        tokens = tokenizer.tokenize(text)
        print(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        print(input_ids)

        model = BertModel.from_pretrained(self.bert_model_dir, output_hidden_states=True)
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_ids)
            embeddings = outputs.last_hidden_state[0]
            print(embeddings.shape)

    def dataset_embedding(self, sentences, labels):
        print('Loading BERT tokenizer...')
        tokenizer = BertTokenizer.from_pretrained(self.bert_model_dir, do_lower_case=True)
        print('example of Berttokenizer below:')
        print(' Original: ', sentences[0])
        print('Tokenized: ', tokenizer.tokenize(sentences[0]))
        print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = BertModel.from_pretrained(self.bert_model_dir, output_hidden_states=True)
        print(device)
        model.to(device)
        # get max length
        max_len = 0
        # For every sentence...
        for sent in sentences:
            # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
            input_ids = tokenizer.encode(sent, add_special_tokens=True)
            # Update the maximum sentence length.
            max_len = max(max_len, len(input_ids))
        print('Max sentence length: ', max_len)

        # Tokenize all of the sentences and map the tokens to thier word IDs.
        after_embedding = []
        input_ids = []
        attention_masks = []
        token_type_ids = []
        # For every sentence...
        print('Start embedding...')
        count = 0
        starttime = time.time()
        for sent in sentences:
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=self.sentence_length,  # Pad & truncate all sentences.
                padding='max_length',
                truncation=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )
            # Add the encoded sentence to the list.
            input_ids_ = encoded_dict['input_ids']
            input_ids.append(torch.tensor(input_ids_))
            attention_masks_ = encoded_dict['attention_mask']
            attention_masks.append(torch.tensor(attention_masks_))
            with torch.no_grad():
                model.eval()
                outputs = model(torch.tensor(input_ids_).to(device), attention_mask=torch.tensor(attention_masks_).to(device))
                embeddings = outputs.last_hidden_state[0]
                after_embedding.append(embeddings.unsqueeze(0))
            # # input_ids = torch.tensor(input_ids_).unsqueeze(0)
            count +=1
            if count % 1000 == 0 and count != 0:
                print('embedding finish {} '.format(count))
                endtime = time.time()
                print("time cost in bert model processing {}".format(endtime - starttime))

        # Convert the lists into tensors.
        # sentences = torch.cat(after_embedding, dim=0)
        sentences = torch.cat(after_embedding, dim=0)
        labels = torch.tensor(labels)

        return sentences, labels

    def get_data(self):
        #clean txt
        new_review = []
        labels = []
        review = pickle.load(open(os.path.join(self.data_dir,'all_data.pkl'), "rb"))
        for word in review['reviewText'][0:]:    #file_in = review_data['reviewText'][0:]
            tmp = re.sub("[^A-z']+", ' ', word).lower()
            tmp = [word for word in tmp.split() if word not in sw] #tokenization
            tmp = [word for word in tmp if d.check(word)] #filter non-English words
            # new_review = new_review.append({'review': tmp}, ignore_index = True)
            new_review.append(" ".join(tmp))
        for rating in review['Rating']:
            if int(rating) > 2:
                labels.append(1)
            else:
                labels.append(0)
        return new_review, labels

    def master(self):
        sentences, labels = self.get_data()
        sentences, labels = self.dataset_embedding(sentences, labels)
        # print(sentences.shape)
        # print(labels.shape)
        return np.array(sentences.to('cpu')), np.array(labels)


if __name__ == '__main__':
    # bert_sentence_embedding()
    # word_embedding()
    embedding_size = 200
    sentence_length = 40
    model_path = '/root/autodl-tmp/projects/transformers/bert-base-uncased'
    data_dir = '/root/autodl-tmp/project_data_file/product_review_senti_analysis'
    Embed = Embedding(embedding_size, sentence_length, model_path, data_dir)
    Embed.master()
