import os
import pickle
import random
import gensim.models.word2vec as word2vec  # word2vec
import numpy as np
import pandas as pd
import collections
import torch
import enchant
import csv
import nltk
# nltk.download('stopwords')
from nltk import word_tokenize #分词函数
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn #得到单词情感得分
import re
import string #本文用它导入标点符号，如!"#$%&
from torch.utils.data import WeightedRandomSampler
global sw
global d
import torch as t
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.utils.data
import re
import numpy as np
import matplotlib.pyplot as plt
from prod_review_Dataset import MyData
from models.LSTM_model import SLCABG
from models.BERT_Embedding import Embedding
sw = stopwords.words('english')
d = enchant.Dict("en_US")


class Sentence_classification(object):
    def __init__(self, path, data_path):
        self.path = path
        self.data_dir = data_path
        self.model_dir = os.path.join(path, 'model')

    def wordtovec_model(self):
        sent = word2vec.Text8Corpus(os.path.join(self.path, 'text8'))
        tmp = word2vec.Word2Vec(sent, vector_size=200)
        tmp.save(os.path.join(self.path, 'text2020.model'))

    def get_sentiment_dict(self, senti_dict):
        sentiment_dict = {}
        for word in senti_dict.key:
          sentiment_dict[word] = float(senti_dict[senti_dict['key'] == word].average)#建立dict
        return sentiment_dict

    def get_word_vectors(self, senti_dict):                             #将词和词向量对应
        senti_dict = pickle.load(open(os.path.join(self.path, 'single_senti_weight_1.pkl'), "rb"))
        vecs = np.loadtxt(os.path.join(self.path, 'vecs.txt'))
        word_2_vec = {}
        word_list = senti_dict['key']
        for i in range(len(word_list)):
            word_2_vec[word_list[i]] = vecs[i]
        return word_2_vec

    def get_weighted_word_vectors(self, senti_dict):
        wordtovec = self.get_word_vectors(senti_dict)
        sentiment_dict = self.get_sentiment_dict(senti_dict)
        word2Vec = {}
        for i in wordtovec.keys():
            word2Vec[i] = sentiment_dict[i] * wordtovec[i]             #有问题暂时没有解决，无法进行相乘操作，输出的暂时仍然是原来的词向量
        return word2Vec


    def Embedding(self):
        senti_dict = pickle.load(open(os.path.join(self.path, 'single_senti_weight_1.pkl'), "rb"))
        model = word2vec.Word2Vec.load(os.path.join(self.path, 'text2020.model'))
        vecs = []
        for word in senti_dict['key']:
            try:
              vec = model[word]
              vecs.append(vec)
            except:
              vec = np.zeros(200)                         #word2vec里没有的词就将其赋值成[1,1,1......1,1]
              vecs.append(vec)
              pass
        vecs = np.array(vecs)
        # vecs存的是word2vec生成的senti_dict中的所有单词对应的词向量
        np.savetxt(os.path.join(self.path, 'vecs.txt'), vecs)
        # 将利用word2vec生成的向量和senti weight相乘
        word2Vec = self.get_weighted_word_vectors(senti_dict)
        return word2Vec


    def get_data(self):
        #clean txt
        new_review = []
        review = pickle.load(open(os.path.join(self.path,'all_data.pkl'), "rb"))
        for word in review['reviewText'][0:]:    #file_in = review_data['reviewText'][0:]
            tmp = re.sub("[^A-z']+", ' ', word).lower()
            tmp = [word for word in tmp.split() if word not in sw] #tokenization
            tmp = [word for word in tmp if d.check(word)] #filter non-English words
            # new_review = new_review.append({'review': tmp}, ignore_index = True)
            new_review.append(tmp)
        return new_review, review

    def process_data(self, sentence_length, words_size, embed_size):
        senti_dict = pickle.load(open(os.path.join(self.path, 'single_senti_weight_1.pkl'), "rb"))
        sentences, review = self.get_data()
        print(review.head())
        print(review.columns)
        frequency = collections.Counter()
        for sentence in sentences:
            for word in sentence:
                frequency[word] += 1
        word2index = dict()
        for i, x in enumerate(frequency.most_common(words_size)):
            word2index[x[0]] = i + 1
        word_2_vec = self.get_weighted_word_vectors(senti_dict)
        word_vectors = torch.zeros(words_size + 1, embed_size)
        for k, v in word2index.items():
            try:
              word_vectors[v,:] = torch.from_numpy(word_2_vec[k])
            except:
              pass
        rs_sentences = []
        for sentence in sentences:
            sen = []
            for word in sentence:
                if word in word2index.keys():
                    sen.append(word2index[word])
                else:
                    sen.append(0)
            if len(sen) < sentence_length:
                sen.extend([0 for _ in range(sentence_length - len(sen))])
            else:
                sen = sen[:sentence_length]
            rs_sentences.append(sen)
        label = [1 for _ in range(17448)]                   #positive total 9793 reviews
        label.extend([0 for _ in range(5193)])              #negative total 467 reviews
        label = np.array(label)
        return rs_sentences, label, word_vectors

        # SL 20->12改动造成错误

    def model_train(self):
        device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
        SENTENCE_LENGTH = 40
        WORD_SIZE = 15000
        EMBED_SIZE = 200
        # epochs:15->12->10->17-》8
        epochs = 100
        Batch_size = 256
        seeds = 4
        random.seed(seeds)

        acc_matrix = pd.DataFrame(None, columns=['acc'])
        r_matrix = pd.DataFrame(None, columns=['r'])
        f1_matrix = pd.DataFrame(None, columns=['f1'])


        # sentences, label, word_vectors = self.process_data(SENTENCE_LENGTH, WORD_SIZE, EMBED_SIZE)
        Embed = Embedding(EMBED_SIZE, SENTENCE_LENGTH, '/root/autodl-tmp/projects/transformers/bert-base-uncased', self.data_dir)
        sentences, label = Embed.master()
        x_train, x_test, y_train, y_test = train_test_split(sentences, label, test_size=0.2)
        # 20->32->34
        positive_total = sum(y_train)
        negative_total = len(y_train) - positive_total
        weight_list = []
        for l in y_train:
            if int(l) == 1:
                weight_list.append(1/negative_total)
            else:
                weight_list.append(1/positive_total)
        train_sampler = WeightedRandomSampler(weights=weight_list, num_samples=int(positive_total*2), replacement=True)
        positive_total_test = sum(y_test)
        negative_total_test = len(y_test) - positive_total_test
        weight_list_test = []
        for l in y_test:
            if int(l) == 1:
                weight_list_test.append(1 / negative_total_test)
            else:
                weight_list_test.append(1 / positive_total_test)
        test_sampler = WeightedRandomSampler(weights=weight_list_test, num_samples=int(positive_total_test * 2), replacement=True)
        train_data_loader = torch.utils.data.DataLoader(MyData(x_train, y_train), batch_size=Batch_size, sampler=train_sampler)
        test_data_loader = torch.utils.data.DataLoader(MyData(x_test, y_test), batch_size=Batch_size, sampler=test_sampler)

        net = SLCABG(EMBED_SIZE, SENTENCE_LENGTH).to(device)  # 发送到GPU中

        # construct loss and optimizer
        optimizer = t.optim.Adam(net.parameters(), 1e-6)
        criterion = nn.CrossEntropyLoss()
        loss_change = []
        tp = 1
        tn = 1
        fp = 1
        fn = 1
        for epoch in range(epochs):
            for i, (cls, sentences) in enumerate(train_data_loader):
                optimizer.zero_grad()  # 梯度归零
                # sentences = sentences.type(t.FloatTensor).to(device)  # to(device)转移到GPU中
                # sentences = torch.tensor(sentences).to(device)
                sentences = sentences.to(device)
                # cls = cls.type(t.LongTensor).to(device)
                # cls = torch.tensor(cls).to(device)
                cls = cls.to(device)
                out = net(sentences)
                # print(torch.max(ou,0))
                _, predicted = torch.max(out.data, 1)
                predict = predicted.cpu().numpy().tolist()
                pred = cls.cpu().numpy().tolist()
                for f, n in zip(predict, pred):
                    if f == 1 and n == 1:
                        tp += 1
                    elif f == 1 and n == 0:
                        fp += 1
                    elif f == 0 and n == 1:
                        fn += 1
                    else:
                        tn += 1
                p = tp / (tp + fp)
                r = tp / (tp + fn)
                f1 = 2 * r * p / (r + p)
                acc = (tp + tn) / (tp + tn + fp + fn)
                loss = criterion(out, cls).to(device)
                loss.backward()
                optimizer.step()
                if (i + 1) % 10 == 0:
                    series_acc = pd.DataFrame(pd.Series({'acc': acc}))
                    acc_matrix = pd.concat([acc_matrix, series_acc], ignore_index=True)
                    series_r = pd.DataFrame(pd.Series({'r': r}))
                    r_matrix = pd.concat([r_matrix, series_r], ignore_index=True)
                    series_f1 = pd.DataFrame(pd.Series({'f1': f1}))
                    f1_matrix = pd.concat([f1_matrix, series_f1], ignore_index=True)
                    loss_change.append(loss.item())
                    print("epoch:", epoch + 1, "step:", i + 1, "loss:", loss.item())
                    print('acc', acc, 'preci', p, 'recall', r, 'f1', f1)
                    if loss.item() < 0.001:
                        break
        net.eval()
        print('==========================================================================================')
        with torch.no_grad():
            tp = 1
            tn = 1
            fp = 1
            fn = 1
            for cls, sentences in test_data_loader:
                # sentences = sentences.type(t.LongTensor).to(device)
                # cls = cls.type(t.LongTensor).to(device)
                sentences = sentences.to(device)
                cls = cls.to(device)
                out = net(sentences)
                _, predicted = torch.max(out.data, 1)
                predict = predicted.cpu().numpy().tolist()
                pred = cls.cpu().numpy().tolist()
                for f, n in zip(predict, pred):
                    if f == 1 and n == 1:
                        tp += 1
                    elif f == 1 and n == 0:
                        fp += 1
                    elif f == 0 and n == 1:
                        fn += 1
                    else:
                        tn += 1
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f1 = 2 * r * p / (r + p)
            acc = (tp + tn) / (tp + tn + fp + fn)
            print('acc', acc, 'p', p, 'r', r, 'f1', f1)
        if acc > 0.95:
            model_name = "acc:" + str(round(acc, 4) * 100) + "_prc:" + str(round(p,4) * 100) + "_rc:" + str(round(r, 4) * 100) + "_f1:" + str(round(f1, 4) * 100) + ".pth"
            torch.save(net, os.path.join(self.model_dir, model_name))
        # 画出loss随时间步变化的趋势
        # 创建折线图
        x_label = [i for i in range(1, len(loss_change) + 1)]
        plt.plot(x_label, loss_change, label='loss')
        # 添加标题和标签
        plt.title('loss_change')
        plt.xlabel('step')
        plt.ylabel('loss_value')
        # 添加图例
        plt.legend()
        # 显示图形
        plt.show()

    def master(self, SENTENCE_LENGTH, WORD_SIZE, EMBED_SIZE):
        path = '/root/autodl-tmp/product_review_doc/prepare'

        # sentences, label, word_vectors = self.process_data(SENTENCE_LENGTH, WORD_SIZE, EMBED_SIZE)
        self.model_train()










# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = '/root/autodl-tmp/project_data_file/product_review_senti_analysis'
    data_path = '/root/autodl-tmp/project_data_file/product_review_senti_analysis/dataset'
    SENTENCE_LENGTH = 40
    WORD_SIZE = 15000
    EMBED_SIZE = 200
    product_review = Sentence_classification(path, data_path)
    product_review.master(SENTENCE_LENGTH, WORD_SIZE, EMBED_SIZE)