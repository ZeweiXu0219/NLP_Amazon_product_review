import os
import torch
import wget
import logging
import time
import datetime
import random
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
# from main import get_data

class Bert_try(object):
    def __init__(self, save_path, bert_path, batch_size, lr, eps, epoch):
        self.data_dir = save_path
        self.batch_size = batch_size
        self.lr = lr
        self.eps = eps
        self.epoch = epoch
        self.bert_model_dir = bert_path

        if torch.cuda.is_available():
            device = torch.device('cuda')
            logging.info('There are %d GPU(s) available.' % torch.cuda.device_count())
            logging.info('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
            logging.info('No GPU available, using the CPU instead.!!!!')
            device = torch.device("cpu")

        self.device = device

    def download_dataset(self):
        logging.info('Downloading dataset...')
        # The URL for the dataset zip file.
        url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'
        # Download the file (if we haven't already)
        if not os.path.exists(os.path.join(self.data_dir, 'cola_public_1.1.zip')):
            wget.download(url, os.path.join(self.data_dir, 'cola_public_1.1.zip'))

    def bert_tokenizer(self, sentences, labels):
        logging.info('Loading BERT tokenizer...')
        tokenizer = BertTokenizer.from_pretrained('/root/autodl-tmp/bert_test/huggingface_transformers/bert-base-uncased', do_lower_case=True)
        print('example of Berttokenizer below:')
        print(' Original: ', sentences[0])
        print('Tokenized: ', tokenizer.tokenize(sentences[0]))
        print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))
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
        input_ids = []
        attention_masks = []
        token_type_ids = []
        # For every sentence...
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
                max_length=max_len,  # Pad & truncate all sentences.
                padding='max_length',
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )
            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])
            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])
        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)

        # Print sentence 0, now as a list of IDs.
        # print('Original: ', sentences[0])
        # print('Token IDs:', input_ids[0])
        return input_ids, attention_masks, labels

    def process_dataset(self, df):
        # Load the dataset into a pandas dataframe.
        # df = pd.read_csv(os.path.join(self.data_dir,'cola_public/raw/in_domain_train.tsv'), delimiter='\t', header=None,
        #                  names=['sentence_source', 'label', 'label_notes', 'sentence'])
        print(df.head())
        print(df.columns)
        # Report the number of sentences.
        print('Number of training sentences: {:,}\n'.format(df.shape[0]))
        input_ids, attention_masks, labels = self.bert_tokenizer(df['sentence'].tolist(), df['label'].tolist())
        # Training & Validation Split
        # Combine the training inputs into a TensorDataset.
        dataset = TensorDataset(input_ids, attention_masks, labels)
        # Create a 90-10 train-validation split.
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size

        # Divide the dataset by randomly selecting samples.
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        print('{:>5,} training samples'.format(train_size))
        print('{:>5,} validation samples'.format(val_size))
        return train_dataset, val_dataset

    def show_param(self, model):
        params = list(model.named_parameters())

        print('The BERT model has {:} different named parameters.\n'.format(len(params)))

        print('==== Embedding Layer ====\n')

        for p in params[0:5]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        print('\n==== First Transformer ====\n')

        for p in params[5:21]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        print('\n==== Output Layer ====\n')

        for p in params[-4:]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    # Function to calculate the accuracy of our predictions vs labels
    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def format_time(self, elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))
        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def prepare_model(self, train_dataset, val_dataset):
        # Load BertForSequenceClassification, the pretrained BERT model with a single
        # linear classification layer on top.
        model = BertForSequenceClassification.from_pretrained(
            self.bert_model_dir,
            # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=2,  # The number of output labels--2 for binary classification.
            # You can increase this for multi-class tasks.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )
        # Tell pytorch to run this model on the GPU.
        model.cuda()
        train_dataloader = DataLoader(dataset=train_dataset,
                                    sampler = RandomSampler(train_dataset), # Select batches randomly
                                    batch_size = self.batch_size # Trains with this batch size.
                                    )
        validation_dataloader = DataLoader(dataset=val_dataset,  # The validation samples.
                                        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
                                        batch_size=batch_size  # Evaluate with this batch size.
                                        )
        self.show_param(model)
        # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
        # I believe the 'W' stands for 'Weight Decay fix"
        optimizer = AdamW(model.parameters(),
                          lr=self.lr,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          eps=self.eps  # args.adam_epsilon  - default is 1e-8.
                          )
        # Number of training epochs. The BERT authors recommend between 2 and 4.
        # We chose to run for 4, but we'll see later that this may be over-fitting the training data.
        # Total number of training steps is [number of batches] x [number of epochs].
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train_dataloader) * self.epoch
        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)

        for _, batch in enumerate(train_dataloader):
            print(batch[0])
            print(batch[1])
            print(batch[2])
        return model, train_dataloader, validation_dataloader, optimizer, scheduler

    def Train(self, model, train_dataloader, validation_dataloader, optimizer, scheduler):
        # Set the seed value all over the place to make this reproducible.
        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        # We'll store a number of quantities such as training and validation loss,
        # validation accuracy, and timings.
        training_stats = []

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        # For each epoch...
        for epoch_i in range(0, self.epoch):

            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epoch))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_train_loss = 0

            # Put the model into training mode. Don't be mislead--the call to
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):

                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = self.format_time(time.time() - t0)

                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using the
                # `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because
                # accumulating the gradients is "convenient while training RNNs".
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                model.zero_grad()

                # Perform a forward pass (evaluate the model on this training batch).
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # It returns different numbers of parameters depending on what arguments
                # arge given and what flags are set. For our useage here, it returns
                # the loss (because we provided labels) and the "logits"--the model
                # outputs prior to activation.
                output = model(b_input_ids,
                                     token_type_ids=None,
                                     attention_mask=b_input_mask,
                                     labels=b_labels)
                # print('the loss is ', loss)
                loss = output.loss
                logits = output.logits
                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value
                # from the tensor.
                total_train_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)

            # Measure how long this epoch took.
            training_time = self.format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            model.eval()

            # Tracking variables
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            # Evaluate data for one epoch
            for batch in validation_dataloader:
                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using
                # the `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():
                    # Forward pass, calculate logit predictions.
                    # token_type_ids is the same as the "segment ids", which
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    # The documentation for this `model` function is here:
                    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                    # Get the "logits" output by the model. The "logits" are the output
                    # values prior to applying an activation function like the softmax.
                    val_output = model(b_input_ids,
                                           token_type_ids=None,
                                           attention_mask=b_input_mask,
                                           labels=b_labels)
                loss = val_output.loss
                logits = val_output.logits
                # Accumulate the validation loss.
                total_eval_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
                total_eval_accuracy += self.flat_accuracy(logits, label_ids)

            # Report the final accuracy for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(validation_dataloader)

            # Measure how long the validation run took.
            validation_time = self.format_time(time.time() - t0)

            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(self.format_time(time.time() - total_t0)))


    def bert_sentence_embedding(self):

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

    def master(self):
        if not os.path.exists(os.path.join(self.data_dir, 'cola_public_1.1.zip')):
            self.download_dataset()
        # Load the dataset into a pandas dataframe.
        df = pd.read_csv(os.path.join(self.data_dir,'cola_public/raw/in_domain_train.tsv'), delimiter='\t', header=None,
                         names=['sentence_source', 'label', 'label_notes', 'sentence'])
        train_dataset, val_dataset = self.process_dataset(df)
        model, train_dataloader, validation_dataloader, optimizer, scheduler = self.prepare_model(train_dataset, val_dataset)
        self.Train(model, train_dataloader, validation_dataloader, optimizer, scheduler)
        self.bert_sentence_embedding()

if __name__ == '__main__':
    save_path = '/root/autodl-tmp/bert_test/dataset'
    bert_path = '/root/autodl-tmp/projects/transformers/bert-base-uncased'
    batch_size = 32
    lr = 2e-5
    eps = 1e-8
    epoch = 4
    try_bert = Bert_try(save_path, bert_path, batch_size, lr, eps, epoch)
    try_bert.master()