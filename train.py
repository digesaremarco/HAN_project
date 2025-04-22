import argparse

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from types import SimpleNamespace

import numpy as np
import torch
import yaml

import evaluate
import han_model
from datasets import load_dataset
from sklearn.datasets import fetch_20newsgroups

import data_loaders


def save_checkpoint(model, optimizer, epoch, loss):
    """ Save the checkpoint of the model """
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    fname = os.path.join('checkpoints', f"checkpoint_epoch_{epoch}.pt")
    info = dict(model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                epoch=epoch,
                loss=loss)
    torch.save(info, fname)
    print(f"Model saved to {fname}")


def grid_search_lr(embedding_matrix, train, validation, loss_fn, opts, device, evaluate_obj):
    """ Grid search for the best learning rate based on accuracy """
    lrs = [1e-2, 1e-3, 1e-4]

    best_lr = None
    best_acc = 0  # Initialize the best accuracy

    for lr in lrs:
        print(f"Testing learning rate: {lr}")

        # Reinitialize the model and optimizer for each learning rate
        model = han_model.HAN(opts, embedding_matrix).to(device)  # Reinitialize the model if necessary
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=opts.optimizer_momentum)

        for epoch in range(3):  # Number of epochs for grid search
            model.train()  # set the model to training mode
            correct_train = 0
            total_train = 0
            for x, y in train:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()  # zero the gradients
                y_pred = model(x)
                y_pred = y_pred.squeeze(1)
                loss = loss_fn(y_pred, y)  # calculate the loss
                loss.backward()
                optimizer.step()

                # Calculate accuracy during training
                _, predicted = torch.max(y_pred, 1)

                correct_train += (predicted == y).sum().item()
                total_train += y.size(0)

            train_acc = correct_train / total_train  # Average accuracy on training
            print(f'Epoch {epoch + 1}, Training Accuracy: {train_acc:.6f}')

            val_acc = evaluate_obj.test_accuracy(model, validation, device)
            print(f'Validation Accuracy: {val_acc:.6f}')

            if val_acc > best_acc:
                best_acc = val_acc
                best_lr = lr

            torch.cuda.empty_cache()

    print(f"Best learning rate: {best_lr} with validation accuracy: {best_acc:.6f}")
    return best_lr


def train_loop(model, train_loader, validation_loader, opts, learning_rate, device, evaluate_obj):
    """ Train the model """
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=opts.optimizer_momentum)
    loss_fn = torch.nn.NLLLoss()  # Negative Log Likelihood Loss

    for epoch in range(opts.epochs):
        model.train()  # set the model to training mode
        losses = []
        corrects = []

        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output.squeeze(1), Y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            corrects.append(
                (torch.argmax(output, dim=1) == Y).sum().item())  # calculate the number of correct predictions

        train_loss = np.mean(losses)
        evaluate_obj.add_loss(train_loss)
        total_samples = sum(Y.size(0) for _, Y in train_loader)
        train_acc = sum(corrects) / total_samples
        print(f"Epoch {epoch + 1}, Training Loss: {train_loss:.6f}, Training Accuracy: {train_acc:.6f}")

        val_acc = evaluate_obj.test_accuracy(model, validation_loader, device)
        evaluate_obj.add_accuracy(val_acc)
        print(f"Validation Accuracy: {val_acc:.6f}")
        val_f1 = evaluate_obj.test_f1_score(model, validation_loader, device)
        evaluate_obj.add_f1_score(val_f1)
        print(f"Validation F1 Score: {val_f1:.6f}")

        torch.cuda.empty_cache()

        # Save the model checkpoint
        if epoch % 5 == 0:
           save_checkpoint(model, optimizer, epoch, train_loss)




def loadSST():
    ''' Load Stanford Sentiment Treebank (SST) dataset '''
    sst_dataset = load_dataset("glue", "sst2")

    train_set = data_loaders.HANTextDataset(sst_dataset['train']['sentence'], sst_dataset['train']['label'], opts)
    embedding_matrix = train_set.embedding_matrix
    vocab = train_set.vocab
    validation_set = data_loaders.HANTextDataset(sst_dataset['validation']['sentence'],
                                                 sst_dataset['validation']['label'], opts,
                                                 vocab=vocab, embedding_matrix=embedding_matrix)
    validation_set, test_set = torch.utils.data.random_split(validation_set, [len(validation_set) - 400, 400])
    return train_set, validation_set, test_set, embedding_matrix


def loadIMDb():
    ''' load IMDb dataset '''
    imdb_dataset = load_dataset("imdb")

    train_set = data_loaders.HANTextDataset(imdb_dataset['train']['text'], imdb_dataset['train']['label'], opts)
    embedding_matrix = train_set.embedding_matrix
    vocab = train_set.vocab
    test_set = data_loaders.HANTextDataset(imdb_dataset['test']['text'], imdb_dataset['test']['label'], opts,
                                           vocab=vocab, embedding_matrix=embedding_matrix)
    test_set, validation_set = torch.utils.data.random_split(test_set, [len(test_set) - 5000, 5000])
    return train_set, validation_set, test_set, embedding_matrix


def load20Newsgroups():
    ''' load 20 Newsgroups dataset '''
    categories = ['rec.sport.baseball', 'rec.sport.hockey']  # 0 is baseball and 1 is hockey
    train_data = fetch_20newsgroups(subset='train', categories=categories)
    test_data = fetch_20newsgroups(subset='test', categories=categories)
    train_set = data_loaders.HANTextDataset(train_data.data, train_data.target, opts)
    embedding_matrix = train_set.embedding_matrix
    vocab = train_set.vocab
    test_set = data_loaders.HANTextDataset(test_data.data, test_data.target, opts,
                                           vocab=vocab, embedding_matrix=embedding_matrix)
    test_set, validation_set = torch.utils.data.random_split(test_set, [len(test_set) - 500, 500])
    return train_set, validation_set, test_set, embedding_matrix


def loadAgNews():
    ''' load AG News dataset, keeping only label 1 and 3 '''
    train_data = load_dataset("sh0416/ag_news", split='train')
    train_data_filtered = train_data.filter(lambda x: x['label'] in [1, 3])
    train_data_filtered = train_data_filtered.map(lambda x: {'label': 0 if x['label'] == 1 else 1,
                                                             'text': x['title'] + ' ' + x['description']})
    test_data = load_dataset("sh0416/ag_news", split='test')
    test_data_filtered = test_data.filter(lambda x: x['label'] in [1, 3])
    test_data_filtered = test_data_filtered.map(lambda x: {'label': 0 if x['label'] == 1 else 1,
                                                             'text': x['title'] + ' ' + x['description']})
    train_set = data_loaders.HANTextDataset(train_data_filtered['text'], train_data_filtered['label'], opts)
    embedding_matrix = train_set.embedding_matrix
    vocab = train_set.vocab
    test_set = data_loaders.HANTextDataset(test_data_filtered['text'], test_data_filtered['label'], opts,
                                           vocab=vocab, embedding_matrix=embedding_matrix)
    test_set, validation_set = torch.utils.data.random_split(test_set, [len(test_set) - 1000, 1000])
    return train_set, validation_set, test_set, embedding_matrix



def main(opts):

    if opts.dataset == 'sst2':
        train_set, validation_set, test_set, embedding_matrix = loadSST()
    elif opts.dataset == 'imdb':
        train_set, validation_set, test_set, embedding_matrix = loadIMDb()
    elif opts.dataset == '20newsgroups':
        train_set, validation_set, test_set, embedding_matrix = load20Newsgroups()
    elif opts.dataset == 'agnews':
        train_set, validation_set, test_set, embedding_matrix = loadAgNews()

    dataloader = data_loaders.MakeDataLoader(opts, train_set, validation_set, test_set)
    train_loader, validation_loader, test_loader = dataloader.train_loader, dataloader.validation_loader, dataloader.test_loader

    model = han_model.HAN(opts, embedding_matrix)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    evaluate_obj = evaluate.Evaluate(opts)

    torch.cuda.empty_cache()

    # Grid search for the best learning rate
    best_lr = grid_search_lr(embedding_matrix, train_loader, validation_loader, torch.nn.NLLLoss(), opts, device, evaluate_obj)

    # Train the model
    print(f"Training the model")
    train_loop(model, train_loader, validation_loader, opts, best_lr, device, evaluate_obj)

    evaluate_obj.plot_accuracy()
    evaluate_obj.plot_f1_score()

    # Evaluate the model on the test set
    print(f"Evaluating the model on the test set")
    test_acc = evaluate_obj.test_accuracy(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.6f}")
    test_f1 = evaluate_obj.test_f1_score(model, test_loader, device)
    print(f"Test F1 Score: {test_f1:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help='YAML Configuration file')
    opts = yaml.load(open(parser.parse_args().config), Loader=yaml.Loader)
    opts = SimpleNamespace(**opts)


    main(opts)