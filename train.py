import argparse
from types import SimpleNamespace

import numpy as np
import torch
import yaml

import Evaluate
import han_model
from datasets import load_dataset

import data_loaders


def save_checkpoint(model, optimizer, epoch, loss):
    """ Save the checkpoint of the model """
    fname = "C:\\Users\\diges\\Desktop\\checkpoint.pth"
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

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                #y_pred = torch.softmax(model(x), dim=1)  # get the prediction
                y_pred = model(x)
                y_pred = y_pred.squeeze(1)
                loss = loss_fn(y_pred, y)  # calculate the loss
                loss.backward()
                optimizer.step()

                # Calculate accuracy during training
                _, predicted = torch.max(y_pred, 1)
                predicted1 = y_pred.argmax(dim=1)

                '''print(f"=== DEBUG: y_pred ===\nShape: {y_pred.shape}\n{y_pred}\n")
                print(f"=== DEBUG: y ===\nShape: {y.shape}\n{y}\n")
                print(f"=== DEBUG: predicted ===\nShape: {predicted.shape}\n{predicted}\n")
                print(f"=== DEBUG: predicted1 ===\nShape: {predicted1.shape}\n{predicted1}\n")'''

                correct_train += (predicted == y).sum().item()
                total_train += y.size(0)

            train_acc = correct_train / total_train  # Average accuracy on training
            print(f'Epoch {epoch + 1}, Training Accuracy: {train_acc:.6f}')

            """# Evaluate the model on the validation set
            model.eval()  # set the model to evaluation mode
            correct_val = 0
            total_val = 0
            with torch.no_grad():  # disable gradient calculation for validation to save memory
                for x, y in validation:
                    x, y = x.to(device), y.to(device)
                    y_pred = torch.softmax(model(x), dim=1)

                    # Calculate accuracy on validation
                    _, predicted = torch.max(y_pred, 1)
                    correct_val += (predicted == y).sum().item()
                    total_val += y.size(0)

            val_acc = correct_val / total_val  # Average accuracy on validation"""

            val_acc = evaluate_obj.test_accuracy(model, validation, device)
            print(f'Validation Accuracy: {val_acc:.6f}')

            if val_acc > best_acc:
                best_acc = val_acc
                best_lr = lr

    print(f"Best learning rate: {best_lr} with validation accuracy: {best_acc:.6f}")
    return best_lr


def train_loop(model, train_loader, test_loader, opts, learning_rate, device, evaluate_obj):
    """ Train the model """
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=opts.optimizer_momentum)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(opts.epochs):
        model.train()  # set the model to training mode
        losses = []
        corrects = []

        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            #output = torch.softmax(model(X), dim=1)  # get the prediction with softmax
            output = model(X)
            loss = loss_fn(output.squeeze(1), Y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            corrects.append(
                (torch.argmax(output, dim=1) == Y).sum().item())  # calculate the number of correct predictions

        train_loss = np.mean(losses)
        evaluate_obj.add_loss(train_loss)
        #train_acc = np.mean(corrects)
        total_samples = sum(Y.size(0) for _, Y in train_loader)
        train_acc = sum(corrects) / total_samples
        print(f"Epoch {epoch + 1}, Training Loss: {train_loss:.6f}, Training Accuracy: {train_acc:.6f}")

        test_acc = evaluate_obj.test_accuracy(model, test_loader, device)
        evaluate_obj.add_accuracy(test_acc)
        print(f"Test Accuracy: {test_acc:.6f}")
        test_f1 = evaluate_obj.test_f1_score(model, test_loader, device)
        evaluate_obj.add_f1_score(test_f1)
        print(f"Test F1 Score: {test_f1:.6f}")

        # Save the model checkpoint
        if epoch % 5 == 0:
            save_checkpoint(model, optimizer, epoch, train_loss)


def main(opts):
    # load Stanford Sentiment Treebank (SST) dataset
    sst_dataset = load_dataset("glue", "sst2")

    '''sst_dataset_train = sst_dataset['train']['sentence']
    sst_dataset_test = sst_dataset['test']['sentence'] # labels are not included
    sst_dataset_validation = sst_dataset['validation']['sentence']
    #sst_dataset2 = sst_dataset_train + sst_dataset_test + sst_dataset_validatio

    #all labels concatenated
    #sst_dataset_labels = sst_dataset['train']['label'] + sst_dataset['test']['label'] + sst_dataset['validation']['label']'''


    '''dataset = data_loaders.HANTextDataset(sst_dataset['train']['sentence'], sst_dataset['train']['label'], opts)
    dataloader = data_loaders.MakeDataLoader(opts, dataset)
    train_loader, test_loader = dataloader.train_loader, dataloader.test_loader'''

    train_set = data_loaders.HANTextDataset(sst_dataset['train']['sentence'], sst_dataset['train']['label'], opts)
    validation_set = data_loaders.HANTextDataset(sst_dataset['validation']['sentence'], sst_dataset['validation']['label'], opts,
                                    vocab=train_set.vocab, embedding_matrix=train_set.embedding_matrix)
    #test_set = data_loaders.HANTextDataset(sst_dataset['test']['sentence'], sst_dataset['test']['label'], opts,
                              #vocab=train_set.vocab, embedding_matrix=train_set.embedding_matrix)


    dataloader = data_loaders.MakeDataLoader(opts, train_set, validation_set)
    train_loader, validation_loader, test_loader = dataloader.train_loader, dataloader.validation_loader, dataloader.test_loader

    # Load the embedding matrix
    #embedding_matrix = dataset.embedding_matrix
    embedding_matrix = train_set.embedding_matrix
    model = han_model.HAN(opts, embedding_matrix)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluate_obj = Evaluate.evaluate(opts)

    # Grid search for the best learning rate
    best_lr = grid_search_lr(embedding_matrix, train_loader, validation_loader, torch.nn.CrossEntropyLoss(), opts, device, evaluate_obj)
    

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
    print(f"Validation F1 Score: {test_f1:.6f}")

    # Save the final model
    torch.save(model.state_dict(), "C:\\Users\\diges\\Desktop\\HAN\\final_model_sst2.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help='YAML Configuration file')
    opts = yaml.load(open(parser.parse_args().config), Loader=yaml.Loader)
    opts = SimpleNamespace(**opts)
    opts.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    '''opts = SimpleNamespace(
        # GRU model parameters
        gru_input_dim=200,
        gru_hidden_dim=50,
        gru_bidirectional=True,
        gru_output_dim=100,
        gru_num_layers=1,

        # Word embedding parameters
        word_embedding_dim=200,
        word_embedding_min_frequency=5,

        # Batch parameters
        batch_size=64,

        # Training parameters
        optimizer_type="SGD",
        optimizer_momentum=0.9,
        learning_rate_search=True,
        epochs=10,

        # Training settings
        train_size=0.8,
        test_size=0.1,
        validation_size=0.1,

        # Tokenization parameters
        tokenization_tool="StanfordCoreNLP",
        tokenization_annotators="tokenize,ssplit",

        # Context vector initialization
        context_vectors_random_init=True,
        context_vectors_dim=100,

        # Dataset parameters
        num_classes=2,

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )'''
    main(opts)

