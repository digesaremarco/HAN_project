import numpy as np
import torch
import han_model


def save_checkpoint(model, optimizer, epoch, loss):
    """ Save the checkpoint of the model """
    fname = "C:\\Users\\diges\\Desktop\\checkpoint.pth"
    info = dict(model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                epoch=epoch,
                loss=loss)
    torch.save(info, fname)
    print(f"Model saved to {fname}")


def test_metrics(model, test_loader, device):
    """ Calculate the accuracy on the test set """
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            output = torch.softmax(model(X), dim=1)
            #output = model(X)
            predicted = torch.argmax(output, dim=1)
            correct += (predicted == Y).sum().item()
            total += Y.size(0)  # Number of samples

    return correct / total  # accuracy = correct / total


def grid_search_lr(embedding_matrix, train, validation, loss_fn, opts, device):
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
                loss = loss_fn(y_pred, y)  # calculate the loss
                loss.backward()
                optimizer.step()

                # Calculate accuracy during training
                _, predicted = torch.max(y_pred, 1)
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

            val_acc = test_metrics(model, validation, device)
            print(f'Validation Accuracy: {val_acc:.6f}')

            if val_acc > best_acc:
                best_acc = val_acc
                best_lr = lr

    print(f"Best learning rate: {best_lr} with validation accuracy: {best_acc:.6f}")
    return best_lr


def train_loop(model, train_loader, test_loader, opts, learning_rate, device):
    """ Train the model """
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            output = torch.softmax(model(X), dim=1)  # get the prediction with softmax
            loss = loss_fn(output, Y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            corrects.append(
                (torch.argmax(output, dim=1) == Y).sum().item())  # calculate the number of correct predictions

        train_loss = np.mean(losses)
        #train_acc = np.mean(corrects)
        total_samples = sum(Y.size(0) for _, Y in train_loader)
        train_acc = sum(corrects) / total_samples
        print(f"Epoch {epoch + 1}, Training Loss: {train_loss:.6f}, Training Accuracy: {train_acc:.6f}")
        test_acc = test_metrics(model, test_loader, device)
        print(f"Test Accuracy: {test_acc:.6f}")

        # Save the model checkpoint
        save_checkpoint(model, optimizer, epoch, train_loss)


