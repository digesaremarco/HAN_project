import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


class Evaluate:

    def __init__(self, opts):
        self.accuracies = []
        self.losses = []
        self.f1_scores = []
        self.epoches = [i + 1 for i in range(opts.epochs)]

    def test_accuracy(self, model, test_loader, device):
        """ Calculate the accuracy on the test set """
        model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():
            for X, Y in test_loader:
                X, Y = X.to(device), Y.to(device)
                output = torch.softmax(model(X), dim=1)
                predicted = torch.argmax(output, dim=1)
                correct += (predicted == Y).sum().item()
                total += Y.size(0)  # Number of samples

        return correct / total  # accuracy = correct / total

    def test_f1_score(self, model, test_loader, device):
        """ Calculate the F1 score on the test set """
        model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for X, Y in test_loader:
                X, Y = X.to(device), Y.to(device)
                output = torch.softmax(model(X), dim=1)
                predicted = torch.argmax(output, dim=1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(Y.cpu().numpy())

        return f1_score(all_targets, all_predictions, average='weighted')

    def add_accuracy(self, accuracy):
        self.accuracies.append(accuracy)

    def add_loss(self, loss):
        self.losses.append(loss)

    def add_f1_score(self, f1_score):
        self.f1_scores.append(f1_score)

    def plot_accuracy(self):
        plt.plot(self.epoches, self.accuracies, label='Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.legend()
        plt.show()

    def plot_f1_score(self):
        plt.plot(self.epoches, self.f1_scores, label='F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.grid()
        plt.legend()
        plt.show()