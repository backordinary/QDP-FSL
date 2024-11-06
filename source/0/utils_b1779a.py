# https://github.com/diracq/quantum-kitchen-sink/blob/7cce2f6caf887c8cf5daea9526d53c719ca40e36/src/utils.py
from sklearn.linear_model import LogisticRegression
from qiskit import IBMQ
from tqdm import tqdm
import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Quantum Kitchen Sink training script')
    parser.add_argument('--n-episodes', type=int, default=20, 
        help='Number of episodes to run QKS.')
    parser.add_argument('--scale', type=int, default=1, 
        help='Standard deviation of QKS\'s normal distribution.')
    parser.add_argument('--distribution', type=str, default='normal', 
        help="The distribution to use for QKS values.")
    parser.add_argument('--n-trials', type=int, default=1000, 
        help='Number of trials to run on each QuantumCircuit')
    parser.add_argument('--qubits', type=int, default=4, 
        help='Number of qubits.')
    parser.add_argument('--img-dir', type=str, default='fig/', 
        help='Directory to save figures')
    parser.add_argument('--no-plot', action='store_true', 
        help='Whether to plot the performance on the test and train datasets.')
    parser.add_argument('--tiling', action='store_true', 
        help='Whether to run QKS with tiling,')
    parser.add_argument('--no-cuda', action='store_true', 
        help='This flag should be passed when running the training script on a machine that does '
             'not support cuda. [NOT USED YET]')
    parser.add_argument('--epochs', type=int, default=100, 
        help='Number of epochs to train the classification model.')
    parser.add_argument('--debug', action='store_true', 
        help='disables comet.ml logging.')
    args = parser.parse_args()
    return args


def load_IBMQ_account():
    # Write the API token to IBM Q
    my_api_token = 'f84c8a32633b1ce7fda84ebc53c33141d96526678b3eba6ed5709619cb2d7378a0770a72831f01ded'\
        '4dd4a9ce691cfc5042e3c622b03129ce5376c5f000b5dcf'
    IBMQ.save_account(my_api_token)

    # Check the connection for IBM Q
    try:
        IBMQ.load_account(overwrite=True)
    except:
        print("""WARNING: There's no connection with the API for remote backends.
                Have you initialized a file with your personal token?
                For now, there's only access to local simulator backends...""")


def logistic_regression(train_dataset, test_dataset):
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(train_dataset.X, train_dataset.y)

    train_acc = lr.score(train_dataset.X, train_dataset.y)
    test_acc = lr.score(test_dataset.X, test_dataset.y)

    print(
        "\naccuracy\n----- \n training: {}\n test:     {}\n"
          .format(train_acc, test_acc)
         )

    train_preds = lr.predict(train_dataset.X)
    test_preds = lr.predict(test_dataset.X)

    return train_preds, test_preds


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(experiment, model, train_loader):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    with experiment.train():
        for item in tqdm(train_loader):
            inputs, labels = item
            inputs = inputs.type(torch.FloatTensor).to(device)
            labels = labels.type(torch.LongTensor).to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()


def test(experiment, model, test_loader, epoch):
    total = 0
    correct = 0

    model.eval()

    with experiment.validate():
        with torch.no_grad():
            for item in test_loader:
                inputs, labels = item
                inputs = inputs.type(torch.FloatTensor).to(device)
                labels = labels.type(torch.LongTensor).to(device)

                logits = model(inputs)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        experiment.log_metric('Accuracy', correct / total, epoch=epoch)
        print('Accuracy of network on test images: %d %%' % (100 * correct / total))


def save(model, PATH):
    print('Saving model...')
    torch.save(model.state_dict(), PATH)

