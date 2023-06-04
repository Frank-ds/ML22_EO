"""Own trial of making a hypertuner"""
import torch
from torch import nn
from torch.optim import Adam
from src.datasets import get_arabic
from src.settings import presets
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from src.models import rnn_models, metrics, train_model


# Step 2: Define the evaluation metric
def evaluate(model, dataloader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            _, predicted = torch.max(output, dim=1)
            y_true.extend(y.tolist())
            y_pred.extend(predicted.tolist())
    return accuracy_score(y_true, y_pred)


# Step 3: Set up a hyperparameter search space
search_space = {"learning_rate": [0.001, 0.01, 0.1], "num_layers": list(range(1, 11))}


# Step 4: Create a training loop
def train_model(hyperparameters):
    model = rnn_models.GRUmodel(
        input_size, hidden_size, hyperparameters["num_layers"], output_size
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=hyperparameters["learning_rate"])

    for epoch in range(num_epochs):
        model.train()
        for x, y in trainstreamer.stream():
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

    # Evaluate the model on the validation set using the evaluation metric
    val_accuracy = evaluate(model, teststreamer)

    return val_accuracy


# Step 5: Perform hyperparameter search
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 13
hidden_size = 128
output_size = 20
num_epochs = 10

trainstreamer, teststreamer = get_arabic(presets)

random_search = RandomizedSearchCV(
    train_model, search_space, n_iter=10, scoring="accuracy"
)
random_search.fit(trainstreamer, teststreamer)

# Step 6: Select the best hyperparameters
best_hyperparameters = random_search.best_params_
