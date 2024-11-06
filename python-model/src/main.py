import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class TitanicModel(nn.Module):
    def __init__(self):
        super(TitanicModel, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.leaky_relu1 = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(10, 2)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.leaky_relu1(self.dropout(self.fc1(x)))
        x = self.fc2(x)
        x = self.softmax(x)

        return x


class TitanicDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == "__main__":
    df = pl.read_csv("dataset/titanic/train.csv")
    df = df.select("Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked")
    numerical = df.select("Pclass", "Age", "SibSp", "Parch", "Survived").with_columns(
        Pclass=pl.col("Pclass") / 3,
        Age=pl.col("Age").fill_null(0.5) / 100,
        SibSp=pl.col("SibSp") / 8,
        Parch=pl.col("Parch") / 6,
    )
    categorical = df.select("Sex", "Embarked").to_dummies()

    df = pl.concat([numerical, categorical], how="horizontal")
    print(df.head())

    X = torch.Tensor(df.drop("Survived").to_numpy())
    y = torch.Tensor(df["Survived"].to_numpy())

    train_ds = TitanicDataset(X[: int(len(X) * 0.8)], y[: int(len(y) * 0.8)])
    valid_ds = TitanicDataset(X[int(len(X) * 0.8) :], y[int(len(X) * 0.8) :])

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=128, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TitanicModel().to(device)
    print(model)
    print(device)

    for epoch in range(100):
        model.train()
        for X_batch, y_batch in tqdm(train_dl):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.long())
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in valid_dl:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch.long())

                print(f"Epoch: {epoch}, Loss: {loss}")

    jit_model = torch.jit.script(model)
    jit_model.save("model.pt")
