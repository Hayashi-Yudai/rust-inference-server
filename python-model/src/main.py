from loguru import logger
import polars as pl
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class TitanicModel(nn.Module):
    def __init__(self):
        super(TitanicModel, self).__init__()
        self.fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x = self.softmax(x)

        return x


class TitanicDataset(Dataset):
    def __init__(self, df: pl.DataFrame):
        self.df = self._preprocess(df)
        
        self.X = torch.Tensor(self.df.drop("Survived").to_numpy())
        self.y = torch.Tensor(self.df["Survived"].to_numpy())
    
    def _preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.select("Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked")
        numerical = df.select("Pclass", "Age", "SibSp", "Parch", "Survived").with_columns(
            Pclass=pl.col("Pclass") / 3,
            Age=pl.col("Age").fill_null(0.5) / 100,
            SibSp=pl.col("SibSp") / 8,
            Parch=pl.col("Parch") / 6,
        )
        categorical = df.select("Sex", "Embarked").to_dummies()

        df = pl.concat([numerical, categorical], how="horizontal")
        
        return df

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

if __name__ == "__main__":
    df = pl.read_csv("dataset/titanic/train.csv")
    train_df = df.head(int(df.height * 0.8))
    valid_df = df.tail(int(df.height * 0.2))

    train_ds = TitanicDataset(train_df)
    valid_ds = TitanicDataset(valid_df)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TitanicModel().to(device)
    logger.info(model)
    logger.info(f"Training Device: {device}")

    best_acc = 0.0
    non_improved_cnt = 0

    for epoch in range(100):
        if non_improved_cnt > 10:
            logger.info(f"Early Stopping! Best accuracy: {best_acc}")
            break

        model.train()
        for X_batch, y_batch in train_dl:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.long())
            loss.backward()
            optimizer.step()

        model.eval()
        total_loss = 0
        counter = 0

        preds = []
        gts = []
        with torch.no_grad():
            for X_batch, y_batch in valid_dl:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch.long())

                preds += y_pred.argmax(dim=1).tolist()
                gts += y_batch.tolist()

                total_loss += loss.item()
                counter += 1
        
        loss_num = total_loss / counter
        acc = accuracy_score(gts, preds)

        if acc < best_acc:
            non_improved_cnt += 1
        else:
            non_improved_cnt = 0

        best_acc = max(best_acc, acc)
        
        logger.info(f"Epoch: {epoch}, Loss: {loss_num}, Accuracy: {acc}")

    jit_model = torch.jit.script(model)
    jit_model.save("model.pt")
