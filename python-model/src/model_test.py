import io

import polars as pl
import torch

torch.jit.load("model.pt")

# Load ScriptModule from io.BytesIO object
with open("model.pt", "rb") as f:
    buffer = io.BytesIO(f.read())

# Load all tensors to the original device
model = torch.jit.load(buffer)
model.eval().to("cuda")

print(model)
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

X = torch.Tensor(df.drop("Survived").to_numpy()).to("cuda")
y = torch.Tensor(df["Survived"].to_numpy())

preds = []
print(len(X))
for i in range(len(X) // 32 + 1):
    x = X[i * 32 : (i + 1) * 32]
    pred = model(x).cpu().detach().numpy().argmax(axis=1)
    preds += pred.tolist()

df = (
    pl.concat([df, pl.DataFrame({"preds": preds})], how="horizontal")
    .select("Survived", "preds")
    .with_columns(right=pl.col("Survived") == pl.col("preds"))
)

total = df.height
right = df.filter(pl.col("right")).height
print(f"total: {total}")
print(f"right: {right}")
