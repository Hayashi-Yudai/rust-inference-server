import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


if __name__ == "__main__":
    model = MyModel()
    print(model)
    print(model(torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))

    jit_model = torch.jit.script(model)
    jit_model.save("model.pt")
