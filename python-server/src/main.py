from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

import torch

app = FastAPI()
model = torch.jit.load("/app/src/python-model/model.pt")


class TitanInputData(BaseModel):
    pclass: int
    sex: str
    age: int
    sibsp: int
    parch: int
    embarked: str

    def to_tensor(self) -> torch.Tensor:
        resp = [self.pclass / 3, self.age / 100, self.sibsp / 8, self.parch / 6]

        if self.sex == 'male':
            resp += [1, 0]
        else:
            resp += [0, 1]

        if self.embarked == 'C':
            resp += [1, 0, 0, 0]
        elif self.embarked == 'Q':
            resp += [0, 1, 0, 0]
        elif self.emberked == 'S':
            resp += [0, 0, 1, 0]
        else:
            resp += [0, 0, 0, 1]

        return torch.Tensor([resp])

@app.get("/")
async def root():
    return {"message": "Pong"}

@app.post('/json')
async def predict(input_data: TitanInputData):
    input_tensor = input_data.to_tensor()
    result = model(input_tensor).cpu().detach().numpy()[0]

    if result[0] > result[1]:
        message = "Died"
        probabity = float(result[0])
    else:
        message = "Survived"
        probability = float(result[1])

    return {
        "message": message,
        "probability": probability
    }
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)