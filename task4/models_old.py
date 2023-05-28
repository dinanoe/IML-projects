import torch.nn as nn

models = {
    "demo": nn.Sequential(
        nn.Linear(1000, 1000),
        nn.Hardswish(),
        nn.Linear(1000, 1000),
        nn.Hardswish(),
        nn.Linear(1000, 1000),
        nn.Hardswish(),
        nn.Linear(1000, 1)
    ),
    "model1": nn.Sequential(
        nn.Linear(1000, 500),
        nn.Hardswish(),
        nn.Linear(500, 500),
        nn.Hardswish(),
        nn.Linear(500, 500),
        nn.Hardswish(),
        nn.Linear(500, 1)
    ),
    "model2": nn.Sequential(
        nn.Linear(1000, 2000),
        nn.Hardswish(),
        nn.Linear(2000, 2000),
        nn.Hardswish(),
        nn.Linear(2000, 2000),
        nn.Hardswish(),
        nn.Linear(2000, 1)
    ),
    "model3": nn.Sequential(
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 500),
        nn.ReLU(),
        nn.Linear(500, 100),
        nn.ReLU(),
        nn.Linear(100, 1)
    ),
    "model4": nn.Sequential(
        nn.Linear(1000, 800),
        nn.Hardswish(),
        nn.Linear(800, 600),
        nn.Hardswish(),
        nn.Linear(600, 400),
        nn.Hardswish(),
        nn.Linear(400, 1)
    ),
    "model5": nn.Sequential(
        nn.Linear(1000, 200),
        nn.Hardswish(),
        nn.Linear(200, 200),
        nn.Hardswish(),
        nn.Linear(200, 200),
        nn.Hardswish(),
        nn.Linear(200, 1)
    ),
    "model6": nn.Sequential(
        nn.Linear(1000, 1000),
        nn.Hardswish(),
        nn.Linear(1000, 500),
        nn.Hardswish(),
        nn.Linear(500, 250),
        nn.Hardswish(),
        nn.Linear(250, 1)
    ),
    "model7": nn.Sequential(
        nn.Linear(1000, 800),
        nn.ReLU(),
        nn.Linear(800, 600),
        nn.ReLU(),
        nn.Linear(600, 400),
        nn.ReLU(),
        nn.Linear(400, 1)
    ),
    "model8": nn.Sequential(
        nn.Linear(1000, 300),
        nn.Hardswish(),
        nn.Linear(300, 300),
        nn.Hardswish(),
        nn.Linear(300, 300),
        nn.Hardswish(),
        nn.Linear(300, 1)
    ),
    "model9": nn.Sequential(
        nn.Linear(1000, 1000),
        nn.Hardswish(),
        nn.Linear(1000, 800),
        nn.Hardswish(),
        nn.Linear(800, 600),
        nn.Hardswish(),
        nn.Linear(600, 1)
    ),
    "model10": nn.Sequential(
        nn.Linear(1000, 400),
        nn.Hardswish(),
        nn.Linear(400, 400),
        nn.Hardswish(),
        nn.Linear(400, 400),
        nn.Hardswish(),
        nn.Linear(400, 1)
    ),
    "model11": nn.Sequential(
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 800),
        nn.ReLU(),
        nn.Linear(800, 600),
        nn.ReLU(),
        nn.Linear(600, 1)
    ),
    "model12": nn.Sequential(
        nn.Linear(1000, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 1)
    ),
    "model13": nn.Sequential(
        nn.Linear(1000, 500),
        nn.Hardswish(),
        nn.Linear(500, 300),
        nn.Hardswish(),
        nn.Linear(300, 100),
        nn.Hardswish(),
        nn.Linear(100, 1)
    ),
    "model14": nn.Sequential(
        nn.Linear(1000, 2000),
        nn.ReLU(),
        nn.Linear(2000, 2000),
        nn.ReLU(),
        nn.Linear(2000, 2000),
        nn.ReLU(),
        nn.Linear(2000, 1)
    ),
    "model15": nn.Sequential(
        nn.Linear(1000, 800),
        nn.Hardswish(),
        nn.Linear(800, 500),
        nn.Hardswish(),
        nn.Linear(500, 200),
        nn.Hardswish(),
        nn.Linear(200, 1)
    ),
    "model16": nn.Sequential(
        nn.Linear(1000, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 1)
    ),
    "model17": nn.Sequential(
        nn.Linear(1000, 600),
        nn.Hardswish(),
        nn.Linear(600, 600),
        nn.Hardswish(),
        nn.Linear(600, 600),
        nn.Hardswish(),
        nn.Linear(600, 1)
    ),
    "model18": nn.Sequential(
        nn.Linear(1000, 300),
        nn.ReLU(),
        nn.Linear(300, 300),
        nn.ReLU(),
        nn.Linear(300, 300),
        nn.ReLU(),
        nn.Linear(300, 1)
    ),
    "model19": nn.Sequential(
        nn.Linear(1000, 1000),
        nn.Hardswish(),
        nn.Linear(1000, 800),
        nn.Hardswish(),
        nn.Linear(800, 400),
        nn.Hardswish(),
        nn.Linear(400, 1)
    ),
    "model20": nn.Sequential(
        nn.Linear(1000, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 100),
        nn.ReLU(),
        nn.Linear(100, 1)
    )
}