import torch.nn as nn

models = {
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
        nn.Linear(1000, 700),
        nn.Hardswish(),
        nn.Linear(700, 500),
        nn.Hardswish(),
        nn.Linear(500, 300),
        nn.Hardswish(),
        nn.Linear(300, 1)
    ),
    "model6": nn.Sequential(
        nn.Linear(1000, 600),
        nn.Hardswish(),
        nn.Linear(600, 400),
        nn.Hardswish(),
        nn.Linear(400, 200),
        nn.Hardswish(),
        nn.Linear(200, 1)
    ),
    "model7": nn.Sequential(
        nn.Linear(1000, 500),
        nn.Hardswish(),
        nn.Linear(500, 300),
        nn.Hardswish(),
        nn.Linear(300, 200),
        nn.Hardswish(),
        nn.Linear(200, 1)
    ),
    "model8": nn.Sequential(
        nn.Linear(1000, 900),
        nn.Hardswish(),
        nn.Linear(900, 700),
        nn.Hardswish(),
        nn.Linear(700, 500),
        nn.Hardswish(),
        nn.Linear(500, 1)
    ),
    "model9": nn.Sequential(
        nn.Linear(1000, 800),
        nn.Hardswish(),
        nn.Linear(800, 500),
        nn.Hardswish(),
        nn.Linear(500, 300),
        nn.Hardswish(),
        nn.Linear(300, 1)
    ),
    "model10": nn.Sequential(
        nn.Linear(1000, 700),
        nn.Hardswish(),
        nn.Linear(700, 400),
        nn.Hardswish(),
        nn.Linear(400, 200),
        nn.Hardswish(),
        nn.Linear(200, 1)
    ),
    "model11": nn.Sequential(
        nn.Linear(1000, 600),
        nn.Hardswish(),
        nn.Linear(600, 300),
        nn.Hardswish(),
        nn.Linear(300, 200),
        nn.Hardswish(),
        nn.Linear(200, 1)
    ),
    "model12": nn.Sequential(
        nn.Linear(1000, 900),
        nn.Hardswish(),
        nn.Linear(900, 600),
        nn.Hardswish(),
        nn.Linear(600, 400),
        nn.Hardswish(),
        nn.Linear(400, 1)
    ),
    "model13": nn.Sequential(
        nn.Linear(1000, 800),
        nn.Hardswish(),
        nn.Linear(800, 500),
        nn.Hardswish(),
        nn.Linear(500, 300),
        nn.Hardswish(),
        nn.Linear(300, 1)
    ),
    "model14": nn.Sequential(
        nn.Linear(1000, 700),
        nn.Hardswish(),
        nn.Linear(700, 400),
        nn.Hardswish(),
        nn.Linear(400, 200),
        nn.Hardswish(),
        nn.Linear(200, 1)
    ),
    "model15": nn.Sequential(
        nn.Linear(1000, 600),
        nn.Hardswish(),
        nn.Linear(600, 300),
        nn.Hardswish(),
        nn.Linear(300, 200),
        nn.Hardswish(),
        nn.Linear(200, 1)
    ),
    "model16": nn.Sequential(
        nn.Linear(1000, 900),
        nn.Hardswish(),
        nn.Linear(900, 700),
        nn.Hardswish(),
        nn.Linear(700, 500),
        nn.Hardswish(),
        nn.Linear(500, 1)
    ),
    "model17": nn.Sequential(
        nn.Linear(1000, 800),
        nn.Hardswish(),
        nn.Linear(800, 500),
        nn.Hardswish(),
        nn.Linear(500, 300),
        nn.Hardswish(),
        nn.Linear(300, 1)
    ),
    "model18": nn.Sequential(
        nn.Linear(1000, 700),
        nn.Hardswish(),
        nn.Linear(700, 400),
        nn.Hardswish(),
        nn.Linear(400, 200),
        nn.Hardswish(),
        nn.Linear(200, 1)
    ),
    "model19": nn.Sequential(
        nn.Linear(1000, 600),
        nn.Hardswish(),
        nn.Linear(600, 300),
        nn.Hardswish(),
        nn.Linear(300, 200),
        nn.Hardswish(),
        nn.Linear(200, 1)
    ),
    "model20": nn.Sequential(
        nn.Linear(1000, 900),
        nn.Hardswish(),
        nn.Linear(900, 600),
        nn.Hardswish(),
        nn.Linear(600, 400),
        nn.Hardswish(),
        nn.Linear(400, 1)
    ),
    "model21": nn.Sequential(
        nn.Linear(1000, 800),
        nn.Hardswish(),
        nn.Linear(800, 500),
        nn.Hardswish(),
        nn.Linear(500, 300),
        nn.Hardswish(),
        nn.Linear(300, 1)
    ),
    "model22": nn.Sequential(
        nn.Linear(1000, 700),
        nn.Hardswish(),
        nn.Linear(700, 400),
        nn.Hardswish(),
        nn.Linear(400, 200),
        nn.Hardswish(),
        nn.Linear(200, 1)
    ),
    "model23": nn.Sequential(
        nn.Linear(1000, 600),
        nn.Hardswish(),
        nn.Linear(600, 300),
        nn.Hardswish(),
        nn.Linear(300, 200),
        nn.Hardswish(),
        nn.Linear(200, 1)
    ),
    "model24": nn.Sequential(
        nn.Linear(1000, 900),
        nn.Hardswish(),
        nn.Linear(900, 700),
        nn.Hardswish(),
        nn.Linear(700, 500),
        nn.Hardswish(),
        nn.Linear(500, 1)
    ),
    "model25": nn.Sequential(
        nn.Linear(1000, 800),
        nn.Hardswish(),
        nn.Linear(800, 500),
        nn.Hardswish(),
        nn.Linear(500, 300),
        nn.Hardswish(),
        nn.Linear(300, 1)
    ),
    "model26": nn.Sequential(
        nn.Linear(1000, 700),
        nn.Hardswish(),
        nn.Linear(700, 400),
        nn.Hardswish(),
        nn.Linear(400, 200),
        nn.Hardswish(),
        nn.Linear(200, 1)
    ),
    "model27": nn.Sequential(
        nn.Linear(1000, 600),
        nn.Hardswish(),
        nn.Linear(600, 300),
        nn.Hardswish(),
        nn.Linear(300, 200),
        nn.Hardswish(),
        nn.Linear(200, 1)
    ),
    "model28": nn.Sequential(
        nn.Linear(1000, 900),
        nn.Hardswish(),
        nn.Linear(900, 700),
        nn.Hardswish(),
        nn.Linear(700, 500),
        nn.Hardswish(),
        nn.Linear(500, 1)
    ),
    "model29": nn.Sequential(
        nn.Linear(1000, 800),
        nn.Hardswish(),
        nn.Linear(800, 500),
        nn.Hardswish(),
        nn.Linear(500, 300),
        nn.Hardswish(),
        nn.Linear(300, 1)
    ),
    "model30": nn.Sequential(
        nn.Linear(1000, 700),
        nn.Hardswish(),
        nn.Linear(700, 400),
        nn.Hardswish(),
        nn.Linear(400, 200),
        nn.Hardswish(),
        nn.Linear(200, 1)
    ),
    "model31": nn.Sequential(
        nn.Linear(1000, 750),
        nn.Hardswish(),
        nn.Linear(750, 550),
        nn.Hardswish(),
        nn.Linear(550, 450),
        nn.Hardswish(),
        nn.Linear(450, 1)
    ),
    "model32": nn.Sequential(
        nn.Linear(1000, 700),
        nn.Hardswish(),
        nn.Linear(700, 550),
        nn.Hardswish(),
        nn.Linear(550, 500),
        nn.Hardswish(),
        nn.Linear(500, 1)
    ),
    "model33": nn.Sequential(
        nn.Linear(1000, 650),
        nn.Hardswish(),
        nn.Linear(650, 500),
        nn.Hardswish(),
        nn.Linear(500, 450),
        nn.Hardswish(),
        nn.Linear(450, 1)
    ),
    "model34": nn.Sequential(
        nn.Linear(1000, 600),
        nn.Hardswish(),
        nn.Linear(600, 550),
        nn.Hardswish(),
        nn.Linear(550, 500),
        nn.Hardswish(),
        nn.Linear(500, 1)
    ),
    "model35": nn.Sequential(
        nn.Linear(1000, 550),
        nn.Hardswish(),
        nn.Linear(550, 500),
        nn.Hardswish(),
        nn.Linear(500, 450),
        nn.Hardswish(),
        nn.Linear(450, 1)
    ),
    "model36": nn.Sequential(
        nn.Linear(1000, 500),
        nn.Hardswish(),
        nn.Linear(500, 550),
        nn.Hardswish(),
        nn.Linear(550, 600),
        nn.Hardswish(),
        nn.Linear(600, 1)
    ),
    "model37": nn.Sequential(
        nn.Linear(1000, 450),
        nn.Hardswish(),
        nn.Linear(450, 500),
        nn.Hardswish(),
        nn.Linear(500, 550),
        nn.Hardswish(),
        nn.Linear(550, 1)
    ),
    "model38": nn.Sequential(
        nn.Linear(1000, 400),
        nn.Hardswish(),
        nn.Linear(400, 450),
        nn.Hardswish(),
        nn.Linear(450, 500),
        nn.Hardswish(),
        nn.Linear(500, 1)
    ),
    "model39": nn.Sequential(
        nn.Linear(1000, 350),
        nn.Hardswish(),
        nn.Linear(350, 400),
        nn.Hardswish(),
        nn.Linear(400, 450),
        nn.Hardswish(),
        nn.Linear(450, 1)
    ),
    "model40": nn.Sequential(
        nn.Linear(1000, 300),
        nn.Hardswish(),
        nn.Linear(300, 350),
        nn.Hardswish(),
        nn.Linear(350, 400),
        nn.Hardswish(),
        nn.Linear(400, 1)
    ),
    "model41": nn.Sequential(
        nn.Linear(1000, 250),
        nn.Hardswish(),
        nn.Linear(250, 300),
        nn.Hardswish(),
        nn.Linear(300, 350),
        nn.Hardswish(),
        nn.Linear(350, 1)
    ),
    "model42": nn.Sequential(
        nn.Linear(1000, 200),
        nn.Hardswish(),
        nn.Linear(200, 250),
        nn.Hardswish(),
        nn.Linear(250, 300),
        nn.Hardswish(),
        nn.Linear(300, 1)
    ),
    "model43": nn.Sequential(
        nn.Linear(1000, 150),
        nn.Hardswish(),
        nn.Linear(150, 200),
        nn.Hardswish(),
        nn.Linear(200, 250),
        nn.Hardswish(),
        nn.Linear(250, 1)
    ),
    "model44": nn.Sequential(
        nn.Linear(1000, 100),
        nn.Hardswish(),
        nn.Linear(100, 150),
        nn.Hardswish(),
        nn.Linear(150, 200),
        nn.Hardswish(),
        nn.Linear(200, 1)
    ),
    "model45": nn.Sequential(
        nn.Linear(1000, 50),
        nn.Hardswish(),
        nn.Linear(50, 100),
        nn.Hardswish(),
        nn.Linear(100, 150),
        nn.Hardswish(),
        nn.Linear(150, 1)
    ),
    "model46": nn.Sequential(
        nn.Linear(1000, 0),
        nn.Hardswish(),
        nn.Linear(0, 50),
        nn.Hardswish(),
        nn.Linear(50, 100),
        nn.Hardswish(),
        nn.Linear(100, 1)
    ),
    "model47": nn.Sequential(
        nn.Linear(1000, 950),
        nn.Hardswish(),
        nn.Linear(950, 900),
        nn.Hardswish(),
        nn.Linear(900, 850),
        nn.Hardswish(),
        nn.Linear(850, 1)
    ),
    "model48": nn.Sequential(
        nn.Linear(1000, 900),
        nn.Hardswish(),
        nn.Linear(900, 850),
        nn.Hardswish(),
        nn.Linear(850, 800),
        nn.Hardswish(),
        nn.Linear(800, 1)
    ),
    "model49": nn.Sequential(
        nn.Linear(1000, 850),
        nn.Hardswish(),
        nn.Linear(850, 800),
        nn.Hardswish(),
        nn.Linear(800, 750),
        nn.Hardswish(),
        nn.Linear(750, 1)
    ),
    "model50": nn.Sequential(
        nn.Linear(1000, 800),
        nn.Hardswish(),
        nn.Linear(800, 750),
        nn.Hardswish(),
        nn.Linear(750, 700),
        nn.Hardswish(),
        nn.Linear(700, 1)
    ),
    "model51": nn.Sequential(
        nn.Linear(1000, 750),
        nn.Hardswish(),
        nn.Linear(750, 700),
        nn.Hardswish(),
        nn.Linear(700, 650),
        nn.Hardswish(),
        nn.Linear(650, 1)
    ),
    "model52": nn.Sequential(
        nn.Linear(1000, 700),
        nn.Hardswish(),
        nn.Linear(700, 650),
        nn.Hardswish(),
        nn.Linear(650, 600),
        nn.Hardswish(),
        nn.Linear(600, 1)
    ),
    "model53": nn.Sequential(
        nn.Linear(1000, 650),
        nn.Hardswish(),
        nn.Linear(650, 600),
        nn.Hardswish(),
        nn.Linear(600, 550),
        nn.Hardswish(),
        nn.Linear(550, 1)
    ),
    "model54": nn.Sequential(
        nn.Linear(1000, 600),
        nn.Hardswish(),
        nn.Linear(600, 550),
        nn.Hardswish(),
        nn.Linear(550, 500),
        nn.Hardswish(),
        nn.Linear(500, 1)
    ),
    "model55": nn.Sequential(
        nn.Linear(1000, 550),
        nn.Hardswish(),
        nn.Linear(550, 500),
        nn.Hardswish(),
        nn.Linear(500, 450),
        nn.Hardswish(),
        nn.Linear(450, 1)
    ),
    "model56": nn.Sequential(
        nn.Linear(1000, 500),
        nn.Hardswish(),
        nn.Linear(500, 450),
        nn.Hardswish(),
        nn.Linear(450, 400),
        nn.Hardswish(),
        nn.Linear(400, 1)
    ),
    "model57": nn.Sequential(
        nn.Linear(1000, 450),
        nn.Hardswish(),
        nn.Linear(450, 400),
        nn.Hardswish(),
        nn.Linear(400, 350),
        nn.Hardswish(),
        nn.Linear(350, 1)
    ),
    "model58": nn.Sequential(
        nn.Linear(1000, 400),
        nn.Hardswish(),
        nn.Linear(400, 350),
        nn.Hardswish(),
        nn.Linear(350, 300),
        nn.Hardswish(),
        nn.Linear(300, 1)
    ),
    "model59": nn.Sequential(
        nn.Linear(1000, 350),
        nn.Hardswish(),
        nn.Linear(350, 300),
        nn.Hardswish(),
        nn.Linear(300, 250),
        nn.Hardswish(),
        nn.Linear(250, 1)
    ),
    "model60": nn.Sequential(
        nn.Linear(1000, 300),
        nn.Hardswish(),
        nn.Linear(300, 250),
        nn.Hardswish(),
        nn.Linear(250, 200),
        nn.Hardswish(),
        nn.Linear(200, 1)
    )
}
