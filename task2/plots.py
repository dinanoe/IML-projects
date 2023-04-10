import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from constants import SEASONS, PRICES

class Plot:
    def __init__(self):
        self.x_name = None
        self.y_name = None
        self.data = []
        self.created = False

    def add(self, x: pd.Series, y: pd.Series, color=None, label=None):
        self.x_name = x.name
        self.y_name = y.name
        x_copy = x.copy()
        y_copy = y.copy()
        self.data.append((x_copy, y_copy, color, label))

    def create(self):
        plt.figure(figsize=(8, 6))
        for i, (x, y, color, label) in enumerate(self.data):
            if not color:
                color = np.random.rand(3)
            if not label:
                label = f"Series {i+1}"
            plt.scatter(x=x, y=y, c=[color], label=label)
        plt.xlabel(self.x_name)
        plt.ylabel(self.y_name)
        plt.legend()
        self.created = True

    def show(self):
        if not self.created:
            self.create()
        plt.show()

    def save(self):
        if not self.created:
            self.create()
        plt.savefig(f'plots/{self.x_name}_{self.y_name}.png', dpi=100)
