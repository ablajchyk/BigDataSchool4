import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from typing import Optional


class Visualisation:
    def __init__(self, data: pd.DataFrame):
        self.data = data.astype(float)

    def correlate(self):
        sb.heatmap(self.data.corr(), annot=True)
        plt.show()


class Analysis:
    def __init__(self, path_tab: str, path_train: str, path_test: str):
        self.data: pd.DataFrame = Optional[pd.DataFrame]
        self.tabular = pd.read_csv(path_tab)
        self.test = pd.read_csv(path_test)
        self.train = pd.read_csv(path_train)

    def unite(self):
        self.data = self.tabular.set_index('ID').join(self.train.set_index('ID'))

    def clear(self):
        self.data = self.data.fillna(self.data.groupby('PERIOD').transform('mean'))

    def drop(self):
        list_of_high_corr = [8, 10, 12, 14, 24, 28, 29, 30, 33, 34, 35, 36, 39, 40, 41]
        self.data = self.data.drop(['V_'+str(x) for x in list_of_high_corr], axis=1)

    def prepare(self):
        self.unite()
        self.clear()
        self.drop()


if __name__ == "__main__":
    analysis = Analysis('tabular_data.csv', 'train_target.csv', 'test_target.csv')
    analysis.prepare()
    visualisation = Visualisation(analysis.data)
    print(analysis.data.head(10))
