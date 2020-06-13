import logging

from PyQt5.QtWidgets import QApplication
import sys



from App import ManagerApp
import pandas as pd
# from matplotlib.colors import ListedColormap
# import seaborn as sns
# import matplotlib.pylab as plt
# import numpy as np
#
# N = 500
#
# my_cmap = ListedColormap(sns.color_palette("hls", N))
#
# data1 = np.random.randn(N)
# data2 = np.random.randn(N)
# colors = np.linspace(0,1,N)
#
# plt.scatter(data1, data2, c=colors, cmap=my_cmap)
# plt.colorbar()
# plt.show()

# plt.plot([0, 1, 2], label="xc", color=(0.3,0.3,0.5))
# plt.show()


# mydict = {2: [1, 2, 3], 10: [10], 0: [4, 5,  6], 1: [21, 22, 23]}
# df = pd.DataFrame.from_dict(mydict, orient='index')
# print(list(df.loc[10]))
# print(df)

logging.basicConfig(level=logging.INFO)

app = QApplication(sys.argv)
ex = ManagerApp()
sys.exit(app.exec_())

