import torch
import matplotlib.pyplot as plt
import pandas as pd
# 读入数据集
data = pd.read_csv('bs1.1.1.csv')

# 计算相关性系数矩阵
corr_matrix = data.corr()

# 提取SOC与Voltage、SOC与Current、SOC与Temperature、SOC与Power之间的相关性系数
soc_voltage_corr = corr_matrix.loc['soc', 'voltage']
soc_current_corr = corr_matrix.loc['soc', 'current']
soc_temperature_corr = corr_matrix.loc['soc', 'temperature']
soc_power_corr = corr_matrix.loc['soc', 'power']

# 创建相关性系数矩阵
corr_heatmap = torch.tensor([[1, soc_voltage_corr, soc_current_corr, soc_temperature_corr, soc_power_corr],
                             [soc_voltage_corr, 1, 0, 0, 0],
                             [soc_current_corr, 0, 1, 0, 0],
                             [soc_temperature_corr, 0, 0, 1, 0],
                             [soc_power_corr, 0, 0, 0, 1]])

# 绘制热力图
plt.imshow(corr_heatmap, cmap='coolwarm')
plt.colorbar()
plt.xticks([0, 1, 2, 3, 4], ['SOC', 'Voltage', 'Current', 'Temperature', 'Power'])
plt.yticks([0, 1, 2, 3, 4], ['SOC', 'Voltage', 'Current', 'Temperature', 'Power'])

# 在每个矩形中心添加对应的值
for i in range(corr_heatmap.shape[0]):
    for j in range(corr_heatmap.shape[1]):
        plt.text(j, i, format(corr_heatmap[i, j], '.2f'), ha="center", va="center", color="black")

plt.show()