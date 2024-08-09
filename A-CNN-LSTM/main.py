import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error
from keras.layers import Attention
from keras.layers import Layer
from keras import backend as K
import random
import tensorflow as tf
import numpy as np
from keras.layers import LSTM, GRU, SimpleRNN, Conv1D, MaxPooling1D
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
plt.rcParams.update({'font.size': 24})
plt.rcParams["font.family"] = "Times New Roman"
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
# 读取数据
data = pd.read_csv('bs311.csv')
# 去除第一列cycle
data.drop(['cycle'], axis=1, inplace=True)

# 数据归一化
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 创建训练数据集
train_data = scaled_data[:439200, :]

# 划分训练数据集和验证数据集
X_train = []
y_train = []
for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i, :])
    y_train.append(train_data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = X_train[-1000:], y_train[-1000:]
X_train, y_train = X_train[:-1000], y_train[:-1000]
# 创建注意力机制层
class AttentionLayer(Layer):
    def __init__(self, attention_dim):
        super(AttentionLayer, self).__init__()
        self.attention_dim = attention_dim

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], self.attention_dim),
                                 initializer="uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(self.attention_dim,),
                                 initializer="uniform", trainable=True)
        self.u = self.add_weight(name="att_context", shape=(self.attention_dim, 1),
                                 initializer="uniform", trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # 对x进行切片并压缩维度，得到新的x_other
        x_other = K.concatenate([x[:, :, :1], x[:, :, 2:]], axis=-1)
        x_other = K.sum(x_other, axis=-1, keepdims=True)
        # 计算注意力权重
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(K.dot(e, self.u), axis=1)
        # 计算加权后的向量
        output = x_other * a
        return a


# 创建模型
model1 = Sequential()
model1.add(Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model1.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
model1.add(MaxPooling1D(pool_size=2))
model1.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
model1.add(MaxPooling1D(pool_size=2))
model1.add(LSTM(units=32, return_sequences=True))
model1.add(AttentionLayer(attention_dim=32))# 添加注意力机制层
model1.add(Flatten())
model1.add(Dense(units=1))
model1.summary()
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 24})
# 编译模型
adam = Adam(learning_rate=0.0005)
model1.compile(loss='mean_squared_error', optimizer=adam)



# 训练模型
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model_checkpoint = ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
history = model1.fit(X_train, y_train, epochs=100, batch_size=256, validation_data=(X_val, y_val), callbacks=[early_stopping, model_checkpoint])

# 预测数据
test_data = scaled_data[439200:, :]
X_test = []
y_test = []
for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, :])
    y_test.append(data['soc'][439200+i-1])
X_test, y_test = np.array(X_test), np.array(y_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 5))


#GRU

model2 = Sequential()
model2.add(LSTM(units=32, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model2.add(Flatten())
model2.add(Dense(units=1))
model2.summary()

adam = Adam(learning_rate=0.0005)
model2.compile(loss='mean_squared_error', optimizer=adam)

# 训练模型
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model_checkpoint = ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
history = model2.fit(X_train, y_train, epochs=100, batch_size=256, validation_data=(X_val, y_val), callbacks=[early_stopping, model_checkpoint])

#CNN+GRU

model3 = Sequential()
model3.add(Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model3.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
model3.add(MaxPooling1D(pool_size=2))
model3.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
model3.add(MaxPooling1D(pool_size=2))
model3.add(GRU(units=32, return_sequences=True))
model3.add(Flatten())
model3.add(Dense(units=1))
model3.summary()

# 编译模型
adam = Adam(learning_rate=0.0005)
model3.compile(loss='mean_squared_error', optimizer=adam)

# 训练模型
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model_checkpoint = ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
history = model3.fit(X_train, y_train, epochs=100, batch_size=256, validation_data=(X_val, y_val), callbacks=[early_stopping, model_checkpoint])


#CNN

# 创建模型
model4 = Sequential()
model4.add(Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model4.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
model4.add(MaxPooling1D(pool_size=2))
model4.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
model4.add(MaxPooling1D(pool_size=2))
model4.add(Flatten())
model4.add(Dense(units=1))
model4.summary()


# 编译模型
adam = Adam(learning_rate=0.0005)
model4.compile(loss='mean_squared_error', optimizer=adam)

# 训练模型
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model_checkpoint = ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
history = model4.fit(X_train, y_train, epochs=100, batch_size=256, validation_data=(X_val, y_val), callbacks=[early_stopping, model_checkpoint])

#LSTM

# 创建模型
model5 = Sequential()
model5.add(GRU(units=32, input_shape=(X_train.shape[1], X_train.shape[2])))
model5.add(Dense(units=1))
model5.summary()

# 编译模型
adam = Adam(learning_rate=0.0005)
model5.compile(loss='mean_squared_error', optimizer=adam)

# 训练模型
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model_checkpoint = ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
history = model5.fit(X_train, y_train, epochs=100,batch_size=256, validation_data=(X_val, y_val), callbacks=[early_stopping, model_checkpoint])

#CNN-LSTM
# 创建模型
model6 = Sequential()
model6.add(Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model6.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
model6.add(MaxPooling1D(pool_size=2))
model6.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
model6.add(MaxPooling1D(pool_size=2))
model6.add(LSTM(units=32, return_sequences=True))
model6.add(Flatten())
model6.add(Dense(units=1))
model6.summary()


# 编译模型
adam = Adam(learning_rate=0.0005)
model6.compile(loss='mean_squared_error', optimizer=adam)

# 训练模型
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model_checkpoint = ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
history = model6.fit(X_train, y_train, epochs=100, batch_size=256, validation_data=(X_val, y_val), callbacks=[early_stopping, model_checkpoint])





# 进行预测
y_pred1= model1.predict(X_test)
y_pred2 = model2.predict(X_test)
y_pred_inv1= scaler.inverse_transform(np.concatenate((y_pred1, X_test[:, -1, 1:]), axis=1))[:, 0]
y_pred_inv2= scaler.inverse_transform(np.concatenate((y_pred2, X_test[:, -1, 1:]), axis=1))[:, 0]
y_pred3 = model3.predict(X_test)
y_pred_inv3= scaler.inverse_transform(np.concatenate((y_pred3, X_test[:, -1, 1:]), axis=1))[:, 0]
y_pred4 = model4.predict(X_test)
y_pred_inv4= scaler.inverse_transform(np.concatenate((y_pred4, X_test[:, -1, 1:]), axis=1))[:, 0]
y_pred5 = model5.predict(X_test)
y_pred_inv5= scaler.inverse_transform(np.concatenate((y_pred5, X_test[:, -1, 1:]), axis=1))[:, 0]
y_pred6 = model6.predict(X_test)
y_pred_inv6= scaler.inverse_transform(np.concatenate((y_pred6, X_test[:, -1, 1:]), axis=1))[:, 0]

abs_error1= np.abs(y_pred_inv1 - y_test)
abs_error2= np.abs(y_pred_inv2 - y_test)
abs_error3= np.abs(y_pred_inv3 - y_test)
abs_error4= np.abs(y_pred_inv4 - y_test)
abs_error5= np.abs(y_pred_inv5 - y_test)
abs_error6= np.abs(y_pred_inv6 - y_test)

rmse1 = np.sqrt(mean_squared_error(y_test,y_pred_inv1))
print('RMSE1:', rmse1)
r21 = r2_score(y_test, y_pred_inv1)
print("R2 score1:", r21)
mape1 = mean_absolute_percentage_error(y_test, y_pred_inv1)
df_mape1 = pd.DataFrame(columns=['Model', 'MAPE'])
df_mape1 = df_mape1._append({'Model': '1', 'MAPE': mape1}, ignore_index=True)
df_mape1.to_csv('mape_values1.csv', index=False)
print("MAPE1:", mape1)
mae1 = np.mean(np.abs(y_test - y_pred_inv1))
print('MAE1:', mae1)

rmse2= np.sqrt(mean_squared_error(y_test,y_pred_inv2))
print('RMSE2:', rmse2)
r22 = r2_score(y_test, y_pred_inv2)
print("R2 score2:", r22)
mape2 = mean_absolute_percentage_error(y_test, y_pred_inv2)
df_mape2 = pd.DataFrame(columns=['Model', 'MAPE'])
df_mape2 = df_mape1._append({'Model': '2', 'MAPE': mape1}, ignore_index=True)
df_mape2.to_csv('mape_values2.csv', index=False)
print("MAPE2:", mape2)
mae2= np.mean(np.abs(y_test - y_pred_inv2))
print('MAE2:', mae2)

rmse3 = np.sqrt(mean_squared_error(y_test,y_pred_inv3))
print('RMSE3:', rmse3)
r23= r2_score(y_test, y_pred_inv3)
print("R2 score3:", r23)
mape3 = mean_absolute_percentage_error(y_test, y_pred_inv3)
df_mape3 = pd.DataFrame(columns=['Model', 'MAPE'])
df_mape3 = df_mape1._append({'Model': '3', 'MAPE': mape1}, ignore_index=True)
df_mape3.to_csv('mape_values3.csv', index=False)
print("MAPE3:", mape3)
mae3 = np.mean(np.abs(y_test - y_pred_inv3))
print('MAE3:', mae3)

rmse4 = np.sqrt(mean_squared_error(y_test,y_pred_inv4))
print('RMSE4:', rmse4)
r24 = r2_score(y_test, y_pred_inv4)
print("R2 score4", r24)
mape4 = mean_absolute_percentage_error(y_test, y_pred_inv4)

print("MAPE4:", mape4)
mae4 = np.mean(np.abs(y_test - y_pred_inv4))
print('MAE4:', mae4)

rmse5 = np.sqrt(mean_squared_error(y_test,y_pred_inv5))
print('RMSE5:', rmse5)
r25 = r2_score(y_test, y_pred_inv5)
print("R2 score5:", r25)
mape5 = mean_absolute_percentage_error(y_test, y_pred_inv5)
print("MAPE5:", mape5)
mae5 = np.mean(np.abs(y_test - y_pred_inv5))
print('MAE5:', mae5)

rmse6 = np.sqrt(mean_squared_error(y_test,y_pred_inv6))
print('RMSE6:', rmse6)
r26 = r2_score(y_test, y_pred_inv6)
print("R2 score6:", r26)
mape6 = mean_absolute_percentage_error(y_test, y_pred_inv6)
print("MAPE6:", mape6)
mae6 = np.mean(np.abs(y_test - y_pred_inv6))
print('MAE6:', mae6)


#mseplt.plot(y_test, color='blue', label='True')
plt.plot(y_test[::20], color='black',  label='True')
plt.plot(y_pred_inv1[::20], color='red', label='A-CNN-LSTM')
plt.plot(y_pred_inv2[::20], color='green', label='GRU')
plt.plot(y_pred_inv3[::20], color='orange', label='CNN-GRU')
plt.plot(y_pred_inv4[::20], color='purple', label='CNN')
plt.plot(y_pred_inv5[::20], color='cyan', label='LSTM')
plt.plot(y_pred_inv6[::20], color='blue', label='CNN-LSTM')
plt.legend()
plt.show()
plt.plot(abs_error1, color='red', label='A-CNN-LSTM')
plt.plot(abs_error2, color='green', label='GRU')
plt.plot(abs_error3, color='orange', label='CNN-GRU')
plt.plot(abs_error4, color='purple', label='CNN')
plt.plot(abs_error5, color='cyan', label='LSTM')
plt.plot(abs_error6, color='blue', label='CNN-LSTM')
plt.legend()
plt.show()


plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()





# 计算PDD和PED
sns.kdeplot(y_pred_inv1, label='PDD')  # PDD
sns.kdeplot(y_pred_inv1 - y_test, label='PED')  # PED
pdd_mean = np.mean(y_pred_inv1)
pdd_std = np.std(y_pred_inv1)
ped_mean = np.mean(y_pred_inv1 - y_test)
ped_std = np.std(y_pred_inv1 - y_test)
print("PDD Mean1:", pdd_mean)
print("PDD Standard Deviation1:", pdd_std)
print("PED Mean1:", ped_mean)
print("PED Standard Deviation1:", ped_std)
# 提取预测结果和观测值的数据
predicted_data = y_pred_inv1  # 模型的预测结果
observed_data = y_test  # 实际观测值
# 计算DAI
dai = (predicted_data - observed_data) / observed_data
dai_mean = np.mean(dai)
print("平均DAI1:", dai_mean)
kl_divergence = entropy(y_test, y_pred_inv1)
print("KL Divergence1:", kl_divergence)



sns.kdeplot(y_pred_inv2, label='PDD')  # PDD
sns.kdeplot(y_pred_inv2 - y_test, label='PED')  # PED
pdd_mean = np.mean(y_pred_inv2)
pdd_std = np.std(y_pred_inv2)
ped_mean = np.mean(y_pred_inv2 - y_test)
ped_std = np.std(y_pred_inv2 - y_test)
print("PDD Mean2:", pdd_mean)
print("PDD Standard Deviation2:", pdd_std)
print("PED Mean2:", ped_mean)
print("PED Standard Deviation2:", ped_std)
# 提取预测结果和观测值的数据
predicted_data = y_pred_inv2  # 模型的预测结果
observed_data = y_test  # 实际观测值
# 计算DAI
dai = (predicted_data - observed_data) / observed_data
dai_mean = np.mean(dai)
print("平均DAI2:", dai_mean)
kl_divergence = entropy(y_test, y_pred_inv2)
print("KL Divergence2:", kl_divergence)



sns.kdeplot(y_pred_inv3, label='PDD')  # PDD
sns.kdeplot(y_pred_inv3 - y_test, label='PED')  # PED
pdd_mean = np.mean(y_pred_inv3)
pdd_std = np.std(y_pred_inv3)
ped_mean = np.mean(y_pred_inv3 - y_test)
ped_std = np.std(y_pred_inv3 - y_test)
print("PDD Mean3:", pdd_mean)
print("PDD Standard Deviation3:", pdd_std)
print("PED Mean3:", ped_mean)
print("PED Standard Deviation3:", ped_std)
# 提取预测结果和观测值的数据
predicted_data = y_pred_inv3  # 模型的预测结果
observed_data = y_test  # 实际观测值
# 计算DAI
dai = (predicted_data - observed_data) / observed_data
dai_mean = np.mean(dai)
print("平均DAI3:", dai_mean)
kl_divergence = entropy(y_test, y_pred_inv3)
print("KL Divergence3:", kl_divergence)




sns.kdeplot(y_pred_inv4, label='PDD')  # PDD
sns.kdeplot(y_pred_inv4 - y_test, label='PED')  # PED
pdd_mean = np.mean(y_pred_inv4)
pdd_std = np.std(y_pred_inv4)
ped_mean = np.mean(y_pred_inv4 - y_test)
ped_std = np.std(y_pred_inv4 - y_test)
print("PDD Mean4:", pdd_mean)
print("PDD Standard Deviation4:", pdd_std)
print("PED Mean4:", ped_mean)
print("PED Standard Deviation4:", ped_std)
# 提取预测结果和观测值的数据
predicted_data = y_pred_inv4  # 模型的预测结果
observed_data = y_test  # 实际观测值
# 计算DAI
dai = (predicted_data - observed_data) / observed_data
dai_mean = np.mean(dai)
print("平均DAI4:", dai_mean)
kl_divergence = entropy(y_test, y_pred_inv4)
print("KL Divergence4:", kl_divergence)




sns.kdeplot(y_pred_inv5, label='PDD')  # PDD
sns.kdeplot(y_pred_inv5 - y_test, label='PED')  # PED
pdd_mean = np.mean(y_pred_inv5)
pdd_std = np.std(y_pred_inv5)
ped_mean = np.mean(y_pred_inv5 - y_test)
ped_std = np.std(y_pred_inv5 - y_test)
print("PDD Mean5:", pdd_mean)
print("PDD Standard Deviation5:", pdd_std)
print("PED Mean5:", ped_mean)
print("PED Standard Deviation5:", ped_std)
# 提取预测结果和观测值的数据
predicted_data = y_pred_inv5  # 模型的预测结果
observed_data = y_test  # 实际观测值
# 计算DAI
dai = (predicted_data - observed_data) / observed_data
dai_mean = np.mean(dai)
print("平均DAI5:", dai_mean)
kl_divergence = entropy(y_test, y_pred_inv5)
print("KL Divergence5:", kl_divergence)





sns.kdeplot(y_pred_inv6, label='PDD')  # PDD
sns.kdeplot(y_pred_inv6 - y_test, label='PED')  # PED
pdd_mean = np.mean(y_pred_inv6)
pdd_std = np.std(y_pred_inv6)
ped_mean = np.mean(y_pred_inv6 - y_test)
ped_std = np.std(y_pred_inv6 - y_test)
print("PDD Mean6:", pdd_mean)
print("PDD Standard Deviation6:", pdd_std)
print("PED Mean6:", ped_mean)
print("PED Standard Deviation6:", ped_std)
# 提取预测结果和观测值的数据
predicted_data = y_pred_inv6  # 模型的预测结果
observed_data = y_test  # 实际观测值
# 计算DAI
dai = (predicted_data - observed_data) / observed_data
dai_mean = np.mean(dai)
print("平均DAI6:", dai_mean)
kl_divergence = entropy(y_test, y_pred_inv6)
print("KL Divergence6:", kl_divergence)



# ...（原来的代码不变）...

# 预测模型并进行逆归一化
# 创建 DataFrame 存储 y_pred_inv5 数据
df_pred1 = pd.DataFrame({'Predicted_SOC': y_pred_inv1})
df_pred2 = pd.DataFrame({'Predicted_SOC': y_pred_inv2})
df_pred3 = pd.DataFrame({'Predicted_SOC': y_pred_inv3})
df_pred4 = pd.DataFrame({'Predicted_SOC': y_pred_inv4})
df_pred5 = pd.DataFrame({'Predicted_SOC': y_pred_inv5})
df_pred6 = pd.DataFrame({'Predicted_SOC': y_pred_inv6})

df_pred1.to_csv('A-CNN-LSTM predicted_soc.csv', index=False)
df_pred2.to_csv('GRU predicted_soc.csv', index=False)
df_pred3.to_csv('CNN-GRU predicted_soc.csv', index=False)
df_pred4.to_csv('CNN predicted_soc.csv', index=False)
df_pred5.to_csv('LSTM predicted_soc.csv', index=False)
df_pred6.to_csv('CNN-LSTM predicted_soc.csv', index=False)

# 创建 DataFrame 存储 y_test 数据
df_test = pd.DataFrame({'True_SOC': y_test})
# 将 y_pred_inv5 导出为 CSV 文件
# 将 y_test 导出为 CSV 文件
df_test.to_csv('true_soc 6duibi.csv', index=False)


