### 1 数据准备
import pandas as pd

## 1.1 数据读取
df = pd.read_csv("C:\\Users\\lcf14\\Desktop\\emotion_dataset_raw.csv")

## 1.2 数据探索
#print(df.head())
#print(df['Emotion'].value_counts())

## 1.3 数据预处理  (使用neattext库)
import neattext.functions as nfx

    # 去除用户信息
df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)

    # 去除句中的停用词
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)
#print(df.head())

    # 文本向量化 (使用词袋模型) (另一种常用的方法为TF-IDF)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['Clean_Text'])

    # 将情感标签转换为数字
emotion_mapping = {'neutral': 0, 'joy': 1, 'sadness': 2, 'fear': 3, 'surprise': 4, 'anger': 5, 'shame': 6, 'disgust': 7}
y = df['Emotion'].map(emotion_mapping)
#print(y)

    # 划分训练集和测试集
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)



### 2 模型训练

## 2.1 OneRule  （如果运行这个算法，需要更改vectorizer的特征值，特征值为10需要30秒，为100需要四分钟）
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import trange

def oneR_classifier(X_train, y_train, X_test):
    best_feature = None
    best_accuracy = 0
    best_predictions = None
    num_features = X_train.shape[1]

    for feature_index in trange(num_features, desc='Finding best feature'):
        # 提取每个特征的所有值
        feature_values = X_train[:, feature_index].toarray().ravel()

        # 基于这个特征创建一个简单的预测模型
        predictions = {}
        for unique_value in np.unique(feature_values):
            if unique_value != 0:
                mask = feature_values == unique_value
                most_common_class = np.bincount(y_train[mask]).argmax()
                predictions[unique_value] = most_common_class

        # 使用这个特征进行预测
        pred_labels = np.array([predictions.get(val, np.random.choice(np.unique(y_train))) for val in X_test[:, feature_index].toarray().ravel()])

        # 计算准确率
        accuracy = accuracy_score(y_test, pred_labels)

        # 更新最佳特征
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_feature = feature_index
            best_predictions = pred_labels

    return best_feature, best_accuracy, best_predictions

    # 运行oneR，获取最佳特征和准确度
best_feature, best_accuracy, best_predictions = oneR_classifier(X_train, y_train, X_test)
print(f"OneR_Accuracy: {best_accuracy}")   # OneR_Accuracy: 0.1431240120706998

    # 获取最佳特征对应的单词
best_feature_name = vectorizer.get_feature_names_out()[best_feature]
print(f"The best feature by OneR is: {best_feature_name}")





## 2.2 Logistic Regression（逻辑回归）
from sklearn.linear_model import LogisticRegression
    # 定义Logistic Regression模型
LR_model = LogisticRegression(max_iter=1000)
    # 拟合模型
LR_model.fit(X_train, y_train)
    # 预测
LR_y_pred = LR_model.predict(X_test)
    # 计算准确率
from sklearn.metrics import accuracy_score
LR_accuracy = accuracy_score(y_test, LR_y_pred)
print(f"LR_Accuracy: {LR_accuracy}")   # 准确率为：0.5640178186521052
"""
    # 使用混淆矩阵来评估结果
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
    # 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, LR_y_pred)
    # 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
"""




## 2.3 k-Nearest Neighbors（k-近邻）
from sklearn.neighbors import KNeighborsClassifier
    # 定义k-Nearest Neighbors模型
KNN_model = KNeighborsClassifier(n_neighbors=5)
    # 拟合模型
KNN_model.fit(X_train, y_train)
    # 预测
KNN_y_pred = KNN_model.predict(X_test)
    # 计算准确率
from sklearn.metrics import accuracy_score
KNN_accuracy = accuracy_score(y_test, KNN_y_pred)
print(f"KNN_Accuracy: {KNN_accuracy}")      # 准确率为：0.4227618910763041
"""
    # 使用混淆矩阵来评估结果
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
    # 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, KNN_y_pred)
    # 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
"""




## 2.4 Naive Bayes（朴素贝叶斯）
from sklearn.naive_bayes import MultinomialNB
    # 定义Naive Bayes模型
NB_model = MultinomialNB()
    # 拟合模型
NB_model.fit(X_train, y_train)
    # 预测
NB_y_pred = NB_model.predict(X_test)
    # 计算准确率
from sklearn.metrics import accuracy_score
NB_accuracy = accuracy_score(y_test, NB_y_pred)
print(f"NB_Accuracy: {NB_accuracy}")        # 准确率为：0.5349906595775255
"""
    # 使用混淆矩阵来评估结果
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
conf_matrix = confusion_matrix(y_test, NB_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
"""




## 2.5 Decision Tree（决策树）
from sklearn.tree import DecisionTreeClassifier
    # 定义Decision Tree模型
DT_model = DecisionTreeClassifier()
    # 拟合模型
DT_model.fit(X_train, y_train)
    # 预测
DT_y_pred = DT_model.predict(X_test)
    # 计算准确率
from sklearn.metrics import accuracy_score
DT_accuracy = accuracy_score(y_test, DT_y_pred)
print(f"DT_Accuracy: {DT_accuracy}")       # 准确率为：0.5082626814197442
"""
    # 使用混淆矩阵来评估结果
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
conf_matrix = confusion_matrix(y_test, DT_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
"""




## 2.6 Gradient Boosting（梯度提升）
from sklearn.ensemble import GradientBoostingClassifier
    # 定义Gradient Boosting模型
GB_model = GradientBoostingClassifier()
    # 拟合模型
GB_model.fit(X_train, y_train)
    # 预测
GB_y_pred = GB_model.predict(X_test)
    # 计算准确率
from sklearn.metrics import accuracy_score
GB_accuracy = accuracy_score(y_test, GB_y_pred)
print(f"GB_Accuracy: {GB_accuracy}")       # 准确率为：0.5026584279350481
"""
    # 使用混淆矩阵来评估结果
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
conf_matrix = confusion_matrix(y_test, GB_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
"""




## 2.7 Support Vector Machine（支持向量机）
from sklearn.svm import SVC
    # 定义SVC模型
#SVM_model = SVC(probability=True)
    # 拟合模型
#SVM_model.fit(X_train, y_train)
# 保存模型
import joblib
#joblib.dump(SVM_model, 'saved_SVM_model.pkl')
SVM_model = joblib.load('saved_SVM_model.pkl')
    # 预测
SVM_y_pred = SVM_model.predict(X_test)
    # 计算准确率
from sklearn.metrics import accuracy_score
SVM_accuracy = accuracy_score(y_test, SVM_y_pred)
print(f"SVM_Accuracy: {SVM_accuracy}")       # 准确率为：0.5765196148871964
"""
    # 使用混淆矩阵来评估结果
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
conf_matrix = confusion_matrix(y_test, SVM_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
"""




## 2.8 Multilayer Perceptron（多层感知机）
from sklearn.neural_network import MLPClassifier
    # 定义Multilayer Perceptron模型
#MLP_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
    # 拟合模型
#MLP_model.fit(X_train, y_train)
# 保存模型
import joblib
#joblib.dump(MLP_model, 'saved_MLP_model.pkl')
MLP_model = joblib.load('saved_MLP_model.pkl')
    # 预测
MLP_y_pred = MLP_model.predict(X_test)
    # 计算准确率
from sklearn.metrics import accuracy_score
MLP_accuracy = accuracy_score(y_test, MLP_y_pred)
print(f"MLP_Accuracy: {MLP_accuracy}")       # 准确率为：0.5466302629688173
"""
    # 使用混淆矩阵来评估结果
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
conf_matrix = confusion_matrix(y_test, MLP_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
"""




## 2.9 Ensemble methods（集成方法，如随机森林）
from sklearn.ensemble import RandomForestClassifier
    # 定义RandomForest模型
#RF_model = RandomForestClassifier(n_estimators=100)
    # 拟合模型
#RF_model.fit(X_train, y_train)
# 保存模型
import joblib
#joblib.dump(RF_model, 'saved_RF_model.pkl')
RF_model = joblib.load('saved_RF_model.pkl')
    # 预测
RF_y_pred = RF_model.predict(X_test)
    # 计算准确率
from sklearn.metrics import accuracy_score
RF_accuracy = accuracy_score(y_test, RF_y_pred)
print(f"RF_Accuracy: {RF_accuracy}")       # 准确率为：0.568472481678402
"""
    # 使用混淆矩阵来评估结果
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
conf_matrix = confusion_matrix(y_test, RF_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
"""


"""
# 输入一句话 (趣味检测)
new_text = str(input())
# 对输入的文本进行预处理
new_text_processed = pd.Series(new_text).str.lower()
new_text_vectorized = vectorizer.transform(new_text_processed)
# 使用训练好的模型进行情绪预测
predicted_emotion = model.predict(new_text_vectorized)[0]
predicted_emotion_word = [key for key, value in emotion_mapping.items() if value == predicted_emotion][0]
print(f"The predicted emotion for the input text is: {predicted_emotion_word}")
"""


### 模型质量评估

# 3.1 绘制ROC曲线图，评估模型质量    (roc_curve计算ROC曲线不支持多类别格式，必须是二分类才能使用)
"""
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

models = [LR_model, KNN_model, NB_model, DT_model, GB_model, SVM_model, MLP_model, RF_model]
model_names = ['LR_model', 'KNN_model', 'NB_model', 'DT_model', 'GB_model', 'SVM_model', 'MLP_model', 'RF_model']

# 假设X_test是测试数据，y_test是测试标签
fig, ax = plt.subplots(figsize=(10, 8))

for model, name in zip(models, model_names):
    # 获取模型对测试集的预测概率
    y_scores = model.predict_proba(X_test)[:, 1]
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    # 绘制ROC曲线
    ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

ax.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic')
ax.legend(loc="lower right")
plt.show()
"""




# 3.2 使用F1值评估模型
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

models = [LR_model, KNN_model, NB_model, DT_model, GB_model, SVM_model, MLP_model, RF_model]
model_names = ['Logistic Regression', 'KNN', 'Naive Bayes', 'Decision Tree', 'Gradient Boosting', 'SVM', 'MLP', 'Random Forest']

f1_scores = []
f1_oneR = f1_score(y_test, best_predictions, average='weighted')
f1_scores.append(f1_oneR)
print(f'OneR F1 Score: {f1_oneR}')
for model, name in zip(models, model_names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    f1_scores.append(f1)
    print(f'{name} F1 Score: {f1}')
# OneR F1 Score: 0.17169862099376054
# Logistic Regression F1 Score: 0.552417311201324
# KNN F1 Score: 0.419277335314708
# Naive Bayes F1 Score: 0.5236681242484474
# Decision Tree F1 Score: 0.5062270115461803
# Gradient Boosting F1 Score: 0.4678656855047288
# SVM F1 Score: 0.5561085375239769
# MLP F1 Score: 0.5395474083641408
# Random Forest F1 Score: 0.5578547936689194

model_names = ['OneR','Logistic Regression', 'KNN', 'Naive Bayes', 'Decision Tree', 'Gradient Boosting', 'SVM', 'MLP', 'Random Forest']
    # 绘制F1值比较图
plt.figure(figsize=(10, 6))
plt.bar(model_names, f1_scores, color='skyblue')
plt.xlabel('Models')
plt.ylabel('F1 Score')
plt.title('F1 Score Comparison of Different Models')
plt.show()
"""



# 3.3 使用准确值评估模型
import matplotlib.pyplot as plt

model_names = ['OneR','Logistic Regression', 'KNN', 'Naive Bayes', 'Decision Tree', 'Gradient Boosting', 'SVM', 'MLP', 'Random Forest']
Accuracy = [best_accuracy, LR_accuracy, KNN_accuracy, NB_accuracy, DT_accuracy, GB_accuracy, SVM_accuracy, MLP_accuracy, RF_accuracy]

plt.figure(figsize=(10, 6))
plt.bar(model_names, Accuracy, color='skyblue')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Different Models')
plt.show()
"""