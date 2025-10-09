```python

# SDSS 天体物理统计项目

# --- 导入库 ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. 数据读取与初步探索 ---
file_path = "SDSS_17K.dat"
df = pd.read_csv(file_path, sep='|')  # 根据实际情况调整分隔符

print(df.head())
print(df.info())

# --- 2. 数据清洗 ---
# 去除缺失值
df = df.dropna()

# 选择主要物理量（例如光度、颜色指标、红移等）
features = ['u', 'g', 'r', 'i', 'z', 'redshift']  # 请根据实际列名调整
X = df[features]

# --- 3. 描述性统计与可视化 ---
print(X.describe())
sns.pairplot(X.sample(500))
plt.show()

# --- 4. PCA 分析 ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:,0], X_pca[:,1], s=5, alpha=0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of SDSS sample")
plt.show()

print("Explained variance ratio:", pca.explained_variance_ratio_)

# --- 5. 聚类分析 ---
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_pca)

plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='viridis', s=5)
plt.title("KMeans Clustering on PCA space")
plt.show()

# --- 6. 监督学习（分类模型） ---
# 假设存在 object_class 列（如 QSO, STAR, GALAXY）
if 'class' in df.columns:
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

# --- 7. 小结与讨论 ---
# 学生可在此部分写出对比不同方法结果的理解和天体物理意义

```
