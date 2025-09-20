import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import zscore
import pandas as pd


df = pd.read_csv("C:/Users/maria/OneDrive/Escritorio/Octavo semestre/Machine Learning/Arbol de decision/dataset_spam_ham_balanceado.csv")


X = df.drop("label", axis=1)
y = df["label"]


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


categorical_features = ["domain_reputation"]
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(), categorical_features)],
    remainder="passthrough"
)


accuracies = []
f1_scores = []

for i in range(50):
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=i, stratify=y
    )
 
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
 
    model = DecisionTreeClassifier(random_state=i)
    model.fit(X_train_transformed, y_train)
    
    
    y_pred = model.predict(X_test_transformed)
    
    
    accuracies.append(accuracy_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))

accuracies = np.array(accuracies)
f1_scores = np.array(f1_scores)


z_scores_acc = zscore(accuracies)
z_scores_f1 = zscore(f1_scores)

print("📊 Resultados globales")
print("Exactitud promedio:", accuracies.mean())
print("F1-score promedio:", f1_scores.mean())
print("Primeros 5 z-scores Exactitud:", z_scores_acc[:5])
print("Primeros 5 z-scores F1:", z_scores_f1[:5])


plt.figure(figsize=(12,6))
plt.plot(range(1,51), accuracies, marker='o', label="Exactitud", alpha=0.7)
plt.plot(range(1,51), f1_scores, marker='s', label="F1-score", alpha=0.7)
plt.xlabel("Ejecución")
plt.ylabel("Valor de la métrica")   
plt.title("Variabilidad de Exactitud y F1-score en 50 ejecuciones")
plt.legend()
plt.grid(True)
plt.show()


model = DecisionTreeClassifier(
    random_state=0, 
    max_depth=3,  
    min_samples_split=50,  
    min_samples_leaf=20 
)
model.fit(X_train_transformed, y_train)


plt.figure(figsize=(20,10))
plot_tree(
    model,
    filled=True,
    feature_names=preprocessor.get_feature_names_out().tolist(),
    class_names=['Ham', 'Spam'],  
    fontsize=10, 
    proportion=True,  
    rounded=True,  
    precision=2  
)
plt.title("Árbol de Decisión para Clasificación de Correos", fontsize=14, pad=20)
plt.tight_layout()
plt.show()


y_pred_final = model.predict(X_test_transformed)
print("\n📊 Métricas del modelo final:")
print(f"Exactitud: {accuracy_score(y_test, y_pred_final):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred_final):.3f}")

