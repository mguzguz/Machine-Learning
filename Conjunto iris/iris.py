import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, accuracy_score

# 1. Cargar dataset
data = load_iris()
X, y = data.data, data.target

# 2. Dividir en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Entrenar un modelo de regresión lineal One-vs-Rest
models = []
for clase in np.unique(y_train):
    y_binary = (y_train == clase).astype(int)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", LinearRegression())
    ])
    model.fit(X_train, y_binary)
    models.append((clase, model))

# 4. Función para predecir la clase
def predecir_planta(nueva_planta):
    scores = [m.predict([nueva_planta])[0] for _, m in models]
    clase_predicha = np.argmax(scores)
    return data.target_names[clase_predicha], clase_predicha

# 5. Evaluación general
def evaluar_modelo():
    y_pred = []
    for x in X_test:
        scores = [m.predict([x])[0] for _, m in models]
        y_pred.append(np.argmax(scores))
    acc = accuracy_score(y_test, y_pred)
    print(f"Exactitud global del modelo con Regresión Lineal: {acc:.2f}")

# 6. Graficar cada regresor (real vs predicho)
def graficar_regresores():
    for clase, model in models:
        y_true = (y_test == clase).astype(int)
        y_pred = model.predict(X_test)

        # Calcular ECM
        ecm = mean_squared_error(y_true, y_pred)

        # Coeficientes y ecuación
        coefs = model.named_steps["reg"].coef_
        intercepto = model.named_steps["reg"].intercept_
        ecuacion = f"y = {intercepto:.2f} + " + " + ".join([f"{c:.2f}*x{i+1}" for i, c in enumerate(coefs)])

        # Gráfico
        plt.figure(figsize=(6, 5))
        plt.scatter(y_true, y_pred, alpha=0.7, edgecolor="k")
        plt.plot([0, 1], [0, 1], "r--", label="y = x (ideal)")
        plt.xlabel("Valor real (0 o 1)")
        plt.ylabel("Valor predicho")
        plt.title(f"Regresión Lineal para clase {data.target_names[clase]}")
        plt.legend()
        plt.text(0.05, 0.9, f"ECM = {ecm:.4f}\n{ecuacion}",
                 transform=plt.gca().transAxes, fontsize=9,
                 bbox=dict(facecolor="white", alpha=0.7))
        plt.show()

# 7. Bucle de clasificación
def clasificador_interactivo():
    while True:
        largo_sepalo = float(input("Ingrese el LARGO del sépalo (cm): "))
        ancho_sepalo = float(input("Ingrese el ANCHO del sépalo (cm): "))
        largo_petalo = float(input("Ingrese el LARGO del pétalo (cm): "))
        ancho_petalo = float(input("Ingrese el ANCHO del pétalo (cm): "))

        nueva = [largo_sepalo, ancho_sepalo, largo_petalo, ancho_petalo]
        clase, clase_idx = predecir_planta(nueva)

        print(f"\nLa planta ingresada fue clasificada como: {clase}\n")

        continuar = input("¿Desea clasificar otra planta? (s/n): ").lower()
        if continuar != "s":
            break


# --- EJECUCIÓN ---
if __name__ == "__main__":
    evaluar_modelo()
    graficar_regresores()
    clasificador_interactivo()
