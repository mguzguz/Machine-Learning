# =============================================
# Proyecto: Clasificación de correos Spam vs Ham
# Modelo: Regresión Logística 
# Autoras: María Camila Guzmán Bello y Ana María Casallas Ariza
# CADI: Machine Learning 802 (Profundizacion I)
# =============================================

import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ------------------------------------------------------------
# 1. Configuración inicial
# ------------------------------------------------------------
CSV_PATH = "C:/Users/Camila Guzman/OneDrive/Escritorio/Universidad/OCTAVO/ML/Actividad2/dataset_spam_ham_balanceado.csv"
OUTPUT_DIR = "experiments_output"
N_RUNS = 15   # número de corridas
TEST_SIZE = 0.30
RANDOM_SEEDS = list(range(1000, 1000 + N_RUNS))
TARGET = "label"

# Mapeo de reputación de dominio
MAPPING = {"buena": 0, "desconocida": 1, "mala": 2}

# Features
FEATURES = [
    "num_links", "num_suspicious_words", "domain_reputation_num",
    "all_caps_ratio", "num_recipients",
    "has_attachment", "spf_pass", "dkim_pass",
    "from_replyto_match", "send_frequency"
]

# Búsqueda de hiperparámetros
PARAM_GRID = {
    "classifier__C": [0.01, 0.1, 1, 10],
    "classifier__penalty": ["l2"],
    "classifier__class_weight": [None, "balanced"]
}
CV = 5
SCORING = "f1_macro"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# 2. Cargar y preparar datos
# ------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

if "domain_reputation_num" not in df.columns:
    df["domain_reputation_num"] = df["domain_reputation"].map(MAPPING)

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    raise RuntimeError(f"Faltan columnas en el CSV: {missing}")

X_full = df[FEATURES].copy()
y_full = df[TARGET].copy()

# ------------------------------------------------------------
# 3. Pipeline
# ------------------------------------------------------------
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), FEATURES)
])

clf = LogisticRegression(solver="liblinear", max_iter=2000)

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", clf)
])

grid = GridSearchCV(pipe, PARAM_GRID, cv=CV, scoring=SCORING, n_jobs=-1)

# ------------------------------------------------------------
# 4. Ejecución de experimentos
# ------------------------------------------------------------
results = []
models_dir = os.path.join(OUTPUT_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

for run_i, seed in enumerate(RANDOM_SEEDS, start=1):
    print(f"\n=== RUN {run_i}/{N_RUNS} (seed={seed}) ===")

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=TEST_SIZE, random_state=seed, stratify=y_full
    )

    t0 = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - t0

    best = grid.best_estimator_
    best_params = grid.best_params_
    cv_best_score = grid.best_score_

    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_spam = f1_score(y_test.map({"ham": 0, "spam": 1}),
                       best.predict_proba(X_test)[:, 1] >= 0.5)
    f1_macro = f1_score(y_test, y_pred, average="macro", labels=["ham", "spam"])

    classifier = best.named_steps["classifier"]
    coefs = classifier.coef_[0]
    intercept = classifier.intercept_[0]
    coef_map = dict(zip(FEATURES, coefs))

    model_name = f"model_run{run_i}_seed{seed}.joblib"
    joblib.dump(best, os.path.join(models_dir, model_name))

    result = {
        "run": run_i,
        "seed": seed,
        "time_s": round(elapsed, 2),
        "best_params": best_params,
        "cv_best_score": cv_best_score,
        "test_accuracy": acc,
        "test_f1_spam": f1_spam,
        "test_f1_macro": f1_macro,
        "intercept": float(intercept),
        "coefs": coef_map,
        "y_test": y_test,   # guardamos para la matriz de confusión
        "y_pred": y_pred
    }
    results.append(result)
    print(f"seed={seed} acc={acc:.4f} f1_spam={f1_spam:.4f}")

# ------------------------------------------------------------
# 5. Guardar resultados
# ------------------------------------------------------------
results_df_rows = []
for r in results:
    row = {
        "run": r["run"],
        "seed": r["seed"],
        "time_s": r["time_s"],
        "cv_best_score": r["cv_best_score"],
        "test_accuracy": r["test_accuracy"],
        "test_f1_spam": r["test_f1_spam"],
        "test_f1_macro": r["test_f1_macro"],
        "intercept": r["intercept"]
    }
    for f in FEATURES:
        row[f"coef__{f}"] = r["coefs"].get(f, np.nan)
    results_df_rows.append(row)

results_df = pd.DataFrame(results_df_rows)
results_df.to_csv(os.path.join(OUTPUT_DIR, "experiments_results.csv"), index=False)

# ------------------------------------------------------------
# 6. Visualización y matriz de confusión del mejor modelo
# ------------------------------------------------------------
best_idx = results_df["test_f1_spam"].idxmax()
best_row = results_df.loc[best_idx]
best_run = int(best_row["run"])
best_seed = int(best_row["seed"])
print(f"\nMejor run = {best_run} (seed={best_seed}) con F1={best_row['test_f1_spam']:.4f}")

best_result = results[best_run - 1]
coef_series = pd.Series(best_result["coefs"]).sort_values(ascending=True)

# Gráfico coeficientes
plt.figure(figsize=(8, 5))
coef_series.plot(kind="barh", color=["red" if v > 0 else "blue" for v in coef_series.values])
plt.xlabel("Coeficiente (log-odds)")
plt.title(f"Coeficientes del mejor modelo (run {best_run})")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "best_run_coefs.png"))
plt.close()

# Matriz de confusión
cm = confusion_matrix(best_result["y_test"], best_result["y_pred"], labels=["ham", "spam"])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title(f"Matriz de confusión (mejor modelo run {best_run})")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "best_run_confusion_matrix.png"))
plt.close()

# ------------------------------------------------------------
# 7. Reporte en Markdown
# ------------------------------------------------------------
md_path = os.path.join(OUTPUT_DIR, "report.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write(f"# Informe de variabilidad - Regresión logística\n")
    f.write(f"**Autoras:** María Camila Guzmán Bello y Ana María Casallas Ariza\n")
    f.write(f"**Materia:** Machine Learning 802\n")
    f.write(f"**Fecha:** {datetime.now().isoformat()}\n\n")

    f.write("## Resumen ejecutivo\n")
    f.write(f"- Dataset usado: `{CSV_PATH}`\n")
    f.write(f"- Corridas ejecutadas: {N_RUNS}\n")
    f.write(f"- Test size: {TEST_SIZE}\n")
    f.write(f"- GridSearchCV (cv={CV}, scoring={SCORING})\n\n")

    f.write("## Resultados por corrida\n")
    f.write(results_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Estadísticas globales\n")
    f.write(f"- F1 (spam) media: {results_df['test_f1_spam'].mean():.4f}\n")
    f.write(f"- F1 (spam) STD: {results_df['test_f1_spam'].std():.4f}\n")
    f.write(f"- Accuracy media: {results_df['test_accuracy'].mean():.4f}\n\n")

    f.write("## Mejor corrida (por F1 spam)\n")
    f.write(f"- Run: {best_run}\n")
    f.write(f"- Seed: {best_seed}\n")
    f.write(f"- Test F1 (spam): {best_row['test_f1_spam']:.4f}\n")
    f.write(f"- Test Accuracy: {best_row['test_accuracy']:.4f}\n")
    f.write(f"- Best params: `{best_result['best_params']}`\n\n")

    f.write("### Ecuación logística (mejor modelo)\n")
    intercept = best_result["intercept"]
    f.write(f"`logit(p) = {intercept:.4f} ")
    for f_name, coef in best_result["coefs"].items():
        sign = "+" if coef >= 0 else "-"
        f.write(f"{sign} {abs(coef):.4f}*{f_name} ")
    f.write("`\n\n")

    f.write("### Gráfica de coeficientes\n")
    f.write(f"![Coeficientes mejor corrida](best_run_coefs.png)\n\n")

    f.write("### Matriz de confusión del mejor modelo\n")
    f.write(f"![Matriz de confusión mejor corrida](best_run_confusion_matrix.png)\n\n")

print(f"Reporte Markdown generado en: {md_path}")
