# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np

import time
import psutil

# Visualizaciones
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree

# Para realizar la clasificación y la evaluación del modelo
# -----------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, cross_val_score, StratifiedKFold, KFold
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    confusion_matrix
)
import shap
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

# Para realizar cross validation
# -----------------------------------------------------------------------
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.preprocessing import KBinsDiscretizer


class AnalisisModelosClasificacion:
    def __init__(self, dataframe, variable_dependiente, random_state=42):
        self.dataframe = dataframe
        self.variable_dependiente = variable_dependiente
        self.random_state = random_state
        self.X = dataframe.drop(variable_dependiente, axis=1)
        self.y = dataframe[variable_dependiente]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, train_size=0.8, random_state=self.random_state, shuffle=True
        )

        # Diccionario de modelos y resultados
        self.modelos = {
            "logistic_regression": LogisticRegression(random_state=self.random_state),
            "tree": DecisionTreeClassifier(random_state=self.random_state),
            "random_forest": RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
            "gradient_boosting": GradientBoostingClassifier(random_state=self.random_state),
            "xgboost": xgb.XGBClassifier(random_state=self.random_state)
        }
        self.resultados = {nombre: {"mejor_modelo": None, "pred_train": None, "pred_test": None} for nombre in self.modelos}

    def ajustar_modelo(self, modelo_nombre, param_grid=None, random_state=42, devolver_objeto=False, entrenamiento_final=False):
        """
        Ajusta el modelo seleccionado con GridSearchCV.
        Si entrenamiento_final=True, el modelo se entrena con todo el conjunto de datos (X, y).
        """
        if modelo_nombre not in self.modelos:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")
        
        modelo = self.modelos[modelo_nombre]

        # Parámetros predeterminados por modelo
        parametros_default = {
            "logistic_regression": {
                'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'saga'],
                'max_iter': [100, 200, 500]
            },
            "tree": {
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            "random_forest": {
                'n_estimators': [15, 25, 50],
                'max_depth': [8, 10, 12, 15],
                'min_samples_split': [1, 2, 5],
                'min_samples_leaf': [1, 2, 4]
            },
            "gradient_boosting": {
                'n_estimators': [100, 150],
                'learning_rate': [0.02, 0.2],
                'max_depth': [3, 4],
                'min_samples_split': [5, 10, 15],
                'min_samples_leaf': [2, 5, 10],
                'subsample': [0.8, 1.0]
            },
            "xgboost": {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [5, 8, 10],
                'min_child_weight': [1, 3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        }
        if param_grid is None:
            param_grid = parametros_default.get(modelo_nombre, {})

        # Decidir los datos a usar según el parámetro `entrenamiento_final`
        if entrenamiento_final:
            print("\n **** Se está entrenando al modelo con TODO el conjunto de datos **** \n")
            X_datos = self.X
            y_datos = self.y
        else:
            X_datos = self.X_train
            y_datos = self.y_train

        if modelo_nombre == "logistic_regression":
            modelo_logistica = LogisticRegression(random_state=self.random_state)
            modelo_logistica.fit(X_datos, y_datos)

            if not entrenamiento_final:
                self.resultados[modelo_nombre]["pred_train"] = modelo_logistica.predict(self.X_train)
                self.resultados[modelo_nombre]["pred_test"] = modelo_logistica.predict(self.X_test)
            else:
                self.resultados[modelo_nombre]["pred_train"] = modelo_logistica.predict(self.X)
            self.resultados[modelo_nombre]["mejor_modelo"] = modelo_logistica

            if devolver_objeto:
                return modelo_logistica
            
        else:
            # Ajuste del modelo con GridSearchCV
            grid_search = GridSearchCV(estimator=modelo, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
            grid_search.fit(X_datos, y_datos)
            print(f"El mejor modelo es {grid_search.best_estimator_}")
            self.resultados[modelo_nombre]["mejor_modelo"] = grid_search.best_estimator_

            if not entrenamiento_final:
                self.resultados[modelo_nombre]["pred_train"] = grid_search.best_estimator_.predict(self.X_train)
                self.resultados[modelo_nombre]["pred_test"] = grid_search.best_estimator_.predict(self.X_test)
            else:
                self.resultados[modelo_nombre]["pred_train"] = grid_search.best_estimator_.predict(self.X)

            if devolver_objeto:
                return grid_search.best_estimator_


    def calcular_metricas(self, modelo_nombre):
        """
        Calcula métricas de rendimiento para el modelo seleccionado, incluyendo AUC, Kappa,
        tiempo de computación y núcleos utilizados.
        """
        if modelo_nombre not in self.resultados:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")
        
        pred_train = self.resultados[modelo_nombre]["pred_train"]
        pred_test = self.resultados[modelo_nombre]["pred_test"]

        if pred_train is None or pred_test is None:
            raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de calcular métricas.")
        
        modelo = self.resultados[modelo_nombre]["mejor_modelo"]

        # Registrar tiempo de ejecución
        start_time = time.time()
        if hasattr(modelo, "predict_proba"):
            prob_train = modelo.predict_proba(self.X_train)[:, 1]
            prob_test = modelo.predict_proba(self.X_test)[:, 1]
        else:
            prob_train = prob_test = None
        elapsed_time = time.time() - start_time

        # Registrar núcleos utilizados
        num_nucleos = getattr(modelo, "n_jobs", psutil.cpu_count(logical=True))

        # Métricas para conjunto de entrenamiento
        metricas_train = {
            "accuracy": accuracy_score(self.y_train, pred_train),
            "precision": precision_score(self.y_train, pred_train, average='weighted', zero_division=0),
            "recall": recall_score(self.y_train, pred_train, average='weighted', zero_division=0),
            "f1": f1_score(self.y_train, pred_train, average='weighted', zero_division=0),
            "kappa": cohen_kappa_score(self.y_train, pred_train),
            "auc": roc_auc_score(self.y_train, prob_train) if prob_train is not None else None,
            "time_seconds": elapsed_time,
            "n_jobs": num_nucleos
        }

        # Métricas para conjunto de prueba
        metricas_test = {
            "accuracy": accuracy_score(self.y_test, pred_test),
            "precision": precision_score(self.y_test, pred_test, average='weighted', zero_division=0),
            "recall": recall_score(self.y_test, pred_test, average='weighted', zero_division=0),
            "f1": f1_score(self.y_test, pred_test, average='weighted', zero_division=0),
            "kappa": cohen_kappa_score(self.y_test, pred_test),
            "auc": roc_auc_score(self.y_test, prob_test) if prob_test is not None else None,
            "tiempo_computacion(segundos)": elapsed_time,
            "nucleos_usados": num_nucleos
        }

        # Combinar métricas en un DataFrame
        return pd.DataFrame({"train": metricas_train, "test": metricas_test}).T

    def plot_matriz_confusion(self, modelo_nombre):
        """
        Plotea la matriz de confusión para el modelo seleccionado.
        """
        if modelo_nombre not in self.resultados:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")

        pred_test = self.resultados[modelo_nombre]["pred_test"]

        if pred_test is None:
            raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de calcular la matriz de confusión.")

        # Matriz de confusión
        matriz_conf = confusion_matrix(self.y_test, pred_test)
        plt.figure(figsize=(8, 6))
        sns.heatmap(matriz_conf, annot=True, fmt='g', cmap='Blues')
        plt.title(f"Matriz de Confusión ({modelo_nombre})")
        plt.xlabel("Predicción")
        plt.ylabel("Valor Real")
        plt.show()
    
    def importancia_predictores(self, modelo_nombre):
        """
        Calcula y grafica la importancia de las características para el modelo seleccionado.
        """
        if modelo_nombre not in self.resultados:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")
        
        modelo = self.resultados[modelo_nombre]["mejor_modelo"]
        if modelo is None:
            raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de calcular importancia de características.")
        
        # Verificar si el modelo tiene importancia de características
        if hasattr(modelo, "feature_importances_"):
            importancia = modelo.feature_importances_
        elif modelo_nombre == "logistic_regression" and hasattr(modelo, "coef_"):
            importancia = modelo.coef_[0]
        else:
            print(f"El modelo '{modelo_nombre}' no soporta la importancia de características.")
            return
        
        # Crear DataFrame y graficar
        importancia_df = pd.DataFrame({
            "Feature": self.X.columns,
            "Importance": importancia
        }).sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(10, 8))  # Aumentar el tamaño de la figura para más espacio
        sns.barplot(x="Importance", y="Feature", data=importancia_df, palette="viridis")
        plt.title(f"Importancia de Características ({modelo_nombre})", fontsize=14)
        plt.xlabel("Importancia", fontsize=12)
        plt.ylabel("Características", fontsize=12)

        # Mejorar legibilidad de las etiquetas
        plt.yticks(fontsize=10, rotation=0, ha="right")  # Reducir tamaño, evitar rotación y alinear a la derecha
        plt.tight_layout()  # Ajustar márgenes automáticamente para evitar solapamiento
        plt.show()

    def plot_shap_summary(self, modelo_nombre):
        """
        Genera un SHAP summary plot para el modelo seleccionado.
        Maneja correctamente modelos de clasificación con múltiples clases.
        """
        if modelo_nombre not in self.resultados:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")

        modelo = self.resultados[modelo_nombre]["mejor_modelo"]

        if modelo is None:
            raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de generar el SHAP plot.")

        # Usar TreeExplainer para modelos basados en árboles
        if modelo_nombre in ["tree", "random_forest", "gradient_boosting", "xgboost"]:
            explainer = shap.TreeExplainer(modelo)
            shap_values = explainer.shap_values(self.X_test)

            # Verificar si los SHAP values tienen múltiples clases (dimensión 3)
            if isinstance(shap_values, list):
                # Para modelos binarios, seleccionar SHAP values de la clase positiva
                shap_values = shap_values[1]
            elif len(shap_values.shape) == 3:
                # Para Decision Trees, seleccionar SHAP values de la clase positiva
                shap_values = shap_values[:, :, 1]
        else:
            # Usar el explicador genérico para otros modelos
            explainer = shap.Explainer(modelo, self.X_test, check_additivity=False)
            shap_values = explainer(self.X_test).values

        # Generar el summary plot estándar
        shap.summary_plot(shap_values, self.X_test, feature_names=self.X.columns)

    def plot_curva_roc(self, modelo_nombre):
        """
        Genera y visualiza la curva ROC para el modelo seleccionado.
        """
        if modelo_nombre not in self.resultados:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")

        modelo = self.resultados[modelo_nombre]["mejor_modelo"]
        if modelo is None:
            raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de visualizar la curva ROC.")
        
        if not hasattr(modelo, "predict_proba"):
            raise ValueError(f"El modelo '{modelo_nombre}' no soporta la predicción de probabilidades.")
        
        prob_test = modelo.predict_proba(self.X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(self.y_test, prob_test)
        auc_value = roc_auc_score(self.y_test, prob_test)
        
        plt.figure(figsize=(8, 6))
        sns.lineplot(x=fpr, y=tpr, color="red", label="Modelo")
        sns.lineplot(x=[0, 1], y=[0, 1], color="grey", linestyle="--", label="Aleatorio")
        plt.title(f"Curva ROC - {modelo_nombre}")
        plt.xlabel("Tasa Falsos Positivos")
        plt.ylabel("Tasa Verdaderos Positivos")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

# Función para asignar colores
def color_filas_por_modelo(row):
    if row["modelo"] == "decision tree":
        return ["background-color: #e6b3e0; color: black"] * len(row)  
    
    elif row["modelo"] == "random_forest":
        return ["background-color: #c2f0c2; color: black"] * len(row) 

    elif row["modelo"] == "gradient_boosting":
        return ["background-color: #ffd9b3; color: black"] * len(row)  

    elif row["modelo"] == "xgboost":
        return ["background-color: #f7b3c2; color: black"] * len(row)  

    elif row["modelo"] == "regresion lineal":
        return ["background-color: #b3d1ff; color: black"] * len(row)  
    
    return ["color: black"] * len(row)

    
