# Tratamiento de datos
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd

# Otros objetivos
# -----------------------------------------------------------------------
import math

# Gráficos
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px



# Métodos estadísticos
# -----------------------------------------------------------------------
from scipy.stats import zscore # para calcular el z-score
from sklearn.neighbors import LocalOutlierFactor # para detectar outliers usando el método LOF
from sklearn.ensemble import IsolationForest # para detectar outliers usando el metodo IF
from sklearn.neighbors import NearestNeighbors # para calcular la epsilon

# Para generar combinaciones de listas
# -----------------------------------------------------------------------
from itertools import product, combinations

# Gestionar warnings
# -----------------------------------------------------------------------
import warnings
warnings.filterwarnings('ignore')

def describe_outliers(dataframe: pd.DataFrame, k=1.5, ordenados = True):

    diccionario_outliers = []
   
    columnas_numericas = dataframe.select_dtypes(np.number).columns

    for columna in columnas_numericas:
        Q1, Q3 = np.nanpercentile(dataframe[columna], (25,75))
        IQR = Q3 - Q1

        limite_inferior = Q1 - (IQR * k)
        limite_superior = Q3 + (IQR * k)

        condicion_inferior = dataframe[columna] < limite_inferior
        condicion_superior = dataframe[columna] > limite_superior

        df_outliers = dataframe[condicion_inferior | condicion_superior]
        
        diccionario_outliers.append({
            'columna': columna,
            'n_outliers': df_outliers.shape[0],
            'limite_inf': limite_inferior,
            'limite_sup': limite_superior,
            '%_outliers': round((df_outliers.shape[0] / dataframe.shape[0]) * 100, 2)
        })

    resultado = pd.DataFrame(diccionario_outliers).sort_values(by='n_outliers', ascending=False) if ordenados == True else pd.DataFrame(diccionario_outliers)
        
    display(resultado)

class GestionOutliersUnivariados:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def _separar_variables_tipo(self):
        """
        Divide el DataFrame en columnas numéricas y de tipo objeto.
        """
        return self.dataframe.select_dtypes(include=np.number), self.dataframe.select_dtypes(include="O")



    def visualizar_outliers_univariados(self, color="blue", whis=1.5, tamano_grafica=(10, 15)):
        """
        Visualiza los outliers univariados mediante boxplots o histogramas.

        Parámetros:
        -----------
        color (str): Color de los gráficos.
        whis (float): Valor para definir el límite de los bigotes en los boxplots.
        tamano_grafica (tuple): Tamaño de la figura.
        """
        tipo_grafica = input("Qué gráfica quieres usar, Histograma (H) o Boxplot(B): ").upper()
        
        num_cols = len(self._separar_variables_tipo()[0].columns)
        num_filas = math.ceil(num_cols / 2)

        _, axes = plt.subplots(num_filas, 2, figsize=tamano_grafica)
        axes = axes.flat

        for indice, columna in enumerate(self.dataframe.select_dtypes(include=np.number).columns):
            if tipo_grafica == "B":
                sns.boxplot(x=columna, data=self.dataframe, color=color, ax=axes[indice], whis=whis,
                            flierprops={'markersize': 4, 'markerfacecolor': 'orange'})
            elif tipo_grafica == "H":
                sns.histplot(x=columna, data=self.dataframe, color=color, ax=axes[indice], bins=50)
            
            axes[indice].set_title(columna)
            axes[indice].set(xlabel=None)

        plt.tight_layout()
        plt.show()


class GestionOutliersMultivariados:

    def __init__(self, dataframe, contaminacion = [0.01, 0.05, 0.1, 0.15]):
        self.dataframe = dataframe
        self.contaminacion = contaminacion

    def separar_variables_tipo(self):
        """
        Divide el DataFrame en columnas numéricas y de tipo objeto.
        """
        return self.dataframe.select_dtypes(include=np.number), self.dataframe.select_dtypes(include="O")

    def visualizar_outliers_bivariados(self, vr, tamano_grafica = (10, 15)):

        num_cols = len(self.separar_variables_tipo()[0].columns)
        num_filas = math.ceil(num_cols / 2)

        fig, axes = plt.subplots(num_filas, 2, figsize=tamano_grafica)
        axes = axes.flat

        for indice, columna in enumerate(self.separar_variables_tipo()[0].columns):
            if columna == vr:
                fig.delaxes(axes[indice])
        
            else:
                sns.scatterplot(x = vr, 
                                y = columna, 
                                data = self.dataframe,
                                ax = axes[indice])
            
            axes[indice].set_title(columna)
            axes[indice].set(xlabel=None, ylabel = None)
        fig.delaxes(axes[-1])
        plt.tight_layout()

    def explorar_outliers_if(self, var_dependiente, indice_contaminacion=[0.01, 0.05, 0.1], estimadores=1000, colores={-1: "red", 1: "grey"}):
        """
        Detecta outliers en un DataFrame utilizando el algoritmo Isolation Forest y visualiza los resultados.

        Params:
            - var_dependiente : str. Nombre de la variable dependiente que se usará en los gráficos de dispersión.
        
            - indice_contaminacion : list of float, opcional. Lista de valores de contaminación a usar en el algoritmo Isolation Forest. La contaminación representa
            la proporción de outliers esperados en el conjunto de datos. Por defecto es [0.01, 0.05, 0.1].
        
            - estimadores : int, opcional. Número de estimadores (árboles) a utilizar en el algoritmo Isolation Forest. Por defecto es 1000.
        
            - colores : dict, opcional. Diccionario que asigna colores a los valores de predicción de outliers del algoritmo Isolation Forest.
            Por defecto, los outliers se muestran en rojo y los inliers en gris ({-1: "red", 1: "grey"}).
        
        Returns:
            Esta función no retorna ningún valor, pero crea y muestra gráficos de dispersión que visualizan los outliers
        detectados para cada valor de contaminación especificado.
        """
    
        df_if = self.dataframe.copy()

        col_numericas = self.separar_variables_tipo()[0].columns.to_list()

        num_filas = math.ceil(len(col_numericas) / 2)

        for contaminacion in indice_contaminacion: 
            
            ifo = IsolationForest(random_state=42, 
                                n_estimators=estimadores, 
                                contamination=contaminacion,
                                max_samples="auto",  
                                n_jobs=-1)
            ifo.fit(self.dataframe[col_numericas])
            prediccion_ifo = ifo.predict(self.dataframe[col_numericas])
            df_if["outlier"] = prediccion_ifo

            fig, axes = plt.subplots(num_filas, 2, figsize=(20, 15))
            axes = axes.flat
            for indice, columna in enumerate(col_numericas):
                if columna == var_dependiente:
                    fig.delaxes(axes[indice])

                else:
                    # Visualizar los outliers en un gráfico
                    sns.scatterplot(x=var_dependiente, 
                                    y=columna, 
                                    data=df_if,
                                    hue="outlier", 
                                    palette=colores, 
                                    style="outlier", 
                                    size=2,
                                    ax=axes[indice])
                    
                    axes[indice].set_title(f"Contaminación = {contaminacion} y columna {columna.upper()}")
                    plt.tight_layout()
                
                        
            if len(col_numericas) % 2 != 0:
                fig.delaxes(axes[-1])
    
    def detectar_outliers_if(self,  contaminacion, n_estimators=1000):
        """
        Detecta outliers en un DataFrame utilizando el algoritmo Isolation Forest.
        """
        df_if = self.dataframe.copy()
        col_numericas = self.separar_variables_tipo()[0].columns.to_list()

        ifo = IsolationForest(random_state=42, n_estimators=n_estimators, contamination=contaminacion, max_samples="auto", n_jobs=-1)
        ifo.fit(self.dataframe[col_numericas])
        prediccion_ifo = ifo.predict(self.dataframe[col_numericas])
        df_if["outlier"] = prediccion_ifo

        return df_if

    def imputar_outliers(self, data, metodo='media'):
        """
        Imputa los valores outliers en las columnas numéricas según el método especificado.
        
        Params:
            - data: DataFrame con los datos incluyendo la columna 'outlier'.
            - metodo: str, método de imputación ('media', 'mediana', 'moda').
        
        Returns:
            - DataFrame con los valores outliers imputados.
        """

        col_numericas = self.separar_variables_tipo()[0].columns.to_list()

        # Diccionario de métodos de imputación
        metodos_imputacion = {
            'media': lambda x: x.mean(),
            'mediana': lambda x: x.median(),
            'moda': lambda x: x.mode()[0]
        }

        if metodo not in metodos_imputacion:
            raise ValueError("Método de imputación no reconocido. Utilice 'media', 'mediana' o 'moda'.")

        for col in col_numericas:
            valor_imputacion = metodos_imputacion[metodo](data.loc[data['outlier'] != -1, col])
            data.loc[data['outlier'] == -1, col] = valor_imputacion
        
        return data.drop("outlier", axis = 1)    
