
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

# Métodos estadísticos
# -----------------------------------------------------------------------
from scipy.stats import zscore  # para calcular el z-score
from sklearn.neighbors import LocalOutlierFactor  # para detectar outliers usando el método LOF
from sklearn.ensemble import IsolationForest  # para detectar outliers usando el método IF
from sklearn.neighbors import NearestNeighbors  # para calcular la epsilon
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler, RobustScaler
from scipy.stats import chi2_contingency

def exploracion_datos(dataframe):

    """
    Realiza una exploración básica de los datos en el DataFrame dado e imprime varias estadísticas descriptivas.

    Parameters:
    -----------
    dataframe : pandas DataFrame. El DataFrame que se va a explorar.

    Returns:
    --------
    None

    Imprime varias estadísticas descriptivas sobre el DataFrame, incluyendo:
    - El número de filas y columnas en el DataFrame.
    - El número de valores duplicados en el DataFrame.
    - Una tabla que muestra las columnas con valores nulos y sus porcentajes.
    - Las principales estadísticas de las variables numéricas en el DataFrame.
    - Las principales estadísticas de las variables categóricas en el DataFrame.

    """

    print(f"El número de filas es {dataframe.shape[0]} y el número de columnas es {dataframe.shape[1]}")

    print("\n----------\n")

    print(f"En este conjunto de datos tenemos {dataframe.duplicated().sum()} valores duplicados")

    
    print("\n----------\n")


    print("Los columnas con valores nulos y sus porcentajes son: ")
    dataframe_nulos = dataframe.isnull().sum()

    display((dataframe_nulos[dataframe_nulos.values >0] / dataframe.shape[0]) * 100)

    print("\n----------\n")
    print("Las principales estadísticas de las variables númericas son:")
    display(dataframe.describe().T)

    print("\n----------\n")
    print("Las principales estadísticas de las variables categóricas son:")
    display(dataframe.describe(include = "O").T)

    print("\n----------\n")
    print("Las características principales del dataframe son:")
    display(dataframe.info())

class Visualizador:
    """
    Clase para visualizar la distribución de variables numéricas y categóricas de un DataFrame.

    Attributes:
    - dataframe (pandas.DataFrame): El DataFrame que contiene las variables a visualizar.

    Methods:
    - __init__: Inicializa el VisualizadorDistribucion con un DataFrame y un color opcional para las gráficas.
    - separar_dataframes: Separa el DataFrame en dos subconjuntos, uno para variables numéricas y otro para variables categóricas.
    - plot_numericas: Grafica la distribución de las variables numéricas del DataFrame.
    - plot_categoricas: Grafica la distribución de las variables categóricas del DataFrame.
    - plot_relacion2: Visualiza la relación entre una variable y todas las demás, incluyendo variables numéricas y categóricas.
    """

    def __init__(self, dataframe):
        """
        Inicializa el VisualizadorDistribucion con un DataFrame y un color opcional para las gráficas.

        Parameters:
        - dataframe (pandas.DataFrame): El DataFrame que contiene las variables a visualizar.
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        """
        self.dataframe = dataframe

    def separar_dataframes(self):
        """
        Separa el DataFrame en dos subconjuntos, uno para variables numéricas y otro para variables categóricas.

        Returns:
        - pandas.DataFrame: DataFrame con variables numéricas.
        - pandas.DataFrame: DataFrame con variables categóricas.
        """
        return self.dataframe.select_dtypes(include=np.number), self.dataframe.select_dtypes(include="O")
    
    def plot_numericas(self, color="grey", tamano_grafica=(15, 5)):
        """
        Grafica la distribución de las variables numéricas del DataFrame.

        Parameters:
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        - tamaño_grafica (tuple, opcional): El tamaño de la figura de la gráfica. Por defecto es (15, 5).
        """
        lista_num = self.separar_dataframes()[0].columns
        fig, axes = plt.subplots(nrows = 2, ncols = math.ceil(len(lista_num)/2), figsize=tamano_grafica, sharey=True)
        axes = axes.flat
        for indice, columna in enumerate(lista_num):
            sns.histplot(x=columna, data=self.dataframe, ax=axes[indice], color=color, bins=20)
        fig.delaxes(axes[-1])
        plt.suptitle("Distribución de variables numéricas")
        plt.tight_layout();

    '''
    def plot_categoricas(self, color="grey", tamano_grafica=(40, 15)):
        """
        Grafica la distribución de las variables categóricas del DataFrame.

        Parameters:
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        - tamaño_grafica (tuple, opcional): El tamaño de la figura de la gráfica. Por defecto es (40, 10).
        """
        dataframe_cat = self.separar_dataframes()[1]
        _, axes = plt.subplots(2, math.ceil(len(dataframe_cat.columns) / 2), figsize=tamano_grafica)
        axes = axes.flat
        for indice, columna in enumerate(dataframe_cat.columns):
            sns.countplot(x=columna, data=self.dataframe, order=self.dataframe[columna].value_counts().index,
                          ax=axes[indice], color=color)
            axes[indice].tick_params(rotation=90)
            axes[indice].set_title(columna)
            axes[indice].set(xlabel=None)

        plt.tight_layout()
        plt.suptitle("Distribución de variables categóricas")

    '''
    def plot_categoricas(self, color="grey", tamano_grafica=(20, 15), espaciado_filas=2):
        """
        Grafica la distribución de las variables categóricas del DataFrame.

        Parameters:
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        - tamano_grafica (tuple, opcional): El tamaño de la figura de la gráfica. Por defecto es (20,15)
        - espaciado_filas (float, opcional): Espaciado entre filas de gráficas. Por defecto es 1.5.
        """
        dataframe_cat = self.separar_dataframes()[1]
        
        # Cambiar las dimensiones de los subplots: 2 gráficas por fila
        filas = math.ceil(len(dataframe_cat.columns) / 2)
        _, axes = plt.subplots(filas, 2, figsize=tamano_grafica)
        axes = axes.flat
        
        for indice, columna in enumerate(dataframe_cat.columns):
            sns.countplot(
                x=columna, 
                data=self.dataframe, 
                order=self.dataframe[columna].value_counts().index, 
                ax=axes[indice], 
                color=color
            )
            axes[indice].tick_params(rotation=90)
            axes[indice].set_title(columna)
            axes[indice].set(xlabel=None)
        
        # Ocultar ejes sobrantes si hay menos columnas que subplots
        for i in range(len(dataframe_cat.columns), len(axes)):
            axes[i].axis("off")
        
        # Ajustar el espacio entre filas de las gráficas
        plt.subplots_adjust(hspace=espaciado_filas)
        
        plt.tight_layout()
        plt.suptitle("Distribución de variables categóricas", y=1.02)


    
    def relacion_numericas(self, variable_dependiente, tamanio=(15,8), paleta="mako"):
        df_numericas= self.separar_dataframes()[0]
        cols_numericas=df_numericas.columns
        nfilas=math.ceil(len(cols_numericas) /2) 
        fig, axes = plt.subplots(nrows= nfilas, ncols= 2, figsize= tamanio)
        axes=axes.flat

        for indice, col in enumerate(cols_numericas):
            if col == variable_dependiente:
                fig.delaxes(axes[indice])
            else:
                sns.scatterplot(x=col ,y=variable_dependiente, data=df_numericas, palette=paleta, ax=axes[indice])

                axes[indice].set_title(f"Relación entre {col} y {variable_dependiente}")
                axes[indice].set_xlabel("")

        fig.delaxes(axes[-1])
        plt.tight_layout()

    def plot_relacion(self, vr, tamano_grafica=(20, 10)):

        lista_num = self.separar_dataframes()[0].columns
        lista_cat = self.separar_dataframes()[1].columns

        fig, axes = plt.subplots(ncols = 2, nrows = math.ceil(len(self.dataframe.columns) / 2), figsize=tamano_grafica)
        axes = axes.flat

        for indice, columna in enumerate(self.dataframe.columns):
            if columna == vr:
                fig.delaxes(axes[indice])
            elif columna in lista_num:
                sns.histplot(x = columna, 
                             hue = vr, 
                             data = self.dataframe, 
                             ax = axes[indice], 
                             palette = "mako", 
                             legend = False)
                
            elif columna in lista_cat:
                sns.countplot(x = columna, 
                              hue = vr, 
                              data = self.dataframe, 
                              ax = axes[indice], 
                              palette = "mako"
                              )

            axes[indice].set_title(f"Relación {columna} vs {vr}",size=25)   

        plt.tight_layout()
    
    def deteccion_outliers(self, color = "grey"):

        """
        Detecta y visualiza valores atípicos en un DataFrame.

        Params:
            - dataframe (pandas.DataFrame):  El DataFrame que se va a usar

        Returns:
            No devuelve nada

        Esta función selecciona las columnas numéricas del DataFrame dado y crea un diagrama de caja para cada una de ellas para visualizar los valores atípicos.
        """

        lista_num = self.separar_dataframes()[0].columns

        fig, axes = plt.subplots(2, ncols = math.ceil(len(lista_num)/2), figsize=(15,5))
        axes = axes.flat

        for indice, columna in enumerate(lista_num):
            sns.boxplot(x=columna, data=self.dataframe, 
                        ax=axes[indice], 
                        color=color, 
                        flierprops={'markersize': 4, 'markerfacecolor': 'orange'})

        if len(lista_num) % 2 != 0:
            fig.delaxes(axes[-1])

        
        plt.tight_layout()

    def correlacion(self, tamano_grafica = (7, 5)):

        """
        Visualiza la matriz de correlación de un DataFrame utilizando un mapa de calor.

        Params:
            - dataframe : pandas DataFrame. El DataFrame que contiene los datos para calcular la correlación.

        Returns:
        No devuelve nada

        Muestra un mapa de calor de la matriz de correlación.

        - Utiliza la función `heatmap` de Seaborn para visualizar la matriz de correlación.
        - La matriz de correlación se calcula solo para las variables numéricas del DataFrame.
        - La mitad inferior del mapa de calor está oculta para una mejor visualización.
        - Permite guardar la imagen del mapa de calor como un archivo .png si se solicita.

        """

        plt.figure(figsize = tamano_grafica )

        mask = np.triu(np.ones_like(self.dataframe.corr(numeric_only=True), dtype = np.bool_))

        sns.heatmap(data = self.dataframe.corr(numeric_only = True), 
                    annot = True, 
                    vmin=-1,
                    vmax=1,
                    cmap="viridis",
                    linecolor="black", 
                    fmt='.1g', 
                    mask = mask)
    

def escalar_datos(data, cols, metodo="robust"):
    """
    Escala los datos de las columnas seleccionadas utilizando diferentes métodos de escalado.

    Parameters:
        data (pd.DataFrame): DataFrame con datos.
        cols (list): Lista de nombres de las columnas a escalar.
        metodo (str): Método de escalado ("minmax", "robust", "standard", "norm"). Default "robust".
    
    Returns:
        pd.DataFrame, object: DataFrame escalado y el scaler utilizado.
    """
    if metodo == "minmax":
        scaler = MinMaxScaler()
    elif metodo == "robust":
        scaler = RobustScaler()
    elif metodo == "standard":
        scaler = StandardScaler()
    elif metodo == "norm":
        return normalize_scaler(data[cols]), None
    
    df_scaled = pd.DataFrame(scaler.fit_transform(data[cols]), columns=cols, index=data.index)
    return df_scaled, scaler


def detectar_orden_var_cat(df, lista_cat, var_respuesta):
    for categorica in lista_cat:
        print(f"Evaluando la variable {categorica.upper()}")
        # Creamos una tabla de contingencia entre cada variable categorica y la variable respuesta
        df_cross_tab = pd.crosstab(df[categorica], df[var_respuesta])
        display(df_cross_tab)
        
        chi2, p, dof, expected = chi2_contingency(df_cross_tab)

        if p <0.05:
            print(f"La variable {categorica} tiene orden.")
        else:
            print(f"La variable {categorica} NO tiene orden.")
        print("_________________________ \n")
