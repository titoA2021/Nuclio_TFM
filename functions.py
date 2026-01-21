# Librerías estándar
import os
import warnings

# Manipulación de datos
import pandas as pd
import numpy as np

# Visualización de datos
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Analisis regex
import re

# Análisis de nulos
import missingno as msno

# Estadística
import scipy.stats as stats

# Configuración de warnings
warnings.filterwarnings('ignore')

def leer_archivo(ruta_completa):
    try:
        _, extension = os.path.splitext(ruta_completa.lower())

        if extension == '.csv':
            df = pd.read_csv(ruta_completa)
        elif extension in ('.xlsx', '.xls'):
            df = pd.read_excel(ruta_completa)
        else:
            print("Error: Formato no compatible")
            return None

        return df

    except FileNotFoundError:
        print(f"Error: Archivo no encontrado en la ruta '{ruta_completa}'.")
        return None

    except Exception as e:
        print(f"Error inesperado: {e}")
        return None




def exploracion_inicial(df, tipo=None):
    """
    Realiza una exploración inicial de un DataFrame y muestra información clave.

    Parámetros:
    df (pd.DataFrame): El DataFrame a explorar.
    tipo (str, opcional): El tipo de exploración. 'simple' muestra menos detalles.

    Imprime:
    Información relevante sobre el DataFrame, incluyendo filas, columnas, tipos de datos,
    estadísticas descriptivas, y valores nulos.
    """

    # Información básica sobre el DataFrame
    num_filas, num_columnas = df.shape
    print(f"¿Cuántas filas y columnas hay en el conjunto de datos?")
    print(f"\tHay {num_filas:,} filas y {num_columnas:,} columnas.")
    print('#' * 90)

    # Exploración simple
    if tipo == '1':

        print("Información detallada del DataFrame:")
        print(df.info())
        print('-' * 100)

        print("Estadísticas descriptivas del DataFrame:")
        display(df.describe(include='all').fillna(''))
        print('-' * 100)

        print("% valores nulos por columna:")
        df_nulos = df.isnull().sum().div(len(df)).mul(100).round(2).reset_index().rename(columns = {'index': 'Col', 0: 'pct'})
        df_nulos = df_nulos.sort_values(by = 'pct', ascending=False).reset_index(drop = True)
        display(df_nulos)
        print('-' * 100)

    elif tipo == '2':
        print("Muestra aleatoria de 5 filas:")
        display(df.sample(n=5))
        print('-' * 100)

        print("Tipo de datos de cada columna:")
        print(df.dtypes)
        print('-' * 100)

        print("Información detallada del DataFrame:")
        print(df.info())
        print('-' * 100)

        print("Estadísticas descriptivas del DataFrame:")
        display(df.describe(include='all').fillna(''))
        print('-' * 100)

        print("Mapa de calor de valores nulos")
        msno.heatmap(df, figsize = (6, 3), fontsize= 9)
        plt.show()
        print('-' * 100)

        print("% valores nulos por columna:")
        df_nulos = df.isnull().sum().div(len(df)).mul(100).round(2).reset_index().rename(columns = {'index': 'Col', 0: 'pct'})
        df_nulos = df_nulos.sort_values(by = 'pct', ascending=False).reset_index(drop = True)
        display(df_nulos)
        print('-' * 100)


    else:
        # Exploración completa
        print("Primer 5 filas:")
        display(df.head())
        print('-' * 100)

        print("Ultimas 5 filas:")
        display(df.tail())
        print('-' * 100)

        print("Muestra aleatoria de 5 filas:")
        display(df.sample(n=5))
        print('-' * 100)

        print("Columnas del DataFrame:")
        print("\n".join(f"\t- {col}" for col in df.columns))
        print('-' * 100)

        print("Tipo de datos de cada columna:")
        print(df.dtypes)
        print('-' * 100)

        print("# columnas por tipo de dato:")
        print(df.dtypes.value_counts())
        print('-' * 100)

        print("Información detallada del DataFrame:")
        print(df.info())
        print('-' * 100)

        print("# valores únicos por columna:")
        print(df.nunique())
        print('-' * 100)

        print("Valores únicos por columna:")
        df_valores_unicos = pd.DataFrame(df.apply(lambda x: x.unique()))
        display(df_valores_unicos)
        print('-' * 100)

        print("Estadísticas descriptivas del DataFrame:")
        display(df.describe(include='all').fillna(''))
        print('-' * 100)

        print("Valores nulos por columna:")
        display(df.isnull().sum())
        print('-' * 100)

        print("% valores nulos por columna:")
        df_nulos = df.isnull().sum().div(len(df)).mul(100).round(2).reset_index().rename(columns = {'index': 'Col', 0: 'pct'})
        df_nulos = df_nulos.sort_values(by = 'pct', ascending=False).reset_index(drop = True)
        display(df_nulos)
        print('-' * 100)

        print("Valores nulos: Visualización")
        msno.bar(df, figsize = (6, 3), fontsize= 9)
        plt.show()
        print('-' * 100)

        print("Visualización de patrones en valores nulos")
        msno.matrix(df, figsize = (6, 3), fontsize= 9, sparkline = False)
        plt.show()
        print('-' * 100)

        print("Mapa de calor de valores nulos")
        msno.heatmap(df, figsize = (6, 3), fontsize= 9)
        plt.show()
        print('-' * 100)

    print('#' * 90)

def coverage_report(df: pd.DataFrame, years):
    """
    Asume columnas: 'Country' y 'Year'

    Devuelve:
    - summary: tabla con cobertura por país (%), #años con dato, #años faltantes
    - missing_years: dict {pais: [años faltantes]}
    - missing_matrix: DataFrame (pais x año) True si falta
    """
    # Filtra solo por años de interés y paises únicos
    d = df.loc[df["Year"].isin(years), ["Country", "Year"]].copy()

    # Si hay duplicados (mismo país-año), nos quedamos con uno
    d = d.drop_duplicates(["Country", "Year"])

    countries = sorted(d["Country"].unique())

    # Tabla para saber si esta presente (pais x año): True si existe registro
    presence = (
        # Asigne una columna temporal 'present' con True
        d.assign(present=True)
        # Pivot table para convertir en tabla de presencia (pais x año), si no hay valores entonces fill_value=False
        .pivot_table(index="Country", columns="Year", values="present", aggfunc="max", fill_value=False)
        # Reindex para asegurar que todos los países y años seleccionados estén presentes
        .reindex(index=countries, columns=years, fill_value=False)
    )

    # La missing matrix se obtiene invirtiendo la de presencia, 
    # Presence: True = dato existe; False = dato falta
    # Missing_matrix: True = falta; False = dato existe
    missing_matrix = ~presence

    # Lista de años faltantes por país
    missing_years = {c: missing_matrix.columns[missing_matrix.loc[c]].tolist() for c in missing_matrix.index}

    # Resumen por país
    years_with_data = presence.sum(axis=1)
    years_missing = missing_matrix.sum(axis=1)
    summary = pd.DataFrame({
        "years_with_data": years_with_data,
        "years_missing": years_missing,
        "coverage_pct": (years_with_data / len(years) * 100).round(1)
    }).sort_values(["coverage_pct", "years_missing"], ascending=[False, True])

    return summary, missing_years, missing_matrix

def plot_missing_heatmap(missing_matrix, title="Missing data heatmap (True = missing)",
                         show_y_labels=True, x_step=3, figsize=(16, 8)):
                         
    """
    missing_matrix: DataFrame booleano (index=Country, columns=Year)
    True = falta dato
    """
    plt.figure(figsize=figsize)
    plt.imshow(missing_matrix.values, aspect="auto", interpolation="nearest")

    # Eje X (años)
    years = list(missing_matrix.columns)
    plt.xticks(
        ticks=list(range(0, len(years), x_step)),
        labels=years[::x_step],
        rotation=90
    )

    # Eje Y (países)
    if show_y_labels:
        countries = list(missing_matrix.index)
        plt.yticks(ticks=range(len(countries)), labels=countries)
    else:
        plt.yticks([])

    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.show()