# %% [markdown]
# <a href="https://www.kaggle.com/code/antoniojess/cryptonewssentiment?scriptVersionId=160742505" target="_blank"><img align="left" alt="Kaggle" title="Open in Kaggle" src="https://kaggle.com/static/images/open-in-kaggle.svg"></a>

# %% [markdown]
# # Importar módulos

# %% [code] {"jupyter":{"outputs_hidden":false}}
##### Para que todo funcione, primero hay que instalar todas las dependencias ######
#!pip install streamlit httpx duckdb sqlalchemy ta alpaca-py seaborn plotly scikit-learn tensorflow

# %% [code] {"jupyter":{"outputs_hidden":false}}
import os
import shutil
import re
import warnings
import httpx
import traceback
from joblib import dump, load
import numpy as np
import pandas as pd
import duckdb
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, GRU, Conv1D, Bidirectional, Dropout, Flatten, MultiHeadAttention
from datetime import datetime as dt, time, timedelta as td
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
import ta
import asyncio
import warnings

warnings.filterwarnings("ignore")

# %% [markdown]
# # Variables globales

# %% [code] {"jupyter":{"outputs_hidden":false}}
LENGTH = 15
TIME_FRAMES = ['H', '4H', 'D']
MODELS_FOLDER = 'C:/Users/Administrador/Downloads/proyecto_bootcamp/models'
BBDD_FILE = "C:/Users/Administrador/Downloads/proyecto_bootcamp/project.duckdb"

# %% [markdown]
# # Funciones

# %% [markdown]
# ## Helpers

# %% [code] {"jupyter":{"outputs_hidden":false}}
async def http_get(url, retries=3, timeout=10.0):
    """
    Realiza una solicitud HTTP GET asincrónica a la URL especificada y devuelve el contenido en formato JSON.

    Parámetros:
    - url (str): La URL a la que se realizará la solicitud GET.
    - retries (int): Número de intentos de reintentos en caso de fallo. Por defecto, se establece en 3.
    - timeout (float): Tiempo máximo, en segundos, para esperar una respuesta antes de lanzar una excepción por tiempo de espera.
                     Por defecto, se establece en 10.0 segundos.

    Retorna:
    - dict: El contenido de la respuesta en formato JSON.

    Excepciones:
    - httpx.ReadTimeout: Se lanza en caso de que se agote el tiempo de espera en la lectura de la respuesta.

    Ejemplo:
    ```python
    url = "https://api.example.com/data"
    data = await http_get(url, retries=2, timeout=15.0)
    print(data)
    ```

    Notas:
    - En caso de fallo, se realizan reintentos según el número especificado en 'retries'.
    - Si se produce un fallo después de todos los reintentos, se lanza una excepción httpx.ReadTimeout.
    """
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(url)
                return r.json()
        except httpx.ReadTimeout:
            if attempt < retries - 1:
                await asyncio.sleep(2)
            else:
                raise

# %% [markdown]
# ## Sentimiento

# %% [markdown]
# ### Índice de miedo y codicia

# %% [code] {"jupyter":{"outputs_hidden":false}}
async def get_fear_greed_index_data(limit):
    """
    Descarga los datos diarios del índice de miedo y codicia de Bitcoin desde la API de alternative.me.

    Parámetros:
    - limit (int or None): El número máximo de datos a recuperar. Si es None, se descargan todos los datos disponibles.
                          Si se proporciona un número, se limita la cantidad de datos descargados.

    Retorna:
    - dict: Datos diarios del índice de miedo y codicia de Bitcoin en formato JSON.

    Ejemplo:
    ```python
    data = await get_fear_greed_index_data(limit=30)
    print(data)
    ```

    Notas:
    - La URL utilizada para la descarga puede incluir un parámetro de límite si se especifica.
    - Los datos se descargan utilizando la función http_get, que maneja la lógica de reintentos y tiempo de espera.
    """
    if limit == None:    
        url = "https://api.alternative.me/fng/"
    else:
        url = f"https://api.alternative.me/fng/?limit={limit}"
    data = await http_get(url)
    # Retorna los datos
    return data

# %% [code] {"jupyter":{"outputs_hidden":false}}
async def get_latest_fgi(fecha_desde):
    """
    Descarga y retorna los datos actualizados del índice de miedo y codicia de Bitcoin desde la fecha indicada.

    Parámetros:
    - fecha_desde (datetime or None): La fecha desde la cual se deben recuperar los datos. Si es None, se descargan los datos desde el 2 de marzo de 2022.

    Retorna:
    - pd.DataFrame: Un DataFrame de pandas que contiene la evolución del índice de miedo y codicia a lo largo del tiempo.

    Ejemplo:
    ```python
    fecha_inicio = dt(year=2022, month=3, day=10)
    df_fgi = await get_latest_fgi(fecha_desde=fecha_inicio)
    print(df_fgi)
    ```

    Notas:
    - La función utiliza la función get_fear_greed_index_data para obtener los datos diarios del índice.
    - Los datos se convierten en un DataFrame de pandas para facilitar su manipulación y análisis.
    - Se eliminan duplicados basados en la columna de fechas y se ordena el DataFrame por fecha.
    """
    # Fecha de último valor fgi
    fecha_ahora = dt.utcnow().replace(second=0, microsecond=0)
    if pd.isna(fecha_desde) or fecha_desde is None:
        fecha_desde = dt(year=2022, month=3, day=2)
    limit = (fecha_ahora - fecha_desde).days + 1
    # Otener los datos
    data = await get_fear_greed_index_data(limit)
    # Retorno lista de elementos
    rows_list = []
    for row in data["data"]:
        fgi_row = {
            "fecha" : dt.fromtimestamp(int(row["timestamp"])).strftime("%Y-%m-%d"),
            "fgi" : float(row["value"]) / 100.0
        }
        rows_list.append(fgi_row)
    # Lista de noticias a dataframe
    df_actualizado = pd.DataFrame(rows_list, columns=["fecha", "fgi"])
    df_actualizado["fecha"] = pd.to_datetime(df_actualizado["fecha"], format="%Y-%m-%d").dt.tz_localize(None)
    df_actualizado["fgi"] = df_actualizado["fgi"].astype(float)
    df_actualizado = df_actualizado.drop_duplicates(subset='fecha', keep='first').set_index('fecha')
    df_actualizado = df_actualizado.sort_index()
    # Retorna el dataframe
    return df_actualizado

# %% [markdown]
# ### Noticias

# %% [code] {"jupyter":{"outputs_hidden":false}}
async def get_alphavantage_data(url_params: str, fecha_desde: str, fecha_hasta: str) -> dict:
    """
    Descarga datos financieros desde la API Alphavantage utilizando los parámetros de URL proporcionados.

    Parámetros:
    - url_params (str): Parámetros adicionales para la URL de la solicitud, como símbolo del activo, intervalo de tiempo, etc.
    - fecha_desde (str): Fecha de inicio en formato YYYY-MM-DD desde la cual se recuperarán los datos.
    - fecha_hasta (str): Fecha de finalización en formato YYYY-MM-DD hasta la cual se recuperarán los datos.

    Retorna:
    - dict: Datos financieros descargados desde Alphavantage en formato JSON.

    Ejemplo:
    ```python
    params = "symbol=MSFT&interval=1day"
    start_date = "2022-01-01"
    end_date = "2022-12-31"
    financial_data = await get_alphavantage_data(url_params=params, fecha_desde=start_date, fecha_hasta=end_date)
    print(financial_data)
    ```

    Notas:
    - La URL se compone de la URL base de Alphavantage, los parámetros de URL proporcionados, y las fechas especificadas.
    - Se utiliza la función http_get para realizar la solicitud HTTP asincrónica.
    """
    api_key = "GPAN46OGIJYO1S0J"
    # Url
    url = f"https://www.alphavantage.co/query?"
    url = f"{url}{url_params}&time_from={fecha_desde}&time_to={fecha_hasta}&apikey={api_key}"
    # debug
    #print(url)
    # Solicitud
    data = await http_get(url)
    # Retorna datos
    return data

# %% [code] {"jupyter":{"outputs_hidden":false}}
async def get_news_sentiment(fecha_desde, fecha_ahora):
    """
    Descarga y retorna el sentimiento de noticias sobre Bitcoin desde la fecha indicada hasta la fecha actual.

    Parámetros:
    - fecha_desde (datetime): Fecha de inicio desde la cual se recuperarán las noticias.
    - fecha_ahora (datetime): Fecha actual hasta la cual se recuperarán las noticias.

    Retorna:
    - pd.DataFrame: Un DataFrame de pandas que contiene el sentimiento de noticias sobre Bitcoin.

    Excepciones:
    - RateLimitExceededException: Se lanza si se excede el límite de solicitudes a la API Alphavantage.

    Ejemplo:
    ```python
    start_date = dt(year=2022, month=1, day=1)
    end_date = dt.utcnow()
    df_sentiment = await get_news_sentiment(fecha_desde=start_date, fecha_ahora=end_date)
    print(df_sentiment)
    ```

    Notas:
    - La función utiliza la función get_alphavantage_data para obtener las noticias financieras.
    - Se maneja la excepción RateLimitExceededException si se excede el límite de solicitudes a la API Alphavantage.
    - Los datos se convierten en un DataFrame de pandas para facilitar su manipulación y análisis.
    """
    class RateLimitExceededException(Exception):
        pass
    # Sentimiento de noticias
    url_params = "function=NEWS_SENTIMENT&tickers=CRYPTO:BTC&sort=EARLIEST&limit=1000"
    # Lista de noticias
    rows_list = []
    # Bucle que obtiene los datos
    while fecha_desde <= fecha_ahora:
        fecha_hasta = fecha_desde + td(days=15)
        try:
            # Obtener las noticias de ese rango temporal
            data = await get_alphavantage_data(url_params, fecha_desde.strftime("%Y%m%dT%H%M"), fecha_hasta.strftime("%Y%m%dT%H%M"))
            if "feed" in data:
                for feed in data["feed"]:
                    # Relleno el dataframe
                    for ticker in feed["ticker_sentiment"]:
                        if ticker["ticker"] == "CRYPTO:BTC":
                            feed_row = {
                                "fecha" : dt.strptime(feed["time_published"], "%Y%m%dT%H%M%S").strftime("%Y-%m-%d %H:%M:%S"),
                                "title" : feed["title"],
                                "summary" : feed["summary"],
                                "url" : feed["url"],
                                "relevance" : float(ticker["relevance_score"]),
                                "score" : float(ticker["ticker_sentiment_score"]),
                                "label" : ticker["ticker_sentiment_label"]
                            }
                            rows_list.append(feed_row)
            elif "Information" in data:
                raise RateLimitExceededException(data["Information"])
        except Exception as e:
            traceback.print_exc()
            break
        fecha_desde = fecha_hasta + td(minutes=1.)
    # Lista de noticias a DataFrame
    df_news = pd.DataFrame(rows_list, columns=["fecha", "title", "summary", "url", "relevance", "score", "label"])
    df_news["fecha"] = pd.to_datetime(df_news["fecha"], format='%Y-%m-%d %H:%M:%S').dt.tz_localize(None)
    df_news["relevance"] = df_news["relevance"].astype(float)
    df_news["score"] = df_news["score"].astype(float)
    # Retorna el dataframe
    return df_news

# %% [code] {"jupyter":{"outputs_hidden":false}}
async def get_latest_news(fecha_desde):
    """
    Descarga y retorna las últimas noticias sobre Bitcoin desde la fecha indicada hasta la fecha actual.

    Parámetros:
    - fecha_desde (datetime or None): La fecha desde la cual se recuperarán las noticias. Si es None, se descargan las noticias desde el 2 de marzo de 2022.

    Retorna:
    - pd.DataFrame: Un DataFrame de pandas que contiene las últimas noticias sobre Bitcoin organizadas por fecha.

    Ejemplo:
    ```python
    fecha_inicio = dt(year=2022, month=3, day=10)
    df_latest_news = await get_latest_news(fecha_desde=fecha_inicio)
    print(df_latest_news)
    ```

    Notas:
    - La función utiliza la función get_news_sentiment para obtener el sentimiento de noticias.
    - Se eliminan duplicados basados en la columna de fechas y se ordena el DataFrame por fecha.
    """
    # Fecha de última noticia
    fecha_ahora = dt.utcnow().replace(second=0, microsecond=0)
    if pd.isna(fecha_desde) or fecha_desde is None:
        fecha_desde = dt(year=2022, month=3, day=2)
    # Se obtinen las últimas noticias
    df_actualizado = await get_news_sentiment(fecha_desde, fecha_ahora)
    df_actualizado = df_actualizado.drop_duplicates(subset='fecha', keep='first').set_index('fecha')
    df_actualizado = df_actualizado.sort_index()
    return df_actualizado

# %% [markdown]
# ## Precios

# %% [code] {"jupyter":{"outputs_hidden":false}}
async def get_prices(fecha_desde, symbols=["BTC/USD"]):
    """
    Obtiene los datos de cotización de Bitcoin con un timeframe de un minuto desde la fecha indicada utilizando la API de Alpaca.

    Parámetros:
    - fecha_desde (datetime): La fecha desde la cual se recuperarán los datos de cotización.
    - symbols (list): Lista de símbolos de activos financieros para los cuales se obtendrán los datos. Por defecto, se establece como ["BTC/USD"].

    Retorna:
    - pd.DataFrame: Un DataFrame de pandas que contiene los datos de cotización de Bitcoin con un timeframe de un minuto.

    Ejemplo:
    ```python
    start_date = dt(year=2022, month=1, day=1)
    df_prices = await get_prices(fecha_desde=start_date, symbols=["BTC/USD"])
    print(df_prices)
    ```

    Notas:
    - La función utiliza la clase CryptoHistoricalDataClient de Alpaca para realizar la solicitud de datos de cotización.
    - Los datos se ajustan y organizan en un DataFrame de pandas para facilitar su manipulación y análisis.
    """
    client = CryptoHistoricalDataClient()
    request_params = CryptoBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Minute,
        start=fecha_desde.strftime("%Y-%m-%dT%H:%M")
    )
    df_price = client.get_crypto_bars(request_params).df
    df_price = df_price.reset_index(level='symbol', drop=True).reset_index()
    df_price = df_price.rename(columns={"timestamp":"fecha", "trade_count":"trades"})
    df_price["fecha"] = pd.to_datetime(df_price["fecha"], format='%Y-%m-%d %H:%M:%S').dt.tz_localize(None)
    column_order = ['fecha', 'open', 'high', 'low', 'close', 'volume', 'trades', 'vwap']
    df_price = df_price[column_order]
    return df_price

# %% [code] {"jupyter":{"outputs_hidden":false}}
async def get_latest_prices(fecha_desde):
    """
    Descarga y retorna los últimos datos de cotización de Bitcoin desde la fecha indicada hasta la fecha actual.

    Parámetros:
    - fecha_desde (datetime or None): La fecha desde la cual se recuperarán los datos de cotización. Si es None, se descargan los datos desde el 2 de marzo de 2022.

    Retorna:
    - pd.DataFrame: Un DataFrame de pandas que contiene los últimos datos de cotización de Bitcoin organizados por fecha.

    Ejemplo:
    ```python
    fecha_inicio = dt(year=2022, month=3, day=10)
    df_latest_prices = await get_latest_prices(fecha_desde=fecha_inicio)
    print(df_latest_prices)
    ```

    Notas:
    - La función utiliza la función get_prices para obtener los datos de cotización.
    - Se eliminan duplicados basados en la columna de fechas y se ordena el DataFrame por fecha.
    """
    # Fecha de último precio
    fecha_ahora = dt.utcnow().replace(second=0, microsecond=0)
    if pd.isna(fecha_desde) or fecha_desde is None:
        fecha_desde = dt(year=2022, month=3, day=2)
    # Se obtinen los últimos precios
    df_actualizado = await get_prices(fecha_desde)
    df_actualizado = df_actualizado.drop_duplicates(subset='fecha', keep='first').set_index('fecha')
    df_actualizado = df_actualizado.sort_index()
    return df_actualizado

# %% [markdown]
# ### Indicadores técnicos

# %% [code] {"jupyter":{"outputs_hidden":false}}
async def agg_ta_indicators(df_prices):
    """
    Agrega indicadores técnicos al DataFrame de cotizaciones de Bitcoin, incluyendo el MACD y RSI.

    Parámetros:
    - df_prices (pd.DataFrame): DataFrame de pandas que contiene los datos de cotización de Bitcoin.

    Retorna:
    - pd.DataFrame: El DataFrame de cotizaciones de Bitcoin con las columnas adicionales de los indicadores técnicos (MACD y RSI).

    Ejemplo:
    ```python
    df_prices = await get_latest_prices(fecha_desde=fecha_inicio)
    df_prices_with_indicators = await agg_ta_indicators(df_prices)
    print(df_prices_with_indicators)
    ```

    Notas:
    - Se utilizan las funciones de la biblioteca ta (Technical Analysis Library) para calcular los indicadores técnicos.
    - Se agregan al DataFrame las columnas 'macd_line', 'macd_signal', 'macd_diff' correspondientes al MACD.
    - Se agrega la columna 'rsi' correspondiente al RSI.
    """
    df_prices['macd_line'] =ta.trend.MACD(close=df_prices['close'], window_slow=26, window_fast=12, window_sign=9, fillna=False).macd()
    df_prices['macd_signal'] =ta.trend.MACD(close=df_prices['close'], window_slow=26, window_fast=12, window_sign=9, fillna=False).macd_signal()
    df_prices['macd_diff'] =ta.trend.MACD(close=df_prices['close'], window_slow=26, window_fast=12, window_sign=9, fillna=False).macd_diff()
    df_prices['rsi']=ta.momentum.RSIIndicator(close=df_prices['close'], window=14, fillna=False).rsi()
    df_prices = df_prices.dropna(how='all')
    return df_prices

# %% [markdown]
# ## Base de datos

# %% [code] {"jupyter":{"outputs_hidden":false}}
async def bbdd_create(BBDD_FILE):
    """
    Crea seis tablas en la base de datos DuckDB para almacenar datos financieros y analíticos.

    Parámetros:
    - BBDD_FILE (str): Ruta del archivo de la base de datos DuckDB.

    Ejemplo:
    ```python
    bbdd_file_path = "/path/to/your/database.db"
    await bbdd_create(BBDD_FILE=bbdd_file_path)
    ```

    Notas:
    - Se crean las tablas 'prices', 'news', y 'fgi' para almacenar datos de cotización, noticias y el índice de miedo y codicia respectivamente.
    - Se crean tablas adicionales para modelos analíticos con diferentes agrupaciones temporales ('D', '4H', 'H').
    """
    try:
        # Crear una conexión a DuckDB
        conn = duckdb.connect(BBDD_FILE)
        #Tabla precios
        prices_table = """
        CREATE TABLE IF NOT EXISTS prices (
            fecha DATETIME UNIQUE,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            close FLOAT,
            volume FLOAT,
            trades FLOAT,
            vwap FLOAT
        )
        """
        conn.execute(prices_table)
        # Noticias
        news_table = """
        CREATE TABLE IF NOT EXISTS news (
            fecha DATETIME UNIQUE,
            title VARCHAR,
            summary VARCHAR,
            url VARCHAR,
            relevance FLOAT,
            score FLOAT,
            label VARCHAR
        )
        """
        conn.execute(news_table)
        # Índice de miedo y codicia
        fgi_table = """
        CREATE TABLE IF NOT EXISTS fgi (
            fecha DATETIME UNIQUE,
            fgi FLOAT
        )
        """
        conn.execute(fgi_table)
        # Crear tablas modelos
        models_columns = [
        'fecha DATETIME UNIQUE',
        'open FLOAT',
        'high FLOAT',
        'low FLOAT',
        'close FLOAT',
        'volume FLOAT',
        'trades FLOAT',
        'vwap FLOAT',
        'news FLOAT',
        'fgi FLOAT',
        'macd_line FLOAT',
        'macd_signal FLOAT',
        'macd_diff FLOAT',
        'rsi FLOAT',
        'close_diff_pct FLOAT'
        ]
    
        for table in TIME_FRAMES:
            models_tables = f'''
            CREATE TABLE IF NOT EXISTS "{table}" (
                {', '.join(models_columns)}
            )
            '''
            conn.execute(models_tables)
        # Cerrar la conexión
        conn.close()
    except Exception as e:
        print(f"Ocurrió un error en la creación de la bbdd: {e}")

# %% [code] {"jupyter":{"outputs_hidden":false}}
async def bbdd_insert(table, df_input):
    """
    Inserta un DataFrame en la tabla especificada en la base de datos DuckDB.

    Parámetros:
    - table (str): Nombre de la tabla en la que se insertarán los datos.
    - df_input (pd.DataFrame): DataFrame de pandas que contiene los datos a insertar.

    Ejemplo:
    ```python
    table_name = "prices"
    await bbdd_insert(table=table_name, df_input=df_prices)
    ```

    Notas:
    - Se utiliza la conexión a la base de datos DuckDB para insertar el DataFrame en la tabla especificada.
    - Se utiliza ON CONFLICT DO NOTHING para evitar la inserción de registros duplicados en la tabla.
    """
    # Insertar el dataframe en su tabla
    try:
        conn = duckdb.connect(BBDD_FILE)
        conn.register('df_input', df_input.reset_index())
        conn.execute(f'INSERT INTO "{table}" SELECT * FROM df_input ON CONFLICT DO NOTHING')
        conn.close()
    except Exception as e:
        print(f"Error en la inserción de datos: {e}")

# %% [code] {"jupyter":{"outputs_hidden":false}}
async def bbdd_query(query):
    """
    Realiza una consulta SQL en la base de datos DuckDB y devuelve el resultado como un DataFrame.

    Parámetros:
    - query (str): Consulta SQL a ejecutar.

    Retorna:
    - pd.DataFrame: DataFrame de pandas que contiene los resultados de la consulta SQL.

    Ejemplo:
    ```python
    sql_query = "SELECT * FROM prices WHERE fecha >= '2022-01-01'"
    result_df = await bbdd_query(query=sql_query)
    print(result_df)
    ```

    Notas:
    - Se utiliza la conexión a la base de datos DuckDB para ejecutar la consulta SQL.
    - Si la consulta devuelve resultados, se devuelve un DataFrame con esos resultados; de lo contrario, se devuelve None.
    """
    # Consulta
    try:
        conn = duckdb.connect(BBDD_FILE)
        df_resp = conn.execute(query).fetchdf()
        conn.close()
        # Verificar
        if not df_resp.empty:
            return df_resp
        else:
            print("El DataFrame está vacío. No se enviará.")
            return None
    except Exception as e:
        print(f"Ocurrió un error en la consulta de datos: {e}")
        return None

# %% [code] {"jupyter":{"outputs_hidden":false}}
async def bbdd_update():
    """
    Actualiza las tablas de la base de datos DuckDB mediante la API y agrupando datos en diferentes timeframes.

    Ejemplo:
    ```python
    await bbdd_update()
    ```

    Notas:
    - Se utiliza un diccionario de funciones (func_dict) que mapea cada tabla con la función correspondiente para actualizarla.
    - Se comprueba la existencia de la base de datos y se crea si no existe.
    - Se actualizan las tres tablas iniciales ('fgi', 'news', 'prices') mediante llamadas a las funciones correspondientes.
    - Se actualizan las tablas adicionales ('H', '4H', 'D') mediante llamadas a la función model_input_data.
    """
    func_dict = {
        'fgi': get_latest_fgi,
        'news': get_latest_news,
        'prices': get_latest_prices
    }
    # Comprobar que la bbdd existe
    if not os.path.exists(BBDD_FILE):
        await bbdd_create(BBDD_FILE)
    # Actualizar datos
    async def update_raw_table(table, function):
        query = f'SELECT CAST(MAX(fecha) AS DATETIME) FROM "{table}"'
        resp = await bbdd_query(query)
        fecha_desde = pd.to_datetime(resp.iloc[0, 0]).to_pydatetime()
        data = await function(fecha_desde)
        await bbdd_insert(table, data)
    tasks = [update_raw_table(table, function) for table, function in func_dict.items()]
    await asyncio.gather(*tasks)
    async def update_model_tables(table):
        query = f'SELECT CAST(MAX(fecha) AS DATETIME) FROM "{table}"'
        resp = await bbdd_query(query)
        fecha_desde = pd.to_datetime(resp.iloc[0, 0]).to_pydatetime()
        data = await model_input_data(table, fecha_desde)
        await bbdd_insert(table, data)
    tasks = [update_model_tables(table) for table in TIME_FRAMES]
    await asyncio.gather(*tasks)

# %% [markdown]
# ## Modelo Deep Learning

# %% [markdown]
# ## Dataset

# %% [code] {"jupyter":{"outputs_hidden":false}}
async def grouping_dataframe(df_fgi, df_news, df_prices, timeframe):
    """
    Agrupa y genera un DataFrame con datos financieros y analíticos para un timeframe específico.

    Parámetros:
    - df_fgi (pd.DataFrame): DataFrame de pandas que contiene datos del índice de miedo y codicia.
    - df_news (pd.DataFrame): DataFrame de pandas que contiene datos de noticias.
    - df_prices (pd.DataFrame): DataFrame de pandas que contiene datos de cotización.
    - timeframe (str): Marco temporal para la agrupación ('H' para 1 hora, '4H' para 4 horas, 'D' para 1 día).

    Retorna:
    - pd.DataFrame: DataFrame de pandas que contiene datos agrupados con indicadores técnicos.

    Ejemplo:
    ```python
    df_fgi = await bbdd_query("SELECT * FROM fgi")
    df_news = await bbdd_query("SELECT * FROM news")
    df_prices = await bbdd_query("SELECT * FROM prices")
    timeframe = 'H'
    grouped_df = await grouping_dataframe(df_fgi, df_news, df_prices, timeframe)
    ```

    Notas:
    - Se utiliza la función resample de pandas para agrupar los datos de precios en el marco temporal especificado.
    - Se calcula el precio ponderado por volumen (vwap) y se aplican diversas funciones de agregación.
    - Se agrega información de noticias y el índice de miedo y codicia según el marco temporal.
    - Se utiliza la función agg_ta_indicators para agregar indicadores técnicos al DataFrame resultante.
    """
    # Agrupación de precios
    df = df_prices.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'trades': 'sum',
        'vwap': lambda x: (x.sum()/len(x)) if len(x) > 0 else 0
    })
    # Agregar noticias
    df['news'] = df_news.resample(timeframe).apply(lambda x: (x['relevance'] * x['score']).sum() / x['relevance'].sum() if x['relevance'].sum() > 0 else 0)
    # Fear and greed index
    if timeframe != 'D':
        df_fgi_expanded = df_fgi.resample(timeframe).ffill()
        df['fgi'] = df_fgi_expanded.reindex(df.index, method='ffill')
    else:
        df['fgi'] = df_fgi.reindex(df.index, method='ffill')
    df = df.dropna(how='all')
    df = await agg_ta_indicators(df)
    return df

# %% [code] {"jupyter":{"outputs_hidden":false}}
async def model_input_data(timeframe, model_data_last_date):
    """
    Obtiene los datos necesarios para entrenar o predecir el modelo para el timeframe proporcionado.

    Parámetros:
    - timeframe (str): Marco temporal para el cual se obtendrán los datos ('H' para 1 hora, '4H' para 4 horas, 'D' para 1 día).
    - model_data_last_date (datetime): Fecha de la última actualización de datos para el modelo.

    Retorna:
    - pd.DataFrame: DataFrame de pandas con los datos agrupados y preparados para la entrada del modelo.

    Ejemplo:
    ```python
    timeframe = 'H'
    last_update_date = dt(year=2022, month=3, day=15)
    df_model_input = await model_input_data(timeframe, model_data_last_date=last_update_date)
    ```

    Notas:
    - Se utiliza un diccionario (df_dict) para almacenar los datos de las tablas 'fgi', 'news', y 'prices'.
    - Se realiza una consulta a la base de datos para obtener los datos necesarios y se filtran por la última fecha de actualización del modelo.
    - Los datos se agrupan y se aplica una transformación de cambio porcentual en el precio de cierre ('close_diff_pct').
    """
    df_dict = {
        'fgi': None,
        'news': None,
        'prices': None
    }
    # Obtener los datos de la BBDD
    for raw_table in df_dict.keys():
        query = f'SELECT CAST(MAX(fecha) AS DATETIME) FROM "{raw_table}"'
        resp = await bbdd_query(query)
        raw_data_last_date = pd.to_datetime(resp.iloc[0, 0]).to_pydatetime()
        if pd.isna(model_data_last_date):
            # debug
            print(f'BBDD vacía: {model_data_last_date}')
            query= f"""
            SELECT *
            FROM "{raw_table}"
            """
            df_dict[raw_table] = await bbdd_query(query)
        else:
            #print(f'BBDD última fecha: {model_data_last_date}')
            intervals = {
                'H': f'{1 * LENGTH*10} hours',
                '4H': f'{4 * LENGTH*10} hours',
                'D': f'{1 * LENGTH*10} days',
            }
            interval_str = intervals[timeframe]
            query = f"""
                SELECT *
                FROM "{raw_table}"
                WHERE fecha >= CAST('{raw_data_last_date}' AS DATETIME) - INTERVAL '{interval_str}'
                ORDER BY fecha DESC;
            """
            df_dict[raw_table] = await bbdd_query(query)
    # Organizar dataframes
    df_fgi = df_dict['fgi']
    df_fgi = df_fgi.set_index('fecha')
    df_news = df_dict['news']
    df_news = df_news.set_index('fecha')
    df_prices = df_dict['prices']
    df_prices = df_prices.set_index('fecha')
    # Agrupar datos
    df = await grouping_dataframe(df_fgi, df_news, df_prices, timeframe)
    # Cambio porcentual
    df['close_diff_pct'] = df['close'].pct_change(fill_method=None) * 100.
    # Eliminiar filas nulas
    df = df.dropna()
    # debug
    #print(f'''ÚLTIMA FECHA: {df.index.max()}''')
    # Return df
    return df

# %% [markdown]
# ### Modelo

# %% [code] {"jupyter":{"outputs_hidden":false}}
async def model_create(input_features):
    """
    Crea un modelo de red neuronal con dos ramas para procesar datos temporales y de atención.

    Parámetros:
    - input_features (int): Número de características de entrada.

    Retorna:
    - tf.keras.Model: Modelo de red neuronal compilado.

    Ejemplo:
    ```python
    input_features = 10
    neural_model = await model_create(input_features)
    ```

    Notas:
    - Se utiliza la API funcional de Keras para crear un modelo con dos ramas.
    - Cada rama consta de una capa convolucional, una capa LSTM de tipo GRU, y una capa de atención de múltiples cabezas.
    - Las dos ramas convergen en una capa densa y una capa lineal de regresión para la salida.
    - La función de pérdida utilizada es 'huber_loss' y el optimizador es 'adam'.
    """
    # Supongamos entradas de 120 pasos de tiempo con 10 características
    input_shape = (LENGTH, input_features)
    inputs = Input(shape=input_shape)
    # Primera rama 01
    branch_0 = Conv1D(filters=128, kernel_size=3, activation='relu')(inputs)
    branch_0 = Dropout(0.5)(branch_0)
    branch_0 = Bidirectional(GRU(64, return_sequences=True, dropout=0.5))(branch_0)
    branch_1 = Conv1D(filters=128, kernel_size=5, activation='relu')(inputs)
    branch_1 = Dropout(0.5)(branch_1)
    branch_1 = Bidirectional(GRU(64, return_sequences=True, dropout=0.5))(branch_1)
    branch_01 = MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.5)(branch_0, branch_1)
    # Primera rama 23
    branch_2 = Conv1D(filters=512, kernel_size=3, activation='relu',)(branch_01)
    branch_2 = Dropout(0.5)(branch_2)
    branch_2 = Bidirectional(GRU(256, return_sequences=True, dropout=0.5))(branch_2)
    branch_3 = Conv1D(filters=512, kernel_size=5, activation='relu')(branch_01)
    branch_3 = Dropout(0.5)(branch_3)
    branch_3 = Bidirectional(GRU(256, return_sequences=True, dropout=0.5))(branch_3)
    branch_23 = MultiHeadAttention(num_heads=8, key_dim=256, dropout=0.5)(branch_2, branch_3)
    branch_23 = Flatten()(branch_23)
    # Capas Densas finales
    dense = Dense(128, activation='relu',)(branch_23)
    dense = Dropout(0.5)(dense)
    output = Dense(1, activation='linear')(dense)
    # Creación y compilación del modelo
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='huber_loss')
    # Mostrar el resumen del modelo
    #model.summary()
    return model

# %% [markdown]
# ### Train model

# %% [code] {"jupyter":{"outputs_hidden":false}}
async def training_generators(df):
    """
    Prepara los generadores de datos para entrenamiento y prueba del modelo.

    Parámetros:
    - df (pd.DataFrame): DataFrame de pandas con los datos de entrada y salida.

    Retorna:
    - Tuple: Generadores de series temporales de entrenamiento y prueba, escaladores para características y objetivo.

    Ejemplo:
    ```python
    df_data = await model_input_data(timeframe='H', model_data_last_date=dt(year=2022, month=3, day=15))
    train_gen, test_gen, X_scaler, y_scaler = await training_generators(df_data)
    ```

    Notas:
    - Se escalan las características utilizando StandardScaler.
    - La columna objetivo ('close_diff_pct') se desplaza y escala para el entrenamiento.
    - Se dividen los datos en conjuntos de entrenamiento y prueba.
    - Se utilizan generadores de series temporales para facilitar el entrenamiento del modelo.
    """
    # Desplazar columna objetivo
    df['close_diff_pct'] = df['close_diff_pct'].shift(-1)
    df = df.dropna()
    # Escalar características
    X = df.drop('close_diff_pct', axis=1).values
    X_scaler = StandardScaler()
    X = X_scaler.fit_transform(X)
    # Escalar columna objetivo
    y = df['close_diff_pct'].values.reshape(-1, 1)
    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y)
    # Separar train y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    # Preparación de las secuencias temporales
    train_generator = TimeseriesGenerator(X_train, y_train, length=LENGTH, batch_size=1)
    test_generator = TimeseriesGenerator(X_test, y_test, length=LENGTH, batch_size=1)
    # Return
    return train_generator, test_generator, X_scaler, y_scaler

# %% [code] {"jupyter":{"outputs_hidden":false}}
async def train_model(table, epochs):
    """
    Entrena el modelo correspondiente a un timeframe específico con datos de su tabla correspondiente.

    Parámetros:
    - table (str): Nombre de la tabla de la base de datos para la cual se entrenará el modelo.
    - epochs (int): Número de épocas de entrenamiento.

    Ejemplo:
    ```python
    await train_model(table='H', epochs=10)
    ```

    Notas:
    - Obtiene los datos de la base de datos para la tabla especificada.
    - Realiza la división de entrenamiento y prueba utilizando la función `training_generators`.
    - Crea y entrena el modelo utilizando la función `model_create`.
    - Implementa Early Stopping como medida de control de parada temprana durante el entrenamiento.
    - Guarda el modelo y los escaladores en archivos para su posterior uso.
    """
    print(f"### Training {table} model ###")
    try:
        # Get data from bbdd
        query= f"""
        SELECT *
        FROM "{table}"
        """
        df = await bbdd_query(query)
        df = df.dropna()
        df = df.set_index('fecha')
        df = df.sort_index()
        # debug
        #print(f'{df.index.min()} | {df.index.max()}')
        # Train & test split
        train_generator, test_generator, X_scaler, y_scaler = await training_generators(df)
        model = await model_create(df.shape[1]-1)
        # control de parada
        early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0.0001,
            patience=2,
            verbose=1,
            mode='min',
            restore_best_weights=True
        )
        # Entrenamiento
        history = await asyncio.to_thread(
            model.fit,
            train_generator,
            validation_data=test_generator,
            epochs=epochs,
            verbose=1,
            callbacks=[early_stopping]
        )
        # print resultados
        print(f"\n{table} model: Train Loss = {min(history.history['loss'])} | Validation Loss = {min(history.history['val_loss'])}\n")
        # Guardar objetos
        model_folder = f"{MODELS_FOLDER}/{table}"
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        dump(X_scaler, f"{model_folder}/X_scaler.joblib")
        dump(y_scaler, f"{model_folder}/y_scaler.joblib")
        model.save(f"{model_folder}/model")
    except Exception as e:
        traceback.print_exc()

async def train_models(epochs):
    tasks = [train_model(table, epochs) for table in TIME_FRAMES]
    await asyncio.gather(*tasks)

# %% [markdown]
# ### Model prediction

# %% [code] {"jupyter":{"outputs_hidden":false}}
async def model_prediction(timeframe):
    """
    Realiza la predicción del modelo correspondiente a un timeframe específico con datos de su tabla correspondiente.

    Parámetros:
    - timeframe (str): Marco de tiempo para el cual se realizará la predicción.

    Retorna:
    - Tuple: Predicción del modelo y DataFrame con los datos de entrada.

    Ejemplo:
    ```python
    prediction, input_data = await model_prediction(timeframe='H')
    ```

    Notas:
    - Obtiene los datos de la base de datos para el timeframe especificado.
    - Prepara los datos de entrada para el modelo.
    - Carga el modelo entrenado y el escalador asociado.
    - Realiza la predicción y revierte la escala para obtener el valor en la unidad original.
    """
    # model path
    model_folder = f"{MODELS_FOLDER}/{timeframe}"
    # Extraer los datos de la bbdd
    query = f'''SELECT * FROM '{timeframe}' ORDER BY fecha DESC LIMIT {LENGTH};'''
    df = await bbdd_query(query)
    df = df.dropna()
    df = df.set_index('fecha')
    df = df.sort_index()
    # Cargar los escaladores
    X_scaler = load(f"{model_folder}/X_scaler.joblib")
    y_scaler = load(f"{model_folder}/y_scaler.joblib")
    # Preparar los datos de entrada al modelo
    X = X_scaler.transform(df.iloc[:, :-1])
    X = np.expand_dims(X, axis=0)
    # Cargar el modelo
    model = load_model(f"{model_folder}/model")
    # Predecir
    yhat = y_scaler.inverse_transform(model.predict(X))
    return yhat[0][0], df

# %% [markdown]
# ## Funciones Streamlit

# %% [code] {"jupyter":{"outputs_hidden":false}}
async def handle_button(timeframe):
    """
    Ejecuta la lógica del programa cuando el usuario selecciona un botón correspondiente a un timeframe.

    Parámetros:
    - timeframe (str): Marco de tiempo para el cual se realizará la operación.

    Retorna:
    - Tuple: Predicción del modelo, DataFrame con los datos de entrada, y DataFrame con noticias del intervalo de tiempo.

    Ejemplo:
    ```python
    prediction, input_data, news_data = await handle_button(timeframe='H')
    ```

    Notas:
    - Actualiza la base de datos mediante `bbdd_update`.
    - Entrena los modelos si no están presentes.
    - Realiza la predicción utilizando `model_prediction`.
    - Obtiene las noticias correspondientes al intervalo de tiempo utilizado por el modelo.
    """
    # Actualizar bbdd
    await bbdd_update()
    # Entrenar los modelos si no lo están
    if not os.path.exists(MODELS_FOLDER):
        await train_models(epochs=100)
    # Hacer predicción
    prediccion, df_model = await model_prediction(timeframe)
    # Obtener las noticias
    min_date = df_model.index.min()
    query=f'''
        SELECT * 
        FROM news
        WHERE fecha >= CAST('{min_date}' AS DATETIME)
        ORDER BY fecha DESC;
    '''
    df_news = await bbdd_query(query)
    df_news = df_news.set_index('fecha').sort_index(ascending=False)
    return prediccion, df_model, df_news

# %% [code] {"jupyter":{"outputs_hidden":false}}
def streamlit_app():
    """
    Configura y ejecuta la aplicación Streamlit para visualizar información financiera y noticias relacionadas con Bitcoin.

    Retorna:
    - None

    Ejemplo:
    ```python
    streamlit_app()
    ```

    Notas:
    - La página está dividida en tres secciones: una columna izquierda con botones de timeframes y noticias, y una columna derecha con gráficos.
    - Utiliza el estado de la sesión de Streamlit para gestionar la selección de timeframe y almacenar datos entre sesiones.
    - Se actualiza la base de datos con `bbdd_update`.
    - Entrena modelos si no están presentes con `train_models`.
    - Realiza predicciones con `model_prediction`.
    - Muestra información financiera y noticias en la interfaz de usuario utilizando gráficos y tablas.
    """
    intervals = {
        'H': f'1 hora',
        '4H': f'4 horas',
        'D': f'1 día'
    }
    # Configuración de la página de Streamlit
    st.set_page_config(page_title="Proyecto 3 Bootcamp", layout="wide")
    # División de la página
    col_izquierda, col_derecha = st.columns([1, 1])
    # Inicializa el estado de la sesión por defecto si es necesario
    if 'timeframe_selected' not in st.session_state:
        st.session_state['timeframe_selected'] = 'D'
        st.session_state['prediccion'], st.session_state['df_model'], st.session_state['df_news'] = asyncio.run(handle_button('D'))
    # Crear un botón para cada timeframe
    with col_izquierda:
        with st.container():
            st.markdown('##### Seleccione TimeFrame:')
            cols = st.columns(4)
            for i, timeframe in enumerate(TIME_FRAMES):
                with cols[i]:
                    if st.button(timeframe):
                        st.session_state['timeframe_selected'] = timeframe
                        st.session_state['prediccion'], st.session_state['df_model'], st.session_state['df_news'] = asyncio.run(handle_button(timeframe))
    # Datos
    df_news = st.session_state.get('df_news')
    df_model = st.session_state.get('df_model')
    prediccion = st.session_state.get('prediccion')
    intervalo = intervals[st.session_state.get('timeframe_selected')]
    # Debug
    #print(f"Predicción {df_model.index[-1]}: {prediccion}")
    # Esto es lo que se ejecuta si el usuario presiona un timeframe
    if 'timeframe_selected' in st.session_state:
        with col_derecha:
            with st.container():
                if df_model is not None:
                    # Crear figura con plotly
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_model.index, y=df_model['close_diff_pct'], mode='lines', name='Cambio de precio', line=dict(color='red')))
                    fig.add_trace(go.Scatter(x=df_model.index, y=df_model['news'], mode='lines', name='Score de noticias', line=dict(color='blue'), yaxis='y2'))
                    fig.update_layout(
                        title='Cambio de precio y sentimiento de noticias de Bitcoin',
                        xaxis_title='Fecha',
                        yaxis_title='Cambio de precio',
                        legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
                    )
                    fig.update_layout(
                        yaxis2=dict(
                            title='Score de noticias',
                            overlaying='y',
                            side='right',
                            showgrid=False,
                        )
                    )
                    fig.update_xaxes(
                        tickangle=45,
                        tickmode='auto',
                        range=[df_model.index.min(), df_model.index.max()],
                    )
                    # Mostrar el gráfico con Streamlit
                    st.plotly_chart(fig, use_container_width=True,)
            with st.container():
                mensaje = f"Predicción cambio de precio en {intervalo}: {'+' if prediccion >= 0 else ''}{100. * prediccion:.3f}%"
                if prediccion > 0:
                    st.success(mensaje)
                elif prediccion == 0:
                    st.warning(mensaje)
                else:
                    st.error(mensaje)
        with col_izquierda:
            with st.container():
                st.markdown('#### Últimas noticias sobre Bitcoin:')
                if df_news is not None:
                    df_news['label'] = df_news['label'].replace({
                        'Bearish': 'Bajista',
                        'Somewhat-Bearish': 'Algo bajista',
                        'Neutral': 'Neutral',
                        'Somewhat-Bullish': 'Algo alcista',
                        'Bullish': 'Alcista'
                    })
                    df_news['title'] = df_news.apply(lambda row: f'<a href="{row["url"]}" target="_blank">{row["title"]}</a>', axis=1)
                    html_table = df_news[['title', 'label']].to_html(header=False, escape=False, index=False)
                    styled_table = f'<div style="max-height:380px; overflow-y:auto;">{html_table}</div>'
                    st.markdown(styled_table, unsafe_allow_html=True)

# %% [markdown]
# ## Main

# %% [code] {"jupyter":{"outputs_hidden":false}}
def main():
    # Streamlit
    streamlit_app()

# %% [markdown]
# # Ejecución de funciones

# %% [code] {"jupyter":{"outputs_hidden":false}}
if __name__ == "__main__":
    main()
