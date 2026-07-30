"""
Script 08: Ejercicio 4. Construccion de los insumos para PyBLP.

Genera dos archivos:
  data/blp_productos.csv  -> product_data (una fila por marca-mercado)
  data/blp_agentes.csv    -> agent_data   (una fila por consumidor simulado)

Correr:  python code/08_datos_blp.py
"""

import numpy as np
import pandas as pd
import pyblp
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
N_TIENDAS_IV = 30  # tiendas usadas para los instrumentos de precios
SEMILLA = 2026

df = pd.read_csv(RAIZ / "data" / "base_limpia.csv")

# ---------- product_data ------------------------------------------------
p = pd.DataFrame({
    "market_ids": df["mercado"],
    "product_ids": df["producto"],
    "firm_ids": df["marca"],          # 4 marcas madre = 4 duenos
    "shares": df["share"],
    "prices": df["precio"],
    "promo": df["descuento"],
    "tienda": df["tienda"],
    "semana": df["semana"],
    "costo": df["costo"],
    "ingreso": df["ingreso"],
})

# dummies de marca madre: van en X2 (coeficientes aleatorios)
for b in range(1, 5):
    p[f"marca{b}"] = (df["marca"] == b).astype(float)

# ---------- instrumentos ------------------------------------------------
# El precio es endogeno: correlaciona con la calidad no observada xi. Se
# instrumenta con (i) el costo del fabricante y (ii) el precio del MISMO
# producto, la MISMA semana, en 30 tiendas de referencia. La logica es la
# de Hausman: esos precios comparten los shocks de costo del laboratorio
# pero no el shock de demanda local del mercado propio.
#
# A diferencia del Ejercicio 3, cada tienda entra como instrumento
# separado y no promediada. La razon es la condicion de orden: theta
# incluye ahora los cinco parametros de dispersion, de modo que hacen
# falta mas instrumentos que los que exigiria la sola endogeneidad.

# Las 30 tiendas se sortean con semilla fija, para que la seleccion sea
# identica en cualquier maquina.
rng = np.random.default_rng(SEMILLA)
tiendas = np.sort(df["tienda"].unique())
elegidas = np.sort(rng.choice(tiendas, size=N_TIENDAS_IV, replace=False))

# pivot_table da vuelta la base: filas = (semana, producto), columnas =
# tienda, celdas = precio. Es la tabla que hay que consultar para saber
# cuanto costaba el producto j en la tienda s durante la semana t.
panel = df.pivot_table(index=["semana", "producto"], columns="tienda",
                       values="precio")

# reindex alinea el panel con las 38.544 filas de la base: para cada
# observacion busca la fila (semana, producto) que le corresponde.
idx = pd.MultiIndex.from_arrays([df["semana"], df["producto"]])
P = panel[elegidas].reindex(idx).to_numpy()          # (38.544, 30)

# Correccion imprescindible: si la tienda de la observacion esta entre las
# 30 elegidas, esa columna contiene su PROPIO precio, que es la variable
# endogena. Ocurre en el 41% de las filas. Se reemplaza por el promedio de
# las otras 29 (leave-one-out).
es_propia = df["tienda"].to_numpy()[:, None] == elegidas[None, :]
media_loo = np.nanmean(np.where(es_propia, np.nan, P), axis=1)
P = np.where(es_propia, media_loo[:, None], P)

p["demand_instruments0"] = df["costo"].to_numpy()
for k in range(N_TIENDAS_IV):
    p[f"demand_instruments{k + 1}"] = P[:, k]

assert p.filter(like="demand_instruments").notna().all().all()
print("instrumentos:", p.filter(like="demand_instruments").shape[1])
print("tiendas de referencia:", list(elegidas))
print("filas con tienda propia entre las 30 (corregidas): %d"
      % es_propia.any(axis=1).sum())

# ---------- agent_data --------------------------------------------------
# La integral que define los shares no tiene forma cerrada, asi que se
# aproxima por un promedio sobre consumidores artificiales. Cada uno
# necesita:
#   - K2 = 5 nodos v_i, uno por cada columna de X2 (4 marcas + precio)
#   - 1 nodo adicional para simular su ingreso
# Se usan sorteos de Halton en lugar de Monte Carlo puro: cubren el
# espacio de manera mas pareja y reducen el error de integracion para un
# mismo numero de nodos.
NS = 500          # consumidores artificiales por mercado
K2 = 5            # columnas de X2
TAU = 0.5         # desvio del log-ingreso DENTRO de cada tienda (supuesto)

integracion = pyblp.build_integration(
    pyblp.Integration("halton", NS, {"seed": SEMILLA}), K2 + 1)
nodos = integracion.nodes            # (500, 6)
pesos = integracion.weights.flatten()  # (500,)
nodos_v = nodos[:, :K2]              # columnas 0-4: los v_i de X2
nodo_ing = nodos[:, K2]              # columna 5: el shock de ingreso

# El ingreso. La base trae un unico log-ingreso por tienda, de modo que
# usarlo tal cual dejaria a todos los hogares de un mercado con el mismo
# ingreso: sigma_I no generaria heterogeneidad alguna dentro del mercado y
# el coeficiente aleatorio perderia su razon de ser. Simulamos entonces
# una distribucion lognormal alrededor de la media de cada tienda:
#     log I_i = m_s + TAU * w_i,   w_i ~ N(0,1)
# TAU es un supuesto: no se observa la dispersion intra-tienda. Como
# referencia, el desvio del log-ingreso ENTRE tiendas es 0,27.
# Se divide por 10.000 para que sigma_I quede en una escala legible.
log_ing_tienda = df.groupby("tienda")["ingreso"].first()
mercados = df[["mercado", "tienda"]].drop_duplicates().reset_index(drop=True)
T = len(mercados)

a = pd.DataFrame({
    "market_ids": np.repeat(mercados["mercado"].values, NS),
    "weights": np.tile(pesos, T),
})
for k in range(K2):
    a[f"nodes{k}"] = np.tile(nodos_v[:, k], T)

m_s = log_ing_tienda.loc[mercados["tienda"]].to_numpy()   # (T,)
# outer: cada mercado t combinado con cada nodo de ingreso
a["income"] = (np.exp(m_s[:, None] + TAU * nodo_ing[None, :]) / 10000.0).ravel()

print("agentes por mercado: %d | nodos por agente: %d" % (NS, K2 + 1))
print("ingreso simulado (x10.000): min %.2f | mediana %.2f | max %.2f"
      % (a["income"].min(), a["income"].median(), a["income"].max()))

# ---------- guardado ----------------------------------------------------
p.to_csv(RAIZ / "data" / "blp_productos.csv", index=False)
a.to_csv(RAIZ / "data" / "blp_agentes.csv", index=False)
print("productos:", p.shape, "| agentes:", a.shape)
print("guardado en data/blp_productos.csv y data/blp_agentes.csv")
