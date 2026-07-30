"""
TP1 - Metodos Econometricos con Aplicacion a IO (UdeSA, 2026)
Script 01: construccion de la base de trabajo.

Correr desde la raiz del proyecto:  python code/01_datos.py

"""

import numpy as np
import pandas as pd
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
EXCEL = RAIZ / "data" / "TP_data.xlsx"
SALIDA = RAIZ / "data" / "base_limpia.csv"

# --- 1. lectura ---------------------------------------------------------
ventas = pd.read_excel(EXCEL, sheet_name="DATA de Medicamentos")
demog = pd.read_excel(EXCEL, sheet_name="Variables demograficas")
demog = demog[["tienda", "cant mujeres", "educacion", "ingreso"]]
demog = demog.rename(columns={"cant mujeres": "prop_mujeres"})

print("ventas:", ventas.shape, "| demograficas:", demog.shape)

# --- 2. mapeo marca -> marca madre y tamano ----------------------------
MAPA_MARCA = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2,
              7: 3, 8: 3, 9: 3, 10: 4, 11: 4}
MAPA_TAM = {1: 25, 2: 50, 3: 100, 4: 25, 5: 50, 6: 100,
            7: 25, 8: 50, 9: 100, 10: 50, 11: 100}

df = ventas.rename(columns={"marca": "producto"}).copy()
df["marca"] = df["producto"].map(MAPA_MARCA)
df["tamano"] = df["producto"].map(MAPA_TAM)

# --- 3. mercado, tamano de mercado y shares ----------------------------
# Un mercado = una tienda en una semana.
df["mercado"] = df["tienda"].astype(str) + "-" + df["semana"].astype(str)

# Tamano de mercado: se calibra con los market shares de la Tabla 1 de la
# consigna, que suman 64%. El share del bien externo es entonces 36% y el
# mercado potencial de cada tienda-semana se obtiene escalando sus ventas
# internas:   M_t = ventas_internas_t / (1 - s_0)
SHARES_CONSIGNA = [8.90, 11.10, 7.60, 9.30, 5.10, 2.20,
                   2.50, 1.00, 4.90, 7.20, 4.20]
SHARE_0 = 1.0 - sum(SHARES_CONSIGNA) / 100.0

df["ventas_internas"] = df.groupby("mercado")["ventas"].transform("sum")
df["M"] = df["ventas_internas"] / (1.0 - SHARE_0)

df["share"] = df["ventas"] / df["M"]
df["share_int"] = df["ventas"] / df["ventas_internas"]
df["share_0"] = SHARE_0

# Variable dependiente del logit (inversion de Berry 1994)
df["y_logit"] = np.log(df["share"]) - np.log(SHARE_0)

# --- 4. merge demograficas ---------------------------------------------
df = df.merge(demog, on="tienda", how="left", validate="many_to_one")
assert df["ingreso"].notna().all(), "hay tiendas sin demografica"

# --- 5. diagnosticos ----------------------------------------------------
print("\n--- diagnosticos ---")
print("mercados (tienda-semana):", df["mercado"].nunique())
print("productos por mercado:", df.groupby("mercado").size().unique())
print("ventas en cero:", int((df["ventas"] == 0).sum()))
print("share del bien externo: %.4f (constante por construccion)" % SHARE_0)
print("suma de shares interiores por mercado: %.4f"
      % df.groupby("mercado")["share"].sum().mean())
print("share individual: media %.4f  [%.4f, %.4f]"
      % (df["share"].mean(), df["share"].min(), df["share"].max()))
print("y_logit: media %.3f  sd %.3f  [%.2f, %.2f]"
      % (df["y_logit"].mean(), df["y_logit"].std(),
         df["y_logit"].min(), df["y_logit"].max()))

print("\nresumen por marca:")
print(df.groupby(["producto", "marca", "tamano"]).agg(
    precio=("precio", "mean"), costo=("costo", "mean"),
    share_pct=("share", lambda s: 100 * s.mean()),
).round(3))

SALIDA.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(SALIDA, index=False)
print("\nguardado en:", SALIDA)
