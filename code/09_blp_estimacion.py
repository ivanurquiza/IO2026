"""
Script 09: Ejercicio 4a. Estimacion del logit con coeficientes aleatorios
(BLP) sobre la muestra completa.

El modelo permite que los gustos varien entre consumidores:
  - beta_ib = sigma_B * v_i  : gusto no observado por cada marca madre
  - alpha_i = alpha + sigma_I * ingreso_i : sensibilidad al precio segun ingreso
Los shares agregados ya no tienen forma cerrada, de modo que PyBLP los
resuelve con el algoritmo de dos loops anidados (loop interno: invertir
los delta por punto fijo; loop externo: minimizar la funcion GMM).

Se estima dos veces, con dos valores iniciales distintos, como pide la
consigna: si ambos convergen al mismo theta, hay evidencia de que no es
un optimo local.

Correr:  python code/09_blp_estimacion.py
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyblp

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "output"
SALIDA.mkdir(exist_ok=True)

# ---- todo lo que se imprime va a la consola y a un log a la vez -------
class Tee:
    def __init__(self, *destinos):
        self.destinos = destinos
    def write(self, texto):
        for d in self.destinos:
            d.write(texto); d.flush()
    def flush(self):
        for d in self.destinos:
            d.flush()

log = open(SALIDA / "blp_log.txt", "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, log)

pyblp.options.verbose = True
pyblp.options.flush_output = True

# ---------- 1. datos ----------------------------------------------------
prod = pd.read_csv(RAIZ / "data" / "blp_productos.csv")
agen = pd.read_csv(RAIZ / "data" / "blp_agentes.csv")

# ---------- 2. centrar el ingreso --------------------------------------
# income_c = income - media. Es una reparametrizacion que no altera el
# ajuste, pero vuelve beta_precio interpretable como el coeficiente de
# precio de un hogar de ingreso medio (en lugar de uno de ingreso cero,
# que no existe) y desacopla el efecto principal del precio de su
# interaccion con el ingreso.
agen["income_c"] = agen["income"] - agen["income"].mean()

print("mercados: %d | filas: %d | agentes: %d"
      % (prod["market_ids"].nunique(), len(prod), len(agen)))

# ---------- 3. las tres formulaciones ----------------------------------
# X1 (lineal): coeficientes iguales para todos. Las 11 dummies de
# marca-tamano entran como C(marca); "0 +" evita la constante porque las
# dummies ya la cubren. Son theta_1 = (alpha, beta), que PyBLP concentra
# fuera resolviendo por 2SLS dado theta_2.
X1 = pyblp.Formulation("0 + prices + promo + C(product_ids)")

# X2 (no lineal): las 4 dummies de marca madre llevan sigma_B (gusto no
# observado); el precio va aqui para interactuar con el ingreso via pi.
X2 = pyblp.Formulation("0 + marca1 + marca2 + marca3 + marca4 + prices")

# demografico: el ingreso centrado, que interactua con el precio -> sigma_I
AG = pyblp.Formulation("0 + income_c")

problema = pyblp.Problem((X1, X2), prod, AG, agen)
print(problema)

# ---------- 4. valores iniciales ---------------------------------------
# sigma (5x5 diagonal): coef. aleatorios NO observados. El orden de X2 es
# [marca1, marca2, marca3, marca4, precio]. El precio queda fijo en 0
# porque su unica heterogeneidad viene del ingreso (via pi), no de un
# shock no observado. OJO: en PyBLP un cero se interpreta como restriccion
# (parametro fijo), no como valor inicial.
# pi (5x1): interacciones con el ingreso. Solo el precio interactua.
INICIALES = {
    "medios": (np.diag([0.5, 0.5, 0.5, 0.5, 0.0]),
               np.array([[0.0], [0.0], [0.0], [0.0], [-0.05]])),
    "altos":  (np.diag([1.5, 1.5, 1.5, 1.5, 0.0]),
               np.array([[0.0], [0.0], [0.0], [0.0], [-0.20]])),
}

# ---------- 5. estimar dos veces y guardar -----------------------------
resultados = {}
for nombre, (sigma0, pi0) in INICIALES.items():
    print("\n" + "=" * 70)
    print("VALOR INICIAL: %s" % nombre)
    print("=" * 70)
    t = time.time()
    res = problema.solve(
        sigma=sigma0,
        pi=pi0,
        method="1s",                    # una etapa, W = (Z'Z)^-1
        se_type="robust",
        optimization=pyblp.Optimization("l-bfgs-b", {"gtol": 1e-5}),
    )
    print(res)
    print("tiempo: %.1f min | objetivo GMM: %.6f"
          % ((time.time() - t) / 60, res.objective.item()))
    resultados[nombre] = res
    with open(SALIDA / f"blp_{nombre}.pkl", "wb") as f:
        pickle.dump(res, f)

print("\n" + "=" * 70 + "\nRESUMEN")
for nombre, res in resultados.items():
    s = np.diag(res.sigma)
    print("  %-7s GMM=%.4f | sigma_B=%s | pi=%.4f | converge=%s"
          % (nombre, res.objective.item(), np.round(s[:4], 3),
             res.pi[4, 0], res.converged))


# ---------- diagnostico de instrumentos --------------------------------
# Los 30 precios del mismo producto en distintas tiendas estan muy
# correlacionados entre si (el laboratorio fija un precio de lista que
# las tiendas mueven poco). Esto genera multicolinealidad en Z, que
# debilita la identificacion de los coeficientes de dispersion sigma.
# Reportamos el numero de condicion de Z como medida de ese problema.
iv = prod.filter(like="demand_instruments")
Z = (iv.values - iv.values.mean(0)) / iv.values.std(0)   # estandarizada
autovalores = np.linalg.eigvalsh(Z.T @ Z / len(Z))
num_condicion = autovalores[-1] / autovalores[0]

corr = iv.iloc[:, 1:].corr().values
corr_media = corr[np.triu_indices(corr.shape[0], k=1)].mean()

print("\n--- diagnostico de instrumentos ---")
print("numero de instrumentos: %d" % iv.shape[1])
print("numero de condicion de Z: %.0f" % num_condicion)
print("correlacion media entre los 30 precios-IV: %.3f" % corr_media)