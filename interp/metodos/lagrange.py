import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import base64
import time
import io

def lagrange(x, y):
    if len(set(x)) != len(x):
        xpol = np.linspace(min(x)-1, max(x)+1, 500)

        plt.plot(x, y, 'r*', label='Puntos dados')
        plt.title('Interpolación por Lagrange')
        plt.legend()
        plt.grid(True)
        plt.xlabel('x')
        plt.ylabel('P(x)')

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        string = base64.b64encode(buf.read()).decode()
        img_uri = f"data:image/png;base64,{string}"

        plt.close()

        return "Error", img_uri
    
    if len(x) != len(y):
        xpol = np.linspace(min(x)-1, max(x)+1, 500)

        plt.plot(x, y, 'r*', label='Puntos dados')
        plt.title('Interpolación por Lagrange')
        plt.legend()
        plt.grid(True)
        plt.xlabel('x')
        plt.ylabel('P(x)')

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        string = base64.b64encode(buf.read()).decode()
        img_uri = f"data:image/png;base64,{string}"

        plt.close()

        return "Error", img_uri
    
    n = len(x)
    tabla = np.zeros((n, n))

    try:

        for i in range(n):
            Li = np.array([1.0])
            den = 1.0
            for j in range(n):
                if j != i:
                    paux = np.array([1.0, -x[j]])
                    Li = np.convolve(Li, paux)
                    den *= (x[i] - x[j])
            tabla[i, :len(Li)] = y[i] * Li / den

        a = np.sum(tabla, axis=0)  # coeficientes del polinomio

        # Eliminar ceros a la izquierda (por si el grado es menor a n-1)
        a = np.trim_zeros(a, 'f')
        grado = len(a) - 1
    except Exception as e:
        print(f"Error en la matriz: {e}")
        xpol = np.linspace(min(x)-1, max(x)+1, 500)

        plt.plot(x, y, 'r*', label='Puntos dados')
        plt.title('Interpolación por Lagrange')
        plt.legend()
        plt.grid(True)
        plt.xlabel('x')
        plt.ylabel('P(x)')

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        string = base64.b64encode(buf.read()).decode()
        img_uri = f"data:image/png;base64,{string}"

        plt.close()

        return "Error", img_uri

    # Construcción del string
    terms = []
    for i, coef in enumerate(a):
        try:

            power = grado - i
            if abs(coef) < 1e-10:
                continue  # Omitir términos nulos
            term = f"{coef:.4f}"
            if power > 0:
                term += f"*x^{power}" if power > 1 else "*x"
            terms.append(term)
        except Exception as e:
            print(f"Error en la matriz: {e}")

            xpol = np.linspace(min(x)-1, max(x)+1, 500)

            plt.plot(x, y, 'r*', label='Puntos dados')
            plt.title('Interpolación por Lagrange')
            plt.legend()
            plt.grid(True)
            plt.xlabel('x')
            plt.ylabel('P(x)')

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            string = base64.b64encode(buf.read()).decode()
            img_uri = f"data:image/png;base64,{string}"

            plt.close()

            return "Error", img_uri
        
    poly_str = " + ".join(terms).replace("+ -", "- ")

    xpol = np.linspace(min(x)-1, max(x)+1, 500)
    p = np.polyval(a, xpol)

    plt.plot(x, y, 'r*', label='Puntos dados')
    plt.plot(xpol, p, 'b-', label=f'Polinomio de grado {grado}')
    plt.title('Interpolación por Lagrange')
    plt.legend()
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('P(x)')

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    string = base64.b64encode(buf.read()).decode()
    img_uri = f"data:image/png;base64,{string}"

    plt.close()

    return poly_str, img_uri

def _calcular_errores_lagrange(x, y, grado):
    """
    Reconstruye el polinomio de Lagrange, lo evalúa en los puntos dados
    y calcula errores + tiempo de ejecución.
    NO toca tu función lagrange() original.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n = len(x)
    if grado is None or grado > n - 1:
        grado = n - 1

    t0 = time.perf_counter()

    # Construimos el polinomio mediante bases de Lagrange
    # Representamos el polinomio como coeficientes en forma estándar:
    # p(x) = c0*x^grado + c1*x^(grado-1) + ... + c_grado
    coef = np.zeros(grado + 1)

    for k in range(n):
        # L_k(x) = prod_{j != k} (x - x_j) / (x_k - x_j)
        # Primero el numerador como polinomio
        Lk = np.array([1.0])   # polinomio 1
        denom = 1.0
        for j in range(n):
            if j == k:
                continue
            # Convolución con (x - x_j)
            Lk = np.convolve(Lk, np.array([1.0, -x[j]]))
            denom *= (x[k] - x[j])

        Lk = Lk / denom  # normalizamos

        # Ajustar tamaño de Lk a grado+1 (por si acaso)
        if len(Lk) < grado + 1:
            Lk = np.pad(Lk, (grado + 1 - len(Lk), 0))
        elif len(Lk) > grado + 1:
            Lk = Lk[-(grado + 1):]

        coef += y[k] * Lk

    # Evaluar polinomio en los puntos originales
    y_aprox = np.polyval(coef, x)

    t1 = time.perf_counter()
    tiempo = t1 - t0

    errores_abs = np.abs(y - y_aprox)
    errores_rel = errores_abs / np.maximum(np.abs(y), 1e-15)
    rmse = np.sqrt(np.mean((y - y_aprox) ** 2))

    return {
        "coeficientes": coef,
        "y_aprox": y_aprox,
        "errores_abs": errores_abs,
        "errores_rel": errores_rel,
        "rmse": rmse,
        "tiempo": tiempo,
    }


def generar_informe_lagrange(x, y, grado, polinomio_str):
    """
    Genera el informe de ejecución en HTML para Lagrange.
    """
    try:
        info = _calcular_errores_lagrange(x, y, grado)
    except Exception as e:
        return f"<p><b>Error al generar el informe de Lagrange:</b> {e}</p>"

    errores_abs = info["errores_abs"]
    errores_rel = info["errores_rel"]
    rmse = info["rmse"]
    tiempo = info["tiempo"]

    informe = f"""
    <h3>Informe de ejecución – Método de Lagrange</h3>
    <p>Se aplicó el método de interpolación de Lagrange para los datos (x, y) con n = {len(x)} puntos,
    construyendo un polinomio de grado {grado}.</p>

    <p>El polinomio obtenido es:</p>
    <pre>p(x) = {polinomio_str}</pre>

    <p>Al evaluar el polinomio en los datos originales se obtuvo:</p>
    <ul>
        <li>Error absoluto máximo: {errores_abs.max():.6e}</li>
        <li>Error absoluto promedio: {errores_abs.mean():.6e}</li>
        <li>Error relativo máximo: {errores_rel.max():.6e}</li>
        <li>RMSE (raíz del error cuadrático medio): {rmse:.6e}</li>
        <li>Tiempo de ejecución: {tiempo:.6f} s</li>
    </ul>
    """

    return informe


def generar_comparacion_errores_lagrange(x, y, grado):
    """
    Devuelve tabla HTML con los distintos tipos de error para Lagrange.
    """
    try:
        info = _calcular_errores_lagrange(x, y, grado)
    except Exception as e:
        return f"<p><b>Error al generar la tabla de errores de Lagrange:</b> {e}</p>"

    errores_abs = info["errores_abs"]
    errores_rel = info["errores_rel"]
    rmse = info["rmse"]
    tiempo = info["tiempo"]

    data = {
        "Tipo de error": [
            "Error absoluto máximo",
            "Error absoluto promedio",
            "Error relativo máximo",
            "RMSE (error cuadrático medio)",
        ],
        "Valor": [
            errores_abs.max(),
            errores_abs.mean(),
            errores_rel.max(),
            rmse,
        ],
        "Tiempo (s)": [
            tiempo,
            tiempo,
            tiempo,
            tiempo,
        ],
    }

    df = pd.DataFrame(data)
    tabla_html = df.to_html(index=False, classes="table table-striped text-center")
    return tabla_html
