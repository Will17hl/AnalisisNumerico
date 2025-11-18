import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import base64
import io
import time  # 👈 para medir tiempo de ejecución


def vandermonde(x, y, grado):
    if len(set(x)) != len(x):
        xpol = np.linspace(min(x)-1, max(x)+1, 500)

        plt.plot(x, y, 'r*', label='Puntos dados')
        plt.title('Interpolación por Vandermonde')  # 👈 antes decía Lagrange
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
        plt.title('Interpolación por Vandermonde')
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
    
    try:
        A = np.vander(x, N=grado + 1, increasing=False)
        a = np.linalg.solve(A, y)
    except Exception as e:
        print(f"Error en la matriz: {e}")
        xpol = np.linspace(min(x)-1, max(x)+1, 500)

        plt.plot(x, y, 'r*', label='Puntos dados')
        plt.title('Interpolación por Vandermonde')
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

    # Construcción de la cadena que representa el polinomio
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
            plt.title('Interpolación por Vandermonde')
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

    np.set_printoptions(precision=4, suppress=True)

    # Crear valores para graficar el polinomio
    xpol = np.linspace(min(x)-1, max(x)+1, 500)
    p = np.polyval(a, xpol)  # Evalúa el polinomio usando los coeficientes

    # Graficar puntos y polinomio
    plt.plot(x, y, 'r*', label='Puntos dados')
    plt.plot(xpol, p, 'b-', label=f'Polinomio de grado {grado}')
    plt.title('Interpolación con matriz de Vandermonde')
    plt.legend()
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    string = base64.b64encode(buf.read()).decode()
    img_uri = f"data:image/png;base64,{string}"

    plt.close()

    return poly_str, img_uri


# ============================================================
#  NUEVAS FUNCIONES PARA INFORME Y COMPARACIÓN DE ERRORES
# ============================================================

def _calcular_errores_vandermonde(x, y, grado):
    """
    Resuelve nuevamente el sistema de Vandermonde para obtener
    coeficientes, evalúa el polinomio en los puntos dados y calcula
    errores y tiempo de ejecución.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    t0 = time.perf_counter()

    A = np.vander(x, N=grado + 1, increasing=False)
    a = np.linalg.solve(A, y)

    # valores aproximados en los puntos originales
    y_aprox = np.polyval(a, x)

    t1 = time.perf_counter()
    tiempo = t1 - t0

    errores_abs = np.abs(y - y_aprox)
    errores_rel = errores_abs / np.maximum(np.abs(y), 1e-15)
    rmse = np.sqrt(np.mean((y - y_aprox) ** 2))

    return {
        "coeficientes": a,
        "y_aprox": y_aprox,
        "errores_abs": errores_abs,
        "errores_rel": errores_rel,
        "rmse": rmse,
        "tiempo": tiempo,
    }


def generar_informe_vandermonde(x, y, grado, polinomio_str):
    """
    Genera el informe de ejecución en HTML para mostrar en la interfaz.
    No modifica nada del comportamiento original de vandermonde().
    """
    try:
        info = _calcular_errores_vandermonde(x, y, grado)
    except Exception as e:
        return f"<p><b>Error al generar el informe de Vandermonde:</b> {e}</p>"

    errores_abs = info["errores_abs"]
    errores_rel = info["errores_rel"]
    rmse = info["rmse"]
    tiempo = info["tiempo"]

    informe = f"""
    <h3>Informe de ejecución – Método de Vandermonde</h3>
    <p>Se aplicó el método de Vandermonde para interpolar los datos (x, y) con n = {len(x)} puntos,
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


def generar_comparacion_errores_vandermonde(x, y, grado):
    """
    Devuelve una tabla HTML con la comparación de tipos de error
    (similar a las del Capítulo 2).
    """
    try:
        info = _calcular_errores_vandermonde(x, y, grado)
    except Exception as e:
        return f"<p><b>Error al generar la tabla de errores de Vandermonde:</b> {e}</p>"

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