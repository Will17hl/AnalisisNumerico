import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import base64
import io
import time

def newtonint(x, y):
    
    if len(set(x)) != len(x):
        xpol = np.linspace(min(x)-1, max(x)+1, 500)

        plt.plot(x, y, 'r*', label='Puntos dados')
        plt.title('Interpolación por newton')
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
        plt.title('Interpolación por newton')
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
        n = len(x)
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)

        # Tabla de diferencias divididas
        tabla = np.zeros((n, n))
        tabla[:, 0] = y
        for j in range(1, n):
            for i in range(j, n):
                tabla[i, j] = (tabla[i, j-1] - tabla[i-1, j-1]) / (x[i] - x[i-j])

        # Coeficientes de la forma de Newton
        coef_newton = tabla[np.arange(n), np.arange(n)]

        # Convertir a forma estándar (coeficientes para np.polyval)
        poly_std = np.array([0.])
        base = np.array([1.])

        for i in range(n):
            term = coef_newton[i] * base
            poly_std = np.pad(poly_std, (len(term) - len(poly_std), 0), 'constant')
            term = np.pad(term, (len(poly_std) - len(term), 0), 'constant')
            poly_std = poly_std + term
            base = np.convolve(base, [1, -x[i]])  # (x - x_i)

        a = poly_std  # Coeficientes del polinomio en forma estándar
        grado = len(a) - 1
    except Exception as e:
        print(f"Error en la matriz: {e}")
        xpol = np.linspace(min(x)-1, max(x)+1, 500)

        plt.plot(x, y, 'r*', label='Puntos dados')
        plt.title('Interpolación por newton')
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

    # Construir string del polinomio
    terms = []
    for i, coef in enumerate(a):
        try:

            power = grado - i
            if abs(coef) < 1e-10:
                continue
            term = f"{coef:.4f}"
            if power > 0:
                term += f"*x^{power}" if power > 1 else "*x"
            terms.append(term)
        except Exception as e:
            print(f"Error en la matriz: {e}")

            xpol = np.linspace(min(x)-1, max(x)+1, 500)

            plt.plot(x, y, 'r*', label='Puntos dados')
            plt.title('Interpolación por newton')
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

    plt.plot(x, y, 'ro', label='Puntos dados')
    plt.plot(xpol, p, 'b-', label=f'Polinomio de grado {grado}')
    plt.title('Interpolación por diferencias divididas de Newton')
    plt.xlabel('x')
    plt.ylabel('P(x)')
    plt.grid(True)
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    string = base64.b64encode(buf.read()).decode()
    img_uri = f"data:image/png;base64,{string}"

    plt.close()

    return poly_str, img_uri

def _calcular_errores_newtonint(x, y, grado):
    """
    Calcula errores del polinomio interpolante asociado a Newton.
    Para los errores usamos el polinomio de interpolación de grado n-1
    obtenido por polyfit (es el mismo polinomio que Newton).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n = len(x)
    if grado is None or grado > n - 1:
        grado = n - 1

    t0 = time.perf_counter()

    # Polinomio interpolante en forma estándar
    coef = np.polyfit(x, y, grado)
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


def generar_informe_newtonint(x, y, grado, polinomio_str):
    """
    Genera el informe de ejecución en HTML para el método de Newton interpolante.
    """
    try:
        info = _calcular_errores_newtonint(x, y, grado)
    except Exception as e:
        return f"<p><b>Error al generar el informe de Newton interpolante:</b> {e}</p>"

    errores_abs = info["errores_abs"]
    errores_rel = info["errores_rel"]
    rmse = info["rmse"]
    tiempo = info["tiempo"]

    informe = f"""
    <h3>Informe de ejecución – Método de Newton interpolante</h3>
    <p>Se aplicó el método de interpolación de Newton para los datos (x, y) con n = {len(x)} puntos,
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


def generar_comparacion_errores_newtonint(x, y, grado):
    """
    Devuelve una tabla HTML con la comparación de los tipos de error
    para el método de Newton interpolante.
    """
    try:
        info = _calcular_errores_newtonint(x, y, grado)
    except Exception as e:
        return f"<p><b>Error al generar la tabla de errores de Newton interpolante:</b> {e}</p>"

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
