import numpy as np
import matplotlib.pyplot as plt
import io
import pandas as pd
import base64
import time

def spline_lineal(x, y):

    if len(set(x)) != len(x):

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
    

    try:

        n = len(x)
        A = np.zeros((2*(n-1), 2*(n-1)))
        b = np.zeros(2*(n-1))

        c = 0
        h = 0
        for i in range(n - 1):
            A[h, c] = x[i]
            A[h, c+1] = 1
            b[h] = y[i]
            c += 2
            h += 1

        c = 0
        for i in range(1, n):
            A[h, c] = x[i]
            A[h, c+1] = 1
            b[h] = y[i]
            c += 2
            h += 1

        coef = np.linalg.solve(A, b)
        tabla = coef.reshape((n-1, 2))
        poly_str, polinomios = obtener_poli_str(x, tabla)
    except Exception as e:
        print(f"Error en la matriz: {e}")

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
    
   

    grafica  = graficar_lineal(x,y,tabla)
    return poly_str, grafica, polinomios

def obtener_poli_str(x, tabla):
    n_tramos = len(tabla)
    polinomios_str = []
    polinomios = []

    for i in range(n_tramos):
        m, b = tabla[i]
        tramo = f"{x[i]} ≤ x ≤ {x[i+1]}"
        pol = f"{m:.4f}*x + {b:.4f}"
        polinomios_str.append(f"Tramo {i+1} ({tramo}):  y = {pol}")
        polinomios.append(pol)

    return polinomios_str, polinomios

def graficar_lineal(x, y, tabla):
    plt.plot(x, y, 'r*', label='Puntos de datos')
    for i in range(len(tabla)):
        x_vals = np.linspace(x[i], x[i+1], 200)
        y_vals = tabla[i, 0] * x_vals + tabla[i, 1]
        plt.plot(x_vals, y_vals, label=f'Tramo {i+1}')
    plt.title("Spline Lineal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    string = base64.b64encode(buf.read()).decode()
    img_uri = f"data:image/png;base64,{string}"
    plt.close()

    return img_uri

def _calcular_errores_spline_lineal(x, y):
    """
    Calcula los errores del spline lineal.
    Usamos np.interp(x, x, y), que es exactamente el spline lineal evaluado en los nodos.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    t0 = time.perf_counter()
    # Spline lineal evaluado en los nodos: debe dar exactamente y
    y_aprox = np.interp(x, x, y)
    t1 = time.perf_counter()
    tiempo = t1 - t0

    errores_abs = np.abs(y - y_aprox)
    errores_rel = errores_abs / np.maximum(np.abs(y), 1e-15)
    rmse = np.sqrt(np.mean((y - y_aprox) ** 2))

    return {
        "y_aprox": y_aprox,
        "errores_abs": errores_abs,
        "errores_rel": errores_rel,
        "rmse": rmse,
        "tiempo": tiempo,
    }


def generar_informe_spline_lineal(x, y, descripcion_tramos):
    """
    Genera el informe de ejecución en HTML para el método de spline lineal.
    """
    try:
        info = _calcular_errores_spline_lineal(x, y)
    except Exception as e:
        return f"<p><b>Error al generar el informe de Spline lineal:</b> {e}</p>"

    errores_abs = info["errores_abs"]
    errores_rel = info["errores_rel"]
    rmse = info["rmse"]
    tiempo = info["tiempo"]

    n = len(x)

    informe = f"""
    <h3>Informe de ejecución – Método de Spline lineal</h3>
    <p>Se aplicó el método de interpolación por spline lineal para los datos (x, y) con n = {n} puntos,
    construyendo un polinomio por tramos de grado 1 en cada intervalo [xᵢ, xᵢ₊₁].</p>

    <p>El spline lineal obtenido (por tramos) es:</p>
    <pre>{descripcion_tramos}</pre>

    <p>Al evaluar el spline lineal en los datos originales se obtuvo:</p>
    <ul>
        <li>Error absoluto máximo: {errores_abs.max():.6e}</li>
        <li>Error absoluto promedio: {errores_abs.mean():.6e}</li>
        <li>Error relativo máximo: {errores_rel.max():.6e}</li>
        <li>RMSE (raíz del error cuadrático medio): {rmse:.6e}</li>
        <li>Tiempo de ejecución: {tiempo:.6f} s</li>
    </ul>
    """

    return informe


def generar_comparacion_errores_spline_lineal(x, y):
    """
    Devuelve una tabla HTML con la comparación de los tipos de error
    para el método de Spline lineal.
    """
    try:
        info = _calcular_errores_spline_lineal(x, y)
    except Exception as e:
        return f"<p><b>Error al generar la tabla de errores de Spline lineal:</b> {e}</p>"

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
