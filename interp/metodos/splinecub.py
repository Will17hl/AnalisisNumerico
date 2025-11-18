import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import time
import pandas as pd

def spline_cubico(x, y):

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
        A = np.zeros((4*(n-1), 4*(n-1)))
        b = np.zeros(4*(n-1))

        c = 0
        h = 0
        for i in range(n - 1):
            A[h, c] = x[i]**3
            A[h, c+1] = x[i]**2
            A[h, c+2] = x[i]
            A[h, c+3] = 1
            b[h] = y[i]
            c += 4
            h += 1

        c = 0
        for i in range(1, n):
            A[h, c] = x[i]**3
            A[h, c+1] = x[i]**2
            A[h, c+2] = x[i]
            A[h, c+3] = 1
            b[h] = y[i]
            c += 4
            h += 1

        c = 0
        for i in range(1, n - 1):
            A[h, c] = 3*x[i]**2
            A[h, c+1] = 2*x[i]
            A[h, c+2] = 1
            A[h, c+4] = -3*x[i]**2
            A[h, c+5] = -2*x[i]
            A[h, c+6] = -1
            b[h] = 0
            c += 4
            h += 1

        c = 0
        for i in range(1, n - 1):
            A[h, c] = 6*x[i]
            A[h, c+1] = 2
            A[h, c+4] = -6*x[i]
            A[h, c+5] = -2
            b[h] = 0
            c += 4
            h += 1

        A[h, 0] = 6 * x[0]
        A[h, 1] = 2
        b[h] = 0
        h += 1

        A[h, -4] = 6 * x[-1]
        A[h, -3] = 2
        b[h] = 0

        coef = np.linalg.solve(A, b)
        tabla = coef.reshape((n-1, 4))
        poly_str, polinomios = obtener_poly_cub_str(x, tabla)
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
    
    grafica  = graficar_cubico(x,y,tabla)
    return poly_str, grafica, polinomios

def obtener_poly_cub_str(x, tabla):
    n_tramos = len(tabla)
    polinomios_str = []
    polinomios = []

    for i in range(n_tramos):
        a, b, c, d = tabla[i]
        tramo = f"{x[i]} ≤ x ≤ {x[i+1]}"
        pol = (f"{a:.4f}*x**3 + {b:.4f}*x**2 + {c:.4f}*x + {d:.4f}")
        polinomios_str.append(f"Tramo {i+1} ({tramo}):  y = {pol}")
        polinomios.append(pol)

    return polinomios_str, polinomios

def graficar_cubico(x, y, tabla):
    plt.plot(x, y, 'r*', label='Puntos de datos')
    for i in range(len(tabla)):
        x_vals = np.linspace(x[i], x[i+1], 200)
        y_vals = (tabla[i, 0]*x_vals**3 + tabla[i, 1]*x_vals**2 +
                  tabla[i, 2]*x_vals + tabla[i, 3])
        plt.plot(x_vals, y_vals, label=f'Tramo {i+1}')
    plt.title("Spline Cúbico")
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

def _evaluar_spline_en_puntos(x, tramos):
    """Evalúa el spline en los nodos x usando los tramos ya calculados."""
    x = np.array(x, dtype=float)
    y_spline = []

    for xp in x:
        if xp <= tramos[0][0]:
            i = 0
        elif xp >= tramos[-1][1]:
            i = len(tramos) - 1
        else:
            # tramos[i]: (xi, xi1, a, b, c, d)
            xs = [t[0] for t in tramos]
            i = np.searchsorted(xs, xp) - 1

        xi, xi1, a, b, c, d = tramos[i]
        t = xp - xi
        yxp = a + b * t + c * t ** 2 + d * t ** 3
        y_spline.append(yxp)

    return np.array(y_spline)


def generar_informe_spline_cub(x, y, descripcion_tramos):
    """
    Genera el texto del informe para mostrar en HTML.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # Recalcular tramos para poder evaluar (pequeña redundancia pero simple)
    n = len(x) - 1
    h = np.diff(x)
    A = np.zeros((n + 1, n + 1))
    b_vec = np.zeros(n + 1)
    A[0, 0] = 1
    A[n, n] = 1
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b_vec[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    M = np.linalg.solve(A, b_vec)

    tramos = []
    for i in range(n):
        xi, xi1 = x[i], x[i + 1]
        hi = h[i]
        Mi, Mi1 = M[i], M[i + 1]
        yi, yi1 = y[i], y[i + 1]

        a = yi
        b = (yi1 - yi) / hi - (2 * Mi + Mi1) * hi / 6
        c = Mi / 2
        d = (Mi1 - Mi) / (6 * hi)

        tramos.append((xi, xi1, a, b, c, d))

    # Errores
    t0 = time.perf_counter()
    y_spline = _evaluar_spline_en_puntos(x, tramos)
    t1 = time.perf_counter()
    tiempo = t1 - t0

    errores = np.abs(y - y_spline)
    err_abs_max = np.max(errores)
    err_abs_prom = np.mean(errores)
    err_rel_max = np.max(errores / np.maximum(np.abs(y), 1e-14))
    rmse = np.sqrt(np.mean(errores ** 2))

    informe = f"""
Se aplicó el método de Spline cúbico para los datos (x, y) con n = {len(x)} puntos,
construyendo un polinomio cúbico por tramos S(x) en cada intervalo [xᵢ, xᵢ₊₁].

El spline cúbico obtenido (por tramos) es:
{descripcion_tramos}

Al evaluar el spline en los datos originales se obtuvo:
• Error absoluto máximo: {err_abs_max:.8e}
• Error absoluto promedio: {err_abs_prom:.8e}
• Error relativo máximo: {err_rel_max:.8e}
• RMSE (raíz del error cuadrático medio): {rmse:.8e}
• Tiempo de ejecución: {tiempo:.6f} s
"""
    return informe


def generar_comparacion_errores_spline_cub(x, y):
    """
    Devuelve una tabla HTML con la comparación de errores del spline cúbico.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x) - 1
    h = np.diff(x)
    A = np.zeros((n + 1, n + 1))
    b_vec = np.zeros(n + 1)
    A[0, 0] = 1
    A[n, n] = 1
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b_vec[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    M = np.linalg.solve(A, b_vec)

    tramos = []
    for i in range(n):
        xi, xi1 = x[i], x[i + 1]
        hi = h[i]
        Mi, Mi1 = M[i], M[i + 1]
        yi, yi1 = y[i], y[i + 1]

        a = yi
        b = (yi1 - yi) / hi - (2 * Mi + Mi1) * hi / 6
        c = Mi / 2
        d = (Mi1 - Mi) / (6 * hi)

        tramos.append((xi, xi1, a, b, c, d))

    # Errores + tiempos
    mediciones = []

    # Error absoluto máximo
    t0 = time.perf_counter()
    y_spline = _evaluar_spline_en_puntos(x, tramos)
    err_abs_max = np.max(np.abs(y - y_spline))
    t1 = time.perf_counter()
    mediciones.append(["Error absoluto máximo", f"{err_abs_max:.8e}", f"{t1 - t0:.6f}"])

    # Error absoluto promedio
    t0 = time.perf_counter()
    err_abs_prom = np.mean(np.abs(y - y_spline))
    t1 = time.perf_counter()
    mediciones.append(["Error absoluto promedio", f"{err_abs_prom:.8e}", f"{t1 - t0:.6f}"])

    # Error relativo máximo
    t0 = time.perf_counter()
    err_rel_max = np.max(np.abs((y - y_spline) / np.maximum(np.abs(y), 1e-14)))
    t1 = time.perf_counter()
    mediciones.append(["Error relativo máximo", f"{err_rel_max:.8e}", f"{t1 - t0:.6f}"])

    # RMSE
    t0 = time.perf_counter()
    rmse = np.sqrt(np.mean((y - y_spline) ** 2))
    t1 = time.perf_counter()
    mediciones.append(["RMSE (error cuadrático medio)", f"{rmse:.8e}", f"{t1 - t0:.6f}"])

    df = pd.DataFrame(mediciones, columns=["Tipo de error", "Valor", "Tiempo (s)"])
    return df.to_html(index=False, classes='table table-striped text-center')