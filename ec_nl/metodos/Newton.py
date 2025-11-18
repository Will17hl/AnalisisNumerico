import matplotlib
matplotlib.use('Agg')
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import io
import base64


def metodo_newton(X0, Tol, Niter, Fun, df_expr):
    fn = []
    xn = []
    E = []
    N = []
    x = X0
    
    
    
    try:
        c = 0
        f = eval(Fun, {"x": x, "math": math})
        derivada = eval(df_expr, {"x": x, "math": math})
        Error = 100
        fn.append(f)
        xn.append(x)
        E.append(Error)
        N.append(c)

        tabla = []
        tabla.append([c, x, f, Error])
    except Exception as e:
        print(f"1: {e}")

        tabla = [[0, 0, 0, 0]]

        df_resultado = pd.DataFrame(tabla, columns=["I", "Xi", "F(Xi)", "E"])
        tabla_html = df_resultado.to_html(index=False, classes='table table-striped text-center')

        x_vals = np.linspace(min(xn) - 1, max(xn) + 1, 100)
        y_vals = [eval(Fun, {"x": val, "math": math}) for val in x_vals]

        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, label=f'f(x) = {Fun}', color='blue')
        plt.axhline(0, color='black', linewidth=1)
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Método de Newton")
        plt.legend()
        plt.grid()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        string = base64.b64encode(buf.read()).decode()
        img_uri = f"data:image/png;base64,{string}"

        plt.close()

        resultado = "Error"

        return resultado, tabla_html, img_uri

    while Error > Tol and f != 0 and derivada != 0 and c < Niter:
        
        try:

            x = x - f / derivada
            derivada = eval(df_expr, {"x": x, "math": math})
            f = eval(Fun, {"x": x, "math": math})
            fn.append(f)
            xn.append(x)
            c += 1
            Error = abs(xn[c] - xn[c - 1])
            N.append(c)
            E.append(Error)
            tabla.append([c, x, f, Error])
        except Exception as e:
            print(f"2: {e}")

            df_resultado = pd.DataFrame(tabla, columns=["I", "Xi", "F(Xi)", "E"])
            tabla_html = df_resultado.to_html(index=False, classes='table table-striped text-center')

            x_vals = np.linspace(min(xn) - 1, max(xn) + 1, 100)
            y_vals = [eval(Fun, {"x": val, "math": math}) for val in x_vals]

            plt.figure(figsize=(8, 6))
            plt.plot(x_vals, y_vals, label=f'f(x) = {Fun}', color='blue')
            plt.axhline(0, color='black', linewidth=1)
            plt.xlabel("x")
            plt.ylabel("f(x)")
            plt.title("Método de Newton")
            plt.legend()
            plt.grid()

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            string = base64.b64encode(buf.read()).decode()
            img_uri = f"data:image/png;base64,{string}"

            plt.close()

            resultado = "Error"

            return resultado, tabla_html, img_uri

        

    if f == 0:
        resultado = f"{x} es raíz de f(x)"
    elif Error < Tol:
        resultado = f"{x} es una aproximación de una raíz de f(x) con tolerancia {Tol}"
    else:
        df_resultado = pd.DataFrame(tabla, columns=["I", "Xi", "F(Xi)", "E"])
        tabla_html = df_resultado.to_html(index=False, classes='table table-striped text-center')

        x_vals = np.linspace(min(xn) - 1, max(xn) + 1, 100)
        y_vals = [eval(Fun, {"x": val, "math": math}) for val in x_vals]

        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, label=f'f(x) = {Fun}', color='blue')
        plt.axhline(0, color='black', linewidth=1)
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Método de Newton")
        plt.legend()
        plt.grid()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        string = base64.b64encode(buf.read()).decode()
        img_uri = f"data:image/png;base64,{string}"

        plt.close()

        resultado = f'Error en la iteracion {Niter}, ultima aproximacion: {x}'

        return resultado, tabla_html, img_uri

    df_resultado = pd.DataFrame(tabla, columns=["I", "Xi", "F(Xi)", "E"])
    tabla_html = df_resultado.to_html(index=False, classes='table table-striped text-center')

    x_vals = np.linspace(min(xn) - 1, max(xn) + 1, 100)
    y_vals = [eval(Fun, {"x": val, "math": math}) for val in x_vals]

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label=f'f(x) = {Fun}', color='blue')
    plt.axhline(0, color='black', linewidth=1)
    plt.scatter(xn[-1], 0, color='red', zorder=5, label=f'Raíz: {round(xn[-1], 4)}')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Método de Newton")
    plt.legend()
    plt.grid()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    string = base64.b64encode(buf.read()).decode()
    img_uri = f"data:image/png;base64,{string}"

    plt.close()

    return resultado, tabla_html, img_uri


def metodo_newtonCS(X0, Tol, Niter, Fun, df_expr):
    fn = []
    xn = []
    E = []
    e = []
    N = []
    x = X0
    
    try:
        c = 0
        error = 100

        f_lambda = lambda x: eval(Fun, {"x": x, "math": math})
        df_lambda = lambda x: eval(df_expr, {"x": x, "math": math})

        
        f = f_lambda(x)
        derivada = df_lambda(x)

        fn.append(f)
        xn.append(x)
        E.append(error)
        e.append(error)
        N.append(c)

        tabla = [[c, x, f, error]]

    except Exception as e:
        print(f"1: {e}")

        tabla = [[0, 0, 0, 0]]

        df_resultado = pd.DataFrame(tabla, columns=["Iteración", "Xi", "F(Xi)", "Error relativo"])
        tabla_html = df_resultado.to_html(index=False, classes='table table-striped text-center')

        x_vals = np.linspace(min(xn) - 1, max(xn) + 1, 100)
        y_vals = [f_lambda(val) for val in x_vals]

        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, label=f'f(x) = {Fun}', color='blue')
        plt.axhline(0, color='black', linewidth=1)
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Método de Newton")
        plt.legend()
        plt.grid()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        string = base64.b64encode(buf.read()).decode()
        img_uri = f"data:image/png;base64,{string}"

        plt.close()

        resultado = "Error"

        return resultado, tabla_html, img_uri

    while error > Tol and abs(f) > Tol and c < Niter:
        
        try:

            x_new = x - f / derivada

            derivada = df_lambda(x_new)
            f = f_lambda(x_new)

            Error = abs(x_new - x)
            error = Error / abs(x_new) if x_new != 0 else Error

            c += 1
            fn.append(f)
            xn.append(x_new)
            N.append(c)
            E.append(Error)
            e.append(error)
            tabla.append([c, x_new, f, error])

            x = x_new
        
        except Exception as e:
            print(f"2: {e}")

            df_resultado = pd.DataFrame(tabla, columns=["Iteración", "Xi", "F(Xi)", "Error relativo"])
            tabla_html = df_resultado.to_html(index=False, classes='table table-striped text-center')

            x_vals = np.linspace(min(xn) - 1, max(xn) + 1, 100)
            y_vals = [f_lambda(val) for val in x_vals]

            plt.figure(figsize=(8, 6))
            plt.plot(x_vals, y_vals, label=f'f(x) = {Fun}', color='blue')
            plt.axhline(0, color='black', linewidth=1)
            plt.xlabel("x")
            plt.ylabel("f(x)")
            plt.title("Método de Newton")
            plt.legend()
            plt.grid()

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            string = base64.b64encode(buf.read()).decode()
            img_uri = f"data:image/png;base64,{string}"

            plt.close()

            resultado = "Error"

            return resultado, tabla_html, img_uri
        
    if abs(f) <= Tol:
        resultado = f"{x} es una raíz de f(x) con tolerancia {Tol}"
    elif error < Tol:
        resultado = f"{x} es una aproximación de una raíz de f(x) con tolerancia {Tol}"
    else:
        df_resultado = pd.DataFrame(tabla, columns=["I", "Xi", "F(Xi)", "E"])
        tabla_html = df_resultado.to_html(index=False, classes='table table-striped text-center')

        x_vals = np.linspace(min(xn) - 1, max(xn) + 1, 100)
        y_vals = [eval(Fun, {"x": val, "math": math}) for val in x_vals]

        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, label=f'f(x) = {Fun}', color='blue')
        plt.axhline(0, color='black', linewidth=1)
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Método de Newton")
        plt.legend()
        plt.grid()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        string = base64.b64encode(buf.read()).decode()
        img_uri = f"data:image/png;base64,{string}"

        plt.close()

        resultado = f'Error en la iteracion {Niter}, ultima aproximacion: {x}'

        return resultado, tabla_html, img_uri

    df_resultado = pd.DataFrame(tabla, columns=["Iteración", "Xi", "F(Xi)", "Error relativo"])
    tabla_html = df_resultado.to_html(index=False, classes='table table-striped text-center')

    x_vals = np.linspace(min(xn) - 1, max(xn) + 1, 100)
    y_vals = [f_lambda(val) for val in x_vals]

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label=f'f(x) = {Fun}', color='blue')
    plt.axhline(0, color='black', linewidth=1)
    plt.scatter(xn[-1], 0, color='red', zorder=5, label=f'Raíz: {round(xn[-1], 4)}')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Método de Newton")
    plt.legend()
    plt.grid()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    string = base64.b64encode(buf.read()).decode()
    img_uri = f"data:image/png;base64,{string}"

    plt.close()

    return resultado, tabla_html, img_uri

def generar_informe_newton(df, fun_str, df_str, x0, tol, niter, tipo_error, tiempo=None):
    """
    df: DataFrame con las iteraciones del método de Newton.
        Se espera algo tipo columnas: i, Xn, f(Xn), Error (aunque tratamos de adaptarnos).
    fun_str: función f(x) como string (ej: "x**3 - x - 2")
    df_str: derivada f'(x) como string
    x0: aproximación inicial
    tol: tolerancia requerida
    niter: máximo de iteraciones
    tipo_error: texto del tipo de error (cifras, relativo, etc.)
    tiempo: tiempo total medido en la vista (segundos)
    """

    # Aseguramos DataFrame
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    # ==========================
    #   MANEJO DEL TIEMPO
    # ==========================
    if tiempo is not None:
        tiempo_str = f"{tiempo:.6f}"
    else:
        tiempo_str = "--"

    # ==========================
    #   DETECCIÓN DE COLUMNAS
    # ==========================
    cols = [str(c) for c in df.columns]

    # Columna de iteración (normalmente la primera)
    col_i = cols[0]

    # Columna de x_n
    posibles_x = ['Xn', 'xn', 'Xi', 'xi', 'x', 'Xm', 'xm']
    col_x = None
    for c in cols:
        if c in posibles_x:
            col_x = c
            break
    if col_x is None and len(cols) >= 2:
        col_x = cols[1]

    # Columna de f(x_n)
    posibles_fx = ['f(Xn)', 'f(xn)', 'f(Xi)', 'f(xi)', 'f(x)', 'f(Xm)', 'f(xm)']
    col_fx = None
    for c in cols:
        if c in posibles_fx:
            col_fx = c
            break
    if col_fx is None and len(cols) >= 3:
        col_fx = cols[2]

    # Columna de Error (si existe)
    posibles_err = ['Error', 'error', 'E', 'err']
    col_err = None
    for c in cols:
        if c in posibles_err:
            col_err = c
            break

    n_iter_real = len(df)
    x_final = df[col_x].iloc[-1]
    fx_final = df[col_fx].iloc[-1]

    if n_iter_real >= 2:
        x_prev = df[col_x].iloc[-2]
        delta = x_final - x_prev
    else:
        x_prev = x_final
        delta = 0.0

    # Si existe columna de error, usamos ese valor como "error del modo"
    if col_err is not None:
        error_final_modo = df[col_err].iloc[-1]
    else:
        # Si no, usamos el error absoluto entre iteraciones
        error_final_modo = abs(delta)

    # 🔧 Manejo de tolerancia (puede venir None)
    if tol is not None:
        convergio_por_tol = abs(error_final_modo) <= tol
    else:
        convergio_por_tol = False

    # 🔧 Manejo de niter (puede venir None)
    if niter is not None:
        uso_todas_iter = (n_iter_real >= niter)
    else:
        uso_todas_iter = False

    # ==========================
    #   TEXTO RESUMEN
    # ==========================
    resumen = []

    resumen.append(
        f"Se aplicó el método de Newton-Raphson a la función f(x) = {fun_str} "
        f"con derivada f'(x) = {df_str}, usando x₀ = {x0} como aproximación inicial, "
        f"tolerancia {tol} y un máximo de {niter} iteraciones."
    )

    resumen.append(
        f"El método realizó {n_iter_real} iteraciones y obtuvo como aproximación final x ≈ {x_final:.8f}."
    )

    resumen.append(
        f"El error final ({tipo_error}) fue de aproximadamente {error_final_modo:.2e}."
    )

    if convergio_por_tol:
        resumen.append(
            "La aproximación cumple el criterio de parada especificado por la tolerancia, "
            "por lo que se considera que el método **convergió adecuadamente**."
        )
    else:
        if uso_todas_iter:
            resumen.append(
                "El método alcanzó el número máximo de iteraciones sin cumplir la tolerancia, "
                "por lo que se considera que **no alcanzó la convergencia deseada**."
            )
        else:
            resumen.append(
                "La tolerancia no se cumplió estrictamente o el método se detuvo por otra condición de parada."
            )

    if n_iter_real <= 5:
        comentario_velocidad = "La convergencia fue muy rápida (pocas iteraciones)."
    elif n_iter_real <= 15:
        comentario_velocidad = "La cantidad de iteraciones se considera moderada."
    else:
        comentario_velocidad = "La cantidad de iteraciones fue alta, lo que indica una convergencia más lenta."

    resumen.append(comentario_velocidad)

    informe_texto = " ".join(resumen)

    # ==========================
    #   COMPARACIÓN DE ERRORES
    # ==========================
    # Error absoluto entre x_n y x_{n-1}
    error_decimales = abs(delta)

    # Error relativo
    error_relativo = abs(delta / x_final) if x_final != 0 else 0.0

    # Error en f(x)
    error_fx = abs(fx_final)

    comparacion_errores = [
        {
            "tipo": "decimales",
            "x_final": x_final,
            "fx_final": fx_final,
            "error_final": error_decimales,
            "tiempo": tiempo_str,
        },
        {
            "tipo": "cifras significativas",
            "x_final": x_final,
            "fx_final": fx_final,
            "error_final": error_relativo,
            "tiempo": tiempo_str,
        },
        {
            "tipo": "relativo_xnm1",
            "x_final": x_final,
            "fx_final": fx_final,
            "error_final": error_relativo,
            "tiempo": tiempo_str,
        },
        {
            "tipo": "fx",
            "x_final": x_final,
            "fx_final": fx_final,
            "error_final": error_fx,
            "tiempo": tiempo_str,
        },
    ]

    informe = {
        "texto": informe_texto,
        "n_iter_real": n_iter_real,
        "xm_final": x_final,
        "error_final": error_final_modo,
        "convergio_por_tol": convergio_por_tol,
        "uso_todas_iter": uso_todas_iter,
        "tipo_error": tipo_error,
        "comparacion_errores": comparacion_errores,
    }

    return informe
