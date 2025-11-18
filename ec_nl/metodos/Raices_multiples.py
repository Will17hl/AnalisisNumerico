import matplotlib
matplotlib.use('Agg')
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

def raices_multiples(X0, Tol, Niter, Fun, df, ddf, Modo):
    tabla = []
    x = X0

    try:
        f = eval(Fun, {"x": x, "math": math})
        df_val = eval(df, {"x": x, "math": math})
        ddf_val = eval(ddf, {"x": x, "math": math})
        c = 0
        Error = 100

        tabla.append([c, x, f, Error])

    except Exception as e:
        print(f"1: {e}")

        tabla = [[0, 0, 0, 0]]

        df = pd.DataFrame(tabla, columns=["Iteración", "Xi", "F(Xi)", "Error"])

        x_vals = np.linspace(X0 - 5, X0 + 5, 400)
        y_vals = [eval(Fun, {"x": val, "math": math}) for val in x_vals]

        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, label=f'f(x) = {Fun}', color='blue')
        plt.axhline(0, color='black', linewidth=1)
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Método de Raíces Múltiples")
        plt.legend()
        plt.grid()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        string = base64.b64encode(buf.read()).decode()
        img_uri = f"data:image/png;base64,{string}"

        plt.close()

        resultado = "Error"

        return resultado, df.to_html(index=False, classes='table table-striped text-center'), img_uri

    while Error > Tol and f != 0 and df_val != 0 and c < Niter:
        
        try:
            x_new = x - (f * df_val) / (df_val**2 - f * ddf_val)
            if (Modo == "cs"):
                Error = abs((x_new - x)/x_new)
            else:
                Error = abs(x_new - x)
            x = x_new
            f = eval(Fun, {"x": x, "math": math})
            df_val = eval(df, {"x": x, "math": math})
            ddf_val = eval(ddf, {"x": x, "math": math})
            c += 1
            tabla.append([c, x, f, Error])
        except Exception as e:
            print(f"2: {e}")

            df = pd.DataFrame(tabla, columns=["Iteración", "Xi", "F(Xi)", "Error"])

            x_vals = np.linspace(X0 - 5, X0 + 5, 400)
            y_vals = [eval(Fun, {"x": val, "math": math}) for val in x_vals]

            plt.figure(figsize=(8, 6))
            plt.plot(x_vals, y_vals, label=f'f(x) = {Fun}', color='blue')
            plt.axhline(0, color='black', linewidth=1)
            plt.xlabel("x")
            plt.ylabel("f(x)")
            plt.title("Método de Raíces Múltiples")
            plt.legend()
            plt.grid()

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            string = base64.b64encode(buf.read()).decode()
            img_uri = f"data:image/png;base64,{string}"

            plt.close()

            resultado = "Error"

            return resultado, df.to_html(index=False, classes='table table-striped text-center'), img_uri
            

    if f == 0:
        resultado = f"{x} es raíz de f(x)"
    elif Error < Tol:
        resultado = f"{x} es una aproximación de una raíz con tolerancia {Tol}"
    else:
        df = pd.DataFrame(tabla, columns=["Iteración", "Xi", "F(Xi)", "Error"])

        x_vals = np.linspace(X0 - 5, X0 + 5, 400)
        y_vals = [eval(Fun, {"x": val, "math": math}) for val in x_vals]

        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, label=f'f(x) = {Fun}', color='blue')
        plt.axhline(0, color='black', linewidth=1)
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Método de Raíces Múltiples")
        plt.legend()
        plt.grid()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        string = base64.b64encode(buf.read()).decode()
        img_uri = f"data:image/png;base64,{string}"

        plt.close()

        resultado = f'Error en la iteracion {Niter}, ultima aproximacion: {x}'

        return resultado, df.to_html(index=False, classes='table table-striped text-center'), img_uri

    df = pd.DataFrame(tabla, columns=["Iteración", "Xi", "F(Xi)", "Error"])

    x_vals = np.linspace(X0 - 5, X0 + 5, 400)
    y_vals = [eval(Fun, {"x": val, "math": math}) for val in x_vals]

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label=f'f(x) = {Fun}', color='blue')
    plt.axhline(0, color='black', linewidth=1)
    plt.scatter(x, 0, color='red', zorder=5, label=f'Raíz: {round(x, 4)}')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Método de Raíces Múltiples")
    plt.legend()
    plt.grid()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    string = base64.b64encode(buf.read()).decode()
    img_uri = f"data:image/png;base64,{string}"

    plt.close()

    return resultado, df.to_html(index=False, classes='table table-striped text-center'), img_uri


def generar_informe_raices_multiples(
    df,
    fun_str,
    df_str,
    ddf_str,
    x0,
    tol,
    niter,
    tipo_error,
    tiempo=None
):
    """
    df: DataFrame con las iteraciones del método de Raíces Múltiples.
        Se espera columnas tipo: iteración, Xi, f(Xi), Error (relativo o absoluto).
    fun_str : str  -> f(x)
    df_str  : str  -> f'(x)
    ddf_str : str  -> f''(x)
    x0      : float -> aproximación inicial
    tol     : float -> tolerancia requerida
    niter   : int   -> máximo de iteraciones
    tipo_error : descripción del tipo de error (cifras, relativo, etc.)
    tiempo  : tiempo total de ejecución en segundos (opcional)
    """

    # Asegurar DataFrame
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
    posibles_fx = ['f(Xn)', 'f(xn)', 'f(Xi)', 'f(xi)', 'f(x)', 'F(Xi)', 'F(xi)', 'F(x)']
    col_fx = None
    for c in cols:
        if c in posibles_fx:
            col_fx = c
            break
    if col_fx is None and len(cols) >= 3:
        col_fx = cols[2]

    # Columna de Error
    posibles_err = ['Error', 'error', 'E', 'err', 'Error relativo']
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

    # Si existe columna de error, usamos ese valor
    if col_err is not None:
        error_final_modo = df[col_err].iloc[-1]
    else:
        error_final_modo = abs(delta)

    # Manejo defensivo por si tol o niter vienen None
    if tol is not None:
        convergio_por_tol = abs(error_final_modo) <= tol
    else:
        convergio_por_tol = False

    if niter is not None:
        uso_todas_iter = (n_iter_real >= niter)
    else:
        uso_todas_iter = False

    # ==========================
    #   TEXTO RESUMEN
    # ==========================
    resumen = []

    resumen.append(
        f"Se aplicó el método de Raíces Múltiples a la función f(x) = {fun_str}, "
        f"con derivada f'(x) = {df_str} y segunda derivada f''(x) = {ddf_str}, "
        f"usando x₀ = {x0} como aproximación inicial, tolerancia {tol} "
        f"y un máximo de {niter} iteraciones."
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
    error_decimales = abs(delta)
    error_relativo = abs(delta / x_final) if x_final != 0 else 0.0
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
