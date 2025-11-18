import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import base64
import io

pd.options.display.float_format = "{:.8f}".format


def biseccion(a, b, tol, niter, fun, Modo):
    print(a, type(a))
    print(b, type(b))
    print(tol, type(tol))
    print(fun, type(fun))

    tabla = []
    resultado = ""
    nlist = []
    xmlist = []
    res = []
    fxmlist = []
    E = []

    try:
        x = a
        fa = eval(fun, {"x": a, "math": math})
        x = b
        fb = eval(fun, {"x": b, "math": math})

    except Exception as e:
        print(f"1: {e}")

        tabla = [[0, 0, 0, 0]]
        df = pd.DataFrame(tabla, columns=["i", "Xm", "f(Xm)", "Error"])

        x_vals = np.linspace(a - 5, b + 5, 400)
        y_vals = [eval(fun, {"x": x, "math": math}) for x in x_vals]

        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, label=f'f(x) = {str(fun)}', color='blue')
        plt.axhline(0, color='black', linewidth=1)
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Método de Bisección")
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

    if fa * fb < 0:
        i = 0
        xm = (a + b) / 2
        x = xm
        fxm = fb = eval(fun, {"x": x, "math": math})
        xmlist.append(fxm)
        error = 100

        tabla.append([i, xm, fxm, error])

        while error > tol and fxm != 0 and i < niter:

            try:
                if fa * fxm < 0:
                    b = xm
                    x = b
                    eval(fun, {"x": x, "math": math})
                else:
                    a = xm
                    x = a
                    fa = fb = eval(fun, {"x": x, "math": math})

                xma = xm
                xm = (a + b) / 2
                x = xm
                fxm = eval(fun, {"x": x, "math": math})

                if Modo == "cs":
                    error = abs((xm - xma) / xm)
                else:
                    error = abs((xm - xma))

                i += 1
                tabla.append([i, xm, fxm, error])

            except Exception as e:
                print(f"2: {e}")
                df = pd.DataFrame(tabla, columns=["i", "Xm", "f(Xm)", "Error"])

                x_vals = np.linspace(a - 5, b + 5, 400)
                y_vals = [eval(fun, {"x": x, "math": math}) for x in x_vals]

                plt.figure(figsize=(8, 6))
                plt.plot(x_vals, y_vals, label=f'f(x) = {str(fun)}', color='blue')
                plt.axhline(0, color='black', linewidth=1)
                plt.scatter(x, 0, color='red', zorder=5, label=f'Raíz: {round(x, 4)}')
                plt.xlabel("x")
                plt.ylabel("f(x)")
                plt.title("Método de Bisección")
                plt.legend()
                plt.grid()

                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                string = base64.b64encode(buf.read()).decode()
                img_uri = f"data:image/png;base64,{string}"

                plt.close()

                resultado = f'Error'

                return resultado, df.to_html(index=False, classes='table table-striped text-center'), img_uri

        if fxm == 0:
            resultado = f"{x} es raíz de f(x)"
            res.append(x)
        elif error <= tol:
            resultado = f"{x} es una aproximación de una raíz con tolerancia {tol}"
        else:
            df = pd.DataFrame(tabla, columns=["i", "Xm", "f(Xm)", "Error"])

            x_vals = np.linspace(a - 5, b + 5, 400)
            y_vals = [eval(fun, {"x": x, "math": math}) for x in x_vals]

            plt.figure(figsize=(8, 6))
            plt.plot(x_vals, y_vals, label=f'f(x) = {str(fun)}', color='blue')
            plt.axhline(0, color='black', linewidth=1)
            plt.scatter(x, 0, color='red', zorder=5, label=f'Raíz: {round(x, 4)}')
            plt.xlabel("x")
            plt.ylabel("f(x)")
            plt.title("Método de Bisección")
            plt.legend()
            plt.grid()

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            string = base64.b64encode(buf.read()).decode()
            img_uri = f"data:image/png;base64,{string}"

            plt.close()

            resultado = f'Error en la iteracion {niter}, ultima aproximacion: {x}'

            return resultado, df.to_html(index=False, classes='table table-striped text-center'), img_uri

    elif fa == 0:
        resultado = f"{a} es raíz de f(x)"
    elif fb == 0:
        resultado = f"{b} es raíz de f(x)"
    else:
        df = pd.DataFrame(tabla, columns=["i", "Xm", "f(Xm)", "Error"])

        x_vals = np.linspace(a - 5, b + 5, 400)
        y_vals = [eval(fun, {"x": x, "math": math}) for x in x_vals]

        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, label=f'f(x) = {str(fun)}', color='blue')
        plt.axhline(0, color='black', linewidth=1)
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Método de Bisección")
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

    df = pd.DataFrame(tabla, columns=["i", "Xm", "f(Xm)", "Error"])

    x_vals = np.linspace(a - 5, b + 5, 400)
    y_vals = [eval(fun, {"x": x, "math": math}) for x in x_vals]

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label=f'f(x) = {str(fun)}', color='blue')
    plt.axhline(0, color='black', linewidth=1)
    plt.scatter(x, 0, color='red', zorder=5, label=f'Raíz: {round(x, 4)}')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Método de Bisección")
    plt.legend()
    plt.grid()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    string = base64.b64encode(buf.read()).decode()
    img_uri = f"data:image/png;base64,{string}"

    plt.close()

    return resultado, df.to_html(index=False, classes='table table-striped text-center'), img_uri


def generar_informe_biseccion(df, fun_str, a, b, tol, niter, tipo_error, tiempo=None):
    """
    df: DataFrame con las iteraciones (columnas esperadas: i, Xm, f(Xm), Error)
    fun_str: string de la función ingresada por el usuario (ej: "x**3 - x - 2")
    a, b: intervalo inicial
    tol: tolerancia solicitada
    niter: número máximo de iteraciones solicitado
    tipo_error: texto del tipo de error elegido (ej: "cifras significativas", "error relativo", etc.)
    """
    if tiempo is not None:
        tiempo_str = f"{tiempo:.6f}"
    else:
        tiempo_str = "--"
        
    # Si por alguna razón df no viene como DataFrame, lo normalizamos:
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df, columns=["i", "Xm", "f(Xm)", "Error"])

    # Nombre de columnas (por si en algún momento cambias)
    col_xm = "Xm" if "Xm" in df.columns else "xm"
    col_fx = "f(Xm)" if "f(Xm)" in df.columns else "f(xm)"
    col_err = "Error" if "Error" in df.columns else "error"

    n_iter_real = len(df)
    xm_final = df[col_xm].iloc[-1]
    fx_final = df[col_fx].iloc[-1]

    # Error que ya trae tu tabla (el del modo elegido)
    error_final_modo = df[col_err].iloc[-1]

    convergio_por_tol = abs(error_final_modo) <= tol
    uso_todas_iter = (n_iter_real >= niter)

    # ─────────────────────────────────────────────
    # 1) TEXTO RESUMEN GENERAL (como ya lo tenías)
    # ─────────────────────────────────────────────
    resumen_general = []

    resumen_general.append(
        f"Se aplicó el método de Bisección a la función f(x) = {fun_str}, "
        f"en el intervalo inicial [{a}, {b}] con tolerancia {tol} y un máximo de {niter} iteraciones."
    )

    resumen_general.append(
        f"El método realizó {n_iter_real} iteraciones y obtuvo como aproximación final x ≈ {xm_final:.8f}."
    )

    resumen_general.append(
        f"El error final ({tipo_error}) fue de aproximadamente {error_final_modo:.2e}."
    )

    if convergio_por_tol:
        resumen_general.append(
            "La aproximación cumple el criterio de parada especificado por la tolerancia, "
            "por lo que se considera que el método **convergió adecuadamente**."
        )
    else:
        if uso_todas_iter:
            resumen_general.append(
                "El método alcanzó el número máximo de iteraciones sin cumplir la tolerancia, "
                "por lo que se considera que **no alcanzó la convergencia deseada**."
            )
        else:
            resumen_general.append(
                "El método no cumplió la tolerancia, pero se detuvo por otra condición de parada."
            )

    # Comentario cualitativo (rápido / lento)
    if n_iter_real <= 10:
        comentario_velocidad = (
            "La cantidad de iteraciones se considera baja, por lo que la convergencia fue relativamente rápida."
        )
    elif n_iter_real <= 25:
        comentario_velocidad = "La cantidad de iteraciones se considera moderada."
    else:
        comentario_velocidad = (
            "La cantidad de iteraciones fue alta, lo que indica una convergencia más lenta."
        )

    resumen_general.append(comentario_velocidad)

    informe_texto = " ".join(resumen_general)

    # ─────────────────────────────────────────────
    # 2) TABLA DE COMPARACIÓN DE TIPOS DE ERROR
    #    (decimales, cifras, relativo_xnm1, fx)
    # ─────────────────────────────────────────────
    # Usamos las dos últimas aproximaciones para construir errores "equivalentes"
    if n_iter_real >= 2:
        xm_prev = df[col_xm].iloc[-2]
        delta = xm_final - xm_prev
    else:
        xm_prev = xm_final
        delta = 0.0

    # Error absoluto entre x_n y x_{n-1}
    error_decimales = abs(delta)

    # Error relativo (respecto a x_n)
    error_relativo = abs(delta / xm_final) if xm_final != 0 else 0.0

    # Error basado en f(x_n)
    error_fx = abs(fx_final)

    # Construimos una lista de dicts para poder mostrarla en template
    comparacion_errores = [
        {
            "tipo": "decimales",
            "x_final": xm_final,
            "fx_final": fx_final,
            "error_final": error_decimales,
            "tiempo":tiempo_str ,  # si luego quieres medir tiempo, aquí lo pones
        },
        {
            "tipo": "cifras significativas",
            "x_final": xm_final,
            "fx_final": fx_final,
            "error_final": error_relativo,
            "tiempo": tiempo_str,
        },
        {
            "tipo": "relativo_xnm1",
            "x_final": xm_final,
            "fx_final": fx_final,
            "error_final": error_relativo,
            "tiempo": tiempo_str,
        },
        {
            "tipo": "fx",
            "x_final": xm_final,
            "fx_final": fx_final,
            "error_final": error_fx,
            "tiempo": tiempo_str,
        },
    ]

    # ─────────────────────────────────────────────
    # 3) Diccionario que se envía al template
    # ─────────────────────────────────────────────
    informe = {
        "texto": informe_texto,
        "n_iter_real": n_iter_real,
        "xm_final": xm_final,
        "error_final": error_final_modo,
        "convergio_por_tol": convergio_por_tol,
        "uso_todas_iter": uso_todas_iter,
        "tipo_error": tipo_error,
        "comparacion_errores": comparacion_errores,
    }

    return informe
