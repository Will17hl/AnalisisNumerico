import matplotlib
matplotlib.use('Agg')
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

pd.options.display.float_format = "{:.16f}".format

def secanteDC( X0, X1, Tol, Niter, Fun):

    nlist = []
    xnlist= []
    fxnlist=[]
    E=[]
    tabla = []
    x = X0
    f0 = eval(Fun, {"x": x, "math": math})
    x = X1
    f1= eval(Fun, {"x": x, "math": math})
    try:
        i = 0
        xnlist.append(X1)
        fxnlist.append(f1)
        nlist.append(i)
        Error = 100
        E.append(Error)
        tabla.append([i, X1, f1, Error])
    except Exception as e:
        print(f"1: {e}")

        tabla = [[0, 0, 0, 0]]

        df_resultado = pd.DataFrame(tabla, columns=["I", "Xi", "F(Xi)", "E"])
        tabla_html = df_resultado.to_html(index=False, classes='table table-striped text-center')

        x_vals = np.linspace(min(xnlist) - 1, max(xnlist) + 1, 100)
        y_vals = [eval(Fun, {"x": val, "math": math}) for val in x_vals]

        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, label=f'f(x) = {Fun}', color='blue')
        plt.axhline(0, color='black', linewidth=1)
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Método de la Secante")
        plt.legend()
        plt.grid()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        string = base64.b64encode(buf.read()).decode()
        img_uri = f"data:image/png;base64,{string}"

        plt.close()

        s = "Error"

        return s, tabla_html, img_uri
    
    while E[i] > Tol and f1 != 0 and i < Niter:
        
        try:

            x=X1-((f1*(X1-X0))/(f1-f0))
            f=eval(Fun, {"x": x, "math": math})
            X0 = X1
            X1 = x
            f0 = f1
            f1 = f
            xnlist.append(X1)
            fxnlist.append(f1)
            i += 1
            Error=abs(X1-X0)
            nlist.append(i)
            E.append(Error)
            tabla.append([i, X1, f1, Error])
            if f == 0:
                s=x
                df = pd.DataFrame({
                    "i": nlist,
                    "Xn": xnlist,
                    "Fxn": fxnlist,
                    "Error": E[i]
                })
                print(df.to_string(index=False))

                print(s,"Es una raiz de f(x)")
                break
            elif Error < Tol:
                s=x
                df = pd.DataFrame({
                    "i": nlist,
                    "Xn": xnlist,
                    "Fxn": fxnlist,
                    "Error": E[i]
                })
                print(df.to_string(index=False), '\n')

                print(s,"es una aproximacion de un raiz de f(x) con una Tolerancia", Tol,'\n')
                break
        except Exception as e:

            print(f"2: {e}")
            df_resultado = pd.DataFrame(tabla, columns=["I", "Xi", "F(Xi)", "E"])
            tabla_html = df_resultado.to_html(index=False, classes='table table-striped text-center')

            x_vals = np.linspace(min(xnlist) - 1, max(xnlist) + 1, 100)
            y_vals = [eval(Fun, {"x": val, "math": math}) for val in x_vals]

            plt.figure(figsize=(8, 6))
            plt.plot(x_vals, y_vals, label=f'f(x) = {Fun}', color='blue')
            plt.axhline(0, color='black', linewidth=1)
            plt.xlabel("x")
            plt.ylabel("f(x)")
            plt.title("Método de la Secante")
            plt.legend()
            plt.grid()

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            string = base64.b64encode(buf.read()).decode()
            img_uri = f"data:image/png;base64,{string}"

            plt.close()

            s = "Error"

            return s, tabla_html, img_uri
    else:
        s=x
        df_resultado = pd.DataFrame(tabla, columns=["I", "Xi", "F(Xi)", "E"])
        tabla_html = df_resultado.to_html(index=False, classes='table table-striped text-center')

        x_vals = np.linspace(min(xnlist) - 1, max(xnlist) + 1, 100)
        y_vals = [eval(Fun, {"x": val, "math": math}) for val in x_vals]

        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, label=f'f(x) = {Fun}', color='blue')
        plt.axhline(0, color='black', linewidth=1)
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Método de la Secante")
        plt.legend()
        plt.grid()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        string = base64.b64encode(buf.read()).decode()
        img_uri = f"data:image/png;base64,{string}"

        plt.close()

        s = f'Error en la iteracion {Niter}, ultima aproximacion: {x}'

        return s, tabla_html, img_uri
    
    df_resultado = pd.DataFrame(tabla, columns=["I", "Xi", "F(Xi)", "E"])
    tabla_html = df_resultado.to_html(index=False, classes='table table-striped text-center')

    x_vals = np.linspace(min(xnlist) - 1, max(xnlist) + 1, 100)
    y_vals = [eval(Fun, {"x": val, "math": math}) for val in x_vals]

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label=f'f(x) = {Fun}', color='blue')
    plt.axhline(0, color='black', linewidth=1)
    plt.scatter(xnlist[-1], 0, color='red', zorder=5, label=f'Raíz: {round(xnlist[-1], 4)}')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Método de la Secante")
    plt.legend()
    plt.grid()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    string = base64.b64encode(buf.read()).decode()
    img_uri = f"data:image/png;base64,{string}"

    plt.close()

    return s, tabla_html, img_uri

def secanteCS( X0, X1, Tol, Niter, Fun):

    nlist = []
    xnlist= []
    fxnlist=[]
    E=[]
    tabla = []
    x = X0
    f0 = eval(Fun, {"x": x, "math": math})
    x = X1
    f1= eval(Fun, {"x": x, "math": math})

    try:
        i = 0
        xnlist.append(X1)
        fxnlist.append(f1)
        nlist.append(i)
        Error = 100
        E.append(Error)
        tabla.append([i, X1, f1, Error])
    except Exception as e:
        print(f"1: {e}")

        tabla = [[0, 0, 0, 0]]

        df_resultado = pd.DataFrame(tabla, columns=["I", "Xi", "F(Xi)", "E"])
        tabla_html = df_resultado.to_html(index=False, classes='table table-striped text-center')

        x_vals = np.linspace(min(xnlist) - 1, max(xnlist) + 1, 100)
        y_vals = [eval(Fun, {"x": val, "math": math}) for val in x_vals]

        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, label=f'f(x) = {Fun}', color='blue')
        plt.axhline(0, color='black', linewidth=1)
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Método de la Secante")
        plt.legend()
        plt.grid()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        string = base64.b64encode(buf.read()).decode()
        img_uri = f"data:image/png;base64,{string}"

        plt.close()

        s = "Error general"

        return s, tabla_html, img_uri
        
    while E[i] >= Tol and f1 != 0 and i < Niter:
        try:
            x=X1-((f1*(X1-X0))/(f1-f0))
            f=eval(Fun, {"x": x, "math": math})
            X0 = X1
            X1 = x
            f0 = f1
            f1 = f
            xnlist.append(X1)
            fxnlist.append(f1)
            i += 1
            Error=abs((xnlist[i]-xnlist[i-1])/xnlist[i])
            nlist.append(i)
            E.append(Error)
            tabla.append([i, X1, f1, Error])
            if f == 0:
                s=x
                df = pd.DataFrame({
                    "i": nlist,
                    "Xn": xnlist,
                    "Fxn": fxnlist,
                    "Error": E[i]
                })
                print(df.to_string(index=False))

                print(s,"Es una raiz de f(x)")
                break
            elif Error < Tol:
                s=x
                df = pd.DataFrame({
                    "i": nlist,
                    "Xn": xnlist,
                    "Fxn": fxnlist,
                    "Error": E[i]
                })
                print(df.to_string(index=False), '\n')

                print(s,"es una aproximacion de un raiz de f(x) con una Tolerancia", Tol,'\n')
                break
        except Exception as e:

            print(f"2: {e}")
            df_resultado = pd.DataFrame(tabla, columns=["I", "Xi", "F(Xi)", "E"])
            tabla_html = df_resultado.to_html(index=False, classes='table table-striped text-center')

            x_vals = np.linspace(min(xnlist) - 1, max(xnlist) + 1, 100)
            y_vals = [eval(Fun, {"x": val, "math": math}) for val in x_vals]

            plt.figure(figsize=(8, 6))
            plt.plot(x_vals, y_vals, label=f'f(x) = {Fun}', color='blue')
            plt.axhline(0, color='black', linewidth=1)
            plt.xlabel("x")
            plt.ylabel("f(x)")
            plt.title("Método de la Secante")
            plt.legend()
            plt.grid()

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            string = base64.b64encode(buf.read()).decode()
            img_uri = f"data:image/png;base64,{string}"

            plt.close()

            s = "Error"

            return s, tabla_html, img_uri 
    else:
        s=x
        df_resultado = pd.DataFrame(tabla, columns=["I", "Xi", "F(Xi)", "E"])
        tabla_html = df_resultado.to_html(index=False, classes='table table-striped text-center')

        x_vals = np.linspace(min(xnlist) - 1, max(xnlist) + 1, 100)
        y_vals = [eval(Fun, {"x": val, "math": math}) for val in x_vals]

        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, label=f'f(x) = {Fun}', color='blue')
        plt.axhline(0, color='black', linewidth=1)
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Método de la Secante")
        plt.legend()
        plt.grid()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        string = base64.b64encode(buf.read()).decode()
        img_uri = f"data:image/png;base64,{string}"

        plt.close()

        s = f'Error en la iteracion {Niter}, ultima aproximacion: {x}'

        return s, tabla_html, img_uri
    
    df_resultado = pd.DataFrame(tabla, columns=["I", "Xi", "F(Xi)", "E"])
    tabla_html = df_resultado.to_html(index=False, classes='table table-striped text-center')

    x_vals = np.linspace(min(xnlist) - 1, max(xnlist) + 1, 100)
    y_vals = [eval(Fun, {"x": val, "math": math}) for val in x_vals]

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label=f'f(x) = {Fun}', color='blue')
    plt.axhline(0, color='black', linewidth=1)
    plt.scatter(xnlist[-1], 0, color='red', zorder=5, label=f'Raíz: {round(xnlist[-1], 4)}')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Método de la Secante")
    plt.legend()
    plt.grid()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    string = base64.b64encode(buf.read()).decode()
    img_uri = f"data:image/png;base64,{string}"

    plt.close()

    return s, tabla_html, img_uri

def generar_informe_secante(df, fun_str, x0, x1, tol, niter, tipo_error, tiempo=None):
    """
    df: DataFrame con las iteraciones del método de Secante.
        Se espera columnas tipo: Iteración, Xi, F(Xi), Error (o similar).
    fun_str: función f(x) como string (ej: "x**3 - x - 2")
    x0, x1: aproximaciones iniciales
    tol: tolerancia requerida
    niter: máximo de iteraciones
    tipo_error: texto del tipo de error (cifras, relativo, etc.)
    tiempo: tiempo total medido en la vista (segundos)
    """

    # Aseguramos DataFrame
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    # --- Tiempo ---
    if tiempo is not None:
        tiempo_str = f"{tiempo:.6f}"
    else:
        tiempo_str = "--"

    # --- Detección de columnas ---
    cols = [str(c) for c in df.columns]

    # Columna de Xi (normalmente la segunda)
    posibles_x = ['Xi', 'xi', 'Xn', 'xn', 'x']
    col_x = None
    for c in cols:
        if c in posibles_x:
            col_x = c
            break
    if col_x is None and len(cols) >= 2:
        col_x = cols[1]

    # Columna de f(Xi)
    posibles_fx = ['F(Xi)', 'f(Xi)', 'f(x)', 'f(Xn)', 'f(xn)']
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

    # Si existe columna de error → usarla como error del modo
    if col_err is not None:
        error_final_modo = df[col_err].iloc[-1]
    else:
        error_final_modo = abs(delta)

    # Manejo de tol/niter por si acaso
    if tol is not None:
        convergio_por_tol = abs(error_final_modo) <= tol
    else:
        convergio_por_tol = False

    if niter is not None:
        uso_todas_iter = (n_iter_real >= niter)
    else:
        uso_todas_iter = False

    # --- Texto resumen ---
    resumen = []

    resumen.append(
        f"Se aplicó el método de la Secante a la función f(x) = {fun_str}, "
        f"usando x₀ = {x0} y x₁ = {x1} como aproximaciones iniciales, "
        f"con tolerancia {tol} y un máximo de {niter} iteraciones."
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

    # --- Comparación de errores ---
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
