import matplotlib
matplotlib.use('Agg')
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import io
import urllib, base64


pd.options.display.float_format = "{:.16f}".format

def punto_fijo(X0, Tol, Niter, Fun, g):
    fn = []
    xn = []
    E = []
    N = []
    x = X0
    f = eval(Fun, {"x": x, "math": math})
    
    try:
        c = 0
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

        df = pd.DataFrame(tabla, columns=["I", "Xi", "F(Xi)", "E"])

        x_vals = np.linspace(X0 - 5, X0 + 5, 400) 
        y_vals = [eval(Fun, {"x": val, "math": math}) for val in x_vals] 

        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, label=f'f(x) = {Fun}', color='blue')
        plt.axhline(0, color='black', linewidth=1)  
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Método de Punto Fijo")
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


    while Error > Tol and f != 0 and c < Niter:

        try:
            x = eval(g, {"x": x, "math": math})
            fe = eval(Fun, {"x": x, "math": math})
            fn.append(fe)
            xn.append(x)
            c += 1
            Error = abs(xn[c] - xn[c - 1])
            N.append(c)
            E.append(Error)
            tabla.append([c, x, fe, Error])
            f = fe
        except Exception as e:
            print(f"2: {e}")

            df = pd.DataFrame(tabla, columns=["I", "Xi", "F(Xi)", "E"])

            x_vals = np.linspace(X0 - 5, X0 + 5, 400) 
            y_vals = [eval(Fun, {"x": val, "math": math}) for val in x_vals] 

            plt.figure(figsize=(8, 6))
            plt.plot(x_vals, y_vals, label=f'f(x) = {Fun}', color='blue')
            plt.axhline(0, color='black', linewidth=1)  
            plt.xlabel("x")
            plt.ylabel("f(x)")
            plt.title("Método de Punto Fijo")
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


    resultado = ""
    if f == 0:
        resultado = f"{x} es raíz de f(x)"
    elif Error < Tol:
        resultado = f"{x} es una aproximación de una raíz de f(x) con tolerancia {Tol}"
    else:
        df = pd.DataFrame(tabla, columns=["I", "Xi", "F(Xi)", "E"])

        x_vals = np.linspace(X0 - 5, X0 + 5, 400) 
        y_vals = [eval(Fun, {"x": val, "math": math}) for val in x_vals] 

        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, label=f'f(x) = {Fun}', color='blue')
        plt.axhline(0, color='black', linewidth=1)  
        plt.scatter(xn[-1], 0, color='red', zorder=5, label=f'Raíz: {round(xn[-1], 4)}')  
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Método de Punto Fijo")
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



    df = pd.DataFrame(tabla, columns=["I", "Xi", "F(Xi)", "E"])


    x_vals = np.linspace(X0 - 5, X0 + 5, 400) 
    y_vals = [eval(Fun, {"x": val, "math": math}) for val in x_vals] 

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label=f'f(x) = {Fun}', color='blue')
    plt.axhline(0, color='black', linewidth=1)  
    plt.scatter(xn[-1], 0, color='red', zorder=5, label=f'Raíz: {round(xn[-1], 4)}')  
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Método de Punto Fijo")
    plt.legend()
    plt.grid()


    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    string = base64.b64encode(buf.read()).decode()
    img_uri = f"data:image/png;base64,{string}"

    plt.close()


    return resultado, df.to_html(index=False, classes='table table-striped text-center'), img_uri




def punto_fijoCS(X0, Tol, Niter, Fun, g):
    fn = []
    xn = []
    E = []
    e = []
    N = []

    x = X0
    f = eval(Fun, {"x": x, "math": math})  

    try:
        c = 0
        Error = 100
        error = 100

        fn.append(f)
        xn.append(x)
        E.append(Error)
        e.append(error)
        N.append(c)

        tabla = []
        tabla.append([c, x, f, error])

    except Exception as e:
        print(f"1: {e}")

        tabla = [[0, 0, 0, 0]]

        df = pd.DataFrame(tabla, columns=["I", "Xi", "F(Xi)", "E"])

        x_vals = np.linspace(X0 - 5, X0 + 5, 400)
        y_vals = [eval(Fun, {"x": val, "math": math}) for val in x_vals]

        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, label=f'f(x) = {Fun}', color='blue')
        plt.axhline(0, color='black', linewidth=1)
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Método de Punto Fijo")
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


    while error > Tol and abs(f) > 1e-12 and c < Niter:
        
        try:

            x_new = eval(g, {"x": x, "math": math})  
            fe = eval(Fun, {"x": x_new, "math": math})  

            Error = abs(x_new - x)  
            error = Error / abs(x_new) if x_new != 0 else float('inf')  

            c += 1
            fn.append(fe)
            xn.append(x_new)
            N.append(c)
            E.append(Error)
            e.append(error)
            tabla.append([c, x_new, fe, error])

            x = x_new  
            f = fe 

        except Exception as e:
            print(f"2: {e}")

            df = pd.DataFrame(tabla, columns=["I", "Xi", "F(Xi)", "E"])

            x_vals = np.linspace(X0 - 5, X0 + 5, 400)
            y_vals = [eval(Fun, {"x": val, "math": math}) for val in x_vals]

            plt.figure(figsize=(8, 6))
            plt.plot(x_vals, y_vals, label=f'f(x) = {Fun}', color='blue')
            plt.axhline(0, color='black', linewidth=1)
            plt.xlabel("x")
            plt.ylabel("f(x)")
            plt.title("Método de Punto Fijo")
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

    resultado = ""
    if abs(f) < 1e-12:
        resultado = f"{x} es raíz de f(x)"
    elif error < Tol:
        resultado = f"{x} es una aproximación de una raíz de f(x) con tolerancia {Tol}"
    else:
        df = pd.DataFrame(tabla, columns=["I", "Xi", "F(Xi)", "E"])

        x_vals = np.linspace(X0 - 5, X0 + 5, 400) 
        y_vals = [eval(Fun, {"x": val, "math": math}) for val in x_vals] 

        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, label=f'f(x) = {Fun}', color='blue')
        plt.axhline(0, color='black', linewidth=1)  
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Método de Punto Fijo")
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

    df = pd.DataFrame(tabla, columns=["I", "Xi", "F(Xi)", "E"])

    x_vals = np.linspace(X0 - 5, X0 + 5, 400)
    y_vals = [eval(Fun, {"x": val, "math": math}) for val in x_vals]

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label=f'f(x) = {Fun}', color='blue')
    plt.axhline(0, color='black', linewidth=1)
    plt.scatter(xn[-1], 0, color='red', zorder=5, label=f'Raíz: {round(xn[-1], 4)}')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Método de Punto Fijo")
    plt.legend()
    plt.grid()


    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    string = base64.b64encode(buf.read()).decode()
    img_uri = f"data:image/png;base64,{string}"

    plt.close()

    return resultado, df.to_html(index=False, classes='table table-striped text-center'), img_uri

def generar_informe_punto_fijo(df, fun_str, g_str, x0, tol, niter, tipo_error, tiempo=None):
    """
    df: DataFrame con las iteraciones del método de Punto Fijo.
        Esperamos columnas tipo: iteración, x_n, g(x_n) o f(x_n), error.
    fun_str: f(x) como string (para evaluar f(x_final) si se puede).
    g_str:  g(x) como string (función de iteración).
    x0: valor inicial.
    tol: tolerancia.
    niter: número máximo de iteraciones.
    tipo_error: descripción (cifras significativas, error relativo, etc.).
    tiempo: tiempo total de ejecución en segundos.
    """

    # Asegurarnos de que tenemos un DataFrame
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    # ---------- Manejo del tiempo ----------
    if tiempo is not None:
        tiempo_str = f"{tiempo:.6f}"
    else:
        tiempo_str = "--"

    # ---------- Detección de columnas ----------
    cols = [str(c) for c in df.columns]

    # Iteración (normalmente primera)
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

    # Columna de g(x_n) o f(x_n)
    posibles_fx = ['g(Xn)', 'g(xn)', 'g(Xi)', 'g(xi)', 'g(x)',
                   'f(Xn)', 'f(xn)', 'f(Xi)', 'f(xi)', 'f(x)', 'f(Xm)', 'f(xm)']
    col_fx = None
    for c in cols:
        if c in posibles_fx:
            col_fx = c
            break
    if col_fx is None and len(cols) >= 3:
        col_fx = cols[2]

    # Columna de Error
    posibles_err = ['Error', 'error', 'E', 'err', 'Error relativo', 'error relativo']
    col_err = None
    for c in cols:
        if c in posibles_err:
            col_err = c
            break

    # ---------- Datos finales ----------
    n_iter_real = len(df)
    x_final = df[col_x].iloc[-1]
    fx_final_tabla = df[col_fx].iloc[-1] if col_fx is not None else None

    if n_iter_real >= 2:
        x_prev = df[col_x].iloc[-2]
        delta = x_final - x_prev
    else:
        x_prev = x_final
        delta = 0.0

    # Error según el modo (columna de error si existe)
    if col_err is not None:
        error_final_modo = df[col_err].iloc[-1]
    else:
        error_final_modo = abs(delta)

    # Tolerancia y niter pueden venir None, así que lo manejamos suave
    if tol is not None:
        convergio_por_tol = abs(error_final_modo) <= tol
    else:
        convergio_por_tol = False

    if niter is not None:
        uso_todas_iter = (n_iter_real >= niter)
    else:
        uso_todas_iter = False

    # ---------- Texto resumen ----------
    resumen = []

    resumen.append(
        f"Se aplicó el método de Punto Fijo a la función f(x) = {fun_str} "
        f"usando la función de iteración g(x) = {g_str}, con x₀ = {x0} como aproximación inicial, "
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

    # ---------- Comparación de errores ----------
    error_decimales = abs(delta)
    error_relativo = abs(delta / x_final) if x_final != 0 else 0.0

    # Usamos el valor de la tabla como "f(x_final)" o "g(x_final)"
    if fx_final_tabla is not None:
        error_fx = abs(fx_final_tabla)
    else:
        # Intento de evaluar f(x_final) directamente
        try:
            x_val = float(x_final)
            fx_val = eval(fun_str, {"x": x_val, "math": math})
            error_fx = abs(fx_val)
        except Exception:
            error_fx = None

    comparacion_errores = [
        {
            "tipo": "decimales",
            "x_final": x_final,
            "fx_final": fx_final_tabla,
            "error_final": error_decimales,
            "tiempo": tiempo_str,
        },
        {
            "tipo": "cifras significativas",
            "x_final": x_final,
            "fx_final": fx_final_tabla,
            "error_final": error_relativo,
            "tiempo": tiempo_str,
        },
        {
            "tipo": "relativo_xnm1",
            "x_final": x_final,
            "fx_final": fx_final_tabla,
            "error_final": error_relativo,
            "tiempo": tiempo_str,
        },
        {
            "tipo": "fx",
            "x_final": x_final,
            "fx_final": fx_final_tabla,
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