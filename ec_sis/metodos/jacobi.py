import numpy as np
import pandas as pd

def jacobi(x0, A, b, tol, niter, modo):
    c = 0
    error = tol + 1
    E = []
    all_x = []

    aux = False

    try:

        D = np.diag(np.diag(A))
        L = -np.tril(A, -1)
        U = -np.triu(A, 1)
        
        T = np.linalg.inv(D).dot(L + U)
        C = np.linalg.inv(D).dot(b)

        sp_radius = max(abs(np.linalg.eigvals(T)))

    except Exception as e:
        print(f"Error en la matriz: {e}")
        return "N/A", "Error", True, 1000
    
    while error > tol and c < niter:
        
        try:

            x1 = T.dot(x0) + C

            if(modo == "cs"):
                error = np.linalg.norm((x1 - x0) / x1, np.inf)
            else:
                error = np.linalg.norm(x1 - x0, np.inf)
                
            E.append(error)
            x0 = x1
            c += 1
            all_x.append(x1)
        except Exception as e:
            print(f"Error2: {e}")

            table_data = {'Iteración': np.arange(1, c + 1)}
            
            for i in range(len(A)):
                table_data[f'x{i+1}'] = [x[i] for x in all_x]
            
            table_data['Error'] = E
            
            df_resultado = pd.DataFrame(table_data)
            tabla_html = df_resultado.to_html(index=False, classes='table table-striped text-center')
            
            return tabla_html, "Error", True, sp_radius

    if error < tol:
        print(f"\nSolución aproximada encontrada con tolerancia {tol}: {x0}")
    else:
        x0 = "Error"
        print(f'⚠️  Fracasó en {niter} iteraciones. ')
        aux = True
    # Crear la tabla con pandas
    table_data = {'Iteración': np.arange(1, c + 1)}
    
    # Agregar las columnas para los valores de x1, x2, ..., xn
    for i in range(len(A)):
        table_data[f'x{i+1}'] = [x[i] for x in all_x]
    
    # Agregar la columna de errores
    table_data['Error'] = E
    
    # Crear el DataFrame
    df_resultado = pd.DataFrame(table_data)
    tabla_html = df_resultado.to_html(index=False, classes='table table-striped text-center')
    
    return tabla_html, x0, aux, sp_radius


def generar_informe_jacobi(
    df,
    A,
    b,
    x0,
    tol,
    niter,
    tipo_error,
    tiempo=None,
    sp_radius=None,
    solucion=None,
):
    """
    Genera el informe HTML para el método de Jacobi.
    """

    iteraciones = len(df)

    # Error final (última fila de la tabla)
    if iteraciones > 0 and "Error" in df.columns:
        error_final = float(df["Error"].iloc[-1])
    else:
        error_final = 0.0

    # Aproximación final
    if solucion is not None and isinstance(solucion, (list, np.ndarray)):
        aprox_final = np.array(solucion, dtype=float).flatten()
    else:
        cols_x = [c for c in df.columns if c.startswith("x")]
        aprox_final = df[cols_x].iloc[-1].values if cols_x else None

    if aprox_final is not None:
        aprox_str = np.array2string(aprox_final, precision=8, separator=", ")
    else:
        aprox_str = "N/A"

    # Representar A, b, x0
    A_str = np.array2string(A, precision=4, separator=", ")
    b_str = np.array2string(np.array(b).flatten(), precision=4, separator=", ")
    x0_str = np.array2string(np.array(x0).flatten(), precision=4, separator=", ")

    # Tipo de error
    if tipo_error == "cs":
        tipo_error_str = "cifras significativas (error relativo)"
    else:
        tipo_error_str = "error absoluto (norma infinito)"

    # Radio espectral
    if sp_radius is not None:
        if sp_radius < 1:
            texto_radio = (
                f"El radio espectral de la matriz de iteración es "
                f"ρ(T) ≈ {sp_radius:.6f} (< 1), por lo que el método de Jacobi "
                "es convergente para este sistema."
            )
        else:
            texto_radio = (
                f"El radio espectral de la matriz de iteración es "
                f"ρ(T) ≈ {sp_radius:.6f} (≥ 1), por lo que el método de Jacobi "
                "puede no ser convergente para este sistema."
            )
    else:
        texto_radio = ""

    # Cumplimiento de la tolerancia
    if error_final <= tol:
        texto_tol = (
            "La aproximación obtenida cumple el criterio de parada especificado por la "
            "tolerancia, por lo que se considera que el método de Jacobi convergió "
            "adecuadamente para este sistema."
        )
    else:
        texto_tol = (
            "El error no alcanzó la tolerancia objetivo dentro del número máximo de "
            "iteraciones, por lo que se considera que el método de Jacobi no convergió "
            "en las condiciones dadas."
        )

    # Tiempo de ejecución
    if tiempo is not None:
        tiempo_str = f"{tiempo:.6f} s"
    else:
        tiempo_str = "N/D"

    informe_html = f"""
    <hr class="mt-5 mb-4">
    <h2>Informe de ejecución – Método de Jacobi</h2>
    <p>
        Se aplicó el método de Jacobi al sistema lineal <code>Ax = b</code>, con:
        <br><strong>A</strong> = {A_str},
        <br><strong>b</strong> = {b_str},
        <br><strong>x</strong><sup>(0)</sup> = {x0_str}.
        <br>Se utilizó una tolerancia de {tol} y un máximo de {niter} iteraciones,
        empleando como criterio de error {tipo_error_str}.
    </p>
    <p>
        El método realizó {iteraciones} iteraciones y obtuvo como aproximación final
        x<sub>aprox</sub> ≈ {aprox_str}. El error final fue aproximadamente
        {error_final:.8e}.
    </p>
    <p>{texto_tol}</p>
    <p>{texto_radio}</p>
    <ul>
        <li><strong>Iteraciones realizadas:</strong> {iteraciones}</li>
        <li><strong>Aproximación final:</strong> {aprox_str}</li>
        <li><strong>Error final:</strong> {error_final:.8e}</li>
        <li><strong>Criterio de error:</strong> {tipo_error_str}</li>
        <li><strong>Tiempo de ejecución:</strong> {tiempo_str}</li>
    </ul>
    """

    return informe_html


def generar_comparacion_errores_jacobi(df, A, b, solucion, tiempo):
    """
    Genera una tabla HTML comparando distintos tipos de error
    para la última iteración del método de Jacobi.
    """

    # Columnas de las variables x1, x2, ..., xn
    cols_x = [c for c in df.columns if c.startswith("x")]
    if len(cols_x) == 0 or len(df) == 0:
        return ""

    # Última aproximación x^(k)
    if solucion is not None and isinstance(solucion, (list, np.ndarray)):
        xk = np.array(solucion, dtype=float).flatten()
    else:
        xk = df[cols_x].iloc[-1].values.astype(float)

    # Aproximación anterior x^(k-1) (si solo hay 1 iteración, usamos la misma)
    if len(df) >= 2:
        xk_1 = df[cols_x].iloc[-2].values.astype(float)
    else:
        xk_1 = xk.copy()

    # Error absoluto entre iteraciones
    err_abs = float(np.linalg.norm(xk - xk_1, np.inf))

    # Error relativo entre iteraciones
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_vec = np.where(xk != 0, (xk - xk_1) / xk, 0.0)
    err_rel = float(np.linalg.norm(rel_vec, np.inf))

    # Norma del residual ||Ax - b||∞
    b_vec = np.array(b).flatten()
    residuo = A @ xk - b_vec
    err_res = float(np.linalg.norm(residuo, np.inf))

    # Representación del vector solución
    x_str = np.array2string(xk, precision=8, separator=", ")

    tiempo_val = float(tiempo) if tiempo is not None else float("nan")

    data = {
        "Tipo de error": [
            "Error absoluto entre iteraciones",
            "Error relativo entre iteraciones (cifras significativas)",
            "Norma del residual ||Ax - b||∞",
        ],
        "x_aprox": [x_str, x_str, x_str],
        "Error final": [err_abs, err_rel, err_res],
        "Tiempo (s)": [tiempo_val, tiempo_val, tiempo_val],
    }

    df_comp = pd.DataFrame(data)

    tabla_html = df_comp.to_html(
        index=False,
        classes="table table-striped text-center",
        float_format=lambda x: f"{x:.8e}",
    )

    return tabla_html
