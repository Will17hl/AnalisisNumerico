from django.shortcuts import render

from .forms import JacobiForm, GausseidelForm, SORForm, TodosForm
from .metodos.jacobi import (
    jacobi,
    generar_informe_jacobi,
    generar_comparacion_errores_jacobi,
)
from .metodos.gauss_seidel import (
    gausseidel,
    generar_informe_gauss_seidel,
    generar_comparacion_errores_gauss_seidel,
)
from .metodos.SOR import (
    SOR,
    generar_informe_SOR,
    generar_comparacion_errores_SOR,
)
from .metodos.parser import parse_matrix

import numpy as np
import pandas as pd
import time

from io import BytesIO, StringIO
from django.http import FileResponse
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet



def jacobi_view(request):
    resultado = None

    if request.method == 'POST':
        form = JacobiForm(request.POST)
        if form.is_valid():
            A_text = form.cleaned_data['A']
            A = parse_matrix(A_text)
            b_text = form.cleaned_data['b']
            b = parse_matrix(b_text)
            x0_text = form.cleaned_data['x0']
            x0 = parse_matrix(x0_text)
            tol = form.cleaned_data['tol']
            niter = form.cleaned_data['niter']
            modo = form.cleaned_data['Modo']

            # Checkbox del informe
            generar_inf = 'generar_informe' in request.POST

            # ⏱ Medimos el tiempo del método
            t0 = time.perf_counter()
            tabla, solucion, error, sp_radius = jacobi(x0, A, b, tol, niter, modo)
            t1 = time.perf_counter()
            tiempo_ejec = t1 - t0

            # Convergencia por radio espectral
            if sp_radius < 1:
                convergencia = 'Radio espectral < 1, por lo tanto el método converge'
            else:
                convergencia = 'Radio espectral > 1, el método no converge'

            informe = ""
            comparacion = ""

            if generar_inf:
                df_resultado = pd.read_html(tabla)[0]

                informe = generar_informe_jacobi(
                    df=df_resultado,
                    A=A,
                    b=b,
                    x0=x0,
                    tol=tol,
                    niter=niter,
                    tipo_error=modo,
                    tiempo=tiempo_ejec,
                    sp_radius=sp_radius,
                    solucion=solucion if not isinstance(solucion, str) else None
                )

                comparacion = generar_comparacion_errores_jacobi(
                    df_resultado, A, b, solucion, tiempo_ejec
                )

            resultado = {
                'solucion': solucion,
                'radio': sp_radius,
                'convergencia': convergencia,
                'tabla': tabla,
                'informe': informe,
                'comparacion_errores': comparacion,
            }

            if error is True:
                return render(
                    request,
                    'errorN.html',
                    {'form': form, 'resultado': resultado, 'tabla': tabla}
                )

            return render(
                request,
                'result_jacobi.html',
                {'form': form, 'resultado': resultado, 'tabla': tabla}
            )
    else:
        form = JacobiForm()

    return render(request, 'jacobi.html', {'form': form})



def gausseidel_view(request):
    resultado = None

    if request.method == 'POST':
        form = GausseidelForm(request.POST)
        if form.is_valid():
            A_text = form.cleaned_data['A']
            A = parse_matrix(A_text)
            b_text = form.cleaned_data['b']
            b = parse_matrix(b_text)
            x0_text = form.cleaned_data['x0']
            x0 = parse_matrix(x0_text)
            tol = form.cleaned_data['tol']
            niter = form.cleaned_data['niter']
            modo = form.cleaned_data['Modo']

            # Leer el checkbox (sin tocar forms.py)
            generar_inf = 'generar_informe' in request.POST

            # ⏱ Medir tiempo de ejecución del método
            t0 = time.perf_counter()
            tabla, solucion, error, sp_radius = gausseidel(x0, A, b, tol, niter, modo)
            t1 = time.perf_counter()
            tiempo_ejec = t1 - t0

            # Mensaje de convergencia
            if sp_radius < 1:
                convergencia = 'Radio espectral < 1, por lo tanto el método converge'
            else:
                convergencia = 'Radio espectral > 1, el método no converge'

            informe = ""
            comparacion = ""

            if generar_inf:
                # Convertimos la tabla HTML a DataFrame
                df_resultado = pd.read_html(tabla)[0]

                # Informe de ejecución
                informe = generar_informe_gauss_seidel(
                    df=df_resultado,
                    A=A,
                    b=b,
                    x0=x0,
                    tol=tol,
                    niter=niter,
                    tipo_error=modo,
                    tiempo=tiempo_ejec,
                    sp_radius=sp_radius,
                    solucion=solucion if not isinstance(solucion, str) else None
                )

                # Tabla de comparación de errores
                comparacion = generar_comparacion_errores_gauss_seidel(
                    df_resultado, A, b, solucion, tiempo_ejec
                )

            resultado = {
                'solucion': solucion,
                'radio': sp_radius,
                'convergencia': convergencia,
                'tabla': tabla,
                'informe': informe,
                'comparacion_errores': comparacion,
            }

            if error is True:
                return render(
                    request,
                    'errorN.html',
                    {'form': form, 'resultado': resultado, 'tabla': tabla}
                )

            return render(
                request,
                'result_gausseidel.html',
                {'form': form, 'resultado': resultado, 'tabla': tabla}
            )
    else:
        form = GausseidelForm()

    return render(request, 'gausseidel.html', {'form': form})



def SOR_view(request):
    resultado = None
    errores = None

    if request.method == 'POST':
        form = SORForm(request.POST)
        if form.is_valid():

            A_text = form.cleaned_data['A']
            A = parse_matrix(A_text)
            b_text = form.cleaned_data['b']
            b = parse_matrix(b_text)
            x0_text = form.cleaned_data['x0']
            x0 = parse_matrix(x0_text)
            tol = form.cleaned_data['tol']
            niter = form.cleaned_data['niter']
            w = form.cleaned_data['w']
            modo = form.cleaned_data['Modo']

            # Checkbox del informe
            generar_inf = 'generar_informe' in request.POST

            # ⏱ medir tiempo del método SOR
            t0 = time.perf_counter()
            tabla, solucion, error, sp_radius = SOR(x0, A, b, tol, niter, w, modo)
            t1 = time.perf_counter()
            tiempo_ejec = t1 - t0

            if sp_radius < 1:
                convergencia = 'Radio espectral < 1, por lo tanto el método converge'
            else:
                convergencia = 'Radio espectral > 1, el método no converge'

            informe = ""
            comparacion = ""

            if generar_inf:
                df_resultado = pd.read_html(tabla)[0]

                informe = generar_informe_SOR(
                    df=df_resultado,
                    A=A,
                    b=b,
                    x0=x0,
                    tol=tol,
                    niter=niter,
                    w=w,
                    tipo_error=modo,
                    tiempo=tiempo_ejec,
                    sp_radius=sp_radius,
                    solucion=solucion if not isinstance(solucion, str) else None
                )

                comparacion = generar_comparacion_errores_SOR(
                    df_resultado, A, b, solucion, tiempo_ejec
                )

            resultado = {
                'solucion': solucion,
                'radio': sp_radius,
                'convergencia': convergencia,
                'tabla': tabla,
                'informe': informe,
                'comparacion_errores': comparacion,
            }

            if error is True:
                return render(
                    request,
                    'errorN.html',
                    {'form': form, 'resultado': resultado, 'tabla': tabla}
                )

            return render(
                request,
                'result_SOR.html',
                {'form': form, 'resultado': resultado, 'tabla': tabla}
            )
    else:
        form = SORForm()

    return render(request, 'SOR.html', {'form': form})



def todos_view(request):
    if request.method == 'POST':
        form = TodosForm(request.POST)
        if form.is_valid():

            generar_inf = 'generar_informe' in request.POST  # checkbox

            try:
                resultados = {}

                A = parse_matrix(form.cleaned_data['A'])
                b = parse_matrix(form.cleaned_data['b'])
                x0 = parse_matrix(form.cleaned_data['x0'])
                tol = form.cleaned_data['tol']
                w1 = form.cleaned_data['w1']
                w2 = form.cleaned_data['w2']
                w3 = form.cleaned_data['w3']
                niter = form.cleaned_data['niter']
                modo = form.cleaned_data['Modo']

                # --------------------
                #  JACOBI
                # --------------------
                t0 = time.perf_counter()
                tabla_j, sol_j, err_j, sp_j = jacobi(x0, A, b, tol, niter, modo)
                t1 = time.perf_counter()
                tiempo_j = t1 - t0

                df_j = pd.read_html(StringIO(tabla_j))[0]

                resultados['Jacobi'] = {
                    'tabla': tabla_j,
                    'df': df_j,
                    'solucion': sol_j,
                    'radio': sp_j,
                    'iteraciones': len(df_j),
                    'tiempo': tiempo_j,
                }

                # --------------------
                #  GAUSS-SEIDEL
                # --------------------
                t0 = time.perf_counter()
                tabla_g, sol_g, err_g, sp_g = gausseidel(x0, A, b, tol, niter, modo)
                t1 = time.perf_counter()
                tiempo_g = t1 - t0

                df_g = pd.read_html(StringIO(tabla_g))[0]

                resultados['Gauss-Seidel'] = {
                    'tabla': tabla_g,
                    'df': df_g,
                    'solucion': sol_g,
                    'radio': sp_g,
                    'iteraciones': len(df_g),
                    'tiempo': tiempo_g,
                }

                # --------------------
                #  SOR w1
                # --------------------
                t0 = time.perf_counter()
                tabla_s1, sol_s1, err_s1, sp_s1 = SOR(x0, A, b, tol, niter, w1, modo)
                t1 = time.perf_counter()
                tiempo_s1 = t1 - t0

                df_s1 = pd.read_html(StringIO(tabla_s1))[0]

                resultados[f'SOR (w={w1})'] = {
                    'tabla': tabla_s1,
                    'df': df_s1,
                    'solucion': sol_s1,
                    'radio': sp_s1,
                    'iteraciones': len(df_s1),
                    'tiempo': tiempo_s1,
                }

                # --------------------
                #  SOR w2
                # --------------------
                t0 = time.perf_counter()
                tabla_s2, sol_s2, err_s2, sp_s2 = SOR(x0, A, b, tol, niter, w2, modo)
                t1 = time.perf_counter()
                tiempo_s2 = t1 - t0

                df_s2 = pd.read_html(StringIO(tabla_s2))[0]

                resultados[f'SOR (w={w2})'] = {
                    'tabla': tabla_s2,
                    'df': df_s2,
                    'solucion': sol_s2,
                    'radio': sp_s2,
                    'iteraciones': len(df_s2),
                    'tiempo': tiempo_s2,
                }

                # --------------------
                #  SOR w3
                # --------------------
                t0 = time.perf_counter()
                tabla_s3, sol_s3, err_s3, sp_s3 = SOR(x0, A, b, tol, niter, w3, modo)
                t1 = time.perf_counter()
                tiempo_s3 = t1 - t0

                df_s3 = pd.read_html(StringIO(tabla_s3))[0]

                resultados[f'SOR (w={w3})'] = {
                    'tabla': tabla_s3,
                    'df': df_s3,
                    'solucion': sol_s3,
                    'radio': sp_s3,
                    'iteraciones': len(df_s3),
                    'tiempo': tiempo_s3,
                }

                # ------------------------------------------------
                #  SI NO HAY CHECKBOX -> MOSTRAR EN PANTALLA
                # ------------------------------------------------
                if not generar_inf:
                    # calcular mejor método por iteraciones
                    mejor_metodo = None
                    menor_iter = float('inf')

                    for metodo, datos in resultados.items():
                        try:
                            it = int(datos['iteraciones'])
                            if it < menor_iter and datos['solucion'] != 'Error':
                                menor_iter = it
                                mejor_metodo = metodo
                        except Exception:
                            continue

                    mensaje_final = None
                    if mejor_metodo:
                        mensaje_final = (
                            f"El mejor método según el menor número de iteraciones es: "
                            f"{mejor_metodo} con {menor_iter} iteraciones."
                        )

                    # renderizar plantilla mostrando resultados
                    return render(
                        request,
                        'todosN.html',
                        {
                            'form': form,
                            'resultados': resultados,
                            'mensaje_final': mensaje_final,
                        }
                    )

                # ------------------------------------------------
                #  SI HAY CHECKBOX -> GENERAR PDF
                # ------------------------------------------------
                # aquí sí usamos las comparaciones de errores
                for metodo, datos in resultados.items():
                    df_met = datos['df']
                    sol_met = datos['solucion']
                    tiempo_met = datos['tiempo']

                    if metodo == 'Jacobi':
                        comp = generar_comparacion_errores_jacobi(
                            df_met, A, b, sol_met, tiempo_met
                        )
                    elif metodo == 'Gauss-Seidel':
                        comp = generar_comparacion_errores_gauss_seidel(
                            df_met, A, b, sol_met, tiempo_met
                        )
                    else:
                        # todos los SOR
                        comp = generar_comparacion_errores_SOR(
                            df_met, A, b, sol_met, tiempo_met
                        )

                    datos['comparacion'] = comp

                # construcción del PDF igual que antes,
                # pero usando el diccionario 'resultados'
                buffer = BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=letter)
                elements = []
                styles = getSampleStyleSheet()

                # Sección por método
                for metodo, datos in resultados.items():
                    elements.append(Paragraph(f"<b>{metodo}</b>", styles['Heading2']))
                    elements.append(
                        Paragraph(
                            f"Solución aproximada: {datos['solucion']}",
                            styles['Normal'],
                        )
                    )
                    elements.append(
                        Paragraph(
                            f"Radio espectral: {datos['radio']:.6f}",
                            styles['Normal'],
                        )
                    )
                    elements.append(Spacer(1, 6))

                    tabla_iter = datos['df']
                    table_data = [list(tabla_iter.columns)] + tabla_iter.values.tolist()
                    elements.append(
                        Table(
                            table_data,
                            style=[
                                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                            ],
                        )
                    )
                    elements.append(Spacer(1, 8))

                    # tabla de comparación de errores
                    elements.append(
                        Paragraph("Comparación de tipos de error", styles['Normal'])
                    )
                    tabla_err = pd.read_html(StringIO(datos['comparacion']))[0]
                    err_data = [list(tabla_err.columns)] + tabla_err.values.tolist()
                    elements.append(
                        Table(
                            err_data,
                            style=[
                                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                            ],
                        )
                    )
                    elements.append(Spacer(1, 16))

                # Resumen y mejor método
                resumen_data = [['Método', 'Solución', 'Iteraciones']]
                mejor_metodo = None
                menor_iter = float('inf')

                for metodo, datos in resultados.items():
                    resumen_data.append(
                        [metodo, str(datos['solucion']), str(datos['iteraciones'])]
                    )
                    try:
                        it = int(datos['iteraciones'])
                        if it < menor_iter and datos['solucion'] != 'Error':
                            menor_iter = it
                            mejor_metodo = metodo
                    except Exception:
                        continue

                elements.append(Spacer(1, 12))
                elements.append(Paragraph("<b>Resumen de métodos</b>", styles['Heading2']))
                elements.append(
                    Table(
                        resumen_data,
                        style=[
                            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                        ],
                    )
                )
                elements.append(Spacer(1, 8))

                if mejor_metodo:
                    mensaje_final = (
                        f"El mejor método según el menor número de iteraciones es: "
                        f"<b>{mejor_metodo}</b> con {menor_iter} iteraciones."
                    )
                else:
                    mensaje_final = (
                        "No se pudo determinar un mejor método por errores en los datos."
                    )

                elements.append(Paragraph(mensaje_final, styles['Normal']))

                doc.build(elements)
                buffer.seek(0)
                return FileResponse(
                    buffer,
                    as_attachment=True,
                    filename="resultados_metodos.pdf"
                )

            except Exception as e:
                resultado = "Error"
                tabla = [f"Error en los datos: {e}"]
                return render(
                    request,
                    'todosN.html',
                    {'form': form, 'resultado': resultado, 'tabla': tabla}
                )

    else:
        form = TodosForm()

    return render(request, 'todosN.html', {'form': form})
