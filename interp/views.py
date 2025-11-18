from django.shortcuts import render
from .forms import VanderForm, LagrangeForm, NewtonintForm, SplineLinIntForm, SplineCubIntForm, TodoForm
from .metodos.vander import (
    vandermonde,
    generar_informe_vandermonde,
    generar_comparacion_errores_vandermonde,
)
from .metodos.lagrange import (
    lagrange,
    generar_informe_lagrange,
    generar_comparacion_errores_lagrange,
)

from .metodos.newtonint import (
    newtonint,
    generar_informe_newtonint,
    generar_comparacion_errores_newtonint,
)

from .metodos.splinelin import (
    spline_lineal,
    generar_informe_spline_lineal,
    generar_comparacion_errores_spline_lineal,
)

from .metodos.splinecub import (
    spline_cubico,
    generar_informe_spline_cub,
    generar_comparacion_errores_spline_cub,
)
from .metodos.parser import parse_matrix

import matplotlib
matplotlib.use('Agg')
import numpy as np
from io import BytesIO
from django.http import FileResponse
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import base64
import pandas as pd

from sympy import sympify, symbols


def vander_view(request):

    if request.method == 'POST':
        form = VanderForm(request.POST)
        if form.is_valid():

            x_text = form.cleaned_data['x']
            x = parse_matrix(x_text)
            y_text = form.cleaned_data['y']
            y = parse_matrix(y_text)
            grado = form.cleaned_data['grado']

            # Ejecutar Vandermonde (polinomio en string + imagen base64)
            pol, grafica = vandermonde(x, y, grado)

            # Si hubo error en el método, se muestra la pantalla de error
            if pol == "Error":
                return render(request, 'errorInt.html', {'pol': pol, 'grafica': grafica})

            # Leer checkbox (no toca forms.py)
            generar_inf = 'generar_informe' in request.POST

            informe = ""
            comparacion = ""

            # Generar informe y comparación SOLO si el usuario marcó el checkbox
            if generar_inf:
                informe = generar_informe_vandermonde(x, y, grado, pol)
                comparacion = generar_comparacion_errores_vandermonde(x, y, grado)

            resultado = {
                'solucion': pol,
                'grafica': grafica,
                'informe': informe,
                'comparacion': comparacion,
            }

            return render(
                request,
                'result_vander.html',
                {
                    'form': form,
                    'resultado': resultado,
                    'grafica': grafica,
                }
            )
    else:
        form = VanderForm()

    return render(request, 'vander.html', {'form': form})


def lagrange_view(request):

    if request.method == 'POST':
        form = LagrangeForm(request.POST)
        if form.is_valid():

            x_text = form.cleaned_data['x']
            x = parse_matrix(x_text)
            y_text = form.cleaned_data['y']
            y = parse_matrix(y_text)

            # El grado lo fijamos como n-1 (no viene del formulario)
            grado = len(x) - 1

            # Tu función original lagrange casi seguro es lagrange(x, y)
            pol, grafica = lagrange(x, y)

            if pol == "Error":
                return render(request, 'errorInt.html', {'pol': pol, 'grafica': grafica})

            # Checkbox (sin tocar forms.py)
            generar_inf = 'generar_informe' in request.POST

            informe = ""
            comparacion = ""

            if generar_inf:
                informe = generar_informe_lagrange(x, y, grado, pol)
                comparacion = generar_comparacion_errores_lagrange(x, y, grado)

            resultado = {
                'solucion': pol,
                'grafica': grafica,
                'informe': informe,
                'comparacion': comparacion,
            }

            return render(
                request,
                'result_lagrange.html',
                {
                    'form': form,
                    'resultado': resultado,
                    'grafica': grafica,
                }
            )
    else:
        form = LagrangeForm()

    return render(request, 'lagrange.html', {'form': form})


def newtonint_view(request):

    if request.method == 'POST':
        form = NewtonintForm(request.POST)  # <-- aquí
        if form.is_valid():

            x_text = form.cleaned_data['x']
            x = parse_matrix(x_text)
            y_text = form.cleaned_data['y']
            y = parse_matrix(y_text)

            grado = len(x) - 1

            # Ajusta al nombre real de tu función del método de Newton
            pol, grafica = newtonint(x, y)   # o newtonint(x, y, grado) si lo pide

            if pol == "Error":
                return render(request, 'errorInt.html', {'pol': pol, 'grafica': grafica})

            generar_inf = 'generar_informe' in request.POST

            informe = ""
            comparacion = ""

            if generar_inf:
                informe = generar_informe_newtonint(x, y, grado, pol)
                comparacion = generar_comparacion_errores_newtonint(x, y, grado)

            resultado = {
                'solucion': pol,
                'grafica': grafica,
                'informe': informe,
                'comparacion': comparacion,
            }

            return render(
                request,
                'result_newtonint.html',
                {
                    'form': form,
                    'resultado': resultado,
                    'grafica': grafica,
                }
            )
    else:
        form = NewtonintForm()  # <-- aquí también

    return render(request, 'newtonint.html', {'form': form})



def spline_lin_view(request):

    if request.method == 'POST':
        form = SplineLinIntForm(request.POST)
        if form.is_valid():

            x_text = form.cleaned_data['x']
            x = parse_matrix(x_text)
            y_text = form.cleaned_data['y']
            y = parse_matrix(y_text)

            # Tu función numérica de spline lineal
            # (si tu función se llama distinto, cámbialo aquí)
            descripcion_tramos, grafica, _ = spline_lineal(x, y)

            if descripcion_tramos == "Error":
                return render(request, 'errorInt.html', {'pol': descripcion_tramos, 'grafica': grafica})

            # Checkbox
            generar_inf = 'generar_informe' in request.POST

            informe = ""
            comparacion = ""

            if generar_inf:
                informe = generar_informe_spline_lineal(x, y, descripcion_tramos)
                comparacion = generar_comparacion_errores_spline_lineal(x, y)

            resultado = {
                'solucion': descripcion_tramos,
                'grafica': grafica,
                'informe': informe,
                'comparacion': comparacion,
            }

            return render(
                request,
                'result_spline_lin.html',
                {
                    'form': form,
                    'resultado': resultado,
                    'grafica': grafica,
                }
            )
    else:
        form = SplineLinIntForm()

    return render(request, 'spline_lin.html', {'form': form})

def spline_cub_view(request):
    resultado = None
    comparacion_html = ""

    if request.method == 'POST':
        form = SplineCubIntForm(request.POST)
        if form.is_valid():
            x_text = form.cleaned_data['x']
            x = parse_matrix(x_text)
            y_text = form.cleaned_data['y']
            y = parse_matrix(y_text)

            generar_inf = 'generar_informe' in request.POST

            descripcion_tramos, grafica, _ = spline_cubico(x, y)

            informe = ""
            if generar_inf:
                informe = generar_informe_spline_cub(x, y, descripcion_tramos)
                comparacion_html = generar_comparacion_errores_spline_cub(x, y)

            resultado = {
                'solucion': descripcion_tramos,
                'grafica': grafica,
                'informe': informe,
                'comparacion': comparacion_html,
            }

            return render(
                request,
                'result_spline_cub.html',
                {'form': form, 'resultado': resultado, 'grafica': grafica}
            )
    else:
        form = SplineCubIntForm()

    return render(request, 'spline_cub.html', {'form': form})


from io import BytesIO
from django.http import FileResponse
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, Table
from reportlab.lib import colors
import base64

from sympy import symbols, sympify

def todo_view(request):
    if request.method == 'POST':
        form = TodoForm(request.POST)
        if form.is_valid():
            resultados = {}

            x_text = form.cleaned_data['x']
            x = parse_matrix(x_text)
            y_text = form.cleaned_data['y']
            y = parse_matrix(y_text)
            nx_text = form.cleaned_data['nx']
            nx = parse_matrix(nx_text)
            ny_text = form.cleaned_data['ny']
            ny = parse_matrix(ny_text)
            grado = form.cleaned_data['grado']

            # checkbox (PDF sí/no)
            generar_pdf = 'generar_informe' in request.POST

            x_sym = symbols('x')

            # -------- Vandermonde --------
            pol, grafica = vandermonde(x, y, grado)
            if pol != "Error":
                expr = sympify(pol)
                eval_point = [expr.subs(x_sym, val).evalf() for val in nx]
                e = abs(eval_point[0] - ny[0])
            else:
                eval_point = None
                e = None
            resultados['Vandermonde'] = {
                'pol': pol,
                'grafica': grafica,
                'eval': eval_point,
                'error': e,
            }

            # -------- Lagrange --------
            pol, grafica = lagrange(x, y)
            if pol != "Error":
                expr = sympify(pol)
                eval_point = [expr.subs(x_sym, val).evalf() for val in nx]
                e = abs(eval_point[0] - ny[0])
            else:
                eval_point = None
                e = None
            resultados['Lagrange'] = {
                'pol': pol,
                'grafica': grafica,
                'eval': eval_point,
                'error': e,
            }

            # -------- Newton interpolante --------
            pol, grafica = newtonint(x, y)
            if pol != "Error":
                expr = sympify(pol)
                eval_point = [expr.subs(x_sym, val).evalf() for val in nx]
                e = abs(eval_point[0] - ny[0])
            else:
                eval_point = None
                e = None
            resultados['Newton interpolante'] = {
                'pol': pol,
                'grafica': grafica,
                'eval': eval_point,
                'error': e,
            }

            # -------- Spline cúbico --------
            pol, grafica, polinomiosC = spline_cubico(x, y)
            if pol != "Error":
                results = []
                errores = []
                for p in polinomiosC:
                    expr = sympify(p)
                    eval_point = [expr.subs(x_sym, val).evalf() for val in nx]
                    e = abs(eval_point[0] - ny[0])
                    results.append(eval_point)
                    errores.append(e)
                e_prom = sum(errores) / len(errores)
            else:
                results = None
                e_prom = None
            resultados['Spline cúbico'] = {
                'pol': polinomiosC if pol != "Error" else pol,
                'grafica': grafica,
                'eval': results,
                'error': e_prom,
            }

            # -------- Spline lineal --------
            pol, grafica, polinomios = spline_lineal(x, y)
            if pol != "Error":
                results = []
                errores = []
                for p in polinomios:
                    expr = sympify(p)
                    eval_point = [expr.subs(x_sym, val).evalf() for val in nx]
                    e = abs(eval_point[0] - ny[0])
                    results.append(eval_point)
                    errores.append(e)
                e_prom = sum(errores) / len(errores)
            else:
                results = None
                e_prom = None
            resultados['Spline lineal'] = {
                'pol': polinomios if pol != "Error" else pol,
                'grafica': grafica,
                'eval': results,
                'error': e_prom,
            }

            # -------- Mejor método (menor error válido) --------
            errores_validos = {
                nombre: datos['error']
                for nombre, datos in resultados.items()
                if datos['error'] is not None
            }
            mejor_metodo = (
                min(errores_validos, key=errores_validos.get)
                if errores_validos else None
            )

            # ========= RAMA 1: generar PDF =========
            if generar_pdf:
                buffer = BytesIO()

                # Página horizontal para tener más espacio a lo ancho
                doc = SimpleDocTemplate(
                    buffer,
                    pagesize=landscape(letter),
                    rightMargin=20,
                    leftMargin=20,
                    topMargin=20,
                    bottomMargin=20,
                )

                # Estilos
                styles = getSampleStyleSheet()
                normal = styles["Normal"]
                small = styles["Normal"]
                small.fontSize = 8
                small.leading = 10

                def P(texto):
                    return Paragraph(str(texto), small)

                elements = []

                elements.append(Paragraph(
                    "<b>Informe de ejecución – Comparación de métodos de interpolación</b>",
                    styles["Heading2"]
                ))
                elements.append(Spacer(1, 12))

                # Detalle por método
                for metodo, datos in resultados.items():
                    elements.append(Paragraph(f"<b>{metodo}</b>", styles["Heading3"]))
                    elements.append(Paragraph("Polinomio / tramos:", normal))
                    elements.append(Paragraph(str(datos['pol']), small))
                    elements.append(Paragraph(f"Evaluación en nx: {datos['eval']}", normal))
                    elements.append(Paragraph(
                        f"Error: {'N/D' if datos['error'] is None else datos['error']}",
                        normal
                    ))

                    if datos['grafica']:
                        try:
                            image_data = base64.b64decode(datos['grafica'].split(',')[1])
                            img = Image(BytesIO(image_data), width=350, height=250)
                            elements.append(img)
                        except Exception:
                            pass

                    elements.append(Spacer(1, 16))

                # ---------- Tabla de resumen comparativo ----------
                resumen_data = [
                    [P('Método'), P('Polinomio / tramos'), P('Evaluación en nx'), P('Error')],
                ]

                for metodo, datos in resultados.items():
                    resumen_data.append([
                        P(metodo),
                        P(datos['pol']),
                        P(datos['eval']),
                        P('N/D' if datos['error'] is None else datos['error']),
                    ])

                # Anchos de columna para que no se corte
                col_widths = [80, 260, 180, 60]

                tabla_resumen = Table(resumen_data, colWidths=col_widths)
                tabla_resumen.setStyle(TableStyle([
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))

                elements.append(Spacer(1, 24))
                elements.append(Paragraph("<b>Resumen comparativo</b>", styles["Heading3"]))
                elements.append(tabla_resumen)

                elements.append(Spacer(1, 16))
                if mejor_metodo:
                    elements.append(Paragraph(
                        f"Mejor método según el menor error: <b>{mejor_metodo}</b> "
                        f"(error = {errores_validos[mejor_metodo]}).",
                        normal
                    ))
                else:
                    elements.append(Paragraph(
                        "No se pudo determinar un mejor método (todos los errores son N/D).",
                        normal
                    ))

                doc.build(elements)
                buffer.seek(0)
                return FileResponse(
                    buffer,
                    as_attachment=True,
                    filename="resultados_metodos_interpolacion.pdf"
                )

            # ========= RAMA 2: mostrar en pantalla =========
            return render(
                request,
                'todosInt.html',
                {
                    'form': form,
                    'resultados': resultados,
                    'mejor_metodo': mejor_metodo,
                }
            )

    else:
        form = TodoForm()

    # GET: solo el formulario
    return render(request, 'todosInt.html', {'form': form})