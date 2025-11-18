from django.shortcuts import render
from .forms import PuntoFijoForm, NewtonForm, BiseccionForm, RaicesMultiplesForm, ReglaFalsaForm, SecanteForm, todosForm
from .metodos import Puntofijo, Newton, Reglafalsa, Secante
from .metodos.Biseccion import biseccion, generar_informe_biseccion
from .metodos.Raices_multiples import raices_multiples
from sympy import symbols, sympify, lambdify
from io import BytesIO, StringIO
from django.http import FileResponse
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, Table, TableStyle
from reportlab.lib import colors
import base64
import pandas as pd
import time
from .metodos.Newton import generar_informe_newton
from .metodos.Puntofijo import generar_informe_punto_fijo
from .metodos.Secante import generar_informe_secante
from .metodos.Raices_multiples import raices_multiples, generar_informe_raices_multiples
from .metodos.Reglafalsa import generar_informe_regla_falsa
import time



def biseccion_view(request):
    """
    Vista del método de Bisección.
    - Resuelve el método con los datos del formulario.
    - Opcional: genera un informe de ejecución si el usuario marca la casilla
'correr_informe' en el formulario.
    """
    if request.method == 'POST':
        form = BiseccionForm(request.POST)
        informe = None

        if form.is_valid():
            data = form.cleaned_data

            # Extraemos explícitamente los parámetros (para el informe)
            a = data.get("a")
            b = data.get("b")
            tol = data.get("tol")
            niter = data.get("niter")
            fun = data.get("fun")
            modo = data.get("Modo")  # 'cs', 'cr', etc. según tu lógica

            # Ejecutar método de bisección
            t0 = time.time()
            resultado, tabla, img = biseccion(a, b, tol, niter, fun, modo)
            tiempo_biseccion = time.time() - t0
            
            if resultado == "Error":
                return render(
                    request,
                    'error.html',
                    {'resultado': resultado, 'tabla': tabla, 'imagen': img}
                )

            # ¿El usuario pidió generar el informe?
            correr_informe = request.POST.get("correr_informe")  # checkbox del template

            if correr_informe == "on":
                # Mapeo simple de modo -> tipo de error
                if modo == "cs":
                    tipo_error = "cifras significativas"
                elif modo == "cr":
                    tipo_error = "error relativo"
                else:
                    tipo_error = "error absoluto"

                try:
                    # Convertir tabla HTML a DataFrame (forma recomendada)
                    df_tabla = pd.read_html(StringIO(tabla))[0]
                    informe = generar_informe_biseccion(
                        df_tabla,
                        fun_str=fun,
                        a=a,
                        b=b,
                        tol=tol,
                        niter=niter,
                        tipo_error=tipo_error,
                        tiempo=tiempo_biseccion,
                    )
                except Exception as e:
                    # Si algo falla, solo no mostramos informe
                    print("Error generando informe de bisección:", e)
                    informe = None

            return render(
                request,
                'result_biseccion.html',
                {
                    'resultado': resultado,
                    'tabla': tabla,
                    'imagen': img,
                    'informe': informe,  # puede ser None si no lo pidió o si falló
                }
            )
    else:
        form = BiseccionForm()

    return render(request, 'biseccion.html', {'form': form})

def regla_falsa(request):
    if request.method == 'POST':
        form = ReglaFalsaForm(request.POST)
        informe = None

        if form.is_valid():
            data = form.cleaned_data

            # Modo y copia sin 'Modo' para pasar al método numérico
            modo = data.get('Modo')
            data_sin_modo = data.copy()
            data_sin_modo.pop('Modo', None)

            # Checkbox del template
            correr_informe = request.POST.get("correr_informe") == "on"

            # Ejecutar Regla Falsa midiendo tiempo
            if modo == 'cs':
                t0 = time.time()
                resultado, tabla, imagen = Reglafalsa.reglafalsaCS(**data_sin_modo)
                tiempo_regla = time.time() - t0
            else:
                t0 = time.time()
                resultado, tabla, imagen = Reglafalsa.reglafalsaDC(**data_sin_modo)
                tiempo_regla = time.time() - t0

            # Si hubo error en el método → página de error
            if resultado == "Error":
                return render(
                    request,
                    'error.html',
                    {'resultado': resultado, 'tabla': tabla, 'imagen': imagen}
                )

            # Tipo de error según el modo
            if modo == "cs":
                tipo_error = "Cifras significativas"
            elif modo == "cr":
                tipo_error = "Error relativo"
            else:
                tipo_error = "Error absoluto"

            # Si el usuario pidió informe → generarlo
            if correr_informe:
                try:
                    df_tabla = pd.read_html(StringIO(tabla))[0]

                    # ⬇⬇ OJO: aquí soportamos mayúsculas y minúsculas
                    a = data.get("a")
                    b = data.get("b")

                    tol = data.get("tol")
                    if tol is None:
                        tol = data.get("Tol")

                    niter = data.get("niter")
                    if niter is None:
                        niter = data.get("Niter")

                    fun = data.get("fun")
                    if fun is None:
                        fun = data.get("Fun")

                    informe = generar_informe_regla_falsa(
                        df=df_tabla,
                        fun_str=fun,
                        a=a,
                        b=b,
                        tol=tol,
                        niter=niter,
                        tipo_error=tipo_error,
                        tiempo=tiempo_regla,
                    )

                except Exception as e:
                    print("Error generando informe de Regla Falsa:", e)
                    informe = None

            # Render del resultado (con o sin informe)
            return render(
                request,
                'result_rf.html',
                {
                    'resultado': resultado,
                    'tabla': tabla,
                    'imagen': imagen,
                    'informe': informe,
                }
            )
    else:
        form = ReglaFalsaForm()

    return render(request, 'regla_falsa.html', {'form': form})


def newton(request):
    if request.method == 'POST':
        form = NewtonForm(request.POST)
        informe = None

        if form.is_valid():
            data = form.cleaned_data

            # Modo (tipo de error)
            modo = data.get('Modo')
            data_sin_modo = data.copy()
            data_sin_modo.pop('Modo', None)

            # Checkbox del template
            correr_informe = request.POST.get("correr_informe") == "on"

            # Ejecutar Newton midiendo tiempo
            if modo == 'cs':
                t0 = time.time()
                resultado, tabla, imagen = Newton.metodo_newtonCS(**data_sin_modo)
                tiempo_newton = time.time() - t0
            else:
                t0 = time.time()
                resultado, tabla, imagen = Newton.metodo_newton(**data_sin_modo)
                tiempo_newton = time.time() - t0

            # Si hubo error en el método numérico
            if resultado == "Error":
                return render(
                    request,
                    'error.html',
                    {'resultado': resultado, 'tabla': tabla, 'imagen': imagen}
                )

            # Tipo de error según el modo
            if modo == "cs":
                tipo_error = "Cifras significativas"
            elif modo == "cr":
                tipo_error = "Error relativo"
            else:
                tipo_error = "Error absoluto"

            # Si el usuario pidió informe -> lo generamos
            if correr_informe:
                try:
                    # Pasar tabla HTML a DataFrame
                    df_tabla = pd.read_html(StringIO(tabla))[0]

                    # 🔹 Usamos el MISMO diccionario que se pasó al método numérico
                    fun_str = (
                        data_sin_modo.get('Fun')
                        or data_sin_modo.get('fun')
                        or ""
                    )
                    df_str = (
                        data_sin_modo.get('df_expr')
                        or data_sin_modo.get('df')
                        or ""
                    )

                    x0_val = (
                        data_sin_modo.get('X0')
                        or data_sin_modo.get('x0')
                    )
                    tol_val = (
                        data_sin_modo.get('Tol')
                        or data_sin_modo.get('tol')
                    )
                    niter_val = (
                        data_sin_modo.get('Niter')
                        or data_sin_modo.get('niter')
                    )

                    informe = generar_informe_newton(
                        df_tabla,
                        fun_str=fun_str,
                        df_str=df_str,
                        x0=x0_val,
                        tol=tol_val,
                        niter=niter_val,
                        tipo_error=tipo_error,
                        tiempo=tiempo_newton
                    )

                except Exception as e:
                    print("Error generando informe de Newton:", e)
                    informe = None

            # Render del resultado con (o sin) informe
            return render(
                request,
                'result_ne.html',
                {
                    'resultado': resultado,
                    'tabla': tabla,
                    'imagen': imagen,
                    'informe': informe,
                }
            )
    else:
        form = NewtonForm()

    return render(request, 'newton.html', {'form': form})

def secante(request):
    if request.method == 'POST':
        form = SecanteForm(request.POST)
        informe = None

        if form.is_valid():
            data = form.cleaned_data

            # Guardamos el modo y creamos copia sin 'Modo'
            modo = data.get('Modo')
            data_sin_modo = data.copy()
            data_sin_modo.pop('Modo', None)

            # Checkbox del template
            correr_informe = request.POST.get("correr_informe") == "on"

            # Ejecutar Secante midiendo tiempo
            if modo == 'cs':
                t0 = time.time()
                resultado, tabla, imagen = Secante.secanteCS(**data_sin_modo)
                tiempo_secante = time.time() - t0
            else:
                t0 = time.time()
                resultado, tabla, imagen = Secante.secanteDC(**data_sin_modo)
                tiempo_secante = time.time() - t0

            # Si hubo error en el método → página de error
            if resultado == "Error":
                return render(
                    request,
                    'error.html',
                    {'resultado': resultado, 'tabla': tabla, 'imagen': imagen}
                )

            # Tipo de error según el modo
            if modo == "cs":
                tipo_error = "Cifras significativas"
            elif modo == "cr":
                tipo_error = "Error relativo"
            else:
                tipo_error = "Error absoluto"

            # Si el usuario pidió informe → generarlo
            if correr_informe:
                try:
                    from io import StringIO
                    df_tabla = pd.read_html(StringIO(tabla))[0]

                    # Intentamos cubrir todos los posibles nombres del form
                    x0 = data.get("a") or data.get("x0") or data.get("X0")
                    x1 = data.get("b") or data.get("x1") or data.get("X1")
                    tol = data.get("tol") or data.get("Tol")
                    niter = data.get("niter") or data.get("Niter")
                    fun = data.get("fun") or data.get("Fun")

                    informe = generar_informe_secante(
                        df_tabla,
                        fun_str=fun,
                        x0=x0,
                        x1=x1,
                        tol=tol,
                        niter=niter,
                        tipo_error=tipo_error,
                        tiempo=tiempo_secante
                    )

                except Exception as e:
                    print("Error generando informe de Secante:", e)
                    informe = None

            # Render del resultado (con o sin informe)
            return render(
                request,
                'result_sc.html',
                {
                    'resultado': resultado,
                    'tabla': tabla,
                    'imagen': imagen,
                    'informe': informe,
                }
            )
    else:
        form = SecanteForm()

    return render(request, 'secante.html', {'form': form})

def raices_multiples_view(request):
    if request.method == 'POST':
        form = RaicesMultiplesForm(request.POST)
        informe = None

        if form.is_valid():
            data = form.cleaned_data

            # Obtenemos el modo (pero NO lo quitamos del diccionario)
            modo = data.get('Modo')

            # Checkbox del template
            correr_informe = request.POST.get("correr_informe") == "on"

            # Ejecutar Raíces Múltiples midiendo tiempo
            t0 = time.time()
            # 👇 aquí usamos **data**, que SÍ incluye 'Modo'
            resultado, tabla, img = raices_multiples(**data)
            tiempo_rm = time.time() - t0

            if resultado == "Error":
                return render(
                    request,
                    'error.html',
                    {'resultado': resultado, 'tabla': tabla, 'imagen': img}
                )

            # Tipo de error según el modo
            if modo == "cs":
                tipo_error = "Cifras significativas"
            elif modo == "cr":
                tipo_error = "Error relativo"
            else:
                tipo_error = "Error absoluto"

            # Generar informe solo si marcaron el checkbox
            if correr_informe:
                try:
                    df_tabla = pd.read_html(StringIO(tabla))[0]

                    # Tomamos los valores sin importar si el campo está en mayúsculas o minúsculas
                    fun_str  = data.get("fun")  or data.get("Fun")
                    df_str   = data.get("df")   or data.get("Df")
                    ddf_str  = data.get("ddf")  or data.get("Ddf") or data.get("DDF")
                    x0_val   = data.get("x0")   or data.get("X0")
                    tol_val  = data.get("tol")  or data.get("Tol")
                    niter_val = data.get("niter") or data.get("Niter") or data.get("NIter")

                    informe = generar_informe_raices_multiples(
                        df_tabla,
                        fun_str=fun_str,
                        df_str=df_str,
                        ddf_str=ddf_str,
                        x0=x0_val,
                        tol=tol_val,
                        niter=niter_val,
                        tipo_error=tipo_error,
                        tiempo=tiempo_rm,
                    )
                except Exception as e:
                    print("Error generando informe de Raíces Múltiples:", e)
                    informe = None

            return render(
                request,
                'result_raices_multiples.html',
                {
                    'resultado': resultado,
                    'tabla': tabla,
                    'imagen': img,
                    'informe': informe,
                }
            )
    else:
        form = RaicesMultiplesForm()

    return render(request, 'raices_multiples.html', {'form': form})


def metodo_punto_fijo(request):
    if request.method == 'POST':
        form = PuntoFijoForm(request.POST)
        informe = None

        if form.is_valid():
            data = form.cleaned_data

            # Modo de error
            modo = data.get('Modo')
            data_sin_modo = data.copy()
            data_sin_modo.pop('Modo', None)

            # Checkbox del template
            correr_informe = request.POST.get("correr_informe") == "on"

            # Ejecutar Punto Fijo midiendo tiempo
            if modo == 'cs':
                t0 = time.time()
                resultado, tabla, imagen = Puntofijo.punto_fijoCS(**data_sin_modo)
                tiempo_pf = time.time() - t0
            else:
                t0 = time.time()
                resultado, tabla, imagen = Puntofijo.punto_fijo(**data_sin_modo)
                tiempo_pf = time.time() - t0

            # Si hubo error en el método
            if resultado == "Error":
                return render(
                    request,
                    'error.html',
                    {'resultado': resultado, 'tabla': tabla, 'imagen': imagen}
                )

            # Tipo de error según el modo
            if modo == "cs":
                tipo_error = "Cifras significativas"
            elif modo == "cr":
                tipo_error = "Error relativo"
            else:
                tipo_error = "Error absoluto"

            # Si el usuario pidió informe -> lo generamos
            if correr_informe:
                try:
                    df_tabla = pd.read_html(StringIO(tabla))[0]

                    # Tomamos los parámetros de la misma data que se usó en el método
                    fun_str = (
                        data_sin_modo.get('Fun')
                        or data_sin_modo.get('fun')
                        or ""
                    )
                    g_str = (
                        data_sin_modo.get('g')
                        or data_sin_modo.get('G')
                        or ""
                    )

                    x0_val = (
                        data_sin_modo.get('X0')
                        or data_sin_modo.get('x0')
                        or data_sin_modo.get('a')  # por si lo llamaste 'a'
                    )
                    tol_val = (
                        data_sin_modo.get('Tol')
                        or data_sin_modo.get('tol')
                    )
                    niter_val = (
                        data_sin_modo.get('Niter')
                        or data_sin_modo.get('niter')
                    )

                    informe = generar_informe_punto_fijo(
                        df_tabla,
                        fun_str=fun_str,
                        g_str=g_str,
                        x0=x0_val,
                        tol=tol_val,
                        niter=niter_val,
                        tipo_error=tipo_error,
                        tiempo=tiempo_pf
                    )

                except Exception as e:
                    print("Error generando informe de Punto Fijo:", e)
                    informe = None

            return render(
                request,
                'result_pf.html',
                {
                    'resultado': resultado,
                    'tabla': tabla,
                    'imagen': imagen,
                    'informe': informe,
                }
            )
    else:
        form = PuntoFijoForm()

    return render(request, 'punto_fijo.html', {'form': form})


def obtener_metricas_error_desde_tabla(tabla_html):
    """
    Recibe la tabla HTML de un método (Bisección, Newton, etc.)
    y devuelve un diccionario con:
      - n_iter
      - x_final
      - fx_final
      - error_modo   (última columna de error si existe)
      - error_dec    (|x_n - x_{n-1}|)
      - error_rel    (|x_n - x_{n-1}| / |x_n|)
      - error_fx     (|f(x_n)|)
    """
    df = pd.read_html(StringIO(tabla_html))[0]
    cols = [str(c) for c in df.columns]

    # ---- detectar columna de x ----
    posibles_x = ['Xn', 'xn', 'Xi', 'xi', 'x', 'Xm', 'xm']
    col_x = None
    for c in cols:
        if c in posibles_x:
            col_x = c
            break
    if col_x is None and len(cols) >= 2:
        col_x = cols[1]   # fallback: segunda columna

    # ---- detectar columna de f(x) ----
    posibles_fx = ['f(Xn)', 'f(xn)', 'f(Xi)', 'f(xi)', 'f(x)', 'F(Xi)', 'F(xi)', 'F(x)', 'F(Xm)', 'f(Xm)']
    col_fx = None
    for c in cols:
        if c in posibles_fx:
            col_fx = c
            break
    if col_fx is None and len(cols) >= 3:
        col_fx = cols[2]  # fallback: tercera columna

    # ---- detectar columna de error ----
    posibles_err = ['Error', 'error', 'E', 'err', 'Error relativo', 'Error relativo ', 'Error relativo_xnm1']
    col_err = None
    for c in cols:
        if c in posibles_err:
            col_err = c
            break

    n_iter = len(df)
    x_final = df[col_x].iloc[-1]
    fx_final = df[col_fx].iloc[-1]

    if n_iter >= 2:
        x_prev = df[col_x].iloc[-2]
        delta = x_final - x_prev
    else:
        x_prev = x_final
        delta = 0.0

    # error del modo (lo que haya en la tabla, si existe)
    if col_err is not None:
        error_modo = df[col_err].iloc[-1]
    else:
        error_modo = abs(delta)

    error_dec = abs(delta)
    error_rel = abs(delta / x_final) if x_final != 0 else 0.0
    error_fx = abs(fx_final)

    return {
        "n_iter": int(n_iter),
        "x_final": float(x_final),
        "fx_final": float(fx_final),
        "error_modo": float(error_modo),
        "error_dec": float(error_dec),
        "error_rel": float(error_rel),
        "error_fx": float(error_fx),
    }


def todos_view(request):
    """
    Genera el informe de ejecución y comparación de TODOS los métodos
    del capítulo 1, ante un error específico (modo).
    - Si el usuario marca "generar_informe": genera un PDF comparativo.
    - Si NO lo marca: muestra los resultados en una página HTML.
    """
    if request.method == 'POST':
        form = todosForm(request.POST)

        if form.is_valid():
            data = form.cleaned_data

            # 🔹 Leer el checkbox del template
            generar_informe = request.POST.get("generar_informe") == "on"

            modo = data.pop('Modo', None)
            resultados = {}

            a = data.get("a")
            b = data.get("b")
            tol = data.get("tol")
            niter = data.get("niter")
            fun = data.get("fun")
            df = data.get("df")
            ddf = data.get("ddf")
            g = data.get("g")

            # 🔹 Descripción del tipo de error según el modo
            if modo == 'cs':
                tipo_error = "cifras significativas"
            elif modo == 'cr':
                tipo_error = "error relativo"
            elif modo == 'ca':
                tipo_error = "error absoluto"
            else:
                tipo_error = "condición de parada"

            # 🔹 Ejecutamos todos los métodos con el mismo conjunto de parámetros
            if modo == 'cs':
                resultados['Punto Fijo'] = {'mensaje': '', 'tabla': '', 'imagen': ''}
                resultados['Secante'] = {'mensaje': '', 'tabla': '', 'imagen': ''}
                resultados['Regla Falsa'] = {'mensaje': '', 'tabla': '', 'imagen': ''}
                resultados['Newton'] = {'mensaje': '', 'tabla': '', 'imagen': ''}
                resultados['Biseccion'] = {'mensaje': '', 'tabla': '', 'imagen': ''}
                resultados['Raices Multiples'] = {'mensaje': '', 'tabla': '', 'imagen': ''}

                resultados['Punto Fijo']['mensaje'], resultados['Punto Fijo']['tabla'], resultados['Punto Fijo']['imagen'] = Puntofijo.punto_fijoCS(a, tol, niter, fun, g)
                resultados['Secante']['mensaje'], resultados['Secante']['tabla'], resultados['Secante']['imagen'] = Secante.secanteCS(a, b, tol, niter, fun)
                resultados['Regla Falsa']['mensaje'], resultados['Regla Falsa']['tabla'], resultados['Regla Falsa']['imagen'] = Reglafalsa.reglafalsaCS(a, b, tol, niter, fun)
                resultados['Newton']['mensaje'], resultados['Newton']['tabla'], resultados['Newton']['imagen'] = Newton.metodo_newtonCS(a, tol, niter, fun, df)
                resultados['Biseccion']['mensaje'], resultados['Biseccion']['tabla'], resultados['Biseccion']['imagen'] = biseccion(a, b, tol, niter, fun, modo)
                resultados['Raices Multiples']['mensaje'], resultados['Raices Multiples']['tabla'], resultados['Raices Multiples']['imagen'] = raices_multiples(a, tol, niter, fun, df, ddf, modo)
            else:
                resultados['Punto Fijo'] = {'mensaje': '', 'tabla': '', 'imagen': ''}
                resultados['Secante'] = {'mensaje': '', 'tabla': '', 'imagen': ''}
                resultados['Regla Falsa'] = {'mensaje': '', 'tabla': '', 'imagen': ''}
                resultados['Newton'] = {'mensaje': '', 'tabla': '', 'imagen': ''}
                resultados['Biseccion'] = {'mensaje': '', 'tabla': '', 'imagen': ''}
                resultados['Raices Multiples'] = {'mensaje': '', 'tabla': '', 'imagen': ''}

                resultados['Punto Fijo']['mensaje'], resultados['Punto Fijo']['tabla'], resultados['Punto Fijo']['imagen'] = Puntofijo.punto_fijo(a, tol, niter, fun, g)
                resultados['Secante']['mensaje'], resultados['Secante']['tabla'], resultados['Secante']['imagen'] = Secante.secanteDC(a, b, tol, niter, fun)
                resultados['Regla Falsa']['mensaje'], resultados['Regla Falsa']['tabla'], resultados['Regla Falsa']['imagen'] = Reglafalsa.reglafalsaDC(a, b, tol, niter, fun)
                resultados['Newton']['mensaje'], resultados['Newton']['tabla'], resultados['Newton']['imagen'] = Newton.metodo_newton(a, tol, niter, fun, df)
                resultados['Biseccion']['mensaje'], resultados['Biseccion']['tabla'], resultados['Biseccion']['imagen'] = biseccion(a, b, tol, niter, fun, modo)
                resultados['Raices Multiples']['mensaje'], resultados['Raices Multiples']['tabla'], resultados['Raices Multiples']['imagen'] = raices_multiples(a, tol, niter, fun, df, ddf, modo)

            # 🔹 Si el usuario NO quiere informe → mostramos todo en HTML
            if not generar_informe:
                return render(
                    request,
                    'result_todos.html',
                    {
                        'resultados': resultados,
                        'tipo_error': tipo_error,
                        'a': a,
                        'b': b,
                        'tol': tol,
                        'niter': niter,
                        'fun': fun,
                    }
                )

            # 🔹 Si SÍ quiere informe → generamos el PDF (tu lógica de antes)
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            elements = []

            descripcion = (
                f"Informe de ejecución y comparación de los métodos de búsqueda de raíces "
                f"para la función f(x) = {fun}, en el intervalo [{a}, {b}], con tolerancia {tol}, "
                f"un máximo de {niter} iteraciones y usando el criterio de {tipo_error}."
            )
            elements.append(Paragraph(descripcion, style=None))
            elements.append(Spacer(1, 12))

            from io import StringIO
            for metodo, datos in resultados.items():
                elements.append(Paragraph(f"<b>{metodo}</b>", style=None))
                elements.append(Paragraph(str(datos['mensaje']), style=None))

                try:
                    tabla_df = pd.read_html(StringIO(datos['tabla']))[0]
                    table_data = [list(tabla_df.columns)] + tabla_df.values.tolist()
                    elements.append(Table(table_data, style=[('GRID', (0, 0), (-1, -1), 1, colors.black)]))
                except Exception as e:
                    elements.append(Paragraph("No se pudo leer la tabla de este método.", style=None))

                if datos['imagen']:
                    try:
                        image_data = base64.b64decode(datos['imagen'].split(',')[1])
                        img = Image(BytesIO(image_data), width=400, height=300)
                        elements.append(img)
                    except Exception:
                        elements.append(Paragraph("No se pudo cargar la gráfica del método.", style=None))

                elements.append(Spacer(1, 12))

            resumen_data = [['Método', 'Raíz aproximada', 'Iteraciones', 'Error final (modo)']]
            comparacion_errores_data = [['Método', 'Error decimales', 'Error relativo', '|f(x_final)|']]

            for metodo, datos in resultados.items():
                try:
                    metricas = obtener_metricas_error_desde_tabla(datos['tabla'])
                    raiz = datos["mensaje"]          # texto que ya muestras en HTML
                    iteraciones = metricas["n_iter"]
                    error_modo = metricas["error_modo"]

                    resumen_data.append([
                        metodo,
                        str(raiz),
                        str(iteraciones),
                        f"{error_modo:.2e}",
                    ])

                    comparacion_errores_data.append([
                        metodo,
                        f"{metricas['error_dec']:.2e}",
                        f"{metricas['error_rel']:.2e}",
                        f"{metricas['error_fx']:.2e}",
                    ])

                except Exception as e:
                    resumen_data.append([metodo, 'Error', 'Error', 'Error'])
                    comparacion_errores_data.append([metodo, 'Error', 'Error', 'Error'])

            mejor_metodo = None
            menor_iteraciones = float('inf')

            for fila in resumen_data[1:]:
                metodo, raiz, iteraciones, err_final = fila
                try:
                    iteraciones_int = int(iteraciones)
                    if iteraciones_int < menor_iteraciones and raiz != 'Error':
                        menor_iteraciones = iteraciones_int
                        mejor_metodo = metodo
                except:
                    continue

            elements.append(Spacer(1, 24))
            if mejor_metodo:
                mensaje_final = (
                    f"El mejor método según el menor número de iteraciones, "
                    f"bajo el criterio de {tipo_error}, es: <b>{mejor_metodo}</b> "
                    f"con {menor_iteraciones} iteraciones."
                )
            else:
                mensaje_final = "No se pudo determinar un mejor método por errores en los datos."

            elements.append(Spacer(1, 24))
            elements.append(Paragraph("<b>Resumen de raíces, iteraciones y error final</b>", style=None))
            elements.append(Table(resumen_data, style=[('GRID', (0, 0), (-1, -1), 1, colors.black)]))

            elements.append(Spacer(1, 12))
            elements.append(Paragraph("<b>Comparación global de errores</b>", style=None))
            elements.append(Table(comparacion_errores_data, style=[('GRID', (0, 0), (-1, -1), 1, colors.black)]))

            elements.append(Spacer(1, 12))
            elements.append(Paragraph(mensaje_final, style=None))


            doc.build(elements)
            buffer.seek(0)
            return FileResponse(buffer, as_attachment=True, filename="resultados_metodos_cap1.pdf")

    else:
        form = todosForm()
    return render(request, 'todos.html', {'form': form})
