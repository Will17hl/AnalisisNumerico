
from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings

from main import views as mainViews
from ec_nl import views as ecViews
from ec_sis import views as ecSViews
from interp import views as interpViews

urlpatterns = [
    path('', mainViews.index, name="Home"),
    path('metodos/', mainViews.metodos, name="Metodos"),
    path('acercade/', mainViews.acerca_de, name="Acerca de"),
    path('biseccion/', ecViews.biseccion_view, name="Biseccion"),
    path('reglafalsa/', ecViews.regla_falsa, name="Regla Falsa"),
    path('newton/', ecViews.newton, name="Newton"),
    path('secante/', ecViews.secante, name="Secante"),
    path('raicesmultiples/', ecViews.raices_multiples_view, name="Raices Multiples"),
    path('puntofijo/', ecViews.metodo_punto_fijo, name="Punto Fijo"),
    path('todos/', ecViews.todos_view, name="Todos"),
    path('jacobi/', ecSViews.jacobi_view, name="Jacobi"),
    path('gausseidel/', ecSViews.gausseidel_view, name='Gausseidel'),
    path('SOR/', ecSViews.SOR_view),
    path('sor/', ecSViews.SOR_view, name='SOR'),
    path('todosNL/', ecSViews.todos_view, name="Todos"),
    path('vander/', interpViews.vander_view, name='Vandermonde'),
    path('lagrange/', interpViews.lagrange_view, name='Lagrange'),
    path('newtonint/', interpViews.newtonint_view, name='NewtonInt'),
    path('splinelin/', interpViews.spline_lin_view, name='SplineLin'),
    path('splinecub/', interpViews.spline_cub_view, name='SplineCub'),
    path('todosInter/', interpViews.todo_view, name="Todos"),

    path('admin/', admin.site.urls),
]


urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)