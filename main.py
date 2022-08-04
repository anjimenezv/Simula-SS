#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from sympy import Eq, dsolve, init_printing, symbols, Function
from scipy.integrate import odeint
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager
from kivy.uix.widget import Widget
from kivy.uix.accordion import Accordion
from kivy.core.window import Window
from kivy.metrics import dp
from kivymd.uix.slider import MDSlider
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.screen import MDScreen
from garden.matplotlib import FigureCanvasKivyAgg

__author__ = "Anfejive"
__copyright__ = "Copyright 2022, SimulaSS"
__credits__ = ["Anfejive"]
__license__ = "GPL"
__version__ = "0.2.0"
__maintainer__ = "anfejive"
__email__ = "anjimenez28@uan.edu.co"
__status__ = "Development"


class IndexScreen(MDScreen):
    def __draw_shadow__(self, origin, end, context=None):
        pass


class AcercaWindow(MDScreen):
    def __draw_shadow__(self, origin, end, context=None):
        pass


class MenuWindow(MDScreen):
    def __draw_shadow__(self, origin, end, context=None):
        pass


class SmalgyecuScreen(MDScreen, Accordion):
    def __draw_shadow__(self, origin, end, context=None):
        pass


class ExSenales(MDScreen, Accordion):

    def __init__(self, **kw):
        super().__init__(**kw)
        self.app = MDApp.get_running_app()

    def __draw_shadow__(self, origin, end, context=None):
        pass

    def ejercicio_demostrativo_signal_cont(self, var_signal, var_signala, var_signalb):
        all_widgets = [self.validate_number(var_signal), self.validate_number(var_signala),
                       self.validate_number(var_signalb)]
        if all(all_widgets):
            t = np.linspace(-0.02, 0.05, 1000)
            var_signal = int(var_signal.text)
            var_signala = int(var_signala.text)
            var_signalb = int(var_signalb.text)
            fig, ax = plt.subplots()
            plt.plot(t, var_signal * np.sin(var_signala * np.pi * var_signalb * t))
            ax.set_xlabel("t", color='white', family='Cambria', size=15)
            ax.set_ylabel("x(t)", color='white', family='Cambria', size=15)
            plt.title('Señal en tiempo continuo x(t)')
            plt.xlim([-0.02, 0.05])
            plt.grid('major')
            canvasK = FigureCanvasKivyAgg(fig, pos_hint={'center_y': .5})
            self.app.add_update_graphic(self, canvasK, 9, 'ejercicio_demostrativo_signal_discont')

    def ejercicio_demostrativo_signal_disc(self, var_muestra):
        all_widgets = [self.validate_number(var_muestra)]
        if all(all_widgets):
            w = 2 * np.pi / 12
            n0 = 0
            ni = np.arange(n0, n0 + int(var_muestra.text), 1)
            senal = np.sin(w * ni)
            np.set_printoptions(precision=4)
            fig, ax = plt.subplots()
            plt.stem(ni, senal)
            ax.set_xlabel("n (muestra)", color='white', family='Cambria', size=15)
            ax.set_ylabel("señal x[n]", color='white', family='Cambria', size=15)
            plt.title('Señal en tiempo discreto')
            plt.grid(True)
            canvasK = FigureCanvasKivyAgg(fig, pos_hint={'center_y': .5})
            self.app.add_update_graphic(self, canvasK, 17, 'ejercicio_demostrativo_signal_discont')

    def ejercicio_demostrativo_senal_par(self, var_m):
        all_widgets = [self.validate_number(var_m)]
        if all(all_widgets):
            T = 2 * np.pi
            f = 1 / T
            w = 2 * np.pi * f
            var_m = int(var_m.text)
            t0 = -(var_m / 2) * T
            tramos = 32
            tn = -t0
            muestras = var_m * tramos + 1
            ti = np.linspace(t0, tn, muestras)
            dt = (tn - t0) / (tramos * var_m)
            senal = np.cos(w * ti)
            desde = -T / 2
            hasta = desde + T + dt
            tperiodo = np.arange(desde, hasta, dt)
            periodo = np.cos(w * tperiodo)
            fig, ax = plt.subplots()
            plt.fill_between(tperiodo, 0, periodo,
                             color='lightblue')
            plt.axvline(x=0, color='blue')
            plt.plot(ti, senal)
            ax.set_xlabel("tiempo", color='white', family='Cambria', size=15)
            ax.set_ylabel("señal x(t)", color='white', family='Cambria', size=15)
            plt.title("Señal PAR")
            plt.grid(True)
            canvasK = FigureCanvasKivyAgg(fig, size_hint_y=.8, pos_hint={'center_y': .5})
            self.app.add_update_graphic(self, canvasK, 9, 'ejercicio_demostrativo_senal_par')

    def ejercicio_demostrativo_senal_impar(self, var_m):
        T = 2 * np.pi
        f = 1 / T
        w = 2 * np.pi * f
        var_m = int(var_m.text)
        t0 = -(var_m / 2) * T
        tramos = 32
        tn = -t0
        muestras = var_m * tramos + 1
        ti = np.linspace(t0, tn, muestras)
        dt = (tn - t0) / (tramos * var_m)
        senal = np.sin(w * ti)
        desde = -T / 2
        hasta = desde + T + dt
        tperiodo = np.arange(desde, hasta, dt)
        periodo = np.sin(w * tperiodo)
        fig, ax = plt.subplots()
        plt.plot(ti, senal)
        plt.fill_between(tperiodo, 0, periodo,
                         color='purple')
        plt.axvline(x=0, color='red')
        ax.set_xlabel("tiempo", color='white', family='Cambria', size=15)
        ax.set_ylabel("señal x(t)", color='white', family='Cambria', size=15)
        plt.title("Señal IMPAR")
        plt.grid(True)
        canvasK = FigureCanvasKivyAgg(fig, size_hint_y=.8, pos_hint={'center_y': .5})
        self.app.add_update_graphic(self, canvasK, 16, 'ejercicio_demostrativo_senal_par')

    def ejercicio_demostrativo_periodica_continua(self, var_m):
        T = 2 * np.pi
        f = 1 / T
        w = 2 * np.pi * f
        t0 = 0
        var_m = int(var_m.text)
        muestras = 51
        desde = T / 4
        tn = var_m * T
        ti = np.linspace(t0, tn, muestras)
        dt = (tn - t0) / (muestras - 1)
        senal = np.cos(w * ti)
        hasta = desde + T + dt
        tperiodo = np.arange(desde, hasta, dt)
        periodo = np.cos(w * tperiodo)
        fig, ax = plt.subplots()
        plt.plot(ti, senal)
        plt.axhline(0, color='gray')
        plt.axvline(0, color='gray')
        plt.fill_between(tperiodo, 0, periodo, facecolor='blue')
        ax.set_xlabel("t", color='white', family='Cambria', size=15)
        ax.set_ylabel("señal x(t)", color='white', family='Cambria', size=15)
        plt.title("Señal periodica continua")
        plt.grid(True)
        canvasK = FigureCanvasKivyAgg(fig, size_hint_y=.8, pos_hint={'center_y': .5})
        self.app.add_update_graphic(self, canvasK, 9, 'ejercicio_demostrativo_periodica_discrecontinua')

    def ejercicio_demostrativo_periodica_discreta(self, var_m):
        N = 8
        w = 2 * np.pi / N
        n0 = 0
        var_m = int(var_m.text)
        muestras = var_m * N + 1
        ni = np.arange(n0, n0 + muestras, 1)
        senal = np.cos(w * ni)
        fig, ax = plt.subplots()
        plt.stem(ni, senal)
        ax.set_xlabel("n \"muestras\"", color='white', family='Cambria', size=15)
        ax.set_ylabel("x[n]", color='white', family='Cambria', size=15)
        plt.title("Señal periodica discreta")
        plt.grid(True)
        canvasK = FigureCanvasKivyAgg(fig, size_hint_y=.8, pos_hint={'center_y': .5})
        self.app.add_update_graphic(self, canvasK, 16, 'ejercicio_demostrativo_periodica_discrecontinua')

    @staticmethod
    def validate_number(widget):
        if not widget.text:
            widget.error = True
            widget.helper_text = 'El campo es requerido'
            widget.helper_text_mode = 'on_error'
        else:
            try:
                int(widget.text)
            except ValueError:
                widget.error = True
                widget.helper_text = 'Solo se permite valores numericos'
                widget.helper_text_mode = 'on_error'
                return False
            else:
                widget.helper_text = ''
                widget.helper_text_mode = 'none'
                return True


class ExSistemas(MDScreen, Accordion):
    def __draw_shadow__(self, origin, end, context=None):
        pass


class ExopMatricial(MDScreen, Accordion):
    def __draw_shadow__(self, origin, end, context=None):
        pass

    def ejercicio_demostrativo(self, widget_id):
        if widget_id == 'ejercicio_demostrativo_matriz':
            A = np.array([[1, 8, 4], [6, 7, 2]])
            B = np.array([[1.6, 8, 5], [-2, 6, 1]])
            C = np.array([[3, 2, 1], [8, 7, 6]], dtype=complex)
            self.ids[widget_id].text = f'Matriz de números enteros:\n[b]{A}[/b]' \
                                       f'\nMatriz de números flotantes:\n[b]{B}[/b]' \
                                       f'\nMatriz de números complejos:\n[b]{C}[/b]'
        elif widget_id == 'ejercicio_demostrativo_igualdad_matriz':
            A = np.array([[4, 8, 3], [10, 9, 5]])
            B = np.array([[4, 1, 3], [7.5, 9, 5]])
            C = np.array([[0.1, 8, 3], [82, 9, 5]])
            self.ids[widget_id].text = f'Matriz A:\n[b]{A}[/b]' \
                                       f'\nMatriz B:\n[b]{B}[/b]' \
                                       f'\nMatriz C:\n[b]{C}[/b]' \
                                       f'\nCompruebe la igualdad de las matrices A y B:\n[b]{np.equal(A, B)}[/b]' \
                                       f'\nCompruebe la igualdad de las matrices A y C:\n[b]{np.equal(A, C)}[/b]'
        elif widget_id == 'ejercicio_demostrativo_operacion_matriz':
            A = np.array([[4, 8, 3], [10, 9, 5]])
            B = np.array([[4, 1, 3], [7.5, 9, 5]])
            self.ids[widget_id].text = f'Matriz A:\n[b]{A}[/b]' \
                                       f'\nMatriz B:\n[b]{B}[/b]' \
                                       f'\nSuma de matriz:\n[b]{A + B}[/b]' \
                                       f'\nResta de matriz:\n[b]{A - B}[/b]'
        elif widget_id == 'ejercicio_demostrativo_multiplicacion_matriz':
            A = np.array([[4, 8, 3], [10, 9, 5]])
            B = np.array([[4, 1], [5, 6], [4, 22]])
            self.ids[widget_id].text = f'Matriz A:\n[b]{A}[/b]' \
                                       f'\nMatriz B:\n[b]{B}[/b]' \
                                       f'\nMatriz multiplicación AB (2x2):\n[b]{np.matmul(A, B)}[/b]' \
                                       f'\nMatriz multiplicación BA (3x3):\n[b]{np.matmul(B, A)}[/b]'


class ApliEDOScreen(MDScreen, Accordion):
    def __draw_shadow__(self, origin, end, context=None):
        pass

    def __init__(self, **kw):
        super().__init__(**kw)
        self.app = MDApp.get_running_app()

    def ejercicio_demostrativo(self, widget_id):
        if widget_id == 'ejercicio_demostrativo_edo':
            init_printing()
            x = symbols('x')
            y = Function('y')
            ed = Eq(y(x).diff() + 3 * x ** 2 * y(x), 6 * x ** 2)
            c = dsolve(ed, y(x))
            self.ids[widget_id].text = f'[b]Resolver:[/b] dy/dx+3x^2y=6x^2' \
                                       f'\n[b]Solución:[/b] {c}'
        elif widget_id == 'ejercicio_demostrativo_edo_2':
            init_printing()
            x = symbols('x')
            y = Function('x')
            ed = 4 * y(x).diff(x, 2) + 12 * y(x).diff() + 9 * y(x)
            c = dsolve(ed, y(x))
            self.ids[widget_id].text = f'[b]Resolver:[/b] 4y^\'\'+12y^\'+9y=0' \
                                       f'\n[b]Solución:[/b] {c}'

    def ejercicio_demostrativo_circuito(self, var_r, var_l, var_c):
        all_widgets = [self.validate_number(var_r), self.validate_number(var_l), self.validate_number(var_c)]
        if all(all_widgets):
            var_r = int(var_r.text)
            var_l = int(var_l.text)
            var_c = int(var_c.text)
            I0 = 1
            Ip0 = 0
            var_t = np.linspace(0, 10, 1000)
            var_i = np.zeros(len(var_t))
            Ip = np.zeros(len(var_t))
            var_i[0] = I0
            Ip[0] = Ip0
            for i in range(1, len(var_t)):
                var_i[i] = var_i[i - 1] + Ip[i - 1] * (var_t[i] - var_t[i - 1])
                Ip[i] = Ip[i - 1] + ((-var_r / var_l) * Ip[i - 1] + (-1 / var_c * var_l) * var_i[i - 1]) * (
                        var_t[i] - var_t[i - 1])
            fig, ax = plt.subplots()

            plt.plot(var_t, var_i)
            ax.set_xlabel("Tiempo", color='white', family='Cambria', size=15)
            ax.set_ylabel("Corriente", color='white', family='Cambria', size=15)
            plt.title('Corriente del circuito RLC')
            plt.grid(True)
            canvasK = FigureCanvasKivyAgg(fig, pos_hint={'center_y': .5})
            self.app.add_update_graphic(self, canvasK, 16, 'ejercicio_demostrativo_circuito')
            var_n = 10000
            var_r = int(var_r)
            var_l = int(var_l)
            var_c = int(var_c)
            var_io = 0
            var_ip0 = 0
            var_t = np.linspace(0, 10, var_n)
            omega = 1 / (var_l * var_c) ** (1 / 2)
            var_v = np.sin(omega * var_t)
            var_i = np.zeros(len(var_t))
            var_ip = np.zeros(len(var_t))
            var_vp = np.zeros(len(var_t))
            var_q = np.zeros(len(var_t))
            var_i[0] = var_io
            var_ip[0] = var_ip0
            var_q[0] = 0
            for i in range(1, len(var_t)):
                var_vp[i] = (var_v[i] - var_v[i - 1]) / (var_t[i] - var_t[i - 1])
                var_i[i] = var_i[i - 1] + var_ip[i - 1] * (var_t[i] - var_t[i - 1])
                var_ip[i] = var_ip[i - 1] + (
                        (-var_r / var_l) * var_ip[i - 1] +
                        (-1 / var_c * var_l) * var_i[i - 1] + var_vp[i - 1] / var_l) * (var_t[i] - var_t[i - 1])
                var_q[i] = var_q[i - 1] + var_i[i - 1] * (var_t[i] - var_t[i - 1])
            fig, ax = plt.subplots()
            plt.plot(var_t, var_i * 1000)
            ax.set_xlabel("Tiempo", color='white', family='Cambria', size=15)
            ax.set_ylabel("Corriente [mA]", color='white', family='Cambria', size=15)
            plt.title('Corrente del circuito RLC')
            plt.grid(True)
            canvasK = FigureCanvasKivyAgg(fig, pos_hint={'center_y': .5})
            self.app.add_update_graphic(self, canvasK, 17, 'ejercicio_demostrativo_circuito')
            fig, ax = plt.subplots()
            plt.plot(var_t, var_q / var_c)
            ax.set_xlabel("Tiempo", color='white', family='Cambria', size=15)
            ax.set_ylabel("Voltaje", color='white', family='Cambria', size=15)
            plt.title('Voltaje en el capacitor')
            plt.grid(True)
            canvasK = FigureCanvasKivyAgg(fig, pos_hint={'center_y': .5})
            self.app.add_update_graphic(self, canvasK, 18, 'ejercicio_demostrativo_circuito')
            fig, ax = plt.subplots()
            plt.plot(var_t, var_v)
            ax.set_xlabel("Tiempo", color='white', family='Cambria', size=15)
            ax.set_ylabel("Voltaje", color='white', family='Cambria', size=15)
            plt.title('Input V')
            plt.grid(True)
            canvasK = FigureCanvasKivyAgg(fig, pos_hint={'center_y': .5})
            self.app.add_update_graphic(self, canvasK, 19, 'ejercicio_demostrativo_circuito')

    @staticmethod
    def validate_number(widget):
        if not widget.text:
            widget.error = True
            widget.helper_text = 'El campo es requerido'
            widget.helper_text_mode = 'on_error'
        else:
            try:
                int(widget.text)
            except ValueError:
                widget.error = True
                widget.helper_text = 'Solo se permite valores numericos'
                widget.helper_text_mode = 'on_error'
                return False
            else:
                widget.helper_text = ''
                widget.helper_text_mode = 'none'
                return True

    def ejercicio_demostrativo_batman(self):
        xs = np.arange(-7.25, 7.25, 0.01)
        ys = np.arange(-5, 5, 0.01)
        x, y = np.meshgrid(xs, ys)
        ec1 = ((x / 7) ** 2 * np.sqrt(abs(abs(x) - 3) / (abs(x) - 3)) + (y / 3) ** 2 * np.sqrt(
            abs(y + 3 / 7 * np.sqrt(33)) / (y + 3 / 7 * np.sqrt(33))) - 1)
        ec2 = (abs(x / 2) - ((3 * np.sqrt(33) - 7) / 112) * x ** 2 - 3 + np.sqrt(1 - (abs(abs(x) - 2) - 1) ** 2) - y)
        ec3 = (9 * np.sqrt(abs((abs(x) - 1) * (abs(x) - .75)) / ((1 - abs(x)) * (abs(x) - .75))) - 8 * abs(x) - y)
        ec4 = (3 * abs(x) + .75 * np.sqrt(abs((abs(x) - .75) * (abs(x) - .5)) / ((.75 - abs(x)) * (abs(x) - .5))) - y)
        ec5 = (2.25 * np.sqrt(abs((x - .5) * (x + .5)) / ((.5 - x) * (.5 + x))) - y)
        ec6 = (6 * np.sqrt(10) / 7 + (1.5 - .5 * abs(x)) * np.sqrt(abs(abs(x) - 1) / (abs(x) - 1)) - (
                6 * np.sqrt(10) / 14) * np.sqrt(4 - (abs(x) - 1) ** 2) - y)
        fig, ax = plt.subplots()
        for f, c in [(ec1, "Yellow"), (ec2, "Yellow"), (ec3, "Yellow"),
                     (ec4, "Yellow"), (ec5, "Yellow"), (ec6, "Yellow")]:
            plt.contour(x, y, f, [0], colors=c)
        ax.set_xlabel("Eje X", color='white', family='Cambria', size=15)
        ax.set_ylabel("Eje Y", color='white', family='Cambria', size=15)
        plt.title('Grafica de Batman')
        plt.grid(True)
        canvasK = FigureCanvasKivyAgg(fig, pos_hint={'center_y': .5})
        self.app.add_update_graphic(self, canvasK, 2, 'ejercicio_demostrativo_batman')

    def ejercicio_demostrativo_masa(self, var_c, var_k, var_m, var_f):
        all_widgets = [self.validate_number(var_c), self.validate_number(var_k), self.validate_number(var_m),
                       self.validate_number(var_f)]
        if all(all_widgets):
            tstart = 0
            tstop = 60
            increment = 0.1
            x_init = [0, 0]
            var_t = np.arange(tstart, tstop + 1, increment)
            var_c = int(var_c.text)
            var_k = int(var_k.text)
            var_m = int(var_m.text)
            var_f = int(var_f.text)
            x = odeint(lambda var_x, _: [var_x[1], (var_f - var_c * var_x[1] - var_k * var_x[0]) / var_m], x_init,
                       var_t)
            x1 = x[:, 0]
            x2 = x[:, 1]
            fig, ax = plt.subplots()
            plt.plot(var_t, x1)
            plt.plot(var_t, x2)
            ax.set_xlabel("t", color='white', family='Cambria', size=15)
            ax.set_ylabel("x(t)", color='white', family='Cambria', size=15)
            ax.legend(["x1", "x2"])
            plt.title('Simulación del sistema masa resorte-amortiguación')
            plt.grid(True)
            canvasK = FigureCanvasKivyAgg(fig, pos_hint={'center_y': .5})
            self.app.add_update_graphic(self, canvasK, 23, 'ejercicio_demostrativo_masa')

    @staticmethod
    def validate_number(widget):
        if not widget.text:
            widget.error = True
            widget.helper_text = 'El campo es requerido'
            widget.helper_text_mode = 'on_error'
        else:
            try:
                int(widget.text)
            except ValueError:
                widget.error = True
                widget.helper_text = 'Solo se permite valores numericos'
                widget.helper_text_mode = 'on_error'
                return False
            else:
                widget.helper_text = ''
                widget.helper_text_mode = 'none'
                return True


class SenalesySistemasScreen(MDScreen, Accordion):
    def __draw_shadow__(self, origin, end, context=None):
        pass


class FormulacionWindow(MDScreen, Accordion):
    def __draw_shadow__(self, origin, end, context=None):
        pass


class CollapsingBoxLayout(MDBoxLayout):
    def __draw_shadow__(self, origin, end, context=None):
        pass

    def __init__(self, **kwargs):
        super(CollapsingBoxLayout, self).__init__(**kwargs)
        self._trigger_update_size = Clock.create_trigger(self._update_size)

    def on_children(self, *_):
        for c in self.children:
            c.bind(size=self._trigger_update_size)
        self._trigger_update_size()

    def _update_size(self, *_):
        if self.size_hint_y is None:
            self.height = max(c.height for c in self.children) if self.children else 0
        if self.size_hint_x is None:
            self.width = max(c.width for c in self.children) if self.children else 0


class SubmenucontrolWindow(MDScreen, Accordion):
    def __draw_shadow__(self, origin, end, context=None):
        pass


class Grafica(MDBoxLayout):
    def __draw_shadow__(self, origin, end, context=None):
        pass

    def __init__(self, **kw):
        super(Grafica, self).__init__(**kw)
        self.orientation = "vertical"
        self.spacing = 10
        self.padding = 20

        self.fig2, self.ax2 = plt.subplots(dpi=80, figsize=(7, 5), facecolor='#000000b7')
        plt.xlim(-11, 11)
        plt.ylim(-8, 8)
        plt.grid(alpha=0.2)
        plt.title("Función seno", color='blue', size=28, family="Kaufmann BT")

        self.ax2.set_facecolor('#6E6D7000')
        self.ax2.axhline(linewidth=2, color='w')
        self.ax2.axvline(linewidth=2, color='w')
        self.ax2.spines['bottom'].set_color('red')
        self.ax2.spines['left'].set_color('blue')
        self.ax2.set_xlabel("Eje  Horizontal", color='white', family='Cambria', size=15)
        self.ax2.set_ylabel("Eje  Vertical", color='white', family='Cambria', size=15)
        self.ax2.tick_params(color='blue', labelcolor='white', direction='out', length=6, width=2)

        self.canvasK = FigureCanvasKivyAgg(self.fig2, size_hint_y=.8)
        nivel = MDSlider(min=0, max=7, value=0, size_hint_y=.1)
        nivel.bind(value=self.valor_slider)

        self.add_widget(self.canvasK)
        self.add_widget(nivel)

    def valor_slider(self, _, value):
        x = np.arange(-4 * np.pi, value * np.pi, 0.01)
        line, = self.ax2.plot(x, value * np.sin(x), color='white', marker='o', linestyle='dotted',
                              markersize=1, linewidth=8)
        self.canvasK.draw()
        line.set_ydata(20)


class MainApp(MDApp):

    title = 'SimulaSS'
    icon = 'imagenes/logo.png'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        plt.style.use("dark_background")
        Builder.load_file('gui_kivy/global.kv')
        Builder.load_file('gui_kivy/main.kv')
        Builder.load_file('gui_kivy/metodosnumericos/algebra_linea_edo.kv')
        Builder.load_file('gui_kivy/metodosnumericos/children/exopmatricial.kv')
        Builder.load_file('gui_kivy/metodosnumericos/children/apliedoscreen.kv')
        Builder.load_file('gui_kivy/interaccion/interaccion.kv')
        Builder.load_file('gui_kivy/control/control.kv')
        Builder.load_file('gui_kivy/senalesysistemas/senalesysistemas.kv')
        Builder.load_file('gui_kivy/senalesysistemas/children/exsenales.kv')
        Builder.load_file('gui_kivy/senalesysistemas/children/exsistemas.kv')
        self.main = ScreenManager()

    def navigation(self, name_screen, animation="next"):
        self.main.current = name_screen
        if animation == "next":
            self.main.transition.direction = "left"
        else:
            self.main.transition.direction = "right"

    @staticmethod
    def add_update_graphic(parent_widget, widget_canvas, index_row, id_widget):
        index_children = (len(parent_widget.ids[id_widget].children)-1) - index_row
        print(parent_widget.ids[id_widget].children)
        old_canvas = parent_widget.ids[id_widget].children[index_children]
        print(f'Index_Children: {index_children}: \nChildren:')
        print(old_canvas)
        if isinstance(old_canvas, Widget) or isinstance(old_canvas, FigureCanvasKivyAgg):
            if isinstance(old_canvas, Widget):
                parent_widget.ids[id_widget].rows_minimum[index_row] = dp(250)
            parent_widget.ids[id_widget].remove_widget(old_canvas)
        print('Despues de eliminado Widget anterior:')
        print(parent_widget.ids[id_widget].children)
        parent_widget.ids[id_widget].add_widget(widget_canvas, index_children)
        print('Despues de agregado Widget Canvas:')
        print(parent_widget.ids[id_widget].children)

    def build(self):
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = "Blue"
        return self.main

    def on_start(self):
        self.main.get_screen('apliEDO').ejercicio_demostrativo_batman()


if __name__ == '__main__':
    Window.size = (450, 600)
    MainApp().run()
