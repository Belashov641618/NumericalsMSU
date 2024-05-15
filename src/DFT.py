from belashovplot import TiledPlot
import numpy


def discrete_fourier_transformation_test(file_name:str= '58.txt'):
    # Импорт данных из файла
    data = []
    with open(file_name, 'r') as file:
        for line in file:
            try:
                data.append(complex(line))
            except Exception as error:
                print(f'Ошибка при серилизации строки {line[:-1]} в complex <{error}>.')

    # Непосредственно вычисления
    signal = numpy.array(data)
    spectrum = numpy.roll(numpy.fft.fftshift(numpy.fft.fft(signal)), -1) / 16
    signal_recalculated = numpy.fft.ifft(numpy.fft.ifftshift(spectrum) * 16)
    signal_deviations = numpy.abs(signal - signal_recalculated)

    # Вычисление рзмера данных
    N = len(signal)

    # Вывод таблиц результатов
    for name, (data, data_range) in {
        'Сигнал':                           (signal,    numpy.arange(0, N)),
        'Спектр':                           (spectrum,  numpy.linspace(-N / 2 + 1, N / 2, N)),
        'Спектр (исправленная нумерация)':  (spectrum,  numpy.fft.fftshift(numpy.fft.fftfreq(N, 1)) * N)
    }.items():
        print(f"{name}: ")
        for n, (x, value) in enumerate(zip(data_range, data), 1):
            print(f'{n}\t{x}\t{value.real}\t{value.imag}')
        print('')


    # Инициализация параметров построения "тайлового" графика для сокращения количества кода
    graphs_data = {
        "Сигнал"                    : signal,
        "Спектр [-N/2+1, N/2]"      : spectrum,
        "Спектр [-N/2-1, N/2]"      : spectrum,
        "Восстановленный сигнал"    : signal_recalculated,
        "Отклонения сигнала"        : signal_deviations
    }
    graph_ranges = [
        numpy.arange(0, N),
        numpy.linspace(-N / 2 + 1, N / 2, N),
        numpy.fft.fftshift(numpy.fft.fftfreq(N, 1)) * N,
        numpy.arange(0, N),
        numpy.arange(0, N)
    ]
    extract_functions = {
        'g' : lambda x: numpy.abs(x),
        'y' : lambda x: numpy.angle(x),
        'r' : lambda x: numpy.real(x),
        'b' : lambda x: numpy.imag(x)
    }
    graph_types = [
        "амплитуда",
        "фаза",
        "реальная часть",
        "мнимая часть"
    ]


    # Настройка аннотаций пространства графиков
    plot = TiledPlot(MaxWidth=12*(21/9), MaxHeight=12)
    plot.FontLibrary.MultiplyFontSize(0.7)
    plot.title('Сравнение')
    plot.description.top("Спектр, полученный быстрым дискретным преобразованием Фурье исходного сигнала и сравнение исходного сигнала с восстановленным из спектра")
    plot.description.row.left   ("Амплитуда",           0)
    plot.description.row.left   ("Фаза",                1)
    plot.description.row.left   ("Реальная часть",      2)
    plot.description.row.left   ("Мнимая часть",        3)
    plot.description.column.top ("Сигнал",              0)
    plot.description.column.top ("Спектр",              1)
    plot.description.column.top ("Спектр (нумерация)",  2)
    plot.description.column.top ("Восстановленный",     3)
    plot.description.column.top ("Отклонения",          4)

    # Построение и добавление описаний
    for col, ((tittle, data), graph_range) in enumerate(zip(graphs_data.items(), graph_ranges)):
        for row, ((color, function), graph_type) in enumerate(zip(extract_functions.items(), graph_types)):
            axes = plot.axes.add(col, row)
            axes.plot(graph_range, function(data), f'.--{color}')
            axes.grid(True)
            plot.graph.description(f'{tittle} : {graph_type}')

    # Вывод графиков на экран
    plot.finalize()
    plot._Figure.savefig('temp.svg')
    plot.show()
