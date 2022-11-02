##############################################################################
# Programa que encuentra todos los picos de una gráfica superiores a una 
# altura introducida por el usuario.

# Su primer paso es graficar los datos para que el usuario pueda identificar
# la altura que poner como límite inferior.

# Tras ello encuentra los máximos y la anchura a mitad de altura de los picos

# El archivo tiene que componerse de dos columnas separadas por tabuladores. La
# primera columna corresponde con el tiempo t y la segunda con la intensidad
# registrada por el contador Geiger
###############################################################################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d


## FUNCIÓN PARA INTERPOLAR DE VALORES DE SAMPLE EN EL DATAFRAME A VALORES EN X
def index_to_xdata(xdata, indices):
    ind = np.arange(len(xdata))
    f = interp1d(ind, xdata)
    return f(indices)


## LEER ARCHIVO TXT Y PONERLO COMO DATAFRAME
def readfile(file):

    with open(file) as f:
        data = f.read().split('\n')
        data = data[:]

    x = np.array([float(row.split('\t')[0]) for row in data])
    # Pasar de tiempo a ángulos
    x = 20 + 0.172*x
    y = np.array([float(row.split('\t')[1]) for row in data])

    d = {'ang': x, 'cuentas': y}

    df = pd.DataFrame(data=d)

    return df, x, y


## ENCONTRAR MÁXIMOS Y MÍNIMOS 
def encontrarMaximosMinimos(df):


    df.plot(x='ang', y='cuentas')
    plt.xlabel('angitud de onda [nm]')
    plt.ylabel('Número de cuentas')
    plt.grid()
    plt.show()

    # Encontrar máximos y mínimos de intensidad y del voltaje de toda la muestra

    height = float(input("Altura minima: "))

    peaks, _ = signal.find_peaks(df.cuentas, height=height)

    x_maximos, y_maximos = df.iloc[peaks].values.T
    
    width, heights, xinf, xsup  = signal.peak_widths(df.cuentas, peaks, rel_height=0.5)
    x = df['ang'].tolist()
    xinf = np.array(index_to_xdata(x, xinf))
    xsup = np.array(index_to_xdata(x, xsup))
    peakWidth = xsup - xinf


    # Plotear la gráfica con máximos y mínimos de toda la muestra 
    # para elegir el rango bueno

    df.plot(x='ang', y='cuentas')
    plt.plot(x_maximos, y_maximos, 'ro')
    plt.xlabel('angitud de onda [nm]')
    plt.ylabel('Número de cuentas')

    plt.grid()
    plt.show()

    return x_maximos, y_maximos, peakWidth


## FUNCIÓN PRINCIPAL
def main(filename):

    # abrir todos los archivos a analizar
    for file in filename:

        df, x, y = readfile(file)

        x_max, y_max, peakWidth = encontrarMaximosMinimos(df)

        for i in range(len(x_max)):
            print(x_max[i], peakWidth[i])


file = ['picos_KCL']
main(file)


    



    


