import matplotlib.pyplot as plt

def plot_rt(T, V, label):
    plt.plot(T, V, label=label)

def plot_derivative(T, dVdT, label):
    plt.plot(T, dVdT, label=label)

def finalize_plot(title, xlabel, ylabel):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
