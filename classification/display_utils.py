import numpy as np
import matplotlib.pyplot as plt




def plot_diagrams(diagrams: list) -> None:
    fig, axes = plt.subplots(len(diagrams), 1, sharex=True)
    for i, diagram in enumerate(diagrams):
        axes[i].plot(diagram[0], diagram[1])
        axes[i].xlabel = diagram[2]
        axes[i].ylabel = diagram[3]
    
    plt.show()