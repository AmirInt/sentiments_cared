import numpy as np
import matplotlib.pyplot as plt




def plot_diagrams(diagrams: list) -> None:
    fig, axes = plt.subplots(1, len(diagrams), sharex=False, sharey=False)
    for i, diagram in enumerate(diagrams):
        axes[i].plot(diagram[0], diagram[1], color='purple')
        axes[i].set_xlabel(diagram[2])
        axes[i].set_ylabel(diagram[3])
    
    plt.show()