import pdb
import torch.nn.functional as F
import numpy as np

def transform_to_image2(data):
    # Initialize the matrix with 100.0 (indicating unused slots)
    matrix = np.full((8, 9), 100.0)
    indices = [
        (0, 3), (0, 4), (0, 5), (0, 6),
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
        (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8),
        (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 8),
        (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 8),
        (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8),
        (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7),
        (7, 3), (7, 4), (7, 5), (7, 6)
    ]
    # Fill the specified indices with data from the input array
    for idx, (i, j) in enumerate(indices):
        matrix[i, j] = data[idx]
    matrix = np.pad(matrix, ((0, 1), (0, 0)), mode='constant', constant_values=100.0)
    masked_matrix = np.where(matrix == 100, np.nan, matrix)
    return masked_matrix

def plot_archetype_matrices(ax, i, archetypes):
    # Define color map limits
    vmin, vmax = -38, 38  # Set the grayscale range

    # Convert to 2D matrix using the transform_to_image2 function
    matrix = transform_to_image2(archetypes[i])

    # Plot the heatmap for the archetype
    im = ax.imshow(matrix, cmap='RdBu', interpolation='nearest', vmin=vmin, vmax=vmax, aspect='auto')
    # ax.set_title(f"AT{i + 1}", fontsize=10)
    ax.text(0.5, 1.05, f"AT{i+1}", ha='center', transform=ax.transAxes)



def main():
    archetypes_csv = "./at17_matrix.csv"
    output_dir = "archetypes_heatmaps"
    plot_archetype_matrices(archetypes_csv, output_dir)

if __name__ == "__main__":
    main()