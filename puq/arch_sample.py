import pandas as pd
import numpy as np
import archetypes as arch

hvf_json = [
    {"x": 0, "y": 3, "index": 0}, {"x": 0, "y": 4, "index": 1}, {"x": 0, "y": 5, "index": 2},
    {"x": 0, "y": 6, "index": 3}, {"x": 1, "y": 2, "index": 4}, {"x": 1, "y": 3, "index": 5},
    {"x": 1, "y": 4, "index": 6}, {"x": 1, "y": 5, "index": 7}, {"x": 1, "y": 6, "index": 8},
    {"x": 1, "y": 7, "index": 9}, {"x": 2, "y": 1, "index": 10}, {"x": 2, "y": 2, "index": 11},
    {"x": 2, "y": 3, "index": 12}, {"x": 2, "y": 4, "index": 13}, {"x": 2, "y": 5, "index": 14},
    {"x": 2, "y": 6, "index": 15}, {"x": 2, "y": 7, "index": 16}, {"x": 2, "y": 8, "index": 17},
    {"x": 3, "y": 0, "index": 18}, {"x": 3, "y": 1, "index": 19}, {"x": 3, "y": 2, "index": 20},
    {"x": 3, "y": 3, "index": 21}, {"x": 3, "y": 4, "index": 22}, {"x": 3, "y": 5, "index": 23},
    {"x": 3, "y": 6, "index": 24}, {"x": 3, "y": 8, "index": 26},
    {"x": 4, "y": 0, "index": 27}, {"x": 4, "y": 1, "index": 28}, {"x": 4, "y": 2, "index": 29},
    {"x": 4, "y": 3, "index": 30}, {"x": 4, "y": 4, "index": 31}, {"x": 4, "y": 5, "index": 32},
    {"x": 4, "y": 6, "index": 33}, {"x": 4, "y": 8, "index": 35},
    {"x": 5, "y": 1, "index": 36}, {"x": 5, "y": 2, "index": 37}, {"x": 5, "y": 3, "index": 38},
    {"x": 5, "y": 4, "index": 39}, {"x": 5, "y": 5, "index": 40}, {"x": 5, "y": 6, "index": 41},
    {"x": 5, "y": 7, "index": 42}, {"x": 5, "y": 8, "index": 43}, {"x": 6, "y": 2, "index": 44},
    {"x": 6, "y": 3, "index": 45}, {"x": 6, "y": 4, "index": 46}, {"x": 6, "y": 5, "index": 47},
    {"x": 6, "y": 6, "index": 48}, {"x": 6, "y": 7, "index": 49}, {"x": 7, "y": 3, "index": 50},
    {"x": 7, "y": 4, "index": 51}, {"x": 7, "y": 5, "index": 52}, {"x": 7, "y": 6, "index": 53}
]

def load_archetype_matrix(filepath):
    """
    Load the archetype matrix from a CSV file.

    Args:
        filepath (str): Path to the CSV file containing the archetype matrix.

    Returns:
        np.ndarray: Archetype matrix as a NumPy array.
    """
    archetypes = pd.read_csv(filepath).to_numpy()
    coordinates = [(entry["x"], entry["y"]) for entry in hvf_json]
    arch2 = []
    for i in range(len(archetypes)):
        vf = [0 for _ in range(81)]
        for j in range(len(archetypes[i])):
            vf[coordinates[j][0] * 9 + coordinates[j][1]] = archetypes[i][j]
        arch2.append(vf)
    return np.array(arch2)


def normalize_data(values, data_min, data_max, archetype_min, archetype_max):
    """
    Normalize the data to the range of archetypes.

    Args:
        values (np.ndarray): Original data values.
        data_min (float): Minimum value in the data.
        data_max (float): Maximum value in the data.
        archetype_min (float): Minimum value in the archetypes.
        archetype_max (float): Maximum value in the archetypes.

    Returns:
        np.ndarray: Normalized values.
    """
    normalized = (values - data_min) / (data_max - data_min)
    scaled = normalized * (archetype_max - archetype_min) + archetype_min
    return scaled


def decompose_hvf_data(aa, hvf_data):
    """
    Perform decomposition using the archetype analysis model.

    Args:
        aa (arch.AA): Trained archetype analysis model.
        hvf_data (np.ndarray): HVF data to be decomposed.

    Returns:
        np.ndarray: Decomposition coefficients.
    """
    return aa.transform(hvf_data)

def reconstruct_from_coefficients(archetype_matrix, coefficients):
    """
    Reconstruct data using archetype coefficients.

    Args:
        archetype_matrix (np.ndarray): Matrix of archetypes.
        coefficients (np.ndarray): Decomposition coefficients.

    Returns:
        np.ndarray: Reconstructed data.
    """
    return archetype_matrix.T @ coefficients


def main():
    # Load archetype matrix
    archetype_matrix_path = "path/to/archetype_matrix.csv"  # Replace with the actual path
    archetype_matrix = load_archetype_matrix(archetype_matrix_path)

    # Initialize archetypes model
    n_archetypes = 17
    aa = arch.AA(n_archetypes=n_archetypes)
    aa.archetypes_ = archetype_matrix

    # Example data (replace with your actual data)
    hvf_data = np.array([[...]])
    data_min, data_max = -37.69, 22.69  # Replace with actual min/max values in your data
    archetype_min, archetype_max = np.min(archetype_matrix), np.max(archetype_matrix)

    # Normalize the data
    normalized_data = normalize_data(hvf_data, data_min, data_max, archetype_min, archetype_max)

    # Decompose the normalized data
    decomposition_coefficients = decompose_hvf_data(aa, normalized_data)

    # Reconstruct the data from decomposition coefficients
    reconstructed_data = reconstruct_from_coefficients(archetype_matrix, decomposition_coefficients[0])

    # Print results for verification
    print("Decomposition Coefficients:", decomposition_coefficients)
    print("Reconstructed Data:", reconstructed_data)


if __name__ == "__main__":
    main()
