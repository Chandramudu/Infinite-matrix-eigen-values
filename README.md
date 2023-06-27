# Infinite-matrix-eigen-values
In this repository, I start with finding the eigen values of an infinite matrix 
import numpy as np

def find_eigenvalues(matrix):
  """Finds the eigenvalues of an infinite matrix.

  Args:
    matrix: The infinite matrix to find the eigenvalues of.

  Returns:
    A list of the eigenvalues of the matrix.
  """

  eigenvalues = []
  for i in range(1, np.inf):
    # Calculate the ith eigenvalue
    eigvalue = np.linalg.eigvals(matrix[0:i, 0:i])[0]

    # Add the eigenvalue to the list
    eigenvalues.append(eigvalue)

  return eigenvalues

if __name__ == "__main__":
  # Create an infinite matrix
  matrix = np.array([[1, 2], [2, 3]])

  # Find the eigenvalues of the matrix
  eigenvalues = find_eigenvalues(matrix)

  # Print the eigenvalues
  print(eigenvalues)
