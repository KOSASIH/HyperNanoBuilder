import matplotlib.pyplot as plt

def fractal(x, y, size, complexity):
    if complexity <= 0:
        return
    
    # Draw the current fractal structure
    plt.plot([x, x + size], [y, y], color='black')
    
    # Recursively generate smaller fractal structures
    new_size = size / 2
    fractal(x, y, new_size, complexity - 1)
    fractal(x + new_size, y, new_size, complexity - 1)

# Example usage
structure_size = 100
structure_complexity = 5
fractal(0, 0, structure_size, structure_complexity)
plt.show()
