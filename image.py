import numpy as np
import random
from colorama import init, Fore, Style
'''
returns a 20x20 array of integers representing the colors of the pixels
0 = white
1 = red
2 = green
3 = blue
4 = yellow
'''


def generate_diagram(count = 1) -> tuple[np.ndarray, np.ndarray[bool]]:
    images = []
    dangerous = []
    for _ in range(count):
        image, danger = _generate_diagram()
        images.append(image)
        dangerous.append(danger)
    return np.array(images), np.array(dangerous)

def _generate_diagram():
    # Initialize a 20x20 array (0 represents white color)
    image = np.zeros((20, 20), dtype=int)

    # Define colors mapping to integers
    colors = {
        "Red": 1,
        "Green": 2,
        "Blue": 3,
        "Yellow": 4
    }

    # Randomly choose whether to start with rows or columns
    start_with_row = random.choice([True, False])

    # Choose four different colors
    chosen_colors = random.sample(list(colors.keys()), 4)

    red_ind = -1
    yellow_ind = -1
    dangerous = False
    for i, _ in enumerate(chosen_colors):
        if (chosen_colors[i] == "Red"):
            red_ind = i
        if (chosen_colors[i] == "Yellow"):
            yellow_ind = i
    if red_ind < yellow_ind:
        dangerous = True

    # First selection
    if start_with_row:
        row = random.randint(0, 19)
        image[row, :] = colors[chosen_colors[0]]
    else:
        col = random.randint(0, 19)
        image[:, col] = colors[chosen_colors[0]]

    # Second selection
    if start_with_row:
        col = random.randint(0, 19)
        image[:, col] = colors[chosen_colors[1]]
    else:
        row = random.randint(0, 19)
        image[row, :] = colors[chosen_colors[1]]

    # Third selection
    if start_with_row:
        while True:
            new_row = random.randint(0, 19)
            if new_row != row:
                break
        image[new_row, :] = colors[chosen_colors[2]]
    else:
        while True:
            new_col = random.randint(0, 19)
            if new_col != col:
                break
        image[:, new_col] = colors[chosen_colors[2]]

    # Fourth selection
    if start_with_row:
        while True:
            new_col = random.randint(0, 19)
            if new_col != col:
                break
        image[:, new_col] = colors[chosen_colors[3]]
    else:
        while True:
            new_row = random.randint(0, 19)
            if new_row != row:
                break
        image[new_row, :] = colors[chosen_colors[3]]

    return (image, dangerous)

def print_diagram(image):
    init()  # Initialize Colorama
    for row in image[0]:
        for pixel in row:
            if pixel == 0:  # White
                print(Style.RESET_ALL + '██', end='')
            elif pixel == 1:  # Red
                print(Fore.RED + '██', end='')
            elif pixel == 2:  # Green
                print(Fore.GREEN + '██', end='')
            elif pixel == 3:  # Blue
                print(Fore.BLUE + '██', end='')
            elif pixel == 4:  # Yellow
                print(Fore.YELLOW + '██', end='')
        print(Style.RESET_ALL)  # Reset to default after each row
    
    print(image[1])

# Generate and print the diagram

if __name__ == "__main__":
    diagram = _generate_diagram()
    print_diagram(diagram)
