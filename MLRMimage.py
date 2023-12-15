import numpy as np
import random



def generate_diagram_hot(count = 1) -> tuple[np.ndarray, np.ndarray]:
    images = []
    wires = []
    for _ in range(count):
        image, wire = _generate_diagram_cut()
        image = image.flatten()
        new_image = []
        for v in image:
            if v == 1:
                new_image.append([1,0,0,0])
            elif v == 2:
                new_image.append([0,1,0,0])
            elif v == 3:
                new_image.append([0,0,1,0])
            elif v == 4:
                new_image.append([0,0,0,1])

        images.append(np.array(new_image).flatten())

        if wire == 1:
            wire = [1,0,0,0]
        elif wire == 2:
            wire = [0,1,0,0]
        elif wire == 3:
            wire = [0,0,1,0]
        elif wire == 4:
            wire = [0,0,0,1]
        wires.append(wire)

    # returns tuple(array of images, array of booleans)
    return np.array(images), np.array(wires)

def _generate_diagram_cut():
    # Initialize a 20x20 array (0 represents white color)
    image = np.zeros((20, 20), dtype=int)

    # Define colors mapping to integers
    colors = {
        "Red": 1,
        "Green": 2,
        "Blue":3,
        "Yellow": 4
    }

    # Randomly choose whether to start with rows or columns
    start_with_row = random.choice([True, False])

    # Choose four different colors
    dangerous = False
    chosen_colors = random.sample(list(colors.keys()), 4)
    while not dangerous:
        chosen_colors = random.sample(list(colors.keys()), 4)

        red_ind = -1
        yellow_ind = -1
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

    return (image, colors[chosen_colors[2]])

def convert_diagram_hot(diagrams: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    '''
    diagrams need to be a tuple like:

    (np.array[image1, image2, ...], np.array[label1, label2, ...])

    imageN is a 2d array 
    '''
    # to store images to return
    images = []
    dangerous = []

    #iterates every image in diagrams
    for index, v in enumerate(diagrams[0]):
        image = v
        image = image.flatten()
        new_image = []
        for i in image:
            if i == 1:
                new_image.append(np.array([1,0,0,0]))
            elif i == 2:
                new_image.append(np.array([0,1,0,0]))
            elif i == 3:
                new_image.append(np.array([0,0,1,0]))
            elif i == 4:
                new_image.append(np.array([0,0,0,1]))
            else:
                new_image.append(np.array([0,0,0,0]))

        images.append(np.array(new_image).flatten())
        dangerous.append(diagrams[1][index])
    # returns tuple(array of images, array of booleans)
    return np.array(images), np.array(dangerous)

def generate_diagram_nonlinear3(count = 1) -> tuple[np.ndarray, np.ndarray]:
    one_hot = {
        1: (1,0,0,0),
        2: (0,1,0,0),
        3: (0,0,1,0),
        4: (0,0,0,1),
        0: (0,0,0,0)
    }

    conved_images = []
    dangers = []

    # tuple of 2 lists, one list of images, other list of labels
    # generate a tuple of diagrams
    diagrams = ([], [])
    for _ in range(count):
        image, label = _generate_diagram_cut()
        diagrams[0].append(image)
        diagrams[1].append(label)

    for _,v in enumerate(zip(diagrams[0], diagrams[1])):
        image, danger = v
        dangers.append(danger)

        # array of convoluted values from 2x2 windows with stride = 1
        conved_image = []
        for row in range(image.shape[0] - 1):
            for col in range(image.shape[1] - 1):
                e11 = image[row, col]
                e12 = image[row, col + 1]
                e21 = image[row + 1, col]
                e22 = image[row + 1, col + 1]
                conved_image.append(np.dot(one_hot[e11], one_hot[e12]) + np.dot(one_hot[e11], one_hot[e21]) + np.dot(one_hot[e11], one_hot[e22]))
            
        assert len(conved_image) == 19*19
        conved_images.append(np.array(conved_image).flatten())

    # combining conved_images and onehot images together into one input
    mergeds = ([], [])
    one_hot_diagrams = convert_diagram_hot(diagrams)
    for i,v in enumerate(one_hot_diagrams[0]):
        merged = []
        merged.extend(v)
        merged.extend(conved_images[i])
        mergeds[0].append(merged)
    mergeds[1].extend(one_hot_diagrams[1].flatten())
    for i,wire in enumerate(mergeds[1]):
        if wire == 1:
            mergeds[1][i] = np.array([1,0,0,0])
        elif wire == 2:
            mergeds[1][i] = np.array([0,1,0,0])
        elif wire == 3:
            mergeds[1][i] = np.array([0,0,1,0])
        elif wire == 4:
            mergeds[1][i] = np.array([0,0,0,1])
        else:
            raise ValueError("Wire value is not 1, 2, 3, or 4 !!!!! WTFFFFFFFF THIS IS NOT POSSIBLE")
    return np.array(mergeds[0]), np.array(mergeds[1])


if __name__ == "__main__":
    images, wires = generate_diagram_nonlinear3(10)
    print(wires)