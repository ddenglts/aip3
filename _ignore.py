# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from image import generate_diagram

# # Assuming input_data and target_data are your data and labels
# data = generate_diagram(5000)
# input_data = np.array([image.flatten() for image in data[0]]) # 100 samples, each is a 20x20 image flattened to 400
# target_data = data[1]  # 100 samples, each is a boolean value

# # Convert numpy arrays to PyTorch tensors
# input_data = torch.tensor(input_data, dtype=torch.float32)
# target = torch.tensor(target_data, dtype=torch.float32)

# # Define the model
# model = nn.Linear(input_data.size(1), 1)

# # Define the loss function and the optimizer
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.1)

# # Train the model
# for epoch in range(1000):
#     optimizer.zero_grad()
#     output = model(input_data)
#     loss = criterion(output, target.view(-1, 1))
#     loss.backward()
#     optimizer.step()

# # Evaluate the model
# with torch.no_grad():
#     data = generate_diagram(5000)
#     input_data = np.array([image.flatten() for image in data[0]]) 
#     input_data = torch.tensor(input_data, dtype=torch.float32)
#     output = model(input_data)
#     prediction = torch.round(torch.sigmoid(output))  # Threshold at 0.5
#     accuracy = (prediction == target.view(-1, 1)).type(torch.float32).mean().item()
#     print(f'Accuracy: {accuracy * 100:.2f}%')

#     import matplotlib.pyplot as plt
#     plt.scatter(data[1], output)
#     plt.show()