import numpy as np
import sklearn.manifold as manifold

from image import *
import matplotlib.pyplot as plt

X, y = generate_diagram_nonlinear1(2500)

# plot tnse
for perp in [1,2,3,4,5,6,7,8,9,10,12,15,17,20,25,30,35,40,45,50,60,70,80,90,100]:
    X_embedded = manifold.TSNE(n_components=2, perplexity=perp).fit_transform(X.copy())
    plt.scatter(X_embedded[:,0], X_embedded[:,1], c=y)
    plt.savefig(f"Timages/tnse{perp}.png")
    plt.clf()
    print("Progress:", perp)