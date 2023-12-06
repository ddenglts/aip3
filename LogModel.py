from image import generate_diagram
import numpy as np

class LogModel:
    '''
    Logistic regression model
    '''
    def train(self, input_data: np.ndarray, output_data: np.ndarray):
        '''
        Train model
        '''

        # init input data matrix x and add x0 = 1 for each row
        self.mat_x = np.insert(input_data, 0, 1, axis=1)
        # init weights
        self.vec_w = np.random.rand(self.mat_x.shape[1])
        