""""

pytorch.py

functions to run and train autopilots using pytorch

"""


import torch.nn as nn
#import torchex.nn as exnn

from .torch_utils import load


class TorchPilot:
    def load(self, model_path):
        self.model = load_model(model_path)

    def shutdown(self):
        pass

    def train_one_epoch(self, epoch, data_gen, save_model_path):
        for xdata in train_gen:
            print(xdata)
    
    def train(self, train_gen, val_gen,
              saved_model_path, epochs=100, steps=100, train_split=0.8,
              verbose=1, min_delta=.0005, patience=5, use_early_stop=True):
        """
        train_gen: generator that yields an array of images an array of

        """

        for epoch in range(epochs):
            loss = self.train_one_epoch(epoch+1, train_gen, save_model_path)
            loss = self.train_one_epoch(epoch+1, val_gen, save_model_path)
            

        hist = self.model.fit_generator(
            train_gen,
            steps_per_epoch=steps,
            epochs=epochs,
            verbose=1,
            validation_data=val_gen,
            callbacks=callbacks_list,
            validation_steps=steps * (1.0 - train_split) / train_split)
        return hist


class TorchLinear(TorchPilot):
    def __init__(self, model=None, num_outputs=None, *args, **kwargs):
        super(TorchLinear, self).__init__(*args, **kwargs)
        if model:
            self.model = model
        elif num_outputs is not None:
            self.model = Linear()
        else:
            self.model = Linear()

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        outputs = self.model.predict(img_arr)
        # print(len(outputs), outputs)
        steering = outputs[0]
        throttle = outputs[1]
        return steering[0][0], throttle[0][0]


    
class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2D(3, 24, (5, 5), (2, 2)),
            nn.ReLU(),
            nn.Conv2D(24, 32, (5, 5), (2, 2)),
            nn.ReLU(),
            nn.Conv2D(32, 64, (5, 5), (2, 2)),
            nn.ReLU(),
            nn.Conv2D(64, 64, (3, 3), (2, 2)),
            nn.ReLU(),
            nn.Conv2D(64, 64, (3, 3), (1, 1)),
            nn.ReLU())

        self.linear = nn.Sequential(
            exnn.Flatten(),
            exnn.Linear(100),
            nn.Dropout(0.1),
            exnn.Linear(50),
            nn.Dropout(0.1),
            )

        self.angle_net = exnn.Linear(1)
        self.throttle_net = exnn.Linear(1)        
        
    def forward(self, x):
        x = self.conv_block(x)
        x = self.linear(x)
        
        angle = self.angle_net(x)
        throttle = self.throttle_net(x)
        
        return angle, throttle

    def loss(self, x, y):
        return None

    def predict(self, x):
        return None
