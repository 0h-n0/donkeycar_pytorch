"""

pytorch.py

functions to run and train autopilots using pytorch

"""

import torch
import torch.utils.data
import torch.nn as nn
import torchex.nn as exnn

from .torch_utils import load_model


class TorchPilot:
    def load(self, model_path):
        _model = load_model(Linear(), model_path)        
        self.model = TorchLinear(_model)
        self.model = _model

    def shutdown(self):
        pass

    def get_optimizer(self, lr=0.001, momentum=0.9, optim_type='SGD'):
        import torch.optim as optim
        try:
            return getattr(optim, optim_type)(self.model.parameters(), lr, momentum)
        except:
            return getattr(optim, optim_type)(self.model.parameters(), lr)

    def train_one_epoch(self, epoch, dataloader, optimizer, saved_model_path=None):
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            if self.use_gpu:
                x = x.to('cuda')
                y = y.to('cuda')                
            output = self.model(x)
            angle_loss, throttle_loss = self.model.loss(output, y)
            loss = angle_loss + throttle_loss
            loss.backward()
            optimizer.step()
            print("epoch {:03d}: total_loss = {:7.5f}, angle = {:7.5f}, throttle_loss = {:7.5f}".format(
                epoch, loss.item(), angle_loss.item(), throttle_loss.item()))
            total_loss += loss
        total_loss /= len(dataloader)
        return total_loss
            
    def train(self, train_gen, val_gen, use_gpu,
              saved_model_path, epochs=100, steps=100, train_split=0.8,
              verbose=1, min_delta=.0005, patience=5, use_early_stop=True):
        """
        train_gen: generator that yields an array of images an array of
        """
        self.use_gpu = use_gpu
        optimizer = self.get_optimizer()

        if self.use_gpu:
            if torch.cuda.is_available():
                self.model.to('cuda')
                torch.backends.cudnn.benchmark = True
            else:
                print('CPU Traning')
                self.use_gpu = False
                
        best_loss = 1000
        
        for epoch in range(epochs):
            if self.use_gpu:
                self.model.to('cuda')
            self.model.train()
            loss = self.train_one_epoch(epoch+1, train_gen, optimizer, saved_model_path)
            print('traning loss', loss.item())
            self.model.eval()
            val_loss = self.train_one_epoch(epoch+1, val_gen, optimizer)
            print('valid loss', val_loss.item())                        
            if best_loss > val_loss.item():
                best_loss = val_loss.item()
                print('best_loss', best_loss)
                self.model.save(saved_model_path)


class TorchLinear(TorchPilot):
    def __init__(self, model=None, num_outputs=None, *args, **kwargs):
        super(TorchLinear, self).__init__(*args, **kwargs)
        if model:
            self.model = model
        elif num_outputs is not None:
            self.model = Linear()
        else:
            self.model = Linear()
        self.model = self.model.eval()
                                                     
    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        outputs = self.model.predict(img_arr)
        # print(len(outputs), outputs)
        print(outputs)
        steering = outputs[0]
        throttle = outputs[1]
        return float(steering[0][0].data), float(throttle[0][0].data)

    
class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 24, (5, 5), (2, 2)),
            nn.ReLU(),
            nn.Conv2d(24, 32, (5, 5), (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (5, 5), (2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 128, (5, 5), (2, 2)),
            nn.ReLU(),
            nn.Conv2d(128, 256, (3, 3), (1, 1)),
            nn.ReLU())

        self.linear = nn.Sequential(
            exnn.Flatten(),
            nn.Linear(2560, 100),
            nn.ReLU(),            
            nn.Dropout(0.1),
            nn.Linear(100, 50),
            nn.Dropout(0.1),
            )

        self.angle_net = nn.Linear(50, 1)
        self.throttle_net = nn.Linear(50, 1)

        self.size_average_mse = nn.MSELoss(reduction='mean')

    def save(self, saved_model_path):
        self = self.to('cpu')
        torch.save(self.state_dict(), saved_model_path)

    @classmethod
    def load(cls, saved_model_path, *args, **kwargs):
        model = model.to('cpu')
        model = cls(*args, **kwargs)
        model.load_state_dict(torch.load(saved_model_path))
        return model
        
    def forward(self, x):
        x = self.conv_block(x)
        x = self.linear(x)
        angle = self.angle_net(x)
        throttle = self.throttle_net(x)
        return angle, throttle

    def loss(self, x, y, angle_weight=0.5):
        angle_loss = self.size_average_mse(x[0], y[0])
        throttle_loss = self.size_average_mse(x[1], y[1])
        loss = angle_weight * angle_loss + throttle_loss * ( 1 - angle_loss)
        return angle_loss, throttle_loss

    def predict(self, x):
        x = torch.FloatTensor(x).permute(0, 3, 1, 2)
        y = self(x)
        return y

    
