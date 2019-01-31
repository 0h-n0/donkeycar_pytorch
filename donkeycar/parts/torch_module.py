"""

pytorch.py

functions to run and train autopilots using pytorch

"""
from tqdm import tqdm
import torch
import torch.utils.data
import torch.nn as nn
import torchex.nn as exnn

from .torch_utils import load_model


class TorchPilot:
    def load(self, model_path):
        import torch
        _model = Linear()
        _model.load_state_dict(torch.load(model_path))
        _model.eval()
        self.model = _model

    def shutdown(self):
        pass

    def get_optimizer(self, lr=0.001, momentum=0.9, optim_type='Adam'):
        import torch.optim as optim
        try:
            return getattr(optim, optim_type)(self.model.parameters(), lr, momentum)
        except:
            return getattr(optim, optim_type)(self.model.parameters(), lr)

    def train_one_epoch(self, epoch, dataloader, optimizer, saved_model_path=None):
        total_loss = 0
        ang_loss = 0
        thr_loss = 0
        pbar = tqdm(total=len(dataloader))
        
        for i, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            if self.use_gpu:
                x = x.to('cuda')
                y = y.to('cuda')                
            output = self.model(x)
            loss, angle_loss, throttle_loss = self.model.loss(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            ang_loss += angle_loss.item()
            thr_loss += throttle_loss.item()
            
            pbar.set_description("epoch {:03d}: tot {:6.4f}, ang = {:6.4f}, thrtle = {:6.4f}".format(
                epoch, total_loss / (i+1), ang_loss / (i+1), thr_loss / (i+1)))
            pbar.update(1)
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
            
        best_loss = 1e10
        
        for epoch in range(epochs):
            if self.use_gpu:
                self.model.to('cuda')
            self.model.train()
            loss = self.train_one_epoch(epoch+1, train_gen, optimizer, saved_model_path)
            print('traning loss', loss, end='')
            self.model.eval()
            val_loss = self.train_one_epoch(epoch+1, val_gen, optimizer)
            print('valid loss', val_loss, end='')                        
            if best_loss > val_loss:
                best_loss = val_loss
                print('best_loss', best_loss, end='')
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
        steering = outputs[0][0].data
        throttle = outputs[0][1].data
        return float(steering), float(throttle)

    
class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()

        self.conv_block = nn.Sequential(
            exnn.Conv2d(24, 3, stride=3, padding=1),
            exnn.Conv2d(24, 3, stride=3, padding=1),
            nn.ReLU(),
            exnn.Conv2d(64, 3, stride=3, padding=1),
            exnn.Conv2d(64, 3, stride=3, padding=1),
            nn.ReLU())
        
        self.linear = nn.Sequential(
            exnn.Flatten(),
            exnn.Linear(500),
            nn.ReLU(),            
            nn.Dropout(0.1),
            exnn.Linear(50),            
            )

        self.angle_net = exnn.Linear(1)
        self.throttle_net = exnn.Linear(1)

        self.mse1 = nn.MSELoss()
        self.mse2 = nn.MSELoss()

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
        return torch.cat((angle, throttle), dim=1)

    def loss(self, x, y, angle_weight=0.5):
        angle_loss = self.mse1(x[:, 0], y[:, 0])
        throttle_loss = self.mse2(x[:, 1], y[:, 1])
        loss = angle_weight * angle_loss + throttle_loss * (1 - angle_weight)
        return loss, angle_loss, throttle_loss

    def predict(self, x):
        x = torch.FloatTensor(x).permute(0, 3, 1, 2) / 255
        y = self(x)
        return y

    
