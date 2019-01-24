import torch

def load_model(model, path_to_model):
    model.load_state_dict(torch.load(path_to_model))
    model.eval()
    return model
    
