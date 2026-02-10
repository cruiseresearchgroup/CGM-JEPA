import torch
from utils.main_utils import get_metrics

class TaskHandler:
    def get_criterion(self):
        raise NotImplementedError
    
    def process_logits(self, logits):
        raise NotImplementedError
    
    def get_metrics(self, predictions, targets, probs):
        raise NotImplementedError
    
    def predict(self, model, x):
        return model.predict(x)
    
    def predict_proba(self, model, x):
        return None

class ClassificationHandler(TaskHandler):
    def get_criterion(self):
        return torch.nn.BCEWithLogitsLoss()
    
    def process_logits(self, logits):
        logits_tensor = torch.cat(logits).squeeze()
        probs_pos = torch.sigmoid(logits_tensor)
        probs = torch.stack([1 - probs_pos, probs_pos], dim=1).cpu().numpy()
        preds = (probs[:, 1] >= 0.5).astype(int)
        return preds, probs
    
    def get_metrics(self, predictions, targets, probs):
        return get_metrics(predictions, targets, probs)
    
    def predict_proba(self, model, x):
        return model.predict_proba(x)