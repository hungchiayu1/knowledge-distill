import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

def download_model():
    
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    
    return model

def create_classification_model(n_classes=10):
    
    model = download_model()
    classification_model = Resnet_Classification(model,n_classes)
    
    
    return classification_model


class Resnet_Classification(nn.Module):
    
    def __init__(self,base_model,n_classes):
        super().__init__()
        self.base_model = base_model
        self.n_classes = n_classes
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(1000,n_classes)
        
    def forward(self,inputs,labels = None,**kargs):
        outputs = self.base_model(inputs,**kargs)
        x = self.dropout(outputs)
        logits = self.classifier(x)
        loss = None
        out = {} ## Output dictionary
        out['logits'] = logits
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1,self.n_classes),labels.view(-1))
            out["loss"] = loss
           
        return out
