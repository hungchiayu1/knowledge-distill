import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

def download_model():
    
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    
    return model

def create_teacher_model(n_classes=10):
    
    model = download_model()
    classification_model = Resnet_Classification(model,n_classes)
    
    
    return classification_model


def create_student_model(n_classes=10):
    
    model = Student(n_classes)
    
    return model

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

    
    

class Student(nn.Module):
    
    def __init__(self,n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.conv2 = nn.Conv2d(64, 8, kernel_size=(3, 3), stride=(1, 1), bias=False)
        self.classifier = nn.Linear(288,self.n_classes)
    
    def distillation_loss(self,logits,labels,teacher_logits,T,alpha):
    
        train_loss_fct = nn.CrossEntropyLoss() 
        train_loss = train_loss_fct(logits,labels) ## Training loss

        distill_loss_fnt =  nn.KLDivLoss() ## Distillation loss

        distill_loss = distill_loss_fnt(F.log_softmax(logits/T,dim=1),F.log_softmax(teacher_logits/T,dim=1))

        ## Total loss
        loss = T*T*0.9*distill_loss+(1-0.9)*train_loss

        return loss


    def forward(self,x,teacher_logits = None,labels=None,T=5,alpha=0.5):
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = nn.functional.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.Flatten()(x)
        logits = self.classifier(x)
        
        
        
        out = {}
        out['logits'] = logits
        
        
       
        if teacher_logits is not None and labels is not None:
            
            out['loss'] = self.distillation_loss(logits,labels,teacher_logits,T,alpha)
            
            return out
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            train_loss = loss_fct(logits,labels)
            out['loss'] = train_loss
        
   
        return out
