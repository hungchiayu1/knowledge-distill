
from load_dataset import download_dataset
from load_model import create_classification_model




def train(classification_model,trainloader,testloader,optimizer,steps = 100,teacher_logits=None):
    
    
    train_step = 0
    running_loss = []
    while True:
        for batch_count,(batch,labels) in enumerate(trainloader):
            if train_step>steps:
                
                return 
            
            if teacher_logits:
                
                teacher_logit = teacher_logits[batch_count]
                    
                output_dict = classification_model(batch,teacher_logits=teacher_logit,labels=labels)
            else:
                output_dict = classification_model(batch,labels=labels)
                
            loss = output_dict['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss.append(loss.item())
           
            if train_step%100 == 0:
                print(f"Step:{train_step}. Train Loss {sum(running_loss)/len(running_loss)}")
                
                running_loss = []
            
            if train_step%500 == 0:
                
                acc,test_loss = test(classification_model,testloader)
                print(f"Step:{train_step}. Validation loss: {test_loss}. Validation Accuracy: {acc}")
        
            train_step+=1
            
            
def test(classification_model,testloader):
    
    test_loss,correct = 0,0
    for batch,labels in testloader:
        
        with torch.no_grad():
            outputs = classification_model(batch,labels=labels)
            logits = outputs['logits']
            preds = torch.argmax(logits,axis=-1)
            test_loss += outputs['loss'].item()
            correct += (preds==labels).sum().item()
    
    
    acc = correct/len(testloader.dataset)
    test_loss = test_loss/len(testloader)
    
    return acc,test_loss

if __name__ == "__main__":
    
    
    trainloader,testloader = download_dataset()
    
    teacher_model = create_classification_model()
    
