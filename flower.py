# Imports here


import matplotlib.pyplot as plt
import PIL
from PIL import Image
import torch
from torch import nn,optim, tensor
import numpy as np
import torch.nn.functional as F
import torchvision
import time
from torchvision import datasets, transforms, models
import json
import argparse



#load the data
def load_data(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])

    testing_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    image_train_data = datasets.ImageFolder(train_dir, transform = training_transforms) 
    image_validation_data = datasets.ImageFolder(valid_dir, transform = validation_transforms)
    image_test_data = datasets.ImageFolder(test_dir, transform = testing_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(image_train_data, batch_size = 64, shuffle = True) 
    validationloader = torch.utils.data.DataLoader(image_validation_data, batch_size = 32, shuffle = True)
    testloader = torch.utils.data.DataLoader(image_test_data, batch_size = 16, shuffle = True)
    return trainloader, validationloader, testloader, image_train_data


#set up our model
def nn_setup(structure, hidden_units1, hidden_units2, learning_rate,power, dropout):
    
    
    if structure == "densenet121":
        model = models.densenet121(pretrained = True)
        num_input = 1024
    elif structure == "vgg16":
        model = models.vgg16(pretrained = True)
        num_input = 25088
        
    elif structure == "alexnet":
        model = models.alexnet(pretrained = True)
        num_input = 9216
    else:
        print(" please select one of these models: densenst121, vgg16, or alexnet")
           
    for param in model.parameters():
        param.requires_grad = False

        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(dropout)),
            ('inputs', nn.Linear(num_input,hidden_units1)),
            ('relu', nn.ReLU()),
            ('hidden_layer1', nn.Linear(hidden_units1, hidden_units2)),
            ('relu1', nn.ReLU()),
            ('hidden_layer2', nn.Linear(hidden_units2, 102)),
            ('output', nn.LogSoftmax(dim=1))
            ]))
        model.classifier = classifier

        criterion = nn.NLLLoss()

        optimizer =  optim.Adam(model.classifier.parameters(), lr = learning_rate)
        if torch.cuda.is_available() and power == 'gpu':
            model.cuda()
        return model, criterion, optimizer, classifier

#train the model and validate it by the validation data to see how is the training doing.

def deep_learning(model, trainloader, epochs, print_every, criterion, optimizer,validationloader, power):
    epochs = epochs
    print_every = print_every
    steps = 0


    if torch.cuda.is_available() and power == 'gpu':
        model.to('cuda')
        
    for e in range(epochs):
        running_loss = 0
        model.train()
        for ii, (inputs, labels) in enumerate(trainloader):
            steps +=1
            start = time.time()
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
            # to zero optimizer
            optimizer.zero_grad()

            #forward and backward passes

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward() # back propagation
            optimizer.step() #Gradient Descent

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                valid_lost = 0
                accuracy = 0

                for ii, (inputs1, labels1) in enumerate(validationloader):
                    optimizer.zero_grad()
                    if torch.cuda.is_available()and power == 'gpu':
                        
                        model.to('cuda')
                    inputs1, labels1 = inputs1.to('cuda'), labels1.to('cuda')
                    
                    with torch.no_grad():
                        outputs = model.forward(inputs1)
                        valid_lost = criterion(outputs, labels1)
                        predict = torch.exp(outputs).data
                        equal = (labels1.data == predict.max(1)[1])
                        accuracy += equal.type_as(torch.FloatTensor()).mean()

                valid_lost = valid_lost / len(validationloader)
                accuracy = accuracy /  len(validationloader)
                end = time.time()
                
                period = end - start
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Traning Loss: {:.4f}".format(running_loss/print_every),
                      "validation Loss: {:.4f}".format(valid_lost),
                      "Accuracy:{:.4f}".format(accuracy),
                      "Time: {:.2f}".format(period))

                running_loss = 0
                model.train()

                
# TODO: Do validation on the test set
def test_data (testloader, model, power):
    
    total = 0
    correct = 0
    if torch.cuda.is_available()and power == 'gpu':
        model.to('cuda')
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct+= (predicted == labels).sum().item()
    print("")
    print('Accuracy of the network on the '+str(total)+' test images: %d %%'% (100 * correct / total))
    print("")
            

# TODO: Save the checkpoint 
def save_checkpoint(checkpoint_path, image_train_data,model,hidden_units1, hidden_units2, learning_rate):
    model.class_to_idx = image_train_data.class_to_idx

    checkpoint= {'inputs': 1024,
                 'output': 102,
                'model': model,
                'hidden_layer1': hidden_units1,
                'hidden_layer2': hidden_units2,
                'learning_rate': learning_rate,
                #'optimizer': optimizer.state_dict(),
                'class_to_idx': model.class_to_idx,
                'state_dict': model.state_dict()}
           
    torch.save(checkpoint, 'checkpoint.pth')
    #return 'checkpoint.pth'


# TODO: Write a function that loads a checkpoint and rebuilds the model

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    inputs = checkpoint['inputs']
    output = checkpoint['output']
    hidden_layer1 = checkpoint['hidden_layer1']
    hidden_layer2 = checkpoint['hidden_layer2']
    learning_rate = checkpoint['learning_rate']
    model = checkpoint['model']
    #model, criterion, optimizer, classifier = nn_setup()
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model



# process images to fit the images that being used during training.
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    img = Image.open(image) 
    
    img.thumbnail((224,224))
    
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    
    img = img.crop((left_margin, bottom_margin, right_margin, top_margin))
    
    image = np.array(img)/225
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean)/ std
    
    image_processed = image.transpose((2,0,1))
    
    return image_processed




# predict function will identify the name of flower

def predict(image_path, model, topk, power):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    ''' 
    # TODO: Implement the code to predict the class from an image file
    if torch.cuda.is_available()and power == 'gpu':
        model.to('cuda:0')
        
    # TODO: Implement the code to predict the class from an image file
    
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    image = image.unsqueeze(0)
    #image = image.float()
    
    with torch.no_grad():
        outputs = model.forward(image.cuda())
    
    prob = torch.exp(outputs)
    
    top_prob, top_indices = prob.topk(topk)
    
    #top_prob = top_prob.detach().type(torch.FloatTensor).numpy().tolist()[0]
    top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0]
    
    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    
    top_classes = [index_to_class[each] for each in top_indices]
    
    return top_prob, top_classes

