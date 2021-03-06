import argparse
import cv2
import numpy as np
import torch
import os
import easygui
from tqdm import tqdm
from torch.autograd import Function
from torchvision import models

def get_classtable():
    classes = []
    
    with open("classes.txt") as lines:
        for line in lines:
            line = line.strip().split(" ", 1)[1]
            line = line.split(", ", 1)[0].replace(" ", "_")
            classes.append(line)
    return classes

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.features, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model.classifier(output)
        
        return target_activations, output


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    print("preprocesing image...")
    for i in tqdm(range(3)):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    print("Done.")
    return input



def show_cam_on_live(img, mask,name):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return  np.uint8(255 * cam) 
 

def show_cam_on_layer_grid(img, masks):
    images = []
    for mask in masks:
        print("maska")
        print(mask)
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        output = np.uint8(255 * cam)
        
        
        images.append(output)
    return images
    #cv2.imwrite('generated_grad/' + name + '_cam.jpg', np.uint8(255 * cam))



def show_cam_on_image(img, mask,name):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite('generated_grad/' + name + '_cam.jpg', np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def grad_all_layers(self, input, layers):
        index = 0
        images = []
        
        for i in range(layers):
           
            self.extractor = ModelOutputs(self.model, [str(i)])
            features, output = self.extractor(input)
            index = np.argmax(output.cpu().data.numpy())
            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0][index] = 1
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            if self.cuda:
                one_hot = torch.sum(one_hot.cuda() * output)
            else:
                one_hot = torch.sum(one_hot * output)

            self.model.features.zero_grad()
            self.model.classifier.zero_grad()
            one_hot.backward(retain_graph=True)

            grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

            target = features[-1]
            target = target.cpu().data.numpy()[0, :]

            weights = np.mean(grads_val, axis=(2, 3))[0, :]
            cam = np.zeros(target.shape[1:], dtype=np.float32)

            print("generating grad-cam...")
            for i, w in tqdm(enumerate(weights)):
                cam += w * target[i, :, :]

            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, (224, 224))
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)
            print("Done.")
            index += 1
            images.append(cam)
        return images

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        print("generating grad-cam...")
        for i, w in tqdm(enumerate(weights)):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        print("Done.")
        return cam


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # replace ReLU with GuidedBackpropReLU
        print("generating Guided BackpropagationReLUMode...")
        for idx, module in tqdm(self.model.features._modules.items()):
            if module.__class__.__name__ == 'ReLU':
                self.model.features._modules[idx] = GuidedBackpropReLU.apply

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        print("Done.")
        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./input_images/input.jpg',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)
def live_setup():
    gradcam = GradCam(model=models.vgg19(pretrained=True), \
                               target_layer_names=["35"], use_cuda=False)
        
    return gradcam

def live(gradcam,img):
    
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    input = preprocess_image(img)
    
    target_index = None
    mask = gradcam(input, target_index)
    
    show_cam_on_live(img, mask)
    
   


def grad_cam_all_layers():
    grad_cam = GradCam(model=models.vgg19(pretrained=True), \
                           target_layer_names=["35"], use_cuda=False)
    classes = get_classtable()
    print("Press Enter to choose image input...",end='')
    input()
    
    file = easygui.fileopenbox()
    if(file):
        img = cv2.imread(file, 1)
        img = np.float32(cv2.resize(img, (224, 224))) / 255
        input_image = preprocess_image(img)
        masks = grad_cam.grad_all_layers(input_image, 35)
    images = show_cam_on_layer_grid(img,masks)
    i = 0
    os.system("rm ./animation/*.jpg")
    for image in images:
        cv2.imwrite('./animation/' + str(i) + '.jpg', image)
        i+=1
    j = 0
    horiz = []
    
    for i in range(5):
        horiz.append(np.concatenate(images[j:j+7], axis=1))
        j+=7
    grid_image = np.concatenate(horiz,axis=0)
    print("Choose name for grid image: ",end="")
    name = input()
    cv2.imwrite('./generated_grad/' + name + '.jpg',grid_image)
    print("generate video(y/n)?",end='')
    video = input()
    if(video == "y"):
        print("Choose file name: ",end='')
        name = input()
        os.system("ffmpeg -r 5 -i ./animation/%d.jpg -vb 20M ./animation/" + name + ".avi")
        os.system("rm ./animation/*.jpg")
    #[cv2.imwrite('./test/' + str(i) + '.jpg',image) for image in images]

def run_from_import():
    
    grad_cam = GradCam(model=models.vgg19(pretrained=True), \
                           target_layer_names=["35"], use_cuda=False)
    classes = get_classtable()
    print("Press Enter to choose image input...",end='')
    input()
    

    file = easygui.fileopenbox()
    if(file):
        img = cv2.imread(file, 1)
        img = np.float32(cv2.resize(img, (224, 224))) / 255
        input_image = preprocess_image(img)
        target_index = None
        mask = grad_cam(input_image, target_index)
    
        print("Choose image name: ",end='')
        name = input()

        show_cam_on_image(img, mask,name)

        gb_model = GuidedBackpropReLUModel(model=models.vgg19(pretrained=True), use_cuda=False)
    
        gb = gb_model(input_image, index=target_index)
        gb = gb.transpose((1, 2, 0))
        cam_mask = cv2.merge([mask, mask, mask])
        cam_gb = deprocess_image(cam_mask*gb)
        gb = deprocess_image(gb)

        cv2.imwrite('generated_grad/'+ name + '_gb.jpg', gb)
        print('generated_grad/'+ name + '_gb.jpg written')
        cv2.imwrite('generated_grad/'+ name + '_cam_gb.jpg', cam_gb)
        print('generated_grad/'+ name + '_cam_gb.jpg written')
        print("Succesfully generated images.")

    else:
        print("File not chosen.")
        
    
    
   
if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """


    args = get_args()    
    print(args)
    exit()
    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    grad_cam = GradCam(model=models.vgg19(pretrained=True), \
                       target_layer_names=["35"], use_cuda=args.use_cuda)

    img = cv2.imread(args.image_path, 1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    input = preprocess_image(img)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None
    mask = grad_cam(input, target_index)

    show_cam_on_image(img, mask)

    gb_model = GuidedBackpropReLUModel(model=models.vgg19(pretrained=True), use_cuda=args.use_cuda)
    gb = gb_model(input, index=target_index)
    gb = gb.transpose((1, 2, 0))
    cam_mask = cv2.merge([mask, mask, mask])
    cam_gb = deprocess_image(cam_mask*gb)
    gb = deprocess_image(gb)

    cv2.imwrite('generated_grad/gb.jpg', gb)
    cv2.imwrite('generated_gradcam_gb.jpg', cam_gb)
