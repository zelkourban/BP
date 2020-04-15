import os
import numpy as np
import copy
from PIL import Image
import matplotlib.cm as mpl_color_map
import torch
from torch.autograd import Variable
from torchvision import models
from torch.optim import Adam
import cv2
from tqdm import tqdm
import time
import grad_cam
from pyfzf.pyfzf import FzfPrompt
import regularized_class_sample_generator

def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr

def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    
    im.save(path)

def preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs
    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        pil_im.thumbnail((224, 224))
    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var

def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im


class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter,image_name):
        self.model = model
        self.image_name = image_name
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        # Create the folder to export images if not exists
        if not os.path.exists('generated_filter'):
            os.makedirs('generated_filter')
        if not os.path.exists('generated_grad'):
            os.makedirs('generated_grad')
        if not os.path.exists('animation'):
            os.makedirs('animation')
        if not os.path.exists('input_images'):
            os.makedirs('input_images')
                        
                
                        

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        self.hook_layer()
        # Generate a random image
        random_image = np.uint8(np.random.uniform(150, 180, (56, 56, 3)))
        # Process image and return variable
        processed_image = preprocess_image(random_image, False)
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 31):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                x = layer(x)
                # Only need to forward until the selected layer is reached
                if index == self.selected_layer:
                    # (forward hook function triggered)
                    break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(processed_image)
            # Save image
            if i % 5 == 0:
                print("saving image at iter ", i)
                im_path = '/content/' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '_with_hooks.jpg'
                save_image(self.created_image, im_path)

    def anim_optimisation(self,scale=1):
        
        # Process image and return variable
                # Generate a random image
                sz = 300
                random_image = np.uint8(np.random.uniform(150, 180, (sz, sz, 3)))
                                
                #random_image = np.uint8(np.random.uniform(150, 180, (sz, sz, 3)))
                # Process image and return variable
                processed_image = preprocess_image(random_image, False)
                frame = 0
                # Define optimizer for the image
                print("Generating animation frames...")
                os.system("rm animation/*")
                #processed_image = preprocess_image(random_image, False)
                #optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
                                  
                for i in tqdm(range(30)):
                  random_image = np.uint8(np.random.uniform(150, 180, (sz, sz, 3)))                              
                  processed_image = preprocess_image(random_image, False)
                  optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
                  for j in tqdm(range(1, 15)):
                      
                      optimizer.zero_grad()
                    # Assign create image to a variable to move forward in the model                    
                      x = processed_image
                      for index, layer in enumerate(self.model):
                        # Forward pass layer by layer
                          x = layer(x)
                          if index == self.selected_layer:
                            # Only need to forward until the selected layer is reached
                            # Now, x is the output of the selected layer
                              break
               
                      self.conv_output = x[0, self.selected_filter]
               
                      loss = -torch.mean(self.conv_output)
                      
                    # Backward
                      loss.backward()
                    # Update image
                      optimizer.step()
                      self.created_image = recreate_image(processed_image)
                                          # Save image               
                      #im_path = './animation/' + str(frame) + '.jpg'
                      #frame+= 1
                      #save_image(self.created_image, im_path)
                    # Recreate image
                      #random_image = val_tfms.denorm(img_var.data.cpu().numpy()[0].transpose(1,2,0))
                  im_path = './animation/' + str(frame) + '.jpg'
                  frame+= 1
                  save_image(self.created_image, im_path)
                  #sz = int(0.6 * sz)  # calculate new image size
                  
                  ##random_image = random_image[sz//4:3*sz//4, sz//4:3*sz//4]
                  #random_image = cv2.resize(random_image, (sz, sz), interpolation = cv2.INTER_CUBIC)
                os.system("ffmpeg -r 10 -i ./animation/%d.jpg -vb 20M ./animation/animation.avi")
                os.system("ffmpeg -i ./animation/animation.avi -vb 20M -filter:v \"minterpolate='mi_mode=mci:mc_mode=aobmc:vsbmc=1:fps=60:me=fss'\" ./animation/animation.mp4")
                
                os.system("rm ./animation/*.jpg")
                os.system("rm ./animation/animation.avi")
                                
                print("Generated animation.avi sucesfully")
                  
                                        
                
        
        

    def visualise_layer_without_hooks(self,scale=1):
        # Process image and return variable
        # Generate a random image
        sz = 256
        random_image = np.uint8(np.random.uniform(150, 180, (sz, sz, 3)))
        # Process image and return variable
        processed_image = preprocess_image(random_image, False)
        # Define optimizer for the image
        for i in tqdm(range(scale)):
          processed_image = preprocess_image(random_image, False)
          optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
          for i in tqdm(range(1, 20)):
              
              optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
              x = processed_image
              for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                  x = layer(x)
                  if index == self.selected_layer:
                    # Only need to forward until the selected layer is reached
                    # Now, x is the output of the selected layer
                      break
            # Here, we get the specific filter from the output of the convolution operation
            # x is a tensor of shape 1x512x28x28.(For layer 17)
            # So there are 512 unique filter outputs
            # Following line selects a filter from 512 filters so self.conv_output will become
            # a tensor of shape 28x28
              self.conv_output = x[0, self.selected_filter]
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
              loss = -torch.mean(self.conv_output)
              ##print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
              loss.backward()
            # Update image
              optimizer.step()
            # Recreate image
               #random_image = val_tfms.denorm(img_var.data.cpu().numpy()[0].transpose(1,2,0))
          sz = int(1.2 * sz)  # calculate new image size
          random_image = cv2.resize(random_image, (sz, sz), interpolation = cv2.INTER_CUBIC)
          #random_image = cv2.blur(random_image,(5,5))
        self.created_image = recreate_image(processed_image)
            # Save image
        
        print("saving image")
        im_path = 'generated_filter/' + self.image_name + '_layer' + str(self.selected_layer) + '_filter' + str(self.selected_filter) + '.jpg'
        save_image(self.created_image, im_path)





nets = ["vgg16","alexnet","googlenet","densenet201"]

def input_print(s):
     printf(s)
     return input()

def printf(s):
    print(s, end= '');


class Net:
    def __init__(self,arch,layer,filter):
        self.arch = arch
        self.layer = layer
        self.filter = filter
        

   
    

    def generate(self):
        image = input_print("Choose image name: ")
        image_name = image + '_layer' + str(self.layer) + '_filter' + str(self.filter) + ".jpg"
        print("\nGenerating " + image_name + "...")
        
        pretrained_model = getattr(models,self.arch)(pretrained=True).features
      
        layer_vis = CNNLayerVisualization(pretrained_model, int(self.layer), int(self.filter),image)
        
        layer_vis.visualise_layer_without_hooks(scale=2)
        print("{}.png succesfully generated".format(image_name))

    def write(self):
        print("\n\nArchitecture: {}\nLayer: {}\nFilter {}".format(self.arch,self.layer,self.filter))
        
        printf("\n\nConfirm?(y/n): ")
        if(input() == 'y'):
            self.generate()
        else:
            main()   
                 
def numInput(s):
    while True:
        n = input_print(s)
        if (testNum(n)):
            break
    return n
       
def testNum(n):
    try:
        int(n)
        return True
    except ValueError:
        return False 
def exiting():
    print("Exiting program...")
    exit()

def rcv_init():
    while True:
        input_print('Press enter to select network architecture...')
        fzf = FzfPrompt()
        network = fzf.prompt(nets)
        classes_list = open('classes_names.txt','r').read().split('\n') 
        classes_dict = {}
        for i in range(len(classes_list)):
            classes_dict[classes_list[i]] = i

        input_print('Press enter to select class...') 
        class_name = fzf.prompt(classes_list)[0]
        iterations = numInput("Select number of iterations (1-infinity): ")
        os.system("clear")
        confirm = input_print('Architecture: {}\nClass: {}\nIterations: {}\nConfirm(y/n)? '.format(network[0],class_name,iterations))
        if(confirm == 'n'):
            os.system("clear")
            continue
        regularized_class_sample_generator.run_from_import(classes_dict[class_name],network[0],int(iterations),class_name)
        break

def layer_vis():
    input_print("Choose network: ")
    ##[ printf(nets[i] + ", ") for i in range(len(nets)-1)]
    ##printf(nets[-1] + ": ")
       
    fzf = FzfPrompt()
    a = fzf.prompt(nets)[0]
    print(a);
                
    l = numInput("Choose layer: 0-255: ")
    f = numInput("Choose filter: 0-55: ")
        
    #while True:
        #anim = input_print("Animate optimisation?(y/n): ")
        #if(anim == "y" or anim == 'yes'):
            #anim = True
            #break
        #elif(anim == "n" or anim == 'no'):
            #anim = False
            #break
        
        
    net = Net(a,l,f)
    net.write()
       

def main():

    if not os.path.exists('generated_grad'):
        os.makedirs('generated_grad')
    if not os.path.exists('animation'):
        os.makedirs('animation')
    if not os.path.exists('input_images'):
        os.makedirs('input_images')
        
    os.system("clear")
    print("--------- CNN Visualizer ---------\n")
    print("1. Filter, layer visualisation\n2. Grad-Cam visualisation\n3. Regularized class visualization\n4. Exit")
    while True:
        p_choose = numInput("Choose option: ")
        
        if(int(p_choose) == 1):
            layer_vis()
            exiting()
        elif(int(p_choose) == 2):
            print("GradCam selected.")
            grad_cam.grad_cam_all_layers()
            exiting()
        elif(int(p_choose) == 3):
            print("RCV selected.")
            rcv_init()
            exiting()
        elif(int(p_choose) == 4):
            exiting()
        os.system("clear")
    
    
    
if __name__ == '__main__':
    main()
  
