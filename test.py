import torch
import torchvision.transforms as standard_transforms
from torch.autograd import Variable

from networks.DPPNet import DPPNet
from networks.Post_processing import Post_processing

import cv2
import numpy as np

import getopt
import sys
import math
import time

arguments_strModelStateDict = './weights/DOPNet_RSOC.pth'
arguments_strImg = './image/P1829.png'
arguments_strOut = './out/result.png'

color_bar = [(0,0,255),(0,255,0)]

arguments_intDevice = 0

mean_std =  ([0.3707408905029297, 0.37291136384010315, 0.3479439616203308], [0.19950968027114868, 0.19257844984531403, 0.19008085131645203])

img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

for strOption, strArgument in \
getopt.getopt(sys.argv[1:], '', [strParameter[2:] + '=' for strParameter in sys.argv[1::2]])[0]:
    if strOption == '--device' and strArgument != '': arguments_intDevice = int(strArgument)  # device number
    if strOption == '--model_state' and strArgument != '': arguments_strModelStateDict = strArgument  # path to the model state
    if strOption == '--img_path' and strArgument != '': arguments_strImg = strArgument  # path to the image
    if strOption == '--out' and strArgument != '': arguments_strOut = strArgument  # path to where the output should be stored

torch.cuda.set_device(arguments_intDevice)

#  If you want to test the average inference speed, please rewrite it yourself to read the images of the whole dataset.
def evaluate(img_path, save_path):
    net = DPPNet(num_classes=2)
    
    net.load_state_dict(torch.load(arguments_strModelStateDict, map_location=lambda storage, loc: storage)['model'], strict=False)
    net.cuda()
    net.eval()

    post_processing  = Post_processing(num_classes=2)
    post_processing.cuda()
    post_processing.eval()
    
    img_ori = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_ori = np.array(img_ori)

    img = img_transform(img_ori)

    img = img.view(1, img.size(0), img.size(1), img.size(2))
    w = math.ceil(img.shape[2] / 16) * 16
    h = math.ceil(img.shape[3] / 16) * 16
    data_list = torch.FloatTensor(1,3,int(w),int(h)).fill_(0)
    data_list[:,:,0:img.shape[2],0:img.shape[3]] = img
    img = data_list

    with torch.no_grad():
        
        img = Variable(img).cuda()
        torch.cuda.synchronize() 
        start_time = time.time()
        outputs, offset1, offset_fin = net.forward(img)
        cnt, result, _ = post_processing(outputs, offset1, offset_fin)
        torch.cuda.synchronize()
        end_time = time.time()
        diff_time = start_time - end_time

    for index in range(len(result)):
        res = result[index]
        for i in range(res.shape[0]):
            img_ori = cv2.circle(img_ori, (int(res[i,0].item()), int(res[i,1].item())), 5, color_bar[index], -1)
    cv2.imwrite(save_path, img_ori)

    print("small vehicle number: {} , large vehicle number: {}".format(cnt[0], cnt[1]))

    print("save pred density map in {} success".format(save_path))

    print("end")


if __name__ == '__main__':
    evaluate(arguments_strImg, arguments_strOut)
