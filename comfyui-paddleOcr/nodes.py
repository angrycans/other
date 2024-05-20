from paddleocr import PaddleOCR, draw_ocr
import numpy as np
from torchvision import transforms
from .utils import tensor2pil,pil2tensor,pil2comfy
import os
import sys
from PIL import Image
import cv2
from paddleocr import PPStructure,draw_structure_result,save_structure_res,draw_ocr
from bs4 import BeautifulSoup


# Load the image using OpenCV
image_path = "path_to_your_image.png"
image = cv2.imread(image_path)



class PaddleOCRNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
               "images": ("IMAGE",),
                "types": (
                    [
                        "ocr",
                        "table",
                    ],
                    {
                        "default": "ocr",
                    },
                ),
               			
            },
        }

    RETURN_TYPES = ("STRING","STRING", "IMAGE")
    RETURN_NAMES = ("Text", "coordinate Text","Out image" )


    FUNCTION = "ocr"

    #OUTPUT_NODE = False


    
 

    def ocr(self,images,types):
        # __dir__ = os.path.dirname(os.path.abspath(__file__))
       
        # #image_path = os.path.join(__dir__, "example.png")
        # print(image_path)


        text = "" 
        coordinate_text = ""
        out_ori_image=None
        if (types=="ocr"):
            print("ocr")
            #image = cv2.imread(image_path)
            #tensorImg = transforms.ToTensor()(image)


            ocr = PaddleOCR()

            for img in images:
                img = tensor2pil(img)
                img = img.convert("RGB")
                # out_ori_image=transforms.ToTensor()(img)
                #out_ori_image=pil2comfy(img)
                # print("world")
                npimg=np.array(img)
                result = ocr.ocr(npimg, cls=False)
                 
                for line in result:
                    for word in line:
                        print(word[1][0])
                        print(word[0])
                        text_content = str(word[1][0]) 
                        text += text_content +"\n" 
                        coordinate_text+=str(word[1][0])+" "+str(word[0] )+"\n"
                
                result = result[0]
                boxes = [line[0] for line in result]
                # txts = [line[1][0] for line in result]
                # scores = [line[1][1] for line in result]
                im_show = draw_ocr(img, boxes)
                im_show = Image.fromarray(im_show)
                out_ori_image=pil2comfy(im_show)

                new_list = [str(x) for x in boxes]
                print(new_list)
        elif (types=="table"):
            print("Table")
            for img in images:
                img = tensor2pil(img)
                img = img.convert("RGB")
                # 将PIL Image转换为NumPy数组
                npimg=np.array(img)
                table_engine = PPStructure(show_log=True)
                # __dir__ = os.path.dirname(os.path.abspath(__file__))
                # image_path = os.path.join(__dir__, "example_table_en.png")
                # img = cv2.imread(image_path)
                img=cv2.cvtColor(npimg, cv2.COLOR_RGB2BGR)
                result = table_engine(img)
                bbox_format2_list = []
                csv_list=[]

                for line in result:
                    line.pop('img')
                
                    if (line['type']=='table'):
                    #if line['type'] == 'table' or line['type'] == 'table_caption':
                        print('table ------->')   
                        print(line)
                        bbox_format1 = line['bbox']
                        x_min, y_min, x_max, y_max = bbox_format1

                        bbox_format2 = [
                            [x_min, y_min],
                            [x_max, y_min],
                            [x_max, y_max],
                            [x_min, y_max]
                        ]
                        #if (line['type']=='')
                        bbox_format2_list.append(bbox_format2)

                       
                        csv_list.append(line['res']['html'])  
                         
                
                print('bbox -------')
                print(bbox_format2_list)

                # text = "" 
                coordinate_text = ""   
                for sublist in bbox_format2_list:
                    line = ""
                    for point in sublist:
                        line += "[" + str(point[0]) + ", " + str(point[1]) + "], "
                    # 移除最后一个逗号并添加换行符
                    line = line[:-2] + "\n"
                    coordinate_text += line

                for subcsvlist in csv_list:
                    text += subcsvlist  +"\n"  
                
                im_show = draw_ocr(img, bbox_format2_list)
                im_show = Image.fromarray(im_show)
                out_ori_image=pil2comfy(im_show)
    

       # return ("hello",tensorImg)
        return (text,coordinate_text,out_ori_image)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "PaddleOCRNode": PaddleOCRNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "PaddleOCRNode": "PaddleOCRNode"
}
