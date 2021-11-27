import cv2
import os
from PIL import Image
import imquality.brisque as brisque
import PIL.Image
import pytesseract
import numpy as np
import easyocr
from matplotlib import pyplot as plt

#for text similarty
from difflib import SequenceMatcher

#for path
import glob
import os

import re
import string

#for excel
import xlwt
from xlwt import Workbook

#for performance
import time



print("===Process Started=== \n")

original_text = "14. Bir toplantıda herkes birbiriyle tokalaşmıştır. Toplam 66 tokalaşma olduğuna göre, toplantıda kaç kişi vardır? A) 8 B) 9 C) 10 D) 11 E) 12"

current_path = os.getcwd()
tesseract_path = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

#Images
images_file_name = "Images"
images_file_path = os.path.join(current_path, images_file_name)
image_format = "*.jpg"

#Result
result_file_name =  "Result"
result_file_path = os.path.join(current_path, result_file_name)
excel_file_name = "Result.xls"


#Control requirements
isExistImageFile = os.path.exists(images_file_path)
if not(isExistImageFile):
    print("Warning! Missing Images file")
    print("Creating Images file")
    os.makedirs(images_file_path)

isExistResultFile = os.path.exists(result_file_path)
if not(isExistResultFile):
    print("Warning! Missing Result file")
    print("Creating Result file")
    os.makedirs(result_file_path)    


#Excel Workbook
wb = Workbook()
sheet = wb.add_sheet(result_file_name)
#Header
header_style = xlwt.easyxf('font: bold 1, color red;')
sheet.write(0, 0, 'Input',header_style)
header_style = xlwt.easyxf('font: bold 1, color green;')
sheet.write(0, 5, 'Output',header_style)
header_style = xlwt.easyxf('font: bold 1, color blue;')
sheet.write(0, 10, 'Result',header_style)
#Input
sheet.write(1, 0, 'Name')
sheet.write(1, 1, 'Path')
sheet.write(1, 2, 'Size')
sheet.write(1, 3, 'Text')
#Output
sheet.write(1, 5, 'Name')
sheet.write(1, 6, 'Path')
sheet.write(1, 7, 'Size')
sheet.write(1, 8, 'Text')
#Result
sheet.write(1, 10, 'Similarty')
sheet.write(1, 11, 'Process Time')



image_path_array = glob.glob(os.path.join(images_file_path, image_format))

pytesseract.pytesseract.tesseract_cmd = tesseract_path

def punctuation(input_text):
    line = input_text.strip('\n')
    line = line.strip('\t')
    return line

#EasyOCR
def EasyOCR(path,lang):
    reader = easyocr.Reader(['tr'], gpu=False)
    easy_ocr = reader.readtext(Image.open(path),paragraph="False", detail=0)
    return easy_ocr

#Image Quality Saver
def ImageQuality(path,quality):
    temp_quality_image = Image.open(path)
    new_path = os.path.join(result_file_path, os.path.basename(path))
    temp_quality_image.save(os.path.join(result_file_path, os.path.basename(path)),quality=quality)
    return new_path

#Compare Similarty
def CompareSimilarty(original,test):
    return SequenceMatcher(None, original, test).ratio()

 #Compare Similarty
def show(header,img):
    cv2.imshow(header,img)
    cv2.waitKey(0)

#Console Write
def output(header,inner):
    print(header)
    print(inner)
    print("\n")
   


idx = 1
for path in image_path_array:
    
    idx = idx + 1   

    output("Original Text",original_text)

    #Process started
    start_time = time.time()

    converted_text = EasyOCR(path,"tr")
    result_st = ""
    for index in converted_text:
        result_st = result_st + index + " "

    punch_text = punctuation(result_st)
    output("Punch Text",punch_text)
    #Process ending
    end_time = time.time()

    compared_ratio = CompareSimilarty(original_text,punch_text)
    output("Similarty Result",compared_ratio)
    
    new_image_path = ImageQuality(path,60)


    sheet.write(idx, 0, os.path.basename(path))
    sheet.write(idx, 1, path)
    sheet.write(idx, 2, os.path.getsize(path))
    sheet.write(idx, 3, original_text)
    sheet.write(idx, 5, os.path.basename(new_image_path))
    sheet.write(idx, 6, new_image_path)
    sheet.write(idx, 7, os.path.getsize(new_image_path))
    sheet.write(idx, 8, punch_text)
    sheet.write(idx, 10, compared_ratio)
    sheet.write(idx, 11, format(end_time-start_time, '.2f'))



wb.save(os.path.join(result_file_path, excel_file_name))



print("\n ===Process Done===")

# #time using for performance test
# print("Process Starting")
# start_time = time.time()




# #a variable for iteration
# a=0


# for i in (file_path_array):

#     #increase iteration to save images
#     a=a+1

#     #read the image
#     i = cv2.imread(i,0)


#     #save the image to determined folder
#     cv2.imwrite("D:/output/" + str(a) + ".jpg",i)
   
    
#     #take the determined folder path to save the image with new quality
#     img_path = "D:/output/" + str(a) + ".jpg"
#     image_file = Image.open(img_path) 
    
    
#     #change the quality of image
#     image_file.save(r"D:/output/" + str(a) + ".jpg"  , quality=60)


#     #get path of image to resize
#     img = "D:/output/" + str(a) + ".jpg"
#     src= cv2.imread(img,0)

    
#     #resizing
#     scale_percent = 60 #percent of original size
#     width = int(src.shape[1] * scale_percent / 400)
#     height = int(src.shape[0] * scale_percent / 400)
#     dim = (width, height)


#     #apply resize process to image
#     img = cv2.resize(src, dim, interpolation = cv2.INTER_AREA)
  
   
#     #image processing
#     filtered = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41,27)
#     #kernel for process
#     kernel = np.ones((1, 1), np.uint8)
#     #first
#     opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
#     closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
   
#     ret1, th1 = cv2.threshold(closing, 0, 255, cv2.THRESH_BINARY)
#     ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     blur = cv2.GaussianBlur(th2, (3, 3), 0)
#     ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     or_image = cv2.bitwise_or(th3, closing)
        
    
#     #write the image with new size and filters
#     cv2.imwrite(r"D:/output/" + str(a) + ".jpg",or_image)
    
    
#     #get the image file and save the images
#     image_file = Image.open(img_path)
#     image_file.save(r"D:/output/" + str(a) + ".jpg"  , quality=60)


# #we will write the process detail to excel file
# #determine a folder path to fill excel file
# file_path_array_for_excel = glob.glob("D:/output/*.jpg")


# #for performance test
# end_time = time.time()
# cycle_time = end_time-start_time
# print(len(file_path_array), "data processing took", format(cycle_time, '.2f'), "sec")
# score = cycle_time/len(file_path_array)
# print("performance score per proccess: ", format(score, '.2f'), "sec")


# #download source and create word set for TurkishNLP 
# obj.download()
# obj.create_word_set()


# #fill the excel file
# for idx, val in enumerate(file_path_array_for_excel):

#     #get the image from val for score
#     imgForScore = cv2.imread(val,0)     
    
#     #string correcting with TurkishNLP
#     lwords = obj.list_words(pytesseract.image_to_string(Image.open(val), lang="tur"))
#     corrected_words = obj.auto_correct(lwords)
#     corrected_string = " ".join(corrected_words)
    
#     #for easyocr
#     reader = easyocr.Reader(['tr'], gpu=True)
#     easy_ocr = reader.readtext(Image.open(val),paragraph="False", detail=0)
    
    
#     sheet.write(idx + 1, 0, val)
#     sheet.write(idx + 1, 1, os.path.basename(val))
#     sheet.write(idx + 1, 2, pytesseract.image_to_string(i))
#     sheet.write(idx + 1, 3, pytesseract.image_to_string(Image.open(val), lang="tur"))
#     sheet.write(idx + 1, 4, corrected_string)
#     sheet.write(idx + 1, 5, easy_ocr)
#     sheet.write(idx + 1, 6, "yaz")
#     sheet.write(idx + 1, 7, format(brisque.score(imgForScore)))


# wb.save('document.xls')


# print("Process Done!")


