# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 09:12:46 2021

@author: yusuf
"""

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

#for test
import time


print("===STARTED=== \n")

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
sheet.write(0, 0, 'Path')
sheet.write(0, 1, 'Name')
sheet.write(0, 2, 'Orijinal Text')
sheet.write(0, 3, 'Tesseract')
sheet.write(0, 4, 'Tesseract and TNLP')
sheet.write(0, 5, 'Easyocr')
sheet.write(0, 6, 'Accuracy')
sheet.write(0, 7, 'Quality')
wb.save(os.path.join(result_file_path, excel_file_name))



image_path_array = glob.glob(os.path.join(images_file_path, image_format))

pytesseract.pytesseract.tesseract_cmd = tesseract_path

#punctuation boşluk temizler ve noktalama işareti ekler.
#boto3 noktalamalara dikkat eder.
#cümle sonlarında nokta var ise doğruluk oranı artar.
def punctuation(input_text):
    line = input_text.strip('\n')
    line = line.strip('\t')
    # removeWhiteSpace = re.sub(' +', ' ', res)
    # removeLastWhiteSpace = removeWhiteSpace.rstrip();
    # return removeLastWhiteSpace
    return line

#Tesseract OCR
def ConvertOCR(path,lang):
    return pytesseract.image_to_string(Image.open(path), lang=lang)

#Tesseract OCR
def EasyOCR(path,lang):
    reader = easyocr.Reader(['tr'], gpu=True)
    easy_ocr = reader.readtext(Image.open(path),paragraph="False", detail=0)
    return easy_ocr


#Image Quality Saver
def ImageQuality(path,quality):
    temp_quality_image = Image.open(path)
    temp_quality_image.save(os.path.join(result_file_path, os.path.basename(path))  , quality=quality)

#Compare Similarty
def CompareSimilarty(original,test):
    return SequenceMatcher(None, original, test).ratio()

 #Compare Similarty
def show(header,img):
    cv2.imshow(header,img)
    cv2.waitKey(0)




for path in image_path_array:
    
  
    #test = cv2.imread(path,0)
    test = cv2.imread(path)
    # cv2.imshow('image',test)
    # cv2.waitKey(0)
    print("original_text :",original_text)
    print("================================")

    converted_text = ConvertOCR(path,"tur")
    print("converted_text :",converted_text)
    print("================================")

    punch_text = punctuation(converted_text)
    print("punch_text:",punch_text)
    print("================================")

    compared_ratio = CompareSimilarty(original_text,punch_text)
    print("Tesseract Similarty:",compared_ratio)


    converted_text = EasyOCR(path,"tr")
  

    result_st = ""
    for index in converted_text:
        result_st = result_st + index + " "

    print("converted_EASY_text :", result_st)
    compared_ratio = CompareSimilarty(original_text,result_st)
    print("EasyOCR Similarty:",compared_ratio)


    #IMAGE PROCESS
    #gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    #show("gray",gray)
    # blur = cv2.GaussianBlur(gray, (3, 3), cv2.BORDER_DEFAULT)
    # show("blur",blur)
    # filtered = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41,27)
    # kernel = np.ones((1, 1), np.uint8)
    # opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)   
    # ret1, th1 = cv2.threshold(closing, 100, 255, cv2.THRESH_BINARY)
    # ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   
    # blur = cv2.GaussianBlur(th2, (11, 5), cv2.BORDER_DEFAULT)
    # ret3, th3 = cv2.threshold(th2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # or_image = cv2.bitwise_or(th3, closing)
    #temp =  cv2.addWeighted(test,1.5,test,-0.5,0,test)
    #kaydet
    # saved = temp
    # cv2.imwrite(os.path.join(result_file_path, os.path.basename(path)),saved)
    # converted_text = ConvertOCR(os.path.join(result_file_path, os.path.basename(path)),"tur")
    # print(converted_text)
    # compared_ratio = CompareSimilarty(original_text,converted_text)
    # print(compared_ratio)    
    # cv2.imshow('image',saved)
    # cv2.waitKey(0)
    
    
    
    ImageQuality(path,60)









print("\n ===DONE===")

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


