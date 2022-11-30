import argparse
import imutils
import cv2
import numpy as np
from imutils import contours
from pytesseract import image_to_string, pytesseract
from pdf2image import convert_from_path
import os
import configparser


def debug(img, path, name):
    if not CONFIG['debug']:
        return False
    elif not path:
        return False
    elif not os.path.exists(path):
        os.mkdir(path)
    cv2.imwrite(path+'/'+str(name)+'.jpg', img) 

def debug_message(msn):
    if not CONFIG['debug']:
        return False
    with open('debug.txt', 'a') as f:
         f.write(str(msn)+'\n')

def get_top_code_area(image, debug_path):
    img_code = image[
                    int(CONFIG['y1_code_area']):int(CONFIG['y2_code_area']), 
                    int(CONFIG['x1_code_area']):int(CONFIG['x2_code_area'])
                    ]
    return img_code


def get_top_code(img_code, debug_path):
    debug(img_code, debug_path, 'code')
    # Grayscale, Gaussian blur, Otsu's threshold
    #image = cv2.imread('1.png')
    gray = cv2.cvtColor(img_code, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Morph open to remove noise and invert image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening
    return image_to_string(invert).strip()[4:]


def get_bubble_area(image, debug_path=False):
    bubble_img = image[
                        int(CONFIG['y1_bubble_area']):int(CONFIG['y2_bubble_area']),
                        int(CONFIG['x1_bubble_area']):int(CONFIG['x2_bubble_area'])
                    ]
    debug(bubble_img, debug_path, 'bubble')
    return bubble_img


def slice_5_regions_bubble(bubble_area_image, debug_path=False):
    slices = []
    
    slice1 = bubble_area_image[
                int(CONFIG['y1_slice1_area']):int(CONFIG['y2_slice1_area']),
                int(CONFIG['x1_slice1_area']):int(CONFIG['x2_slice1_area'])
            ]
    slice2 = bubble_area_image[
                int(CONFIG['y1_slice2_area']):int(CONFIG['y2_slice2_area']),
                int(CONFIG['x1_slice2_area']):int(CONFIG['x2_slice2_area'])
            ]
    slice3 = bubble_area_image[
                int(CONFIG['y1_slice3_area']):int(CONFIG['y2_slice3_area']),
                int(CONFIG['x1_slice3_area']):int(CONFIG['x2_slice3_area'])
            ]
    slice4 = bubble_area_image[
                int(CONFIG['y1_slice4_area']):int(CONFIG['y2_slice4_area']),
                int(CONFIG['x1_slice4_area']):int(CONFIG['x2_slice4_area'])
            ]
    slice5 = bubble_area_image[
                int(CONFIG['y1_slice5_area']):int(CONFIG['y2_slice5_area']),
                int(CONFIG['x1_slice5_area']):int(CONFIG['x2_slice5_area'])
            ]

    slices.append(slice1)
    debug(slice1, debug_path, 'slice1')

    slices.append(slice2)
    debug(slice2, debug_path, 'slice2')

    slices.append(slice3)
    debug(slice3, debug_path, 'slice3')

    slices.append(slice4)
    debug(slice4, debug_path, 'slice4')

    slices.append(slice5)
    debug(slice5, debug_path, 'slice5')

    return slices


def resize_image(img):
    h, w, _ = img.shape
    img = cv2.resize(img, (int(w/2), int(h/2)))
    return img

def get_question_image(img_slice, question_position):
    dimensions = img_slice.shape
    height = img_slice.shape[0]
    width = img_slice.shape[1]
    question_height = int(height/18)
    pos1 = (question_height*(question_position))+question_position
    pos2 = (question_height*(question_position+1))+question_position
    question_img = img_slice[pos1:pos2]
    return question_img

def find_question_bubble_contours(img_slice, question_number, question_position, debug_path=False):
    question_img = get_question_image(img_slice, question_position)
    question_number = question_number+1
    #get the question contours
    gray = cv2.cvtColor(question_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    edged = cv2.Canny(blurred, 75, 200)
    thresh = cv2.threshold(blurred, 0, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(c.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    #identify the bubbles
    questionCnts = []
    bubblesCount = 0
    for c in cnts:
        # compute the bounding box of the contour, then use the
        # bounding box to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # in order to label the contour as a question, region
        # should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        #debug_message(str(cont)+' '+str(w)+' '+str(h)+' AR:'+str(ar))
        if w >= 40 and h >= 40 and w <= 90 and h <= 90 and ar >= 0.5 and ar <= 1.2:
            #print(w, h, ar)
            bubblesCount = bubblesCount + 1
            questionCnts.append(c)

    questionCnts = contours.sort_contours(questionCnts)[0]
    
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, questionCnts, -1, 255, -1)
    debug(mask,debug_path, 'edged-cnts-q'+str(question_number))
    return {'questionCnts': questionCnts, 'bubblesCount':bubblesCount}


def get_filled_bubble(img_slice, question_number, question_position, questionCnts, bubblesCount, debug_path):
    def opcaoescolhida(opcao):
        if opcao==0:
            return 'A'
        elif opcao==1:
            return 'B'
        elif opcao==2:
            return 'C'
        elif opcao==3:
            return 'D'
        elif opcao==4:
            return 'E'
        else:
            return opcao


    question_number = question_number+1
    question_img = get_question_image(img_slice, question_position)
    gray = cv2.cvtColor(question_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    thresh = cv2.threshold(blurred, 0, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C | cv2.THRESH_OTSU)[1]
    
    opcaoEscolhida = '_'
    cnts_opcao = []

    if bubblesCount != 5:
        opcaoEscolhida = input("Não consegui ler. Informe a resposta: ")
    else:
        for (opcao, cnts) in enumerate(questionCnts):
            # construct a mask that reveals only the current
            # "bubble" for the question
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [cnts], -1, 255, -1)
            debug(mask, debug_path, str(question_number)+'-MASK'+str(opcao))
            debug(thresh, debug_path, str(question_number)+'-TH'+str(opcao))
            
            #show_images('teste', [mask])
            # apply the mask to the thresholded image, then
            # count the number of non-zero pixels in the
            # bubble area
            #debug(thresh, debug_path, str(count)+'-'+str(opcao))
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            debug_message(str(opcao)+' '+str(total))
            #print(opcaoEscolhida, escolhidaAnterior)
            if total> int(CONFIG['filled_bubble_sensibility']):                    
                if opcaoEscolhida!='_':
                    #opcaoEscolhida = 
                    opcaoEscolhida = input("Dúvida entre opções "+opcaoescolhida(opcao)+" ou "+opcaoescolhida(escolhidaAnterior)+": ")
                    #opcaoEscolhida = '*'
                else:
                    opcaoEscolhida = opcao
                    cnts_opcao = cnts
                    escolhidaAnterior = opcao

    if opcaoEscolhida=='_':
        opcaoEscolhida = input("Informe a resposta: ")
    retorno = opcaoescolhida(opcaoEscolhida)
    debug_message('Opcao escolhida :'+retorno)
    debug_message('-------------------')
    print('Opção Escolhida: '+retorno)
    return retorno
    


def get_replies(image, page, debug_path=False):
    
    #resize slice in 50% - better results
    image = resize_image(image)

    #get the card code on top
    img_code = get_top_code_area(image, debug_path)
    code = get_top_code(img_code, debug_path)
    replies = code
    print('Matricula '+code)
    debug_message("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    debug_message(code)
    
    #buble area
    img_bubble_area = get_bubble_area(image, debug_path)

    #slice buble area
    img_slices = slice_5_regions_bubble(img_bubble_area, debug_path)

    question_position = 0
    for question_number in range(int(number_of_questions)):    
        debug_message('Questão: '+str(question_number+1)+' => Slice:'+str(int((question_number)/18))+' - Position '+str(question_position))
        print('Matricula '+code+'=> Questão: '+str(question_number+1))
        qcnts = find_question_bubble_contours(img_slices[int((question_number)/18)], question_number, question_position, debug_path)
        replies = replies+','+str(get_filled_bubble(img_slices[int((question_number)/18)], question_number, question_position, qcnts['questionCnts'], qcnts['bubblesCount'], debug_path))
        if question_position==17:
            question_position=0
        else:
            question_position = question_position+1

        

    return replies

def load_template_conf(template_file):
    config = configparser.ConfigParser()
    config.read(template_file)
    confs = {}
    for section in config.sections():
        for key in config[section]:
            confs[key] = config[section][key]
    return confs

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", required=True,
	help="path to the input PDF")
ap.add_argument("-q", "--questions", required=True,
	help="number of questions")
ap.add_argument("-t", "--template", required=True,
	help="template INI file")
args = vars(ap.parse_args())

number_of_questions = args['questions']

CONFIG = load_template_conf(args['template'])

path = CONFIG['path']
poppler_path = CONFIG['poppler_path']
#poppler_path =path+'poppler-22.11.0/Library/bin/'
img_path = CONFIG['img_path']
pytesseract.tesseract_cmd = CONFIG['tesseract_cmd']
print(poppler_path)

pages = convert_from_path(args["file"], 500, poppler_path=poppler_path)

cont = 0

for page in pages:
    cont = cont + 1
    img = img_path+str(cont)+'/OUT.jpg'
    if not os.path.exists(img_path+str(cont)):
        os.mkdir(img_path+str(cont))
    #print(img)
    page.save(img, 'JPEG')
    image = cv2.imread(img)
    #print(get_replies(image, img_path+str(cont)))
    with open('respostas.csv', 'a') as f:
         f.write(get_replies(image, cont, img_path+str(cont))+'\n')
