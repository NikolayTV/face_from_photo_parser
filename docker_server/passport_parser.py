from pathlib import Path
from glob import glob
from tqdm import tqdm
from mtcnn.mtcnn import MTCNN
import cv2
import fnmatch
import os
import re



def resize_by_height(img, new_height):
    '''
    Input: image, new_height
    Returns: image with new height and scaled width
    '''
    height, width = img.shape[:2]
    resize_multiple = new_height / height
    img = cv2.resize(img, None, fx=resize_multiple, fy=resize_multiple, interpolation=cv2.INTER_AREA)
    return img


def get_face_crop(img):
    results = anfas_detector.detect_faces(img)
    if results == []: 
        return results, None

    x, y, w, h = results[0]['box']

    y_border, x_border = img.shape[:2]
    x1, y1, x2, y2 = max(0, x), max(0, y), min(x_border, x+w), min(y_border, y+h)

    face_crop = img[y1:y2, x1:x2]
    return face_crop, round(results[0]['confidence'], 2)


def get_all_fnames(base_folder):
    all_fnames = glob(str(Path(base_folder, '**', '*')), recursive=True)
    all_imgs = []
    patterns = ['*jpg', '*jpeg', '*png']

    for pattern in patterns:
        match = re.compile(fnmatch.translate(pattern), re.IGNORECASE).match
        valid_pths = [pth for pth in all_fnames if match(pth)]
        all_imgs.extend(valid_pths)
                      
    return all_imgs
         
    


anfas_detector = MTCNN(steps_threshold = [0.4, 0.6, 0.6], min_face_size = 100)

base_folder = '../example'

result_base_folder = Path(f'{base_folder}_result')
if not os.path.exists(result_base_folder):
    os.mkdir(result_base_folder)
    
all_imgs = get_all_fnames(base_folder)

print(f'Найдено изображений: {len(all_imgs)}')
for num, img_path in enumerate(tqdm(all_imgs[:])):
    try:
        if num % 100 == 0:
            with open('progress.txt', 'w') as f:
                f.write(str(num))

        folder_name = Path(img_path).parts[-2]
        img_name = Path(img_path).parts[-1]
        
        img = cv2.imread(img_path)
        if img.shape[0] > img.shape[1]:
            img = cv2.rotate(img, rotateCode = 0)

        img = resize_by_height(img, 820)

        face_crop, confidence = get_face_crop(img)
        if len(face_crop) == 0: 
            img = cv2.rotate(img, rotateCode = 0)
            face_crop, confidence = get_face_crop(img)
            if len(face_crop) == 0: 
                img = cv2.rotate(img, rotateCode = 0)
                face_crop, confidence = get_face_crop(img)
                if len(face_crop) == 0: 
                    img = cv2.rotate(img, rotateCode = 0)
                    face_crop, confidence = get_face_crop(img)
                    continue
                    
            
        if not os.path.exists(result_base_folder/folder_name):
            os.mkdir(result_base_folder/folder_name)
            
        cv2.imwrite(str(result_base_folder/folder_name/Path(str(confidence)+'_'+img_name)), face_crop)
        

    except Exception as exp:
        print(exp, img_path)