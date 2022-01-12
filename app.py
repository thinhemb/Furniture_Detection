import torch
import streamlit as st
import numpy as np
import time
import cv2
from PIL import Image
import os
import io
labels= {
    0: "Bed",
    1: "Table",
    2: "Couch",
}   
def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

@st.cache
def load_img(file):
    images=Image.open(file)
    return images




def predict(image_dir,model_path):
    
    model = torch.load(model_path,map_location='cpu')

    model.training = False
    model.eval()
    
    image= cv2.imread(image_dir)
    image_orig = image.copy()

    rows, cols,cns= image.shape
     
    smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
    min_side = 608
    max_side = 1024
    scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
    largest_side = max(rows, cols)

    if largest_side * scale > max_side:
        scale = max_side / largest_side

        # resize the image with the computed scale
    image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
    rows, cols, cns = image.shape

    pad_w = 32 - rows % 32
    pad_h = 32 - cols % 32

    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = image.astype(np.float32)
    image = new_image.astype(np.float32)
    image /= 255
    image -= [0.485, 0.456, 0.406]
    image /= [0.229, 0.224, 0.225]
    image = np.expand_dims(image, 0)
    image = np.transpose(image, (0, 3, 1, 2))

    with torch.no_grad():

        image = torch.from_numpy(image)
        # if torch.cuda.is_available():
        #     image = image.cuda()
        s = time.time()
        # print(image.shape, image_orig.shape, scale)
        scores, classification, transformed_anchors = model(image.float())
        print('Elapsed time: {}'.format(time.time() - s))
        id = np.where(scores.cpu() > 0.5)

        for j in range(id[0].shape[0]):
            bbox = transformed_anchors[id[0][j], :]

            x1 = int(bbox[0] / scale)
            y1 = int(bbox[1] / scale)
            x2 = int(bbox[2] / scale)
            y2 = int(bbox[3] / scale)
            label_name = labels[int(classification[id[0][j]])]
            print(bbox, classification.shape)
            score = scores[j]
            caption = '{} {:.3f}'.format(label_name, score)
                # draw_caption(img, (x1, y1, x2, y2), label_name)
            draw_caption(image_orig, (x1, y1, x2, y2), caption)
            cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            st.write(label_name)
        dir='./result/'+image_dir.split('/')[-1]
        # print(dir)
        cv2.imwrite(dir, image_orig)
        if image_orig is not None:
            img=Image.open(dir)
            st.image(img, caption='Show image')
    
    # if st.button('Dectection'):
        # st.write('True')

        # 
if __name__ == '__main__':
    st.title("Nhận diện đồ nội thất")
    
    file= st.file_uploader("Upload your file",type=['png','jpg'])
    image_dir=''
    if file is not None:
        img=load_img(file)
        st.image(img, caption='Your photo want to recognize',width=500)
        # st.write(dir(file))
        image_dir=os.path.join("./fileDir",file.name)
        with open(image_dir,"wb") as f:
            f.write((file).getbuffer())
			
        st.success("file successful")
    model_path='./retinanet_30 epochs.pt'
    if image_dir is not '':
        predict(image_dir,model_path)
    st.image("label_train.png",width=500)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("Bed")
        st.image('./result/378detections.jpg')

    with col2:
        st.header("Table")
        st.image("./result/27detections.jpg")

    with col3:
        st.header("Couch")
        st.image("./result/58detections.jpg")