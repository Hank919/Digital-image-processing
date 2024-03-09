from ultralytics import YOLO
import os
import cv2
from PIL import Image
import numpy
import numpy as np
# 加载预训练的YOLOv8n-seg分割模型
model = YOLO('YOLOv8_pre_weight.pt')




# source = Image.open('./training_dataset/image/47.jpg')
folder_path = './training_dataset/image/'

# 取得資料夾中的所有檔案
image_files = os.listdir(folder_path)
i=0
for image_file in image_files:
    print(image_file)
     # 組合完整的影像檔案路徑
    image_path = os.path.join(folder_path, image_file)
    source = cv2.imread(image_path)
    # 使用 YOLO 模型進行預測
    results=model.predict(image_path, save=False, conf=0.1)
# results = model.predict(source, save=False, conf=0.1, show_labels = False)
    H,W,_ = source.shape
    total_mask = np.zeros((H, W))
    print(H)
    print(W)
    for result in results:
      im_array = result.plot(boxes = False, labels = False,probs =True)
      im = Image.fromarray(im_array[...,::-1])
      im.save(f"./training_dataset/Detect_origin_DL/{image_file}")

    if(results[0].masks != None):
      for j,mask in enumerate(result.masks.data):
        mask = mask.cpu().numpy()*255
        mask = cv2.resize(mask,(W,H))
        # print(mask.shape)
        total_mask += mask
        # cv2.imwrite(f"./training_dataset/black_mask/{image_file}",mask)
        
    cv2.imwrite(f"./training_dataset/Detect_mask_DL/{image_file}",total_mask)
    i+=1

# folder_path = './training_dataset/image/'

# # 取得資料夾中的所有檔案
# image_files = os.listdir(folder_path)

# # 迭代處理每個影像檔案
# for image_file in image_files:
#     # 組合完整的影像檔案路徑
#     image_path = os.path.join(folder_path, image_file)
    
#     # 使用 YOLO 模型進行預測
#     model.predict(image_path, save=True, conf=0.5)