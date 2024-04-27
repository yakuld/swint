import cv2
from facenet_pytorch import MTCNN
import os
# from google.colab.patches import cv2_imshow
from tqdm import tqdm

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# Load MTCNN model
detector = MTCNN(device='cuda:1')

# # Open video file
# cap = cv2.VideoCapture('/content/train_00000003.mp4')
loc = '/home/yakul/dataset/DFDC/video_frames/videos'
ratio = 1.0
video_list = [os.path.join(loc, x) for x in os.listdir(loc) if 'video' in x]
fail = 0
shape_fail = 0

for video in tqdm(video_list):
    # ret, frame = cap.read()
    # if not ret:
    #     break

    for file in os.listdir(video):
      # Detect faces in the frame
      frame = cv2.imread(os.path.join(video, file))
      if frame.shape[0] < 300:
        shape_fail += 1
        continue
      try:
        boxes, _ = detector.detect(frame)
      
      
      # if not boxes:
      #   continue
      # Draw bounding boxes around detected faces
        for box in boxes:
            x, y, x1, y1 = [int(i) for i in box]
            width = x1 - x
            height = y1 - y
            # cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            offset_x = width * 0
            offset_y = height * 0
            x -= offset_x
            y -= offset_y
            width += 2 * offset_x
            height += 2 * offset_y
            _ratio = height / width

            if(_ratio < ratio):
              a = (ratio - _ratio) * width
              height += a
              y -= a/2
            else:
              a = ((_ratio - ratio) / ratio) * width
              width += a
              x -= a/2

            x = int(x)
            y = int(y)
            width = int(width)
            height = int(height)
            frame = frame[y : y + height,x : x + width , :]
            os.remove(os.path.join(video, file))
            os.chdir(video)
            cv2.imwrite(file, frame)
            break
        
      except Exception as e:
        fail += 1

print('Done')
print('Shape Fail = ',shape_fail)
print('Exception Fail = ',fail)

