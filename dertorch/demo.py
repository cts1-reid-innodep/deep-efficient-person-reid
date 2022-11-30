import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import glob

from tqdm import tqdm
from queue import Queue
from darknet.darknet.darknet import *
from darknet.darknet.darknet_images import *

videoDB = glob.glob("/home/snu1/sample/maskreid/*.mp4")

if __name__ == '__main__':

    # 0. prepare OD model.
    print("First, initiate OD model.")
    
    # prepare OD
    weights = "/home/snu1/darknet/darknet/gyolo/GYOLOv3_416_136733.weights"
    config_file = "/home/snu1/darknet/darknet/gyolo/GYOLOv3_416_136733.cfg"
    data_file = "/home/snu1/darknet/darknet/gyolo/gyolov3.data"
    network, class_names, class_colors = load_network(
        config_file,
        data_file,
        weights,
        batch_size=1
    )

    # 1. Show Query Video, Get Query Frame

    while True:
        try:
            video_id = int(input("Please type videoID (1~20) which you want : "))
        except:
            print("your input can't be integer.")
            continue
        if video_id not in range(1, 21):
            print("video id not in (1~20)")
        else:
            break

    video_path = videoDB[video_id]

    #cv2.namedWindow('Re-ID', cv2.WINDOW_NORMAL)
    # 1920, 1080 (original)
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    print(f"fps: {fps}")
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            show_frame = cv2.resize(frame, (256, 256))
            cv2.imshow('Re-ID Sample', show_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    frame_count = int(video.get(cv2.CAP_PROP_POS_FRAMES))
    # query_frame = frame[:,:,::-1]
    query_frame = cv2.resize(frame, (416, 416))
    video.release()
    cv2.destroyAllWindows()

    # 2. Get Query Candidates, (by OD) show candidate images.
    query_OD_image, candidates = image_detection(
        frame, network, class_names, class_colors, .25
        )
    print(np.shape(query_frame))
    print(np.shape(query_OD_image))   

    candidate_dict = {}
    for object_name, person_id, box_points in candidates:
        box_points = bbox2points(box_points)
        candidate_dict[person_id] = {'box_points': box_points}
        candidate_dict[person_id]['numpy'] = query_frame[box_points[1]: box_points[3],box_points[0]:box_points[2], :]
        print(f"{person_id} ; {object_name}, box_point {box_points}")


    #show_image = cv2.resize(image, (1024, 512))
    # TODO DISCUSS : 한개 씩 보이게? or 다같이.
    print("Ready to select candidate, press s key.")
    while True:
        cv2.imshow('OD image',query_OD_image)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
    while True:
        candidate_id = input("Which candidate is your pick? (type person_id (ex) 57.94 ): ")
        if candidate_id not in candidate_dict.keys():
            print("candidate id out of range, please check id.")
        else:
            break 

    cv2.destroyAllWindows()

    # while True:
    #     candidate_id = input("Which candidate is your pick? (type person_id (ex) 57.94 ): ")
    #     if candidate_id not in candidate_dict.keys():
    #         print("candidate id out of range, please check id.")
    #     else:
    #         break 
    query_candidate = candidate_dict[candidate_id]
    while True:
        cv2.imshow('query image',query_candidate['numpy'])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    # 3. Get other videos' frames' candidates.
    # TODO DISCUSS : 데모 방식 변경으로 한 영상에서 query뽑아서, 다른 영상에서 추적하는 것으로 변경.
    # 이는 기존과 달리 시간 순으로 모든 영상에서 찾는 것이 어려워, 해당 방법 이용하기로.

    while True:
        try:
            gallery_video_id = int(input("Please type videoID (1~20) where you want to find query: "))
        except:
            print("your input can't be integer")
            continue
        if gallery_video_id not in range(1, 21):
            print("video id not in (1~20)")
        elif gallery_video_id == video_id:
            print("gallery video is equal to query video, please select another video.")
        else:
            break

    gallery_video_path = videoDB[gallery_video_id]
    gallery_frames = []
    gallery_candidate_dict = {}

    gallery_video = cv2.VideoCapture(gallery_video_path)
    print("Video Processing start")
    while gallery_video.isOpened():
        ret, frame = gallery_video.read()

        if ret:
            # 5초마다 프레임 추출
            if (int(gallery_video.get(cv2.CAP_PROP_POS_FRAMES)) % (fps *5)) == 0:
                gallery_frames.append(frame)
        else:
            break
    print("Video Processing end")
    # TODO : get gallery_frame's candidates
    print(f"gallery frame # : {len(gallery_frames)}")
    for i, gallery_frame in enumerate(gallery_frames):
        _, candidates = image_detection(
            gallery_frame, network, class_names, class_colors, .25
            )
        for object_name, person_id, box_points in candidates:
            if person_id in gallery_candidate_dict.keys():
                print("already exist person")
                continue
            box_points = bbox2points(box_points)
            gallery_candidate_dict[person_id] = {'box_points': box_points}
            gallery_candidate_dict[person_id]['numpy'] = query_frame[box_points[1]: box_points[3],box_points[0]:box_points[2], :]
            print(f"{i}th frame {person_id} ; {object_name}, box_point {box_points}")
    print(f"gallery candidate # : {len(gallery_candidate_dict.keys())}")

    # gallery_candidates = [images] (=gallery_boxes)

    # 4. get feature with re-ID model
    query_candidate_tensor = torch.Tensor(cv2.resize(query_candidate['numpy'], (256, 128)))
    gallery_candidates_tensor = [torch.Tensor(cv2.resize(gallery_candidate_dict[gc_key]['numpy'], (256, 128))) for gc_key in gallery_candidate_dict.keys()]
    re_id_inputs = torch.stack([query_candidate_tensor] + gallery_candidates_tensor)
    print(re_id_inputs.shape)
    re_id_model = 
    re_id_model = Backbone(num_classes=len(gallery_candidate_dict.keys()), model_name="resnet50_ibn_a")
    
    # TODO : load test weight 
    re_id_model.load_param("")

    with torch.no_grad():
        _, re_id_outputs = re_id_model(re_id_inputs)
    

    # 5. use re-ranking

    # 6. show top-5 candidate
    