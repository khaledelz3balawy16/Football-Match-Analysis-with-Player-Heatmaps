#Team classifier Model 
import supervision as sv
from tqdm import tqdm
from TeamSplit import TeamClassifier
from ultralytics import YOLO

model_path = r"E:\Football Analysis system\Parameters\Player.pt"
video_path = r"E:\Football Analysis system\Video\uc.mp4"
PLAYER_ID = 2
STRIDE = 5

frame_generator = sv.get_video_frames_generator(source_path=video_path, stride=STRIDE)

crops = []
for frame in tqdm(frame_generator, desc='collecting crops'):
    model = YOLO(model_path)
    result = model.predict(frame,device='cuda')[0]
    detections = sv.Detections.from_ultralytics(result)
    detections =detections[detections.confidence>0.3]
    players_detections = detections[detections.class_id == PLAYER_ID]
    players_crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
    crops += players_crops

team_classifierrr = TeamClassifier(device="cuda")
team_classifierrr.fit(crops)

#Save Model 
import pickle 
classifier_save_path = r"E:\Football Analysis system\Parameters\team_classifier2.pkl"  # Path to save the classifier
# Save the trained team classifier to a file
with open(classifier_save_path, 'wb') as f:
    pickle.dump(team_classifierrr, f)

print(f"Team classifier saved to {classifier_save_path}")