from feat.detector import Detector
import cv2
import pandas as pd
import load_config as config


class FacialService:

  loadConfig = config.yaml_load().get("feat-fy")
  faceModel = loadConfig['face_model']
  landmarkModel = loadConfig['landmark_model']
  auModel = loadConfig['au_model']
  facePoseModel = loadConfig['facepose_model']
  emotionModel = loadConfig['emotion_model']
  identityModel = loadConfig['identity_model']


  def __init__(self):
    self.model = Detector(
    face_model = self.faceModel,
    landmark_model = self.landmarkModel,
    au_model = self.auModel,
    emotion_model = self.emotionModel,
    facepose_model= self.facePoseModel,
    identity_model = self.identityModel
    )


  """
  Method to extraction facial data from detections 

  Parameters
  ----------
  img: is path file input

  Return
  ------
  Frame Data related AUS and Head
  """
  def extractionFacial(self, person):
    landmarks = self.detectLandmarks(person)
    dataAu = self.detectAus(frame = person, landmarks = landmarks)
    dataAu.rename(columns = {0:"AU01",1:"AU02",2:"AU03",3:"AU04",4:"AU05",5:"AU06",6:"AU07",7:"AU08"
      ,8:"AU09",9:"AU10",10:"AU11",11:"AU12",12:"AU13",13:"AU14",14:"AU15",15:"AU16",16:"AU17",17:"AU18",18:"AU19",19:"AU20"}, inplace = True)

    dataPose = self.detectFacePose(person, landmarks)
    dataPose.rename(columns = {0:"Pitch", 1:"Roll", 2:"Yaw"}, inplace = True)

    dataFacial = pd.concat([dataAu,dataPose], axis = 1)
    if type(dataFacial) == list:
      dataFacial = pd.DataFrame(dataFacial)
    return dataFacial


  def detectFaces(self, frame):
    return self.model.detect_faces(frame)

  def detectLandmarks(self, frame):
    detected_faces = self.detectFaces(frame = frame)
    return self.model.detect_landmarks(frame = frame, detected_faces = detected_faces)

  def detectFacePose(self, frame, landmarks=None):
    data = self.model.detect_facepose(frame = frame, landmarks = landmarks)['poses']
    if type(data) == list:
      return pd.DataFrame(data.pop())
    return data

  def detectAus(self, frame, landmarks):
    data = self.model.detect_aus(frame = frame, landmarks = landmarks)
    if type(data) == list:
      return pd.DataFrame(data.pop())
    return data





