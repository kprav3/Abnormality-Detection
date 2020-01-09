# Abnormality Detection in Musculoskeletal Radiographs

## Deep learning techniques based on convolutional networks to analyse radiographs for abnormality detection to aid radiologists and doctors in choosing adequate therapy.

Abnormalities in the bone include fracture, hardware, dengenrative joint diseases and other abnormalities including lesions and subluxations.

The CRISP-DM process to used to break the process into six major phases

Our product is a medical expert system that can support physicians in diagnosing patients swifty in resourse-poor settings where there generally is a shortage of medical practitioners and delay in providing diagnonsis. 

The Stanford MURA dataset was used with persmission from the Stanford Machine Learning Group. The dataset contained 14,863 patient studies where each study was manually labelled by radiologists as abnormal or normal. The dataset contained 7 bones - Shoulder, Humerus, Elbow, Forearm, Wrist, Hand and Finger. Each study contained 2-3 perspectives of the same bone.

The MURA dataset was arranged under subfolders. Each bone contains a seperate folder inside which carried patient study number appended with positive or negative. Positive stands for positive for abnormality and vice versa. Each patient study has two-three perspectives of the bone. For our training model, we created two folders - positive and negative. 
Using simple queries,positive and negative radiographs for all bones were extracted and placed them in their respective folders. 
Note: The model learnt on all the 7 bones together and not each bone induvidually.

Convolutional neural network architectures were used to train the model and compare them based on various evaluation metrics. 
Models include a simple CNN, tinyVGG, and variants of VGG and ResNet.







