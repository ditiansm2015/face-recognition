from model import create_model
import cv2
from align import AlignDlib
import numpy as np
import os.path
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
import warnings

#Class to return path of dataset images

class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file)


#Funtion to load metadata of the dataset images

def load_metadata(path):
    metadata = []
    for i in os.listdir(path):
        for f in os.listdir(os.path.join(path, i)):
            # Check file extension. Allow only jpg/jpeg/pngb' files.
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)


# Function to load images

def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]


# Function to generate face allignment

def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), 
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)



#............................................................................................#

print('\n\n\n\nPackage Intializations Completed\n\n\n\n')

#Loading metadata
metadata = load_metadata('images')

# Creating CNN model 
nn4_small2_pretrained = create_model()

print('1. Model Created Successfully')


#Loading pre-trained weights
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')

print('2. Weights Loaded Successfully')


# Initialize the OpenFace face alignment utility
alignment = AlignDlib('models/landmarks.dat')

print('3. Openface utility intialized Successfully')


#Generating the embedding for dataset images
embedded = np.zeros((metadata.shape[0], 128))

for i, m in enumerate(metadata):
    img = load_image(m.image_path())
    img = align_image(img)

    # scale RGB values to interval [0,1]
    img = (img / 255.).astype(np.float32)

    # obtain embedding vector for image
    embedded[i] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]


print('4. Embedding generating successfully')


#Training KNN and SVM classifier for classification
targets = np.array([m.name for m in metadata])

encoder = LabelEncoder()
encoder.fit(targets)

# Numerical encoding of identities
y = encoder.transform(targets)

train_idx = np.arange(metadata.shape[0]) % 2 != 0
test_idx = np.arange(metadata.shape[0]) % 2 == 0

# 50 train examples of 10 identities (5 examples each)
X_train = embedded[train_idx]
# 50 test examples of 10 identities (5 examples each)
X_test = embedded[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
svc = LinearSVC()

knn.fit(X_train, y_train)
svc.fit(X_train, y_train)

acc_knn = accuracy_score(y_test, knn.predict(X_test))
acc_svc = accuracy_score(y_test, svc.predict(X_test))

print(f'KNN accuracy = {acc_knn}, SVM accuracy = {acc_svc}')
print('5. Classifier loaded succesfully')

print('6. Intializing Webcam........')

# Suppress LabelEncoder warning
warnings.filterwarnings('ignore')

#Using trained model for real time predection

x1=1
x2=1
y1=1
y2=1
font=cv2.FONT_HERSHEY_SIMPLEX
embedded = 0;

cap = cv2.VideoCapture(0)
cap.open(0)

while(True):
    try:
        
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        flipped=cv2.flip(frame,1)
    
        bb = alignment.getLargestFaceBoundingBox(flipped)
    
        y1=bb.top()
        x1=bb.left()
        y2=bb.top()+bb.height()
        x2=bb.left()+bb.width()
        x3=bb.left()
        y3=y2+50
            
        aligned = alignment.align(96, flipped, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
        img = (aligned / 255.).astype(np.float32)
        embedded= nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
        example_prediction = svc.predict([embedded])
        example_identity = encoder.inverse_transform(example_prediction)[0]
    
        cv2.putText(flipped,example_identity,(x3,y3),font,1,(0,0,255),1,cv2.LINE_AA)
        cv2.imshow('frame',cv2.rectangle(flipped, (x1,y1 ), (x2, y2), (0,0,255), 2))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    except Exception as e:
        continue

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


print('7. Interface exited')

