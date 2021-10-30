import cv2
import numpy as np
from keras_vggface.vggface import VGGFace
# 如果报错，打开keras_vggface/models.py将报错的import改为from keras.utils.layer_utils import get_source_inputs
from scipy.spatial.distance import cosine

detector = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
gallery = np.load('./gallery.npy', allow_pickle=True)
threshold = 0.5
print('initialization finished')
video = cv2.VideoCapture(1)
video.set(5, 10)

while True:
	(status, frame) = video.read()

	if not status:
		print('failed to capture video')
		break

	faces = detector.detectMultiScale(cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))

	for face in faces:
		(x, y, w, h) = face
		region = frame[y:y + h, x:x + w]

		region = cv2.resize(region, (224, 224))
		region = region.astype('float64')
		region = region.reshape(1, 224, 224, 3)

		enc = model.predict(region)

		name = 'unknown'
		max_similarity = 0

		for friend in gallery:
			similarity = 1 - cosine(enc, friend[1])
			if similarity > max_similarity and similarity > threshold:
				name = friend[0]

		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
		cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)
		print('face detected: ', name)

	cv2.imshow('face detection', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

video.release()
cv2.destroyAllWindows()