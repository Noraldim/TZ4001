import cv2


video = cv2.VideoCapture('test2.mp4')

prototxt = 'deploy.prototxt.txt'    
caffemodel='res10_300x300_ssd_iter_140000.caffemodel'

model = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

while True:

    ret, frame = video.read()
    

    if not ret:
        break

    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))


    model.setInput(blob)

    detections = model.forward()


    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            x1, y1, x2, y2 = (detections[0, 0, i, 3:7] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]).astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            
            label = '/f prototxt: {:.2f}%'.format(confidence * 100)
            
      
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Objects Detected', frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
