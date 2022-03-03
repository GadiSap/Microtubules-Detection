import cv2
import math
import csv


# Upload video
name = "20211221_LandingRate_k20_001"
mt = cv2.VideoCapture(name + '.avi')
# Out video and csv files
out = cv2.VideoWriter(name + '_out.avi', 0, 5.0, (512,512))
mtcount = open(name + '_count.csv', 'w', newline = '')
writer = csv.writer(mtcount)


frame_num = 0 #frame counter

while True:
    ret, frame = mt.read()
    # Image to gray
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Choose area of interest (roi)
    height, width = gray_frame.shape
    roi = gray_frame [0: height, 0: width]

    # Object detection
    ret1, thresh = cv2.threshold(roi, 20, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    count_mt = 0
    mt_locations = []

    #Filtering the contours for microtubules
    for i in contours:
        area = cv2.contourArea(i)
        (x, y), (w, h), tet =cv2.minAreaRect(i)
        if area > 5: #filter by size
            #if (w > 5 and h < 2) or (w < 2 and h) > 5 : #filter by shape
                cv2.drawContours(roi, contours, -1, (255, 0, 0), 1)
                count_mt += 1
                mt_locations.append ([count_mt, x, y]) # writing locations
                cv2.putText(frame, str(count_mt), (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 255), 1) #add numbers to video

    # write video
    out.write(frame)
    # Show video
    cv2.imshow("Original", frame)
    cv2.imshow("ROI", roi)
    cv2.imshow("Threshold", thresh)
    print (count_mt)
    frame_num += 1
    writer.writerow([frame_num, count_mt, mt_locations])

    key = cv2.waitKey(1)
    if key == 27:
        break
mt.release()
cv2.destroyAllWindows
mtcount.close()

