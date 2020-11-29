#%%
import cv2
import os
import string

class directoryMaker:
    def __init__(self):
        if 'Signs' not in os.listdir():
            os.mkdir('Signs')
            
    def sub_folder_maker(self, clas='Train', alphabets=True, numbers=True, special=True):
        if clas not in os.listdir('Signs'):
            os.mkdir('Signs/'+clas)
            if alphabets==True:
                for letter in string.ascii_lowercase:
                    if letter not in os.listdir('Signs/'+clas):
                        os.mkdir('Signs/'+clas+'/'+letter)
            if numbers==True:
                for number in '13456789':
                    if number not in os.listdir('Signs/'+clas):
                        os.mkdir('Signs/'+clas+'/'+number)
            if special==True:
                if 'space' not in os.listdir('Signs/'+clas):
                    os.mkdir('Signs/'+clas+'/space')
                if 'break' not in os.listdir('Signs/'+clas):
                    os.mkdir('Signs/'+clas+'/break')
                
class signMaker:
    background = None
    accumulated_weight = 0.5
    
    #suit coordinated that make you comfortable while making signs
    #remember to keep the same coordinates in real_time_isl.py 
    #it has been found to give better results when both of them are at the same place
    ROI_top = 100
    ROI_bottom = 300
    ROI_right = 150
    ROI_left = 350

    def __init__(self):
        pass

    def cal_accum_avg(self, frame, accumulated_weight):
        self.background
        
        if self.background is None:
            self.background = frame.copy().astype("float")
            return None
        cv2.accumulateWeighted(frame, self.background, accumulated_weight)

    def segment_hand(self, frame, threshold=25):
        self.background
        
        diff = cv2.absdiff(self.background.astype("uint8"), frame)
        _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        if len(contours)==0:
            return None
        else:
            hand_segment_max_cont = max(contours, key=cv2.contourArea)
            return (thresholded, hand_segment_max_cont)

    def capture_sign(self, sign='1', clas='Train', num_img=300):
        cam = cv2.VideoCapture(0)
        num_frames = 0
        num_imgs_taken = 1
        
        while True:
            ret, frame = cam.read()
            frame = cv2.flip(frame, 1)
            frame_copy = frame.copy()
            roi = frame[self.ROI_top:self.ROI_bottom, self.ROI_right:self.ROI_left]
            gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)
        
            if num_frames < 60:
                self.cal_accum_avg(gray_frame, self.accumulated_weight)
                if num_frames <= 59:
                    cv2.putText(frame_copy, 'CAPTURING BACKGROUND INFO,', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame_copy, 'PLEASE WAIT.', (20, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif num_frames <= 300:         
                hand = self.segment_hand(gray_frame)                
                cv2.putText(frame_copy, 'Position your hand.', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame_copy, 'Sign for '+str(sign)+'.', (20, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                if hand is not None:
                    thresholded, hand_segment = hand
                    cv2.drawContours(frame_copy, [hand_segment + (self.ROI_right, self.ROI_top)], -1, (255, 0, 0), 1)
                    cv2.imshow("Thresholded Sign", thresholded)
            else: 
                hand = self.segment_hand(gray_frame)
                if hand is not None:
                    thresholded, hand_segment = hand
                    cv2.drawContours(frame_copy, [hand_segment + (self.ROI_right, self.ROI_top)], -1, (255, 0, 0),1)            
                    cv2.putText(frame_copy, str(num_imgs_taken)+' images for '+str(sign), (166, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.imshow("Thresholded Sign", thresholded)
                    
                    if num_imgs_taken <= num_img:
                        cv2.imwrite('Signs/'+clas+'/'+str(sign)+'/'+str(num_imgs_taken)+'.jpg', thresholded)
                    else:
                        break
                    num_imgs_taken+=1
                else:
                    cv2.putText(frame_copy, 'No hand detected!', (162, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
            cv2.rectangle(frame_copy, (self.ROI_left, self.ROI_top), (self.ROI_right, self.ROI_bottom), (30, 200, 0), 3)
            cv2.putText(frame_copy, 'Press esc to exit.', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            num_frames+= 1
            cv2.imshow("Indian Sign Language Generator", frame_copy)
            k = cv2.waitKey(1) & 0xFF
        
            if k == 27:
                break
        
        cv2.destroyAllWindows()
        cam.release()
        
#%%
ISL_directory = directoryMaker()
ISL_directory.sub_folder_maker()
ISL_directory.sub_folder_maker(clas='Test')

#%%
# keep num_img for Train: 301
# keep num_img for Test: 76
ISL_capture = signMaker()
ISL_capture.capture_sign(sign='z', clas='Test', num_img=76)