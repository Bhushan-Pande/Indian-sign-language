#%%
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from statistics import mode
from autocorrect import Speller

#%%
class realtimeClassifier:
    word_dict = {0:'1', 1:'3', 2:'4', 3:'5', 4:'6', 5:'7', 6:'8', 7:'9', 8:'a', 9:'b', 10:'break', 11:'c', 12:'d',
                 13:'e', 14:'f', 15:'g', 16:'h', 17:'i', 18:'j', 19:'k', 20:'l', 21:'m', 22:'n', 23:'o', 24:'p',
                 25:'q', 26:'r', 27:'s', 28:'space', 29:'t', 30:'u', 31:'v', 32:'w', 33:'x', 34:'y', 35:'z'}
    
    background = None
    accumulated_weight = 0.5
    
    ROI_top = 100
    ROI_bottom = 300
    ROI_right = 150
    ROI_left = 350 
    
    def __init__(self, model_path):
        self.cnn = load_model(model_path)
    
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

    def capture_sign(self):
        white_img = np.zeros([520, 320, 1], dtype=np.uint8)
        white_img.fill(255)
        cam = cv2.VideoCapture(0)
        num_frames = 0
        selected_char = ['_']*50
        i=0
        word = ''
        line = ''
        spell = Speller(lang='en')
        names = ['bhushan', 'shobhit', 'kewal']
        
        while True:
            ret, frame = cam.read()
            frame = cv2.flip(frame, 1)
            white_copy = white_img.copy()
            frame_copy = frame.copy()
            roi = frame[self.ROI_top:self.ROI_bottom, self.ROI_right:self.ROI_left]
            gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)
            
            if num_frames < 70:
                self.cal_accum_avg(gray_frame, self.accumulated_weight)
                cv2.putText(frame_copy, 'CAPTURING BACKGROUND INFO,', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame_copy, 'PLEASE WAIT.', (20, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else: 
                hand = self.segment_hand(gray_frame)
                if hand is not None:
                    thresholded, hand_segment = hand
                    cv2.drawContours(frame_copy, [hand_segment + (self.ROI_right, self.ROI_top)], -1, (255, 0, 0),1)
                    cv2.imshow('Thresholded Sign', thresholded)
                    thresholded = cv2.resize(thresholded, (64, 64))
                    thresholded = np.reshape(thresholded, (1, thresholded.shape[0], thresholded.shape[1], 1))
                    classified = self.cnn.predict(thresholded)
                    selected_char[i] = self.word_dict[np.argmax(classified)]
                    i = (i+1)%50
                    cv2.putText(frame_copy, self.word_dict[np.argmax(classified)], (242, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(frame_copy, 'No hand detected!', (162, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    selected_char = ['_']*50
                    i = 0
            if i==49:
                try:
                    frequent = mode(selected_char)
                except:
                    frequent = selected_char[-1]
                if frequent not in ['break', 'space']:
                    word = word+frequent
                elif frequent == 'break':
                    if len(word) > 0:
                        word = word[:-1]
                    else:
                        line = line[:-1]
                else:
                    if word not in names:
                        word = spell(word)
                    line = line+word+' ' 
                    word = ''
                    
            cv2.putText(white_copy, line+word, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.imshow('Transcript', white_copy)
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
model_path = 'weights-improvement-06-0.82-0.000200.hdf5'
ISL = realtimeClassifier(model_path)
ISL.capture_sign()