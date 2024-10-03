import random
import cv2
import numpy as np

class Filter:
    def __init__(self, outputs=None):
        self.outputs = outputs if outputs is not None else []

    def send(self, frame):
        for output in self.outputs:
            output.process(frame)


class PinkFilter(Filter):
    def process(self, frame):
        pink_frame = frame.copy()
        pink_frame[:, :, 2] = np.minimum(frame[:, :, 2] + 100, 255)
        self.send(pink_frame)


class ShakingFilter(Filter):
    def process(self, frame):
        rows, cols, _ = frame.shape
        shake_frame = frame.copy()
        max_shift = 10
        dx = random.randint(-max_shift, max_shift)
        dy = random.randint(-max_shift, max_shift)

        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shake_frame = cv2.warpAffine(shake_frame, M, (cols, rows))
        self.send(shake_frame)


class HeartEffectFilter(Filter):
    def process(self, frame):
        heart_frame = frame.copy()
        center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
        radius = 40
        cv2.circle(heart_frame, (center_x - radius, center_y - radius), radius, (0, 0, 255), -1)
        cv2.circle(heart_frame, (center_x + radius, center_y - radius), radius, (0, 0, 255), -1)
        center_y+=2
        points = np.array([[center_x - 2*radius-2.5, center_y - radius], 
                           [center_x + 2*radius+2.5, center_y - radius], 
                           [center_x, center_y + radius * 2]], np.int32)
        cv2.fillPoly(heart_frame, [points], (0, 0, 255))

        self.send(heart_frame)


class MirrorEffectFilter(Filter):
    def __init__(self, outputs=None):
        super().__init__(outputs)
        self.mirrored = False
        
    def process(self, frame):
        mirrored_frame = cv2.flip(frame, 1)
        self.mirrored = True
        self.send(mirrored_frame)
        
        
class DisplayFilter(Filter):
    def process(self, frame):
        cv2.imshow("Processed Video", frame)
        cv2.waitKey(1)


class InputDisplayFilter(Filter):
    def process(self, frame):
        cv2.imshow("Original Video", frame)
        self.send(frame)


def main():
    cap = cv2.VideoCapture(0)
    display = DisplayFilter()
    mirror = MirrorEffectFilter(outputs=[display])
    shake = ShakingFilter(outputs=[mirror])
    pink = PinkFilter(outputs=[shake])
    heart = HeartEffectFilter(outputs=[pink])
    input_display = InputDisplayFilter(outputs=[heart])
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        input_display.process(frame)
        

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
