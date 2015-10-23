"""
The application which reproduces the results of the Python application
webcam-pulse-detector developed by Tristan Hearn.
"""

# Author: Baptiste Lefebvre <baptiste.lefebvre@ens.fr>
# License: MIT


import datetime
import time

import cv2 as cv
import numpy as np



device_ = 0

class Camera(object):
    def __init__(self, device=device_):
        self.device = cv.VideoCapture(device)
        if not self.device:
            raise Exception("Camera not accessible.")
        self.shape = self.getFrame().shape
    
    def getFrame(self):
        _, frame = self.device.read()
        return frame
    
    def release(self):
        self.device.release()
        return

class CascadeDetection(object):
    """
    Detects objects using pre-trained Haar cascade files and OpenCV.
    """
    def __init__(self, function, scaleFactor = 1.3, minNeighbors = 4,
                 minSize = (75, 75), flags = cv.CASCADE_SCALE_IMAGE,
                 persist = True, smooth = 10.0, return_one = True):
        self.frame_in = None
        self.detected = np.array([[0, 0, 2, 2]])
        self.scaleFactor = scaleFactor
        self.persist = persist # keep last detected location v.s. overwrite with None
        self.minNeighbors = minNeighbors
        self.minSize = minSize
        self.return_one = return_one # return either one detection location or all
        self.flags = flags
        self.smooth = smooth
        self.cascade = cv.CascadeClassifier(function)
        self.find = True
        self.last_center = [0, 0]
        return
    
    def toggle(self):
        self.find = not self.find
        return self.find
    
    def on(self):
        if not self.find:
            self.toggle()
        return
    
    def off(self):
        if self.find:
            self.toggle()
        return
    
    def shift(self, detected):
        x, y, w, h = detected
        center = np.array([x + 0.5 * w, y + 0.5 * h])
        shift = np.linalg.norm(center - self.last_center)
        diag = np.sqrt(w ** 2 + h ** 2)
        self.last_center = center
        return shift
    
    def execute(self):
        if not self.find:
            return
        detected = self.cascade.detectMultiScale(self.frame_in,
                                                 scaleFactor = self.scaleFactor,
                                                 minNeighbors = self.minNeighbors,
                                                 minSize = self.minSize,
                                                 flags = self.flags)
        if not isinstance(detected, np.ndarray):
            return
        # Regularization against jitteryness.
        if self.smooth:
            if self.shift(detected[0]) < self.smooth:
                return
        if self.return_one:
            width = detected[0][2]
            height = detected[0][3]
            for i in range(1, len(detected)):
                if width < detected[i][2] and height < detected[i][3]:
                    detected[0] = detected[i]
                    width = detected[i][2]
                    height = detected[i][3]
            self.detected[0] = detected[0]
        else:
            self.detected = detected
        return

class FaceDetector(CascadeDetection):
    """
    Detects a human face in a frame.
    """
    def __init__(self, minSize = (50, 50), smooth = 10.0, return_one = True):
        #function = "xml/haarcascade_frontalface_default.xml"
        function = "xml/haarcascade_frontalface_alt.xml"
        #function = "xml/haarcascade_frontalface_alt2.xml"
        #function = "xml/haarcascade_frontalface_tree.xml"
        super(FaceDetector, self).__init__(function,
                                           minSize = minSize,
                                           smooth = smooth,
                                           return_one = return_one)
        self.foreheads = np.array([])
        return
    
    def get_foreheads(self):
        """
        Defines forehead location using offsets & multiplicative scalings.
        """
        fh_x = 0.5
        fh_y = 0.18
        fh_w = 0.25
        fh_h = 0.15
        forh = []
        for rect in self.detected:
            x, y, w, h = rect
            x = x + w * fh_x
            y = y + h * fh_y
            w = w * fh_w
            h = h * fh_h
            x = x - w / 2.0
            y = y - h / 2.0
            forh.append([int(x), int(y), int(w), int(h)])
        self.foreheads = np.array(forh)
        return
    
    def execute(self):
        super(FaceDetector, self).execute()
        if self.detected[0][2] != 2:
            self.get_foreheads()
        return


# Image frame analysis components written to operate only on inputted regions.

class ProcessRect(object):
    """
    Process inputted rectangles into an inputted frame.
    """
    def __init__(self, channels = [0, 1, 2], zerochannels = []):
        self.frame_in = None
        self.rects_in = None
        self.frame_out = None
        self.channels = channels
        self.zerochannels = zerochannels
        return
    
    def execute(self):
        temp = np.array(self.frame_in)
        if 0 < self.rects_in.size:
            for rect in self.rects_in:
                if len(self.frame_in.shape) == 3:
                    for chan in self.channels:
                        temp[:, :, chan] = self.process(rect, temp[:, :, chan])
                    x, y, w, h = rect
                    for chan in self.zerochannels:
                        temp[y:y+h, x:x+w, chan] = 0 * temp[y:y+h, x:x+w, chan]
                else:
                    temp = self.process(rect, temp)
        self.frame_out = temp
        return
    
    def process(self):
        return

class VariableEqualizerBlock(ProcessRect):
    """
    Equalizes the contrast in a specific region within the inputted frame.
    """
    def __init__(self, channels = [0, 1, 2], zerochannels = []):
        super(VariableEqualizerBlock, self).__init__(channels = channels,
                                                     zerochannels = zerochannels)
        self.beta = 0.0
        self.alpha = 1.0
        return
    
    def process(self, rect, frame):
        x, y, w, h = rect
        subimg = np.array(frame[y:y+h, x:x+w])
        subimg = self.beta * subimg + self.alpha * cv.equalizeHist(subimg)
        frame[y:y+h, x:x+w] = subimg
        return frame

class FrameSlices(object):
    """
    Collect slices of inputted frame using rectangle specifications.
    """
    def __init__(self, channels = [0, 1, 2]):
        self.frame_in = None
        self.rects_in = None
        self.slices = [np.array([0, 0])]
        self.combined = None
        self.zero_mean = None
        self.channels = channels
        return
    
    def combine(self, left, right):
        """
        Stack images horizontally.
        """
        h = max(left.shape[0], right.shape[0])
        w = left.shape[1] + right.shape[1]
        hoff = left.shape[0]
        shape =list(left.shape)
        shape[0] = h
        shape[1] = w
        comb = np.zeros(tuple(shape), left.dtype)
        comb[:left.shape[0], :left.shape[1]] = left
        comb[:right.shape[0], left.shape[1]:] = right
        # TODO: understand why not
        #comb[:right.shape[0], right.shape[1]:] = right
        return comb
    
    def execute(self):
        comb = 150 * np.ones((2, 2))
        if 0 < self.rects_in.size:
            self.slices = []
            for x, y, w, h in self.rects_in:
                output = self.frame_in[y:y+h, x:x+w]
                self.slices.append(output)
                comb = self.combine(output, comb)
        self.combined = comb
        self.zero_mean = self.slices[0].mean()
        return


# Some 1D signal procesing methods used for the analysis of image frames.

class PhaseController(object):
    """
    Outputs aither a convex combination of two floats generated from an inputted
    phase angle, or a set of two default values.
    """
    def __init__(self, default_a, default_b, state = False):
        self.state = state
        self.default_a = default_a
        self.default_b = default_b
        self.phase = 0.0
        self.alpha = self.default_a
        self.beta = self.default_b
        return
    
    def toggle(self):
        self.state = not self.state
        return self.state
    
    def on(self):
        if not self.state:
            self.toggle()
        return
    
    def off(self):
        if self.state:
            self.toggle()
        return
    
    def execute(self):
        if self.state:
            t = (np.sin(self.phase) + 1.0) / 2.0
            t = 0.9 * t + 0.1
            self.alpha = t
            self.beta = 1.0 - t
        else:
            self.alpha = self.default_a
            self.beta = self.default_b
        return

class BufferFFT(object):
    """
    Collects data from a connected input float over each run and buffers it
    internally into lists of a given maximum size.
    """
    def __init__(self, n = 322, spike_limit = 5.0):
        self.n = n
        self.data_in = None
        self.samples = []
        self.fps = 1.0
        self.times = []
        self.fft = np.array([])
        self.freqs = np.array([])
        self.interpolated = np.zeros(2)
        self.even_times = np.zeros(2)
        self.spike_limit = spike_limit
        self.ready = False
        self.size = None
        return
    
    def get_fft(self):
        n = len(self.times)
        self.fps = float(n) / (self.times[-1] - self.times[0])
        self.even_times = np.linspace(self.times[0], self.times[-1], n)
        interpolated = np.interp(self.even_times, self.times, self.samples)
        interpolated = np.hamming(n) * interpolated
        self.interpolated = interpolated
        interpolated = interpolated - np.mean(interpolated)
        fft = np.fft.rfft(interpolated)
        self.freqs = float(self.fps) / n * np.arange(n / 2 + 1)
        return fft
    
    def find_offset(self):
        N = len(self.samples)
        for i in range(2, N):
            samples = self.samples[i:]
            delta = max(samples) - min(samples)
            if delta < self.spike_limit:
                return (N - i)
    
    def reset(self):
        N = self.find_offset()
        self.ready = False
        self.times = self.times[N:]
        self.samples = self.samples[N:]
        return
    
    def execute(self):
        self.samples.append(self.data_in)
        self.times.append(time.time())
        self.size = len(self.samples)
        if self.n < self.size:
            self.ready = True
            self.samples = self.samples[-self.n:]
            self.times = self.times[-self.n:]
        if 4 < self.size:
            self.fft = self.get_fft()
            if self.spike_limit:
                if self.spike_limit < max(self.samples) - min(self.samples):
                    self.reset()
        return

class BandProcess(object):
    """
    Component to isolate specific frequancy bands.
    """
    def __init__(self, limits = [0.0, 3.0], make_filtered = True,
                 operation = "pass"):
        self.freqs_in = None
        self.fft_in = None
        self.freqs = None
        self.make_filtered = make_filtered
        self.filtered = None
        self.fft = None
        self.limits = limits
        self.operation = operation
        self.hz = 0.0
        self.peak_hz = 0.0
        self.phase = 0.0
        return

    def execute(self):
        if self.operation == "pass":
            idx = np.where((self.limits[0] < self.freqs_in)
                           & (self.freqs_in < self.limits[1]))
        else:
            idx = np.where((self.freq_in < self.limits[0])
                           & (self.limits[1] < self.freqs_in))
        self.freqs = self.freqs_in[idx]
        self.fft = np.abs(self.fft_in[idx]) ** 2
        if self.make_filtered:
            fft_out = 0 * self.fft_in
            fft_out[idx] = self.fft_in[idx]
            if 2 < len(fft_out):
                self.filtered = np.fft.irfft(fft_out)
                self.filtered = self.filtered / np.hamming(len(self.filtered))
        try:
            maxidx = np.argmax(self.fft)
            self.peak_hz = self.freqs[maxidx]
            self.phase = np.angle(self.fft_in)[idx][maxidx]
        except ValueError:
            pass

class Cardiac(BandProcess):
    """
    Isolates portions of a pre-computed time series FFT corresponding to human
    heartbeats.
    """
    def __init__(self, bpm_limits = [50, 160]):
        super(Cardiac, self).__init__()
        self.bpm = 0.0
        self.limits = [bpm_limits[0] / 60.0, bpm_limits[1] / 60.0]
        return
    
    def execute(self):
        super(Cardiac, self).execute()
        self.freqs = 60 * self.freqs
        self.bpm = 60 * self.peak_hz
        return


class ShowBPMText(object):
    """
    Shows the estimated BPM in the image frame.
    """
    def __init__(self):
        self.frame_in = None
        self.frame_out = None
        self.ready = False
        self.bpm = 0.0
        self.x = 0
        self.y = 0
        self.fps = 0.0
        self.size = 0.0
        self.n = 0
    
    def execute(self):
        if self.ready:
            col = (0, 255, 0)
            text = "%0.1f bpm" % self.bpm
            tsize = 2
        else:
            col = (100, 255, 100)
            gap = max(0, (self.n - self.size)) / self.fps
            text = "(estimate: %0.1f bpm, wait %0.0f s)" % (self.bpm, gap)
            tsize = 1
        flip_code = 1 # around the y-axis
        frame_tmp = cv.flip(self.frame_in, flip_code)
        width = frame_tmp.shape[1]
        cv.putText(frame_tmp, text, (width - self.x, self.y), cv.FONT_HERSHEY_PLAIN,
                   tsize, col)
        self.frame_out = frame_tmp
        return


# Assembler to detect a human face in an image frame, and then isolate the
# forehead.

class FindFacesGetPulse(object):
    """
    Detects a human face in an image frame.
    """
    def __init__(self, bpm_limits = [50, 160], data_spike_limit = 13.0,
                 face_detector_smoothness = 10):
        self.frame_in = None
        self.frame_out = None
        # TODO: remove
        #self.face_detector = None
        # Find faces within the grayscaled and contrast-adjusted input image.
        self.face_detector = FaceDetector(smooth = face_detector_smoothness)
        # TODO: remove
        #self.fft = None
        # Collects data over time to compute a 1D temporal FFT.
        self.fft = BufferFFT(n = 245, spike_limit = data_spike_limit)
        # Takes in a computed FFT and estimates cardiac data.
        self.cardiac = Cardiac(bpm_limits = bpm_limits)
        self.faces = None
        self.bpm_limits = bpm_limits
        self.data_spike_limit = data_spike_limit
        self.face_detector_smoothness = face_detector_smoothness
        return
    
    def run(self):
        # Splits input color image into R, G and B channels.
        _, green_frame, _ = cv.split(self.frame_in)
        # Converts input color image to grayscale.
        gray_frame = cv.cvtColor(self.frame_in, cv.COLOR_BGR2GRAY)
        # Equalizes contrast on the grayscaled input image.
        equalize_frame = cv.equalizeHist(gray_frame)
        # TODO: remove
        ## Find faces within the grayscaled and contrast-adjusted input image.
        #self.face_detector = FaceDetector(smooth = self.face_detector_smoothness)
        # Collects subimage samples of the detected faces.
        face_frame_slices = FrameSlices()
        # Collects subimage samples of the detected foreheads.
        forehead_frame_slices = FrameSlices()
        # Highlights the location of the detected faces using contrast
        # equalization.
        face_equalizer = VariableEqualizerBlock(channels = [0, 1, 2])
        # Highlights the location of the detected foreheads using contrast
        # equalization (green channel only).
        forehead_equalizer = VariableEqualizerBlock(channels = [1], zerochannels = [0, 2])
        # TODO: remove
        ## Collects data over time to compute a 1D temporal FFT.
        #self.fft = BufferFFT(n = 245, spike_limit = self.data_spike_limit)
        # TODO: remove
        ## Takes in a computed FFT and estimates cardiac data.
        #self.cardiac = Cardiac(bpm_limits = self.bpm_limits)
        # Toggles flashing of the detected foreheads in sync with the detected
        # heartbeat.
        phase_controller = PhaseController(default_a = 1.0,
                                           default_b = 0.0,
                                           state = True)
        # Show the BPM.
        show_bpm_text = ShowBPMText()
        
        # Pass the contrast adjusted grayscale image to the face detector
        self.face_detector.frame_in = equalize_frame
        self.face_detector.execute()
        
        # Pass the original image frame and the detected faces locations to the
        # face highlighter
        face_equalizer.frame_in = self.frame_in
        face_equalizer.rects_in = self.face_detector.detected
        face_equalizer.execute()
        
        # Pass the original image frame and the detected face locations to the
        # forehead highlighter
        forehead_equalizer.frame_in = face_equalizer.frame_out
        forehead_equalizer.rects_in = self.face_detector.foreheads
        forehead_equalizer.execute()
        
        # Pass the original image frame and detected face locations to the face
        # subimage collector
        face_frame_slices.rects_in = self.face_detector.detected
        face_frame_slices.frame_in = equalize_frame
        face_frame_slices.execute()
        
        # Pass the green channel of the original image frame and the detected
        # face locations to the forehead subimage collector
        forehead_frame_slices.rects_in = self.face_detector.foreheads
        forehead_frame_slices.frame_in = green_frame
        forehead_frame_slices.execute()
        
        # Send the mean of the first detected forehead subimage (green channel)
        # to the buffering FFT component
        # TODO: understand why commented
        #self.fft.data_in = forehead_frame_slices.slices[0]
        self.fft.data_in = forehead_frame_slices.zero_mean
        self.fft.execute()
        
        # Send FFT outputs (the FFT and the associated frequencies in hertz) to
        # the cardiac data estimator
        self.cardiac.fft_in = self.fft.fft
        self.cardiac.freqs_in = self.fft.freqs
        self.cardiac.execute()
        
        # Connect the estimated heartbeat phase to the forehead flashing
        # controller
        phase_controller.phase = self.cardiac.phase
        phase_controller.state = self.fft.ready
        phase_controller.execute()
        
        # Connect the phase_controller to the forehead highlighter
        forehead_equalizer.alpha = phase_controller.alpha
        forehead_equalizer.beta = phase_controller.beta
        
        # Connect collection of all detected faces for output
        self.faces = face_frame_slices.combined
        
        # Text display of estimated cardiac rythm
        show_bpm_text.frame_in = forehead_equalizer.frame_out
        show_bpm_text.bpm = self.cardiac.bpm
        show_bpm_text.x = self.face_detector.detected[0][0] + self.face_detector.detected[0][2]
        show_bpm_text.y = self.face_detector.detected[0][1]
        show_bpm_text.fps = self.fft.fps
        show_bpm_text.size = self.fft.size
        show_bpm_text.n = self.fft.n
        show_bpm_text.execute()
        
        show_bpm_text.ready = self.fft.ready
        
        #flip_code = 1 # around the y-axis
        #self.frame_out = cv.flip(self.frame_in, flip_code)
        self.frame_out = show_bpm_text.frame_out
        
        return

class GetPulseApp(object):
    """
    Python application that finds a fae in a webcam stream, then isolates the
    forehead. Then the average green-light intensity in the forehead region is
    gathered over time, and the detected person's pulse is estimated.
    """
    def __init__(self):
        # Imaging device.
        self.camera = Camera()
        self.height = 0
        self.width = 0
        self.pressed = 0
        
        # Analysis of received image frames.
        self.processor = FindFacesGetPulse(bpm_limits = [50, 160],
                                           data_spike_limit = 2500.0,
                                           face_detector_smoothness = 10.0)
        
        # Initialize parameters for the cardiac data plot.
        self.bpm_plot = False
        self.plot_title = "Cardiac info - raw signal, filter signal, and PSD"
        
        # Maps keystrokes to specified methods.
        self.key_controls = {
            "s" : self.toggle_search,
            "d" : self.toggle_display_plot,
            "f" : self.write_csv
        }
        return
    
    def toggle_search(self):
        """
        Toggles a motion lock on the processor's face detection
        """
        state = self.processor.face_detector.toggle()
        if not state:
            self.processor.fft.reset()
        print("face detection lock = " + str(not state))
        return
    
    def toggle_display_plot(self):
        """
        Toggles the data display.
        """
        if self.bpm_plot:
            print("bpm plot disabled")
            self.bpm_plot = False
            cv.destroyWindow(self.plot_title)
        else:
            print("bpm plot enabled")
            self.bpm_plot = True
            self.make_bpm_plot()
            cv.moveWindow(self.plot_title, self.width, 0)
        return
    
    def write_csv(self):
        """
        Writes current data to a csv file
        """
        bpm = " " + str(int(self.processor.cardiac.bpm))
        filename = "data/" + str(datetime.datetime.now()).split(".")[0] + bpm + " bpm.csv"
        data = np.transpose(np.array([self.processor.fft.times, self.processor.fft.samples]))
        np.savetxt(filename, data, delimiter=',')
        print("csv file saved")
        return
    
    def make_bpm_plot(self):
        """
        Creates and/or updates the data display.
        """
        data = [[self.processor.fft.times, 
                 self.processor.fft.samples],
                [self.processor.fft.even_times[4:-4], 
                 self.processor.cardiac.filtered[4:-4]],
                [self.processor.cardiac.freqs, 
                 self.processor.cardiac.fft]]
        size = (280, 640)
        margin = 25
        name = self.plot_title
        labels = [False, False, True]
        skip = [3, 3, 4]
        showmax = [False, False, "bpm"]
        bg = None
        #bg = self.processor.grab_faces.slices[0]
        label_ndigits = [0, 0, 0]
        showmax_digits = [0, 0, 1]
        
        for x, y in data:
            if len(x) < 2 or len(y) < 2:
                return
        
        n_plots = len(data)
        w = float(size[1])
        h = size[0] / float(n_plots)
        
        z = np.zeros((size[0], size[1], 3))
        
        if isinstance(bg, np.ndarray):
            wd = int(bg.shape[1] / bg.shape[0] * h )
            bg = cv.resize(bg, (wd, int(h)))
            if len(bg.shape) == 3:
                r = combine(bg[:, :, 0], z[:, :, 0])
                g = combine(bg[:, :, 1], z[:, :, 1])
                b = combine(bg[:, :, 2], z[:, :, 2])
            else:
                r = combine(bg, z[:, :, 0])
                g = combine(bg, z[:, :, 1])
                b = combine(bg, z[:, :, 2])
            z = cv.merge([r, g, b])[:, :-wd,]    
        
        i = 0
        P = []
        for x, y in data:
            x = np.array(x)
            y = -np.array(y)
            
            xx = (w-2*margin)*(x - x.min()) / (x.max() - x.min())+margin
            yy = (h-2*margin)*(y - y.min()) / (y.max() - y.min())+margin + i*h
            mx = max(yy)
            if labels:
                if labels[i]:
                    for ii in xrange(len(x)):
                        if ii%skip[i] == 0:
                            col = (255,255,255)
                            ss = '{0:.%sf}' % label_ndigits[i]
                            ss = ss.format(x[ii]) 
                            cv.putText(z,ss,(int(xx[ii]),int((i+1)*h)),
                                       cv.FONT_HERSHEY_PLAIN,1,col)           
            if showmax:
                if showmax[i]:
                    col = (0,255,0)    
                    ii = np.argmax(-y)
                    ss = '{0:.%sf} %s' % (showmax_digits[i], showmax[i])
                    ss = ss.format(x[ii]) 
                    #"%0.0f %s" % (x[ii], showmax[i])
                    cv.putText(z,ss,(int(xx[ii]),int((yy[ii]))),
                               cv.FONT_HERSHEY_PLAIN,2,col)
            
            try:
                pts = np.array([[x_, y_] for x_, y_ in zip(xx,yy)],np.int32)
                i+=1
                P.append(pts)
            except ValueError:
                pass #temporary
        """ 
        #Polylines seems to have some trouble rendering multiple polys for some people
        for p in P: cv.polylines(z, [p], False, (255,255,255),1)
        """
        #hack-y alternative:
        for p in P:
            for i in xrange(len(p)-1):
                cv.line(z,tuple(p[i]),tuple(p[i+1]), (255,255,255),1)    
        cv.imshow(name, z)
        return
    
    def key_handler(self):
        """
        Handle keystrokes.
        """
        time = 10 #ms
        self.pressed = cv.waitKey(time)
        if self.pressed == 27:
            print("Exiting...")
            self.camera.device.release()
            exit()
        else:
            for key in self.key_controls.keys():
                if self.pressed == ord(key):
                    self.key_controls[key]()
        return
    
    def main_loop(self):
        """
        Single iteration of the application's main loop.
        """
        # Get current image frame from the camera.
        input_frame = self.camera.getFrame()
        self.height, self.width, _ = input_frame.shape
        
        # Set current image frame to the processor's input.
        self.processor.frame_in = input_frame
        # Process the image frame to perform all needed analysis.
        self.processor.run()
        # Collect the output frame for display.
        output_frame = self.processor.frame_out
        
        # Show the processed/annotated output frame.
        cv.imshow("Processed", output_frame)
        
        # Create and/or update the raw data display if needed.
        if self.bpm_plot:
            self.make_bpm_plot()
        
        # Handle any key presses.
        self.key_handler()
        
        return



if __name__ == "__main__":
    App = GetPulseApp()
    while True:
        App.main_loop()
