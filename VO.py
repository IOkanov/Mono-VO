# odnose se na frame_stage atribut VisualOdometry klase
STAGE_FIRST_FRAME = 0 # detekcija značajki prvog frame
STAGE_SECOND_FRAME = 1 # detekcija značajki drugog frame i njihovo pronalaženje u prvom frame-u
STAGE_DEFAULT_FRAME = 2 # opci postupak

kMinNumFeature = 1000

import cv2
import numpy as np
import time

fMATCHING_DIFF = 1  

lk_params = dict(winSize=(21, 21), 
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def KLTFeatureTracking(image_ref, image_cur, px_ref):

    kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)
    kp1, st, err = cv2.calcOpticalFlowPyrLK(image_cur, image_ref, kp2, None, **lk_params)

    d = abs(px_ref - kp1).reshape(-1, 2).max(-1)
    good = d < fMATCHING_DIFF 
    #print(good)
    if len(d) == 0:
        print('Error: No matches where made.')
    elif list(good).count(True) <= 5: 
        print('Warning: No match was good.')
        return kp1, kp2

    n_kp1 = []
    n_kp2 = []
    for i, good_flag in enumerate(good):
        if good_flag:
            n_kp1.append(kp1[i])
            n_kp2.append(kp2[i])

    n_kp1, n_kp2 = np.array(n_kp1, dtype=np.float32), np.array(n_kp2, dtype=np.float32)

    d = abs(n_kp1 - n_kp2).reshape(-1, 2).max(-1)

    diff_mean = np.mean(d)

    return n_kp1, n_kp2, diff_mean


# Parametri za SHI-TOMASI detektor
#feature_params = dict(maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)
feature_params = dict(maxCorners=500, qualityLevel=0.3, minDistance=5, blockSize=7)

##### UKOLIKO BI SE KORISTIO BFMatcher (LV6-7) #####
# Bila bi potrebna izmjena - bilo bi potrebno pamtiti deskriptore (dodatni atributi), umjesto KLTFeatureTracking() funkcije pozivati BFFeatureTracking() funkciju
'''
def BFFeatureTracking(kp_prev, kp_cur, des_prev, des_cur, k):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_prev, des_cur, k)
    
    # Ovaj korak nije potreban za ORB, ali jeste za SIFT
    px_prev = []
    px_cur = []
    for match_prev, match_cur in matches:
        if match_prev.distance < 0.75*match_cur.distance:
            px_prev.append(kp_prev[match_prev.queryIdx].pt)
            px_cur.append(kp_cur[match.trainIdx].pt)
            
    px_prev = np.array(px_prev, dtype = np.float32)
    px_cur = np.array(px.cur, dtype = np.float32)
    
    d = abs(px_prev - px_cur).reshape(-1, 2).max(-1)
    diff_mean = np.mean(d)
    
    return px_prev, px_cur, diff_mean
'''

class VisualOdometry:
    
    # intrin_mat predstavlja matricu intrinzicnih parametara kamere 
    # detector predstavlja string (key s obzirom da ce moguci detektori biti pohranjeni u dict) koji se odnosi na odabrani feature detector - ORB, FAST, SIFT i SHI-TOMASI
    def __init__(self, intrin_mat, detector, ground_truth = None, use_abs_scale = False):
        # frame_stage definira koja funkcija ce se pozivati zavisno od rednog broja frame-a
        # cur_frame odgovara trenutnom frame-u dok prev_frame odgovara prethodnom frame-u
        # px_ref pamti one znacajke koje nisu match-ane dok px_prev pamti one izdvojene znacajke koju su match-ane sa prethodne slike        
        
        self.frame_stage = 0
        self.cur_frame = None 
        self.prev_frame = None 
        self.frame_id = 0
        self.skip_frame = False
        self.cur_R = None
        self.cur_t = None
        self.px_ref = None
        self.px_prev = None
        self.px_cur = None
        self.cur_roi = None
        self.prev_roi = None
        self.K = intrin_mat
        self.scale = 0
        self.cur_cloud = None
        self.prev_cloud = None
        self.ground_truth = ground_truth
        self.use_abs_scale = use_abs_scale
        self.detect_name = detector
        self.detectors = {'FAST' : cv2.FastFeatureDetector_create(threshold = 25, nonmaxSuppression = True), 'SHI-TOMASI' : 'SHI - TOMASI', 'ORB' : cv2.ORB_create(nfeatures = 2000, scoreType = cv2.ORB_FAST_SCORE), 'SIFT' : cv2.xfeatures2d.SIFT_create(), 'SURF' : cv2.xfeatures2d.SURF_create()} 
                          
        
        self.detector = self.detectors[detector]
        # Koristeno za racunanje vremena izvrsavanja
        self.avg_time = 0
        self.error = 0
        # Kada bi se koristio BFMatcher.knnMatch(self.des_prev, self.des_cur, k = 2)
        # self.des_prev = None
        # self.des_cur = None
        # self.kp_prev = None
        # self.kp_cur = None
    
    def frameSkip(self, pixel_diff):
        # Testirati za razlicite udaljenosti
        return pixel_diff < 3
    
    def getAbsoluteScale(self, frame_id):
        # na osnovu mjerenja sa imu ili prema spremljenim koordinatama racuna apsolutnu skalu
        # frame_id ce cuvati u trenutku poziva ove funkcije id prethodnog frame-a
        x_prev = self.ground_truth[self.frame_id][0]
        y_prev = self.ground_truth[self.frame_id][1]
        z_prev = self.ground_truth[self.frame_id][2]
        x_cur = self.ground_truth[frame_id][0]
        y_cur = self.ground_truth[frame_id][1]
        z_cur = self.ground_truth[frame_id][2]
        return np.sqrt((x_cur - x_prev)*(x_cur - x_prev) + (y_cur - y_prev)*(y_cur - y_prev) + (z_cur - z_prev)*(z_cur - z_prev))
    
    def getRelativeScale(self):
        
        min_idx = min([self.cur_cloud.shape[0], self.prev_cloud.shape[0]])
        ratios = []
        for i in range(1, min_idx):
            X_cur_i = self.cur_cloud[i]
            X_cur_j = self.cur_cloud[i - 1]
            X_prev_i = self.prev_cloud[i]
            X_prev_j = self.prev_cloud[i - 1]
            
            # Ne bi se trebalo moci desiti dijeljene sa nulom
            if np.linalg.norm(X_cur_j - X_cur_i) != 0:
                ratios.append(np.linalg.norm(X_prev_j - X_prev_i) / np.linalg.norm(X_cur_j - X_cur_i)) 
        ratio = np.median(ratios)
        return ratio
    
    def triangulatePoints(self, R, t):
        
        P0 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]])
        P0 = self.K.dot(P0)
        P1 = np.hstack((R, t))
        P1 = self.K.dot(P1)
        point1 = self.px_prev.reshape(2, -1)
        point2 = self.px_cur.reshape(2, -1)
        
        return cv2.triangulatePoints(P0, P1, point1, point2).reshape(-1, 4)[:, :3]
    
    def detectNewFeatures(self, cur_img):
        
        # izdvajanje znacajki za odabrani detektor 
        if self.detect_name == 'FAST' or self.detect_name == 'SURF':
            feature_pts = self.detector.detect(cur_img, None)
            # izdvajamo samo (u,v) koordinate i one moraju biti pohranjeni kao tip float
            feature_pts = np.array([x.pt for x in feature_pts], dtype=np.float32)
        elif self.detect_name == 'SHI-TOMASI':
            feature_pts = cv2.goodFeaturesToTrack(cur_img, **feature_params)
            feature_pts = np.array([x for x in feature_pts], dtype=np.float32).reshape((-1, 2))
        elif self.detect_name == 'ORB' or self.detect_name == 'SIFT':
            kp, des = self.detector.detectAndCompute(cur_img,None)
            feature_pts = np.array([x.pt for x in kp], dtype=np.float32)

        return feature_pts
    
    ################## PRVI FRAME ##################    
    def processFirstFrame(self):
        # potrebno je samo izdvojiti značajke i podesiti konstantu frame_stage za narednu iteraciju
        
        # mozemo proslijediti ili cijelu sliku ili roi 
        self.px_ref = self.detectNewFeatures(self.cur_frame)
        self.frame_stage = STAGE_SECOND_FRAME
    
    ################## DRUGI FRAME ##################
    def processSecondFrame(self):
        # potrebno je izdvojiti znacajke sa nove slike, te ih match-ati sa znacajka sa prve slike i podesiti frame_stage atribut; estimirati esencijalnu matricu te na osnovu nje odredit matricu R i vektor t 
        
        # mozemo proslijediti ili cijelu sliku ili roi 
        prev_img = self.prev_frame
        cur_img = self.cur_frame
        # Match-anje izdvojenih znacajki 
        # Ako bismo testirali BFMatcher u ovoj liniji bismo pozivali funkciju BFFeatureMatcher(self.kp_prev, self.kp_cur, self.des_prev, self.des_cur, k)
        self.px_prev, self.px_cur, _diff = KLTFeatureTracking(prev_img, cur_img, self.px_ref)
        
        E, mask = cv2.findEssentialMat(self.px_cur, self.px_prev, self.K, method = cv2.RANSAC, prob = 0.999, threshold = 1.0)
        _, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_prev, self.K)
        
        # Triangulaciju trenutno vrsimo kako bismo dobili relativnu skalu - potrebno je nadograditi da ju dobijamo na osnovu IMU npr.
        self.cur_cloud = self.triangulatePoints(self.cur_R, self.cur_t)
        
        self.frame_stage = STAGE_DEFAULT_FRAME
        # Moguce potraziti nove skroz, a ne koristiti samo one koje su match-ane
        #self.px_ref = self.detectNewFeatures(cur_img)
        self.px_ref = self.px_cur
        self.prev_cloud = self.cur_cloud
    
    ################## SVI NAREDNI FRAME-OVI ##################
    def processFrame(self, frame_id):
        # cini isto sto i processSecondFrame() s time da ovaj kod vrijedi za svaku i-tu iteraciju za i>2
        
        # mozemo proslijediti ili cijelu sliku ili roi 
        prev_img = self.prev_frame
        cur_img = self.cur_frame
        
        self.px_prev, self.px_cur, px_diff = KLTFeatureTracking(prev_img, cur_img, self.px_ref)
        
        self.skip_frame = self.frameSkip(px_diff)
        if self.skip_frame:
            # self.prev_frame, self.prev_cloud i px_ref trebaju ostati isti
            return
        
        E, mask = cv2.findEssentialMat(self.px_cur, self.px_prev, self.K, method = cv2.RANSAC, prob = 0.999, threshold = 1.0)
        _, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_prev, self.K)
        
        self.cur_cloud = self.triangulatePoints(R, t)
        
        if self.ground_truth and self.use_abs_scale:
            self.scale = self.getAbsoluteScale(frame_id)
        else:
            self.scale = self.getRelativeScale()
        
        # TODO - Da li forsirati kretanje unaprijed jer je ipak auto ne moze uvis ili udesno a da stoji u mjestu?
        #if (t[2] > t[0] and t[2] > t[1]):
        # Test - bez skaliranja
        #self.cur_t = self.cur_t + self.cur_R.dot(t)
        # Test - dodatno skaliranje sa vrijednoscu iz opsega 0.5 - 0.6
        #self.cur_t = self.cur_t + 0.62 * self.scale * self.cur_R.dot(t)
        # Opci slucaj
        self.cur_t = self.cur_t + self.scale * self.cur_R.dot(t)
        self.cur_R = R.dot(self.cur_R)
    
        # TODO - mozda potraziti nove skroz, a ne koristiti samo one koje su match-ane
        #self.px_ref = self.detectNewFeatures(cur_img)
        if self.px_cur.shape[0] < kMinNumFeature:
            self.px_cur = self.detectNewFeatures(cur_img)
        
        # frame_stage je vec podesen
        #self.frame_stage = STAGE_DEFAULT_FRAME  
        self.px_ref = self.px_cur
        self.prev_cloud = self.cur_cloud
    
    def update(self, img, frame_id):
        
        # TODO - dodati exeption za odgovarajuće dimenzije (mora biti grayscale)
        
        self.cur_frame = img
        # img.shape[0] je visina, img.shape[1] je sirina; 0,0 je gornji lijevi ugao
        # TODO - postaviti da je roi podesiv uvodeci param. per_height, per_width
        self.cur_roi = img[img.shape[0]:int(img.shape[0]*0.6), 0:img.shape[1]]
        
        start_time = 0
        stop_time = 0
        if self.frame_stage == STAGE_DEFAULT_FRAME:
            start_time = time.clock()
            self.processFrame(frame_id)
            stop_time = time.clock()
            # Rekurzivno racunanje prosjecnog vremena trajanja jedne iteracije izrazene u sekundama
            self.avg_time = (self.avg_time*(frame_id - 1) + stop_time - start_time)/frame_id
            # NAPOMENA: smisleno je racunati error samo ako nije preskocen frame, a samim time lokacija u .csv file-u
            if not self.skip_frame:
                # NAPOMENA: voditi racuna da su y i z koordinate zamijenjene u self.cur_t
                difference = np.sqrt((self.ground_truth[frame_id][0] - self.cur_t[0])**2 + (self.ground_truth[frame_id][1] - self.cur_t[2])**2)
                self.error = (self.error * (frame_id - 1) + difference[0])/frame_id
        elif self.frame_stage == STAGE_SECOND_FRAME:
            self.processSecondFrame()
        elif self.frame_stage == STAGE_FIRST_FRAME:
            self.processFirstFrame()
                                     
        if self.skip_frame:
            self.skip_frame = False
            return False
        
        self.frame_id = frame_id
        # Samo onda kada se ova cijela iteracija izvrsi se nasljeduju nove ref. tacke, point cloud i frame
        self.prev_frame = self.cur_frame
        self.prev_roi = self.cur_roi
        
        return True