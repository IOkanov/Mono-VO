import numpy as np 
import glob
import cv2
import math
import time
import csv

from VO import VisualOdometry
import matplotlib.pyplot as plt

# test_1 -> images_total = 386 
# test_2 -> images_total = 90
# test_3 -> images_total = 312 

# NAPOMENA: Da bi se mogao koristiti idalje SIFT i SURF potrebno je instalirati:
# pip install opencv-python==3.4.2.17
# pip install opencv-contrib-python==3.4.2.17
# TODO - napraviti da se parametri citaju iz nekog .txt file-a
K = np.array([[640.0, 0.0, 640.0], [0.0, 640.0, 360], [0, 0, 1]])
# Moguci detectori: ORB, FAST, SIFT, SURF, SHI - TOMASI
#detector = 'SHI-TOMASI'
folder = 'test_1'
image_format = 'png'
images_total = 386
file = open('test_1/koordinate_1.csv')
reader = csv.reader(file)

def leftToRight(vectors):
    # Za test_1 i test_2 - rotacija oko z-ose za 90 stepeni
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    # Za test_3 - nije potrebna rotacija
    #R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    right_vectors = [R.dot(np.array([x, y, z])) for (x, y, z) in vectors]
    X = right_vectors[0][0]
    Y = right_vectors[0][1]
    right_vectors = [(-(x - X), y - Y, z) for (x, y, z) in right_vectors]
    return right_vectors

def trajectoryWindow(window, x, y, z, img_id):
    # Crtanje trajektorije na osnovu dobivenih vektora translacije
    
    # paziti da se image koordinate ne poklapaju sa svjetskim x_i - x_w, y_i - z_w, z_i - y_w 
    draw_x, draw_y = int(x) + 290, int(z) + 90
    cv2.circle(window, (draw_x, draw_y), 1, (img_id * 255 / images_total, 255 - img_id * 255 / images_total, 0), 1)
    cv2.rectangle(window, (10, 20), (600, 60), (0, 0, 0), -1)
    text = "Koordinate: x=%2fm y=%2fm z=%2fm" % (x, z, y)
    cv2.putText(window, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
    cv2.imshow('Trajektorija', window)

    return window

def VOPlot(t_v_orb, t_v_fast, t_v_sift, t_v_surf, ground_truth):
    # Konacna trajektorija
    plt.figure(1)
    VO_ORB, = plt.plot(*zip(*t_v_orb), marker='o', color='b', label='ORB')
    VO_FAST, = plt.plot(*zip(*t_v_fast), marker='o', color='r', label='FAST')
    VO_SIFT, = plt.plot(*zip(*t_v_sift), marker='o', color='c', label='SIFT')
    VO_SURF, = plt.plot(*zip(*t_v_surf), marker='o', color='m', label='SURF')
    ground_truth = [(x, y) for (x, y, z) in ground_truth]
    GT, = plt.plot(*zip(*ground_truth), marker='o', color='g', label='Ground Truth')
    plt.legend(handles=[VO_ORB, VO_FAST, VO_SIFT, VO_SURF, GT])
    plt.axis('equal')
    plt.grid()
    plt.show()

    return

def GTPlot(ground_truth):
    # Konacna trajektorija
    plt.figure(1)
    ground_truth = [(x, y) for (x, y, z) in ground_truth]
    GT, = plt.plot(*zip(*ground_truth), marker='o', color='g', label='Ground Truth')
    plt.legend(handles=[GT])
    plt.grid()
    plt.show()

    return

def run():
            
    #traj = np.zeros((600, 600, 3), dtype = np.uint8)
    t_v_orb = []
    t_v_fast = []
    t_v_sift = []
    t_v_surf = []
    #t_v_st = []
    ground_truth = []    
    
    for row in reader:
        if not row:
            continue
        ground_truth.append((float(row[0]),float(row[1]), float(row[2])))
        
    ground_truth = leftToRight(ground_truth)    
    #GTPlot(ground_truth)
    #return
    
    # Moguce je proslijedi ground truth, ali ako nije proslijeden racuna se relativna skala
    vo_orb = VisualOdometry(K, 'ORB', ground_truth)
    vo_fast = VisualOdometry(K, 'FAST', ground_truth)
    vo_sift = VisualOdometry(K, 'SIFT', ground_truth)
    vo_surf = VisualOdometry(K, 'SURF', ground_truth)
    #vo_st = VisualOdometry(K, 'SHI-TOMASI')
    images = glob.glob(folder + '/*.' + image_format)
    img_id = 0
    
    
    for i, img_path in enumerate(images):
        
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            return
        
        img = cv2.imread(img_path, 0)
        
        # Korak predprocesiranja - povecava kontrast
        clahe = cv2.createCLAHE(clipLimit = 5.0)
        img = clahe.apply(img)
        
        if vo_orb.update(img, img_id):
            if img_id == 0:
                t_v_orb.append((0, 0))
            else:
                t_v_orb.append((vo_orb.cur_t[0], vo_orb.cur_t[2]))
            
        if vo_fast.update(img, img_id):
            if img_id == 0:
                t_v_fast.append((0, 0))
            else:
                t_v_fast.append((vo_fast.cur_t[0], vo_fast.cur_t[2]))
                
        if vo_sift.update(img, img_id):
            if img_id == 0:
                t_v_sift.append((0, 0))
            else:
                t_v_sift.append((vo_sift.cur_t[0], vo_sift.cur_t[2]))
                
        if vo_surf.update(img, img_id):
            if img_id == 0:
                t_v_surf.append((0, 0))
            else:
                t_v_surf.append((vo_surf.cur_t[0], vo_surf.cur_t[2]))
                
        
        cv2.imshow('Ruta', img)
            
        #time.sleep(0.025)
        #print(img_id)
        img_id += 1            
    
    print('Detektor --- vrijeme --- greska')
    print('ORB: ', vo_orb.avg_time, ', ', vo_orb.error)
    print('FAST: ', vo_fast.avg_time, ', ', vo_fast.error)
    print('SIFT: ', vo_sift.avg_time, ', ', vo_sift.error)
    print('SURF: ', vo_surf.avg_time, ', ', vo_surf.error)
    #print('SHI-TOMASI: ', vo_st.avg_time, ', ', vo_st.error)
    VOPlot(t_v_orb, t_v_fast, t_v_sift, t_v_surf, ground_truth)
    return
    
if __name__ == '__main__':
    
    run()
        
        