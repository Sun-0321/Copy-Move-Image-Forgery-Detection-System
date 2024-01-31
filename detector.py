import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from collections import Counter

def siftInvocation(image):   
    sift = cv2.xfeatures2d.SIFT_create()
    grayScale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp, desc = sift.detectAndCompute(grayScale, None)
    return kp, desc

def readImage(img):  
    print(img)  
    return cv2.imread(img)


def showImage(img):   
    img = imutils.resize(img,width=600)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def clusterisation(points1, points2, threshold, metric):
   
    points = np.vstack((points1, points2))        
    distMatrixCluster = pdist(points, metric='euclidean') 
    Z = hierarchy.linkage(distMatrixCluster, metric)
    
    cluster = hierarchy.fcluster(Z, t=threshold, criterion='inconsistent', depth=4)    
    cluster, points = filterOutliers(cluster, points)
    n = int(np.shape(points)[0]/2)
    return cluster, points[:n], points[n:]


def detect_feature_for_image(kpts, dept):   
    
    norm = cv2.NORM_L2  
    k = 10      
    matchOBJ = cv2.BFMatcher(norm)    
    matches = matchOBJ.knnMatch(dept, dept, k)      

    ratio = 0.45
    similarityMatched1 = []
    similarityMatched2 = []

    for elem in matches:
        k = 1   
        while elem[k].distance < ratio * elem[k + 1].distance:  
            k += 1
        for i in range(1, k):            
            if pdist(np.array([kpts[elem[i].queryIdx].pt, kpts[elem[i].trainIdx].pt]), "euclidean") > 10:
                similarityMatched1.append(kpts[elem[i].queryIdx])
                similarityMatched2.append(kpts[elem[i].trainIdx])

    similarPoints1 = [match.pt for match in similarityMatched1]
    similarPoints2 = [match.pt for match in similarityMatched2]

    if len(similarPoints1) > 0:
        points = np.hstack((similarPoints1, similarPoints2))    
        distinctPoints = np.unique(points, axis=0)   
        return np.float32(distinctPoints[:, 0:2]), np.float32(distinctPoints[:, 2:4])
    else:
        return None, None




def filterOutliers(clstr, pts):    
    clstrCnt = Counter(clstr)
    container = []  
    for key in clstrCnt:
        if clstrCnt[key] <= 3:
            container.append(key)

    indices = np.array([])   
    for i in range(len(container)):
        indices = np.concatenate([indices, np.where(clstr == container[i])], axis=None)

    indices = indices.astype(int)
    indices = sorted(indices, reverse=True)

    for i in range(len(indices)):   
        pts = np.delete(pts, indices[i], axis=0)

    for i in range(len(container)):  
        clstr = clstr[clstr != container[i]]

    return clstr, pts


def displayImageToConsole(img, p1, p2, clusters):
    
    plt.imshow(img)
    plt.axis('off')

    clrs = clusters[:np.shape(p1)[0]]
    plt.scatter(p1[:, 0], p1[:, 1], c=clrs, s=30)

    for cord1, cord2 in zip(p1, p2):
        x1 = cord1[0]
        y1 = cord1[1]
        x2 = cord2[0]
        y2 = cord2[1]

        plt.plot([x1, x2], [y1, y2], 'c', linestyle=":")

    plt.savefig("results.png", bbox_inches='tight', pad_inches=0)
    plt.clf()


def detectCopyMove(img):  
    
    kp, desc = siftInvocation(img)
    point1, point2 = detect_feature_for_image(kp, desc)    

    if point1 is None:        
        return False

    clusters, point1, point2 = clusterisation(point1, point2, 2.2, 'ward')
    if len(clusters) == 0 or len(point1) == 0 or len(point2) == 0:        
        return False
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    displayImageToConsole(img, point1, point2, clusters)
    return True
