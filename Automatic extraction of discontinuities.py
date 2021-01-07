import numpy as np
import math
from math import sqrt,acos,pi,atan,fabs
import time
import os
import random
import heapq
from scipy.spatial import cKDTree
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
from sklearn import metrics

# Computing time
start_time = time.time()

def Normal_vector(data, Triangulation):
    """Calculate the normal vector and the centroid of the triangle"""
    print("Start to calculate the normal vector of each triangle...")
    print('The current time is：', time.asctime(time.localtime(time.time())))
    m, n = np.shape(Triangulation)
    normal_vector = np.zeros((m, 3))     # Define the set of normal vectors of each triangle
    Triangle_centroid = np.zeros((m, 3)) # Define the centroid set of each triangle
    # Calculate the normal vector of the ith triangle
    for i in range(m):
        # Vertex coordinate index
        triangle = Triangulation[i]
        vertice1 = data[triangle[0]]
        vertice2 = data[triangle[1]]
        vertice3 = data[triangle[2]]
        Triangle_centroid[i] = np.mean(data[triangle],0)
        # Finding the normal vector of a triangle
        ab = vertice2 - vertice1
        ac = vertice3 - vertice1
        vector_k = [ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0]]
        # Finding unit normal vector
        mold = math.sqrt(vector_k[0] ** 2 + vector_k[1] ** 2 + vector_k[2] ** 2)
        vector_k = np.array(vector_k) / mold
        normal_vector[i] = vector_k

    normal_vector = np.round(normal_vector, decimals=6)
    Triangle_centroid = np.round(Triangle_centroid, decimals=6)
    return normal_vector,Triangle_centroid

def density_distance(density,Normals):
    """Distance between sample points"""
    print('Start MPCA algorithm...')
    print('The current time is：', time.asctime(time.localtime(time.time())))

    m = len(Normals)
    distance = np.zeros((m,1))
    for i in range(m):
        index = []
        density_i = density[i]
        for j in range(m):
            if density[j] > density_i:
                index.append(j)
        if len(index) > 1:
            # cKDTree
            tree = cKDTree(Normals[index])  ## Create kdtree
            distance_index = tree.query(Normals[i], k=2)  ## Euclidean distance index (search includes itself)
            Normals_near = Normals[index[distance_index[1][0]]]  # Nearest point
            # if cos_sim(Normals[i],Normals_near) < 0:
            #     Normals_near = - Normals_near
            # distance_i = np.sqrt(np.sum((Normals[i]-Normals_near)**2))
            d = abs(Normals[i,0]*Normals_near[0]+Normals[i,1]*Normals_near[1]+Normals[i,2]*Normals_near[2])
            if d > 1:
                d = 1
            distance_i = acos(d)
            distance[i] = distance_i
        elif len(index) == 1:
            # if cos_sim(Normals[i],Normals[j]) < 0:
            #     Normals[j] = - Normals[j]
            # distance_i = np.sqrt(np.sum((Normals[i] - Normals[j]) ** 2))
            d = abs(Normals[i, 0] * Normals_near[0] + Normals[i, 1] * Normals_near[1] + Normals[i, 2] * Normals_near[2])
            if d > 1:
                d = 1
            distance_i = acos(d)
            distance[i] = distance_i
        else:
            distance[i] = pi / 2
    return distance

class Cluster_center:
    """Select initial cluster center"""
    def __init__(self, k,cluster_center):
        self.cluster_center = cluster_center
        self.k = k
        self.cluster = []

    def mold(self, D, U):
        U = np.array(U)
        Dist = []
        if U.ndim == 2:
            for i in range(len(U)):
                dist = Cosine_function(D, U[i])
                Dist.append(dist)
        else:
            U = U.reshape(1,3)
            dist = Cosine_function(D, U)
            Dist.append(dist)
        return Dist

    def Start_choosing(self):
        self.cluster.append(self.cluster_center[0])
        self.cluster_center = np.delete(self.cluster_center, 0, axis=0)
        index = []
        for i in range(len(self.cluster_center)):
            Dist = self.mold(self.cluster_center[i],self.cluster)
            if all([v > 10 for v in Dist]):# The distance between main directions should be greater than 10 degrees
                self.cluster.append(self.cluster_center[i])
                index.append(i)
            if len(self.cluster) >= self.k:
                break
        self.cluster_center = np.delete(self.cluster_center, index, axis=0)

        if len(self.cluster) < self.k:
            kk = self.k - len(self.cluster)
            self.cluster.extend(self.cluster_center[0:kk])

        self.cluster = np.array(self.cluster)
        return self.cluster

def Discontinuous_segmentation(Normals,Triangulation,Triangle_centroid, k,cluster_center):
    """Kmeans clustering"""
    print("Start clustering...")
    print('The current time is：', time.asctime(time.localtime(time.time())))

    m, n = np.shape(Normals)
    clusterAssement = np.mat(np.zeros([m, 1]))
    iters = 0
    while iters < 1:
        for i in range(m):
            minDist = np.inf  # The initial setting is infinity
            minIndex = -1
            for j in range(k):
                distJ = Cosine_function(cluster_center[j, :], Normals[i, :])
                if distJ < minDist:
                    # The angle between each sample point and the main direction should be less than 30 degrees
                    if distJ <= 30:
                        minDist = distJ
                        minIndex = j
                    else:
                        minIndex = -1

            clusterAssement[i, :] = minIndex

        clusterAssement = clusterAssement.astype(int)
        iters += 1
    Normals_group = []
    Triangulation_group = []
    Triangle_centroid_group = []

    for i in range(k):
        bb = [m for m in range(len(clusterAssement)) if clusterAssement[m] == i]
        Normals_group.append(Normals[bb])
        Triangulation_group.append(Triangulation[bb])
        Triangle_centroid_group.append(Triangle_centroid[bb])

    return Normals_group,Triangulation_group,Triangle_centroid_group,clusterAssement

def normals_adjust(Normals,clusterAssement,k):
    """Adjusting normal vector -- KDE kernel density estimation"""
    for cent in range(k):
        normals_index = [m for m in range(len(clusterAssement)) if clusterAssement[m] == cent]
        normals = Normals[normals_index]
        # data = normals
        # values = data.T
        # kde = gaussian_kde(values)  # Construction of multivariate kernel density evaluation function
        # density = kde(values)  # Given the sample point, calculate the density of the sample point
        # max_density_normals = data[np.argmax(density)]  # Normal vector with maximum density
        # max_density_index = np.argmax(density)  # Normal vector index with maximum density
        # a = normals[max_density_index]
        a = normals[0]
        for j in range(len(normals)):
            similarity = cos_sim(a,normals[j])
            if similarity < 0:
                Normals[normals_index[j]] = -Normals[normals_index[j]]
    return Normals

def HDBScan(Normals_group,Triangulation_group,Triangle_centroid_group, r, minpts, points_min, path):
    """Recognition of discontinuities based on hdbscan algorithm"""
    print('Start to split the set of discontinuities...')
    print('The current time is：', time.asctime(time.localtime(time.time())))

    face_qualified = []
    face_Unqualified = []
    for i in range(len(Triangle_centroid_group)):
        X = np.array(Triangle_centroid_group[i])
        Y = np.array(Triangulation_group[i])
        print(len(X))
        print(len(Y))
        # labels = HDBSCAN(min_samples=minpts,min_cluster_size=points_min).fit_predict(X)
        labels = DBSCAN(eps=r, min_samples=minpts, algorithm='kd_tree').fit_predict(X)
        print(labels)
        max_label = np.max(labels)
        print(max_label)
        a = 0
        for j in range(max_label+1):
            face_point_index = [m for m in range(len(labels)) if labels[m] == j]
            face = Y[face_point_index]
            if len(face_point_index) > points_min:
                a += 1
                face = np.array(face)
                face_qualified.append(face)
                np.savetxt(path + "Optimization_Face/" + "%d-" % (i + 1) + "%d.txt"%a, face, fmt='%0.0f')
            else:
                face = np.array(face)
                face_Unqualified.append(face)

    return face_qualified,face_Unqualified

def Plane_fitting(path,data,face_qualified):
    """Plane fitting"""
    print('Start plane fitting...')
    print('The current time is：', time.asctime(time.localtime(time.time())))
    files = os.listdir(path + "Optimization_Face/")
    Dip = []    # dip angle
    Azimuth = []  # inclination
    Parameters = []
    F = 0
    for file in files:  # traverse folder
        position = path + "Optimization_Face/" + file
        with open(position, 'r') as f:
            lines = f.readlines()
            rows = len(lines)
            mesh = np.zeros((rows, 3))
            row = 0
            for line in lines:
                line = line.strip().split(' ')
                mesh[row, :] = line[:]
                row += 1
        ##From plane data to point cloud data
        mesh = mesh.astype(int)
        Position = []
        for i in range(len(mesh)):
            for j in range(3):
                Position.append(data[mesh[i,j]])
        Position = np.array(Position)
        ##Remove duplicate points
        Position = np.unique(Position,axis=0)
        np.savetxt(path + "Face/" + file, Position, fmt='%0.6f')

        # RANSAC algorithm for computing plane normal vector
        SIZE = len(Position)
        iters = 200  # The maximum number of iterations, each time to get a better estimate will optimize the value of iters
        sigma = 0.035  # Acceptable difference between data and model
        # Parameter estimation of the best model
        best_a = 0
        best_b = 0
        best_c = 0
        best_d = 0
        pretotal = 0  # Number of interior points
        P = 0.95  # The probability of getting the right model is expected
        for i in range(iters):
            # Randomly select three points in the data to solve the model
            sample_index = np.arange(Position.shape[0])  # Turn each row of the matrix into a list index
            np.random.shuffle(sample_index)
            sample_index = Position[sample_index[0:3]]  # Randomly select three points of the data
            x1 = sample_index[0, 0]
            x2 = sample_index[1, 0]
            x3 = sample_index[2, 0]
            y1 = sample_index[0, 1]
            y2 = sample_index[1, 1]
            y3 = sample_index[2, 1]
            z1 = sample_index[0, 2]
            z2 = sample_index[1, 2]
            z3 = sample_index[2, 2]
            # ax + by + cz + d = 0 Find out a, b, c, d
            a = y1 * (z2 - z3) + y2 * (z3 - z1) + y3 * (z1 - z2)
            b = z1 * (x2 - x3) + z2 * (x3 - x1) + z3 * (x1 - x2)
            c = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
            d = -a * x1 - b * y1 - c * z1
            # Calculate the number of interior points
            total_inlier = 0
            for index in Position:
                # ax + by + cz + d = 0 z_estimate = a * index[0] + b * index[1] + c
                if abs(a * index[0] + b * index[1] + c * index[2] + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2) < sigma:
                    total_inlier = total_inlier + 1

            # Judge whether the current model is better than the previous estimated model
            if total_inlier > pretotal:
                #iters = math.log(1 - P) / math.log(1 - pow(total_inlier / SIZE, 3))
                pretotal = total_inlier
                best_a = a
                best_b = b
                best_c = c
                best_d = d
                # Judge whether the current model has met more than half of the points
                if total_inlier > int(0.9 * SIZE):
                    break
            #print(pretotal)
        AA = best_a
        BB = best_b
        CC = best_c
        DD = best_d
        A = AA / math.sqrt(AA ** 2 + BB ** 2 + CC ** 2)
        B = BB / math.sqrt(AA ** 2 + BB ** 2 + CC ** 2)
        C = CC / math.sqrt(AA ** 2 + BB ** 2 + CC ** 2)
        D = DD / math.sqrt(AA ** 2 + BB ** 2 + CC ** 2)
        Parameters.append(A)
        Parameters.append(B)
        Parameters.append(C)
        Parameters.append(D)

        qingxiang = 0
        if B == 0:
            qingjiao = math.fabs(math.atan(A / C) * 180 / (math.pi))
            if A > 0:
                qingxiang = 0.5 * 180
            elif A < 0:
                qingxiang = 1.5 * 180
            elif A == 0:
                qingxiang = random.randint(0,180)
        else:
            qingjiao = math.fabs(math.atan(math.sqrt(A ** 2 + B ** 2) / C) * 180 / (math.pi))
            if (A > 0 and B > 0 and C > 0) or (A < 0 and B < 0 and C < 0):
                qingxiang = math.atan(A / B) * 180 / (math.pi)
            elif (A > 0 and B > 0 and C < 0) or (A > 0 and B < 0 and C > 0) or (A < 0 and B < 0 and C > 0) or (A < 0 and B > 0 and C < 0):
                qingxiang = math.atan(A / B) * 180 / (math.pi) + 180
            elif (A > 0 and B < 0 and C < 0) or (A < 0 and B > 0 and C > 0):
                qingxiang = math.atan(A / B) * 180 / (math.pi) + 360
        Dip.append(qingjiao)
        Azimuth.append(qingxiang)
        ###################################################################
    ##Save the file into a TXT file
    Dip = np.array(Dip).reshape(len(Dip),1)
    Azimuth = np.array(Azimuth).reshape(len(Azimuth),1)
    ##ABC应改为单位向量，D/平方根A**2+B**2+C**2
    Parameters = np.array(Parameters).reshape(len(Azimuth),4)
    np.savetxt(path + "Occurrence/"+ "Dip.txt",Dip,fmt='%0.2f')
    np.savetxt(path + "Occurrence/" + "Azimuth.txt", Azimuth, fmt='%0.2f')
    np.savetxt(path + "Occurrence/" + "Parameters.txt", Parameters, fmt='%0.3f')
    return Dip,Azimuth,Parameters

def Cosine_function(A, B):
    """Arccosine function"""
    result = 0
    for i in range(len(A)):
        result += A[i] * B[i]
    if abs(result) > 1:
        result = 1
    Dist = acos(abs(result)) * (180 / pi)
    return Dist

def cos_sim(vector_a, vector_b):
    """Calculate the similarity between two vectors[-1,1]"""
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    return num

if __name__ == '__main__':
    path = "C:/Users/Administrator/Desktop/files/code/Artifical Quarry Slope/"
    # 1、Loading point cloud data
    print("Read point cloud data...")
    print('The current time is：', time.asctime(time.localtime(time.time())))
    data = np.loadtxt(path + "data_resample.txt")
    data = np.round(data, decimals=6)
    print('Point cloud data is：')
    print(len(data))
    # 2、Loading plane data
    print("Read plane data...")
    print('The current time is：', time.asctime(time.localtime(time.time())))
    Triangulation = np.loadtxt(path + "data_triangle.txt", dtype=int, usecols=(1, 2, 3), unpack=False)
    print('The plane of the triangle is：')
    print(len(Triangulation))
    # 3、Calculate normal vector and triangle centroid
    Normals, Triangle_centroid = Normal_vector(data, Triangulation)
    print('The normal vector of point cloud is：')
    np.savetxt(path + "data_normals.txt", Normals, fmt='%0.6f')
    print(len(Normals))
    print('The centroid of triangle is：')
    print(len(Triangle_centroid))
    # DPCA algorithm
    # 4、Loading sample density value (In CloudCompare)
    density = np.loadtxt(path + 'data_density.txt', dtype=int)
    print('The sample density value is：')
    #print(density)
    print(max(density))
    # Distance between sample points
    # distance = density_distance(density, Normals)
    # np.savetxt(path + 'data_distance.txt', distance, fmt='%0.6f')
    distance = np.loadtxt(path + 'data_distance.txt')
    print('The sample distance is：')
    #print(distance)
    print(max(distance))
    y = (density * distance.T).flatten().tolist()
    print('The density of the sample is：')
    print(max(y))
    ############################
    ## 聚类中心集合
    ####????如果出现相同的数，默认输出一个数
    cluster_center_index = list(map(y.index, heapq.nlargest(20, y)))  # 找出最大的20个数
    print(type(cluster_center_index))
    cluster = Normals[cluster_center_index]
    print('The cluster center set is：')
    print(cluster)
    ###########################
    Silhouette_validity = []   # Effectiveness index value
    Silhouette = -1
    cluster_numbers = [2,3,4,5,6]
    n = 0
    for k in cluster_numbers:
        n += 1
        print('The current time is：', time.asctime(time.localtime(time.time())))
        Cc = Cluster_center(k,cluster)
        cluster_center = Cc.Start_choosing()
        print('Cluster center：')
        print(cluster_center)
        # Grouping discontinuities by kmeans clustering
        print("开始第%d次聚类..." %n)
        Normals_group,Triangulation_group,Triangle_centroid_group,clusterAssement= \
            Discontinuous_segmentation(Normals,Triangulation,Triangle_centroid,k,cluster_center)
        # Adjust normal vector
        Normals = normals_adjust(Normals, clusterAssement, k)
        # np.savetxt(path + 'Normals.txt', Normals, fmt='%0.6f')
        # Determination of k-cluster number  Silhouette_validity
        labels = np.array(clusterAssement).reshape(len(clusterAssement), )
        # metric='euclidean'/'manhattan'/'cosine'
        score = metrics.silhouette_score(Normals, labels, metric='cosine')
        print(score)
        Silhouette_validity.append(score)
        if Silhouette < score:
            Silhouette = score
            X = cluster_center
            Y = clusterAssement
            Z = Normals_group
            E = Triangulation_group
            F = Triangle_centroid_group
    print('The cluster center is：')
    cluster_center = X
    print(cluster_center)
    print('Label as：')
    clusterAssement = Y
    print(clusterAssement)
    print('Silhouette validity value is：')
    print(Silhouette_validity)
    k = cluster_numbers[Silhouette_validity.index(max(Silhouette_validity))]
    print('The optimal number of clusters is：%d' % k)
    Normals_group = Z
    Triangulation_group = E
    Triangle_centroid_group = F
    #################################
    # Segmentation of discontinuities by hdbscan clustering
    r = 0.1   # 采样值的2倍
    minpts = 4
    points_min = 100
    face_qualified, face_Unqualified = HDBScan(Normals_group,Triangulation_group,Triangle_centroid_group, r, minpts, points_min, path)
    print('Qualified discontinuities are：%d'%(len(face_qualified)))
    print('Unqualified discontinuities are：%d'%(len(face_Unqualified)))
    # Plane fitting
    Dip, Azimuth, Parameters = Plane_fitting(path, data, face_qualified)
    print('Dip is：')
    # print(Dip)
    print('Azimuth is：')
    # print(Azimuth)

    end_time = time.time()
    print("Total time：", (end_time - start_time)/60)