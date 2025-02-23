import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import math
from math import sqrt, acos, degrees
from shapely.geometry import Polygon
import xlsxwriter

# 加载模型
sleep_model = joblib.load(r"test/sleep-24-new.joblib")
geti_model = joblib.load(r"test/new-geti-gai1.joblib")

#加载数据集以便对其标准化
sleep_datasets = pd.read_excel(r'test/sleep-30.xlsx',header=None)
X_sleep_datasets = sleep_datasets.iloc[:, :-1]  # 特征
scaler_sleep_datasets = StandardScaler()
X_scaled_1 = scaler_sleep_datasets.fit_transform(X_sleep_datasets)

geti_datasets = pd.read_excel(r'test/new.xlsx',header=None)
X_geti_datasets = geti_datasets.iloc[:, :-1]  # 特征
scaler_geti_datasets = StandardScaler()
X_scaled_2 = scaler_geti_datasets.fit_transform(X_geti_datasets)


def preventing_the_impossible(lst,huanjing_point,save_per):

    for i in range(len(lst)):
        if lst[i] == 4: #检查是否为真的攀爬
            a_ratio = intersection_area_ratio(save_per[i+1][0],huanjing_point[1],huanjing_point[2],huanjing_point[3],huanjing_point[4])
            if a_ratio < 0.3:
                lst[i] = lst[i-1]
        elif lst[i] == 1: #检查是否为真的饮水
            if distance(huanjing_point[0],save_per[i+1][1]) > 50: # 不太可能为饮水
                if lst[i-1] != 1:
                    lst[i] = lst[i-1]
                else:
                    lst[i] = 5
        elif lst[i] == 2: #检查是否为真的行走
            box_center_last = [save_per[i][0][0]+save_per[i][0][2]/2,save_per[i][0][1]+save_per[i][0][3]/2]
            box_center_current = [save_per[i+1][0][0]+save_per[i+1][0][2]/2,save_per[i+1][0][1]+save_per[i+1][0][3]/2]
            tail_last = save_per[i][6]
            tail_current = save_per[i+1][6]
            if distance(box_center_last,box_center_current) < 3 or distance(tail_last,tail_current) < 3:
                if lst[i-1] != 2:
                    lst[i] = lst[i-1]
                else:
                    lst[i] = 5

def intersection_area_ratio(box,p1,p2,p3,p4):
    x1,y1 = box[0],box[1]
    x2,y2 = box[0]+box[2],box[1]
    x3,y3 = box[0]+box[2],box[1]+box[3]
    x4,y4 = box[0],box[1]+box[3]

    x5,y5 = p1[0],p1[1]
    x6,y6 = p2[0],p2[1]
    x7,y7 = p3[0],p3[1]
    x8,y8 = p4[0],p4[1]

    # Calculate area of Quadrilateral 1
    poly1 = Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
    area_quadrilateral1 = poly1.area

    # Calculate area of Quadrilateral 2
    poly2 = Polygon([(x5, y5), (x6, y6), (x7, y7), (x8, y8)])

    # Calculate intersection area
    intersection = poly1.intersection(poly2)
    intersection_area_value = intersection.area

    # Calculate the ratio of intersection area to Quadrilateral 1 area
    if area_quadrilateral1 > 0:
        intersection_ratio = intersection_area_value / area_quadrilateral1
    else:
        intersection_ratio = 0.0

    return intersection_ratio



def filter_error(behavior_list):
    for i in range(len(behavior_list)):
        if 5<i<len(behavior_list)-5:
            if behavior_list[i - 2] == behavior_list[i - 1] == behavior_list[i + 1] == behavior_list[i + 2]:
                behavior_list[i] = behavior_list[i-1]


def short_eat_2_rear(lst):
    count = 0
    for i in range(len(lst)):
        if lst[i] == 0:
            count += 1
        else:
            if count < 30:
                for j in range(i - count, i):
                    lst[j] = 7
            count = 0
    # 处理结尾的情况
    if count < 30:
        for j in range(len(lst) - count, len(lst)):
            lst[j] = 7

def short_rest_2_sniffing(lst):
    count = 0
    for i in range(len(lst)):
        if lst[i] == 6:
            count += 1
        else:
            if count < 10:
                for j in range(i - count, i):
                    lst[j] = 5
            count = 0
    # 处理结尾的情况
    if count < 10:
        for j in range(len(lst) - count, len(lst)):
            lst[j] = 5

def short_grooming_2_sniffing(lst):
    count = 0
    for i in range(len(lst)):
        if lst[i] == 3:
            count += 1
        else:
            if count < 4:
                for j in range(i - count, i):
                    lst[j] = 5
            count = 0
    # 处理结尾的情况
    if count < 4:
        for j in range(len(lst) - count, len(lst)):
            lst[j] = 5

def one_drinking_2_sniffing(lst):
    for i in range(len(lst)):
        if i > 5 and i < len(lst)-5:
            if lst[i] == 1 and lst[i-1] != 1 and lst[i+1] != 1:
                lst[i] = lst[i-1]

def one_rearing_2_sniffing(lst):
    for i in range(len(lst)):
        if i > 5 and i < len(lst)-5:
            if lst[i] == 7 and lst[i-1] != 7 and lst[i+1] != 7:
                lst[i] = lst[i-1]

def one_walking_2_sniffing(lst):
    for i in range(len(lst)):
        if i > 5 and i < len(lst)-5:
            if lst[i] == 2 and lst[i-1] == 5 and lst[i+1] == 5:
                lst[i] = 5
def true_rest_remove_sniffing(lst):
    # 2、3、4个连续嗅探夹在静止之间，改为静止
    for i in range(len(lst)):
        if i > 5 and i < len(lst)-5:
            if lst[i] == 5 and lst[i+1] == 5 and lst[i-1] == 6 and lst[i+2] == 6:
                lst[i] = 6
                lst[i+1] = 6
            elif lst[i] == 5 and lst[i+1] == 5 and lst[i+2] == 5 and lst[i-1] == 6 and lst[i+3] == 6:
                lst[i] = 6
                lst[i + 1] = 6
                lst[i+2] = 6
            elif lst[i] == lst[i+1] == lst[i+2] == lst[i+3]== 5 and lst[i-1]==lst[i+4]==6:
                lst[i] = 6
                lst[i + 1] = 6
                lst[i + 2] = 6
                lst[i+3] = 6

def time_domain_features(data):
    # kurt = kurtosis(data)
    mean = np.mean(data)
    median = np.median(data)
    std_dev = np.std(data)
    max_val = np.max(data)

    return mean, median, std_dev, max_val

def frequency_domain_features(data, sample_rate = 30):
    fft_result = np.fft.fft(data)
    psd = np.abs(fft_result) ** 2
    freq = np.fft.fftfreq(len(data), d=1 / sample_rate)

    kurt_psd = kurtosis(psd)
    skew_psd = skew(psd)

    psd_01_1Hz = np.mean(psd[(freq >= 0.1) & (freq <= 1)])
    psd_1_3Hz = np.mean(psd[(freq > 1) & (freq <= 3)])
    psd_3_5Hz = np.mean(psd[(freq > 3) & (freq <= 5)])
    psd_5_8Hz = np.mean(psd[(freq > 5) & (freq <= 8)])
    psd_8_15Hz = np.mean(psd[(freq > 8) & (freq <= 15)])
    total_psd = np.sum(psd)
    max_psd = np.max(psd)
    min_psd = np.min(psd)
    avg_psd = np.mean(psd)
    std_psd = np.std(psd)

    return kurt_psd, skew_psd, psd_01_1Hz, psd_1_3Hz, psd_3_5Hz, psd_5_8Hz, psd_8_15Hz, total_psd, max_psd, min_psd, avg_psd, std_psd

def tezheng_sleep(list1):

    dx = []
    dy = []
    dw = []
    dh = []
    juli = []
    ds = []
    jieguo = []

    for i in range(len(list1)):
        if i > 0:
            dx.append(abs((list1[i][0] + list1[i][2]/2) - (list1[i-1][0] + list1[i-1][2]/2)))
            dy.append(abs((list1[i][1] + list1[i][3]/2) - (list1[i-1][1] + list1[i-1][3]/2)))
            dw.append(abs(list1[i][2]-list1[i-1][2]))
            dh.append(abs(list1[i][3]-list1[i-1][3]))
            ddx = (list1[i][0] + list1[i][2]/2) - (list1[i-1][0] + list1[i-1][2]/2)
            ddy = (list1[i][1] + list1[i][3]/2) - (list1[i-1][1] + list1[i-1][3]/2)
            juli.append(np.sqrt(ddx ** 2 + ddy ** 2))
            ds.append(abs(list1[i][2]*list1[i][3]-list1[i-1][2]*list1[i-1][3]))

    for i in range(4):
        jieguo.append(time_domain_features(dx)[i])
        jieguo.append(time_domain_features(dy)[i])
        jieguo.append(time_domain_features(dw)[i])
        jieguo.append(time_domain_features(dh)[i])
        jieguo.append(time_domain_features(juli)[i])
        jieguo.append(time_domain_features(ds)[i])
    # for i in range(12):
    #     jieguo.append(frequency_domain_features(dx)[i])
    #     jieguo.append(frequency_domain_features(dy)[i])
    #     jieguo.append(frequency_domain_features(dw)[i])
    #     jieguo.append(frequency_domain_features(dh)[i])
    #     jieguo.append(frequency_domain_features(juli)[i])
    #     jieguo.append(frequency_domain_features(ds)[i])

    return jieguo

def sleep_behavior(boxs_300):
    """
    输入的[[x1,y1,w,h],[x1,y1,w,h],.....]
    """
    jieguo = tezheng_sleep(boxs_300)

    jieguo_biaozhunhua = scaler_sleep_datasets.transform([jieguo])

    pred_1 = sleep_model.predict(jieguo_biaozhunhua)[0]

    return pred_1

# 欧式距离函数
def distance(point1,point2):  # point1 = [x1,y1] point2 = [x2,y2]
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
def angle_3_point(vertex,p1,p2):
    #计算的是以vertex为顶点的，p1,p2为两边的角度
    # 计算三条边的长度
    x1,y1,x2,y2,x3,y3 = vertex[0],vertex[1],p1[0],p1[1],p2[0],p2[1]
    d1 = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    d2 = sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
    d3 = sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)

    # 使用余弦定理计算角度

    if d1 != 0 and d2 != 0:
        cos_angle = (d1 ** 2 + d2 ** 2 - d3 ** 2) / (2 * d1 * d2)
    else:
        cos_angle = 1

    if cos_angle > 1:
        cos_angle = 1
    elif cos_angle < -1:
        cos_angle = -1
    angle_rad = acos(cos_angle)
    angle_deg = degrees(angle_rad)
    return round(angle_deg, 0)
def zai_or_buzai(nose,p1,p2,p3,p4):
    x = nose[0]
    y = nose[1]

    polygon = [(p1[0],p1[1]),(p2[0],p2[1]),(p3[0],p3[1]),(p4[0],p4[1])]
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    if inside:
        return 1
    else:
        return 0



def distance_point_to_line_segment(point, point1, point2):
    x, y = point
    x1, y1 = point1
    x2, y2 = point2

    # 计算点到线段的垂直距离
    dx = x2 - x1
    dy = y2 - y1
    if dx == dy == 0:  # 如果线段是一个点，直接计算点到点的距离
        return math.hypot(x - x1, y - y1)
    # 计算投影点的参数 t
    t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
    # 计算投影点的坐标
    px = x1 + t * dx
    py = y1 + t * dy
    # 如果投影点在线段上，则返回点到投影点的距离，否则返回点到线段两端点的最小距离
    if (x1 <= px <= x2 or x2 <= px <= x1) and (y1 <= py <= y2 or y2 <= py <= y1):
        return math.hypot(x - px, y - py)
    else:
        return min(math.hypot(x - x1, y - y1), math.hypot(x - x2, y - y2))


def nearest_segment_to_point(point, points):
    x, y = point
    min_distance = float('inf')
    nearest_segment = None
    for i in range(len(points)):
        point1 = points[i]
        point2 = points[(i + 1) % len(points)]
        distance = distance_point_to_line_segment([x, y], point1, point2)
        if distance < min_distance:
            min_distance = distance
            nearest_segment = point1, point2
    return nearest_segment

def zhixin(p1,p2,p3,p4):
    points = [p1,p2,p3,p4]
    x_sum = 0
    y_sum = 0
    for point in points:
        x_sum += point[0]
        y_sum += point[1]
    centroid_x = x_sum / len(points)
    centroid_y = y_sum / len(points)
    return [centroid_x, centroid_y]

def geti_behavior(list_biaoge,j,huanjing):

    # print("为什么会出错",len(list_biaoge))
    # 当前帧
    i = j + 1
    # print("为什么会出错", i)
    # 当前帧
    box_x1, box_y1 = list_biaoge[i][0][0],list_biaoge[i][0][1]
    box_w, box_h = list_biaoge[i][0][2],list_biaoge[i][0][3]
    box_center_x, box_center_y = box_x1 + box_w / 2, box_y1 + box_h / 2
    nose = list_biaoge[i][1]
    left = list_biaoge[i][2]
    right = list_biaoge[i][3]
    bozi = list_biaoge[i][4]
    center = list_biaoge[i][5]
    tail = list_biaoge[i][6]

    # 上一帧
    box_x1_1, box_y1_1 = list_biaoge[j][0][0],list_biaoge[j][0][1]
    box_w_1, box_h_1 = list_biaoge[j][0][2],list_biaoge[j][0][3]
    box_center_x_1, box_center_y_1 = box_x1_1 + box_w_1 / 2, box_y1_1 + box_h_1 / 2
    nose_1 = list_biaoge[j][1]
    left_1 = list_biaoge[j][2]
    right_1 = list_biaoge[j][3]
    bozi_1 = list_biaoge[j][4]
    center_1 = list_biaoge[j][5]
    tail_1 = list_biaoge[j][6]


    # 特征 ：当前帧与之前帧的差异值
    nose_dis = distance(nose_1, nose)
    left_dis = distance(left_1, left)
    right_dis = distance(right_1, right)
    bozi_dis = distance(bozi_1, bozi)
    center_dis = distance(center_1, center)
    tail_dis = distance(tail_1, tail)
    box_center_dis = distance([box_center_x_1, box_center_y_1], [box_center_x, box_center_y])
    box_ds = abs(box_w_1 * box_h_1 - box_h * box_w)
    nose_left_dd = abs(distance(nose, left) - distance(nose_1, left_1))
    nose_right_dd = abs(distance(nose, right) - distance(nose_1, right_1))
    left_bozi_dd = abs(distance(left, bozi) - distance(left_1, bozi_1))
    right_bozi_dd = abs(distance(right, bozi) - distance(right_1, bozi_1))
    bozi_center_dd = abs(distance(bozi, center) - distance(bozi_1, center_1))
    center_tail_dd = abs(distance(center, tail) - distance(center_1, tail_1))
    nose_tail_dd = abs(distance(nose, tail) - distance(nose_1, tail_1))


    # 当前帧的值
    nose_left = distance(nose, left)
    nose_right = distance(nose, right)
    left_bozi = distance(left, bozi)
    right_bozi = distance(right, bozi)
    bozi_center = distance(bozi, center)
    center_tail = distance(center, tail)
    nose_tail = distance(nose, tail)
    # 头部转角（身体中心到脖子的线段与脖子到鼻子线段的夹角）
    head_angle = angle_3_point(bozi, nose, center)
    # 身体转角（尾巴根部到身体中心的线段与身体中心到脖子线段的夹角）
    body_angle = angle_3_point(center, bozi, tail)
    # 头部夹角（左耳与鼻子连线与右耳与鼻子连线夹角）
    head_left_right = angle_3_point(nose, left, right)
    box_area = box_w * box_h
    nose_water = distance(nose, huanjing[0])
    #
    nose_food_bool = zai_or_buzai(nose, huanjing[1], huanjing[2], huanjing[3], huanjing[4])
    nose_tiesi_center = distance(nose, zhixin(huanjing[1], huanjing[2], huanjing[3], huanjing[4]))
    left_tiesi_center = distance(left, zhixin(huanjing[1], huanjing[2], huanjing[3], huanjing[4]))
    right_tiesi_center = distance(right, zhixin(huanjing[1], huanjing[2], huanjing[3], huanjing[4]))
    bozi_tiesi_center = distance(bozi, zhixin(huanjing[1], huanjing[2], huanjing[3], huanjing[4]))
    center_tiesi_center = distance(center, zhixin(huanjing[1], huanjing[2], huanjing[3], huanjing[4]))
    tail_tiesi_center = distance(tail, zhixin(huanjing[1], huanjing[2], huanjing[3], huanjing[4]))
    # 先确定是那条边，再计算6个点与最近边的距离
    # 找到最近的线段
    nearest_segment = nearest_segment_to_point(nose, [huanjing[5], huanjing[6], huanjing[7], huanjing[8]])
    if nearest_segment[0] == huanjing[5] and nearest_segment[1] == huanjing[6]:
        nose_bian = distance_point_to_line_segment(nose, huanjing[5], huanjing[6])
        if not zai_or_buzai(nose, huanjing[5], huanjing[6], huanjing[7], huanjing[8]):
            nose_bian = -nose_bian

        left_bian = distance_point_to_line_segment(left, huanjing[5], huanjing[6])
        if not zai_or_buzai(left, huanjing[5], huanjing[6], huanjing[7], huanjing[8]):
            left_bian = -left_bian

        right_bian = distance_point_to_line_segment(right, huanjing[5], huanjing[6])
        if not zai_or_buzai(right, huanjing[5], huanjing[6], huanjing[7], huanjing[8]):
            right_bian = -right_bian

        bozi_bian = distance_point_to_line_segment(bozi, huanjing[5], huanjing[6])
        if not zai_or_buzai(bozi, huanjing[5], huanjing[6], huanjing[7], huanjing[8]):
            bozi_bian = -bozi_bian

        center_bian = distance_point_to_line_segment(center, huanjing[5], huanjing[6])
        if not zai_or_buzai(center, huanjing[5], huanjing[6], huanjing[7], huanjing[8]):
            center_bian = -center_bian

        tail_bian = distance_point_to_line_segment(tail, huanjing[5], huanjing[6])
        if not zai_or_buzai(tail, huanjing[5], huanjing[6], huanjing[7], huanjing[8]):
            tail_bian = -tail_bian
    elif nearest_segment[0] == huanjing[6] and nearest_segment[1] == huanjing[7]:
        nose_bian = distance_point_to_line_segment(nose, huanjing[6], huanjing[7])
        if not zai_or_buzai(nose, huanjing[5], huanjing[6], huanjing[7], huanjing[8]):
            nose_bian = -nose_bian

        left_bian = distance_point_to_line_segment(left, huanjing[6], huanjing[7])
        if not zai_or_buzai(left, huanjing[5], huanjing[6], huanjing[7], huanjing[8]):
            left_bian = -left_bian

        right_bian = distance_point_to_line_segment(right, huanjing[6], huanjing[7])
        if not zai_or_buzai(right, huanjing[5], huanjing[6], huanjing[7], huanjing[8]):
            right_bian = -right_bian

        bozi_bian = distance_point_to_line_segment(bozi, huanjing[6], huanjing[7])
        if not zai_or_buzai(bozi, huanjing[5], huanjing[6], huanjing[7], huanjing[8]):
            bozi_bian = -bozi_bian

        center_bian = distance_point_to_line_segment(center, huanjing[6], huanjing[7])
        if not zai_or_buzai(center, huanjing[5], huanjing[6], huanjing[7], huanjing[8]):
            center_bian = -center_bian

        tail_bian = distance_point_to_line_segment(tail, huanjing[6], huanjing[7])
        if not zai_or_buzai(tail, huanjing[5], huanjing[6], huanjing[7], huanjing[8]):
            tail_bian = -tail_bian
    elif nearest_segment[0] == huanjing[7] and nearest_segment[1] == huanjing[8]:
        nose_bian = distance_point_to_line_segment(nose, huanjing[7], huanjing[8])
        if not zai_or_buzai(nose, huanjing[5], huanjing[6], huanjing[7], huanjing[8]):
            nose_bian = -nose_bian

        left_bian = distance_point_to_line_segment(left, huanjing[7], huanjing[8])
        if not zai_or_buzai(left, huanjing[5], huanjing[6], huanjing[7], huanjing[8]):
            left_bian = -left_bian

        right_bian = distance_point_to_line_segment(right, huanjing[7], huanjing[8])
        if not zai_or_buzai(right, huanjing[5], huanjing[6], huanjing[7], huanjing[8]):
            right_bian = -right_bian

        bozi_bian = distance_point_to_line_segment(bozi, huanjing[7], huanjing[8])
        if not zai_or_buzai(bozi, huanjing[5], huanjing[6], huanjing[7], huanjing[8]):
            bozi_bian = -bozi_bian

        center_bian = distance_point_to_line_segment(center, huanjing[7], huanjing[8])
        if not zai_or_buzai(center, huanjing[5], huanjing[6], huanjing[7], huanjing[8]):
            center_bian = -center_bian

        tail_bian = distance_point_to_line_segment(tail, huanjing[7], huanjing[8])
        if not zai_or_buzai(tail, huanjing[5], huanjing[6], huanjing[7], huanjing[8]):
            tail_bian = -tail_bian
    elif nearest_segment[0] == huanjing[8] and nearest_segment[1] == huanjing[5]:
        nose_bian = distance_point_to_line_segment(nose, huanjing[8], huanjing[5])
        if not zai_or_buzai(nose, huanjing[5], huanjing[6], huanjing[7], huanjing[8]):
            nose_bian = -nose_bian

        left_bian = distance_point_to_line_segment(left, huanjing[8], huanjing[5])
        if not zai_or_buzai(left, huanjing[5], huanjing[6], huanjing[7], huanjing[8]):
            left_bian = -left_bian

        right_bian = distance_point_to_line_segment(right, huanjing[8], huanjing[5])
        if not zai_or_buzai(right, huanjing[5], huanjing[6], huanjing[7], huanjing[8]):
            right_bian = -right_bian

        bozi_bian = distance_point_to_line_segment(bozi, huanjing[8], huanjing[5])
        if not zai_or_buzai(bozi, huanjing[5], huanjing[6], huanjing[7], huanjing[8]):
            bozi_bian = -bozi_bian

        center_bian = distance_point_to_line_segment(center, huanjing[8], huanjing[5])
        if not zai_or_buzai(center, huanjing[5], huanjing[6], huanjing[7], huanjing[8]):
            center_bian = -center_bian

        tail_bian = distance_point_to_line_segment(tail, huanjing[8], huanjing[5])
        if not zai_or_buzai(tail, huanjing[5], huanjing[6], huanjing[7], huanjing[8]):
            tail_bian = -tail_bian

    current_frame = [nose_left, nose_right, left_bozi, right_bozi, bozi_center, center_tail, nose_tail, head_angle,
                     body_angle, head_left_right, box_area, nose_water, nose_food_bool, nose_tiesi_center,
                     left_tiesi_center, right_tiesi_center, bozi_tiesi_center, center_tiesi_center, tail_tiesi_center,
                     nose_bian, left_bian, right_bian, bozi_bian, center_bian, tail_bian]
    shang_1 = [nose_dis, left_dis, right_dis, bozi_dis, center_dis, tail_dis, box_center_dis, box_ds, nose_left_dd,
               nose_right_dd, left_bozi_dd, right_bozi_dd, bozi_center_dd, center_tail_dd, nose_tail_dd]

    tezheng = current_frame + shang_1

    jieguo_biaozhunhua = scaler_geti_datasets.transform([tezheng])
    pred_1 = geti_model.predict(jieguo_biaozhunhua)[0]

    return pred_1


def Occurs_n_times(lst):
    count = 0
    for i in range(len(lst)):
        if lst[i] == 9:
            count += 1
        else:
            if count < 599:
                for j in range(i - count, i):
                    lst[j] = 10
            count = 0
    # 处理结尾的情况
    if count < 599:
        for j in range(len(lst) - count, len(lst)):
            lst[j] = 10
    return lst

#原来的
# def count_consecutive_numbers(lst):
#     """
#     代码会输出每个数字连续相邻大于等于三次的个数以及总的个数
#     """
#     result = {}
#     current_number = None
#     count = 0
#
#     for number in lst:
#         if number == current_number:
#             count += 1
#         else:
#             if count >= 3:
#                 if current_number not in result:
#                     result[current_number] = []
#                 result[current_number].append(count)
#             current_number = number
#             count = 1
#
#     if count >= 3:
#         if current_number not in result:
#             result[current_number] = []
#         result[current_number].append(count)
#
#     return result

#加上时间的
def count_consecutive_numbers(lst):
    result = {}
    current_number = None
    count = 0
    start_index = None

    for index, number in enumerate(lst):
        if number == current_number:
            count += 1
        else:
            if count >= 3:
                if current_number not in result:
                    result[current_number] = [[], []]
                result[current_number][0].append(count)
                result[current_number][1].append(start_index)
            current_number = number
            count = 1
            start_index = index

    if count >= 3:
        if current_number not in result:
            result[current_number] = [[], []]
        result[current_number][0].append(count)
        result[current_number][1].append(start_index)

    return result



def save_xlsx(mouse_one_geti,mouse_two_geti,mouse_three_geti,s_12,s_13,s_23,s_123,wanzhenglujing):

    m1_0, m1_1, m1_2, m1_3, m1_4, m1_5, m1_6, m1_7, m1_9 = [], [], [], [], [], [], [], [], []
    m2_0, m2_1, m2_2, m2_3, m2_4, m2_5, m2_6, m2_7, m2_9 = [], [], [], [], [], [], [], [], []
    m3_0, m3_1, m3_2, m3_3, m3_4, m3_5, m3_6, m3_7, m3_9 = [], [], [], [], [], [], [], [], []

    for number, counts in count_consecutive_numbers(mouse_one_geti).items():
        if number == 0:
            m1_0 = counts
        elif number == 1:
            m1_1 = counts
        elif number == 2:
            m1_2 = counts
        elif number == 3:
            m1_3 = counts
        elif number == 4:
            m1_4 = counts
        elif number == 5:
            m1_5 = counts
        elif number == 6:
            m1_6 = counts
        elif number == 7:
            m1_7 = counts
        elif number == 8:
            m1_8 = counts
        elif number == 9:
            m1_9 = counts

    for number, counts in count_consecutive_numbers(mouse_two_geti).items():
        if number == 0:
            m2_0 = counts
        elif number == 1:
            m2_1 = counts
        elif number == 2:
            m2_2 = counts
        elif number == 3:
            m2_3 = counts
        elif number == 4:
            m2_4 = counts
        elif number == 5:
            m2_5 = counts
        elif number == 6:
            m2_6 = counts
        elif number == 7:
            m2_7 = counts
        elif number == 8:
            m2_8 = counts
        elif number == 9:
            m2_9 = counts

    for number, counts in count_consecutive_numbers(mouse_three_geti).items():
        if number == 0:
            m3_0 = counts
        elif number == 1:
            m3_1 = counts
        elif number == 2:
            m3_2 = counts
        elif number == 3:
            m3_3 = counts
        elif number == 4:
            m3_4 = counts
        elif number == 5:
            m3_5 = counts
        elif number == 6:
            m3_6 = counts
        elif number == 7:
            m3_7 = counts
        elif number == 8:
            m3_8 = counts
        elif number == 9:
            m3_9 = counts
    m12_1 = []
    m12_2 = []
    m12_3 = []
    m12_A0 = []
    m12_A1 = []
    m12_A2 = []
    m12_A3 = []
    m12_A4 = []
    m12_B0 = []
    m12_B1 = []
    m12_B2 = []
    m12_B3 = []
    m12_B4 = []
    for number, counts in count_consecutive_numbers(s_12).items():
        if number == 1:
            m12_1 = counts
        elif number == 2:
            m12_2 = counts
        elif number == 3:
            m12_3 = counts
        elif number == "A0":
            m12_A0 = counts
        elif number == "A1":
            m12_A1 = counts
        elif number == "A2":
            m12_A2 = counts
        elif number == "A3":
            m12_A3 = counts
        elif number == "A4":
            m12_A4 = counts
        elif number == "B0":
            m12_B0 = counts
        elif number == "B1":
            m12_B1 = counts
        elif number == "B2":
            m12_B2 = counts
        elif number == "B3":
            m12_B3 = counts
        elif number == "B4":
            m12_B4 = counts

    m13_1 = []
    m13_2 = []
    m13_3 = []
    m13_A0 = []
    m13_A1 = []
    m13_A2 = []
    m13_A3 = []
    m13_A4 = []
    m13_B0 = []
    m13_B1 = []
    m13_B2 = []
    m13_B3 = []
    m13_B4 = []
    for number, counts in count_consecutive_numbers(s_13).items():
        if number == 1:
            m13_1 = counts
        elif number == 2:
            m13_2 = counts
        elif number == 3:
            m13_3 = counts
        elif number == "A0":
            m13_A0 = counts
        elif number == "A1":
            m13_A1 = counts
        elif number == "A2":
            m13_A2 = counts
        elif number == "A3":
            m13_A3 = counts
        elif number == "A4":
            m13_A4 = counts
        elif number == "B0":
            m13_B0 = counts
        elif number == "B1":
            m13_B1 = counts
        elif number == "B2":
            m13_B2 = counts
        elif number == "B3":
            m13_B3 = counts
        elif number == "B4":
            m13_B4 = counts

    m23_1 = []
    m23_2 = []
    m23_3 = []
    m23_A0 = []
    m23_A1 = []
    m23_A2 = []
    m23_A3 = []
    m23_A4 = []
    m23_B0 = []
    m23_B1 = []
    m23_B2 = []
    m23_B3 = []
    m23_B4 = []
    for number, counts in count_consecutive_numbers(s_23).items():
        if number == 1:
            m23_1 = counts
        elif number == 2:
            m23_2 = counts
        elif number == 3:
            m23_3 = counts
        elif number == "A0":
            m23_A0 = counts
        elif number == "A1":
            m23_A1 = counts
        elif number == "A2":
            m23_A2 = counts
        elif number == "A3":
            m23_A3 = counts
        elif number == "A4":
            m23_A4 = counts
        elif number == "B0":
            m23_B0 = counts
        elif number == "B1":
            m23_B1 = counts
        elif number == "B2":
            m23_B2 = counts
        elif number == "B3":
            m23_B3 = counts
        elif number == "B4":
            m23_B4 = counts

    m123 = []
    for number, counts in count_consecutive_numbers(s_123).items():
        if number == 1:
            m123 = counts

    # 保存行为为xslx文件
    workbook = xlsxwriter.Workbook(wanzhenglujing)  # 创建一个excel文件
    worksheet1 = workbook.add_worksheet("mouse1")
    worksheet2 = workbook.add_worksheet("mouse2")
    worksheet3 = workbook.add_worksheet("mouse3")
    worksheet4 = workbook.add_worksheet("mouse12")
    worksheet5 = workbook.add_worksheet("mouse13")
    worksheet6 = workbook.add_worksheet("mouse23")
    worksheet7 = workbook.add_worksheet("mouse123")

    worksheet1.write('A1', '进食')
    worksheet1.write('B1', '饮水')
    worksheet1.write('C1', '行走')
    worksheet1.write('D1', '梳理')
    worksheet1.write('E1', '攀爬')
    worksheet1.write('F1', '嗅探')
    worksheet1.write('G1', '静止')
    worksheet1.write('H1', '支撑站立')
    worksheet1.write('I1', '睡觉')
    worksheet1.write_column(1, 0, m1_0)
    worksheet1.write_column(1, 1, m1_1)
    worksheet1.write_column(1, 2, m1_2)
    worksheet1.write_column(1, 3, m1_3)
    worksheet1.write_column(1, 4, m1_4)
    worksheet1.write_column(1, 5, m1_5)
    worksheet1.write_column(1, 6, m1_6)
    worksheet1.write_column(1, 7, m1_7)
    # worksheet1.write_column(1, 8, m1_8)
    worksheet1.write_column(1, 8, m1_9)

    worksheet2.write('A1', '进食')
    worksheet2.write('B1', '饮水')
    worksheet2.write('C1', '行走')
    worksheet2.write('D1', '梳理')
    worksheet2.write('E1', '攀爬')
    worksheet2.write('F1', '嗅探')
    worksheet2.write('G1', '静止')
    worksheet2.write('H1', '支撑站立')
    worksheet2.write('I1', '睡觉')
    worksheet2.write_column(1, 0, m2_0)
    worksheet2.write_column(1, 1, m2_1)
    worksheet2.write_column(1, 2, m2_2)
    worksheet2.write_column(1, 3, m2_3)
    worksheet2.write_column(1, 4, m2_4)
    worksheet2.write_column(1, 5, m2_5)
    worksheet2.write_column(1, 6, m2_6)
    worksheet2.write_column(1, 7, m2_7)
    # worksheet2.write_column(1, 8, m2_8)
    worksheet2.write_column(1, 8, m2_9)

    worksheet3.write('A1', '进食')
    worksheet3.write('B1', '饮水')
    worksheet3.write('C1', '行走')
    worksheet3.write('D1', '梳理')
    worksheet3.write('E1', '攀爬')
    worksheet3.write('F1', '嗅探')
    worksheet3.write('G1', '静止')
    worksheet3.write('H1', '支撑站立')
    worksheet3.write('I1', '睡觉')
    worksheet3.write_column(1, 0, m3_0)
    worksheet3.write_column(1, 1, m3_1)
    worksheet3.write_column(1, 2, m3_2)
    worksheet3.write_column(1, 3, m3_3)
    worksheet3.write_column(1, 4, m3_4)
    worksheet3.write_column(1, 5, m3_5)
    worksheet3.write_column(1, 6, m3_6)
    worksheet3.write_column(1, 7, m3_7)
    # worksheet3.write_column(1, 8, m3_8)
    worksheet3.write_column(1, 8, m3_9)

    worksheet4.write('A1', "相互嗅探")
    worksheet4.write('B1', '两只靠近')
    worksheet4.write('C1', '相互打架')
    worksheet4.write('D1', '前者嗅探后者面部')
    worksheet4.write('E1', '前者嗅探后者身体')
    worksheet4.write('F1', '前者嗅探后者尾部')
    worksheet4.write('G1', '前者追逐后者')
    worksheet4.write('H1', '前者打后者')
    worksheet4.write('I1', '后者嗅探前者面部')
    worksheet4.write('J1', '后者嗅探前者身体')
    worksheet4.write('K1', '后者嗅探前者尾部')
    worksheet4.write('L1', '后者追逐前者')
    worksheet4.write('M1', '后者打前者')
    worksheet4.write_column(1, 0, m12_1)
    worksheet4.write_column(1, 1, m12_2)
    worksheet4.write_column(1, 2, m12_3)
    worksheet4.write_column(1, 3, m12_A0)
    worksheet4.write_column(1, 4, m12_A1)
    worksheet4.write_column(1, 5, m12_A2)
    worksheet4.write_column(1, 6, m12_A3)
    worksheet4.write_column(1, 7, m12_A4)
    worksheet4.write_column(1, 8, m12_B0)
    worksheet4.write_column(1, 9, m12_B1)
    worksheet4.write_column(1, 10, m12_B2)
    worksheet4.write_column(1, 11, m12_B3)
    worksheet4.write_column(1, 12, m12_B4)

    worksheet5.write('A1', "相互嗅探")
    worksheet5.write('B1', '两只靠近')
    worksheet5.write('C1', '相互打架')
    worksheet5.write('D1', '前者嗅探后者面部')
    worksheet5.write('E1', '前者嗅探后者身体')
    worksheet5.write('F1', '前者嗅探后者尾部')
    worksheet5.write('G1', '前者追逐后者')
    worksheet5.write('H1', '前者打后者')
    worksheet5.write('I1', '后者嗅探前者面部')
    worksheet5.write('J1', '后者嗅探前者身体')
    worksheet5.write('K1', '后者嗅探前者尾部')
    worksheet5.write('L1', '后者追逐前者')
    worksheet5.write('M1', '后者打前者')
    worksheet5.write_column(1, 0, m13_1)
    worksheet5.write_column(1, 1, m13_2)
    worksheet5.write_column(1, 2, m13_3)
    worksheet5.write_column(1, 3, m13_A0)
    worksheet5.write_column(1, 4, m13_A1)
    worksheet5.write_column(1, 5, m13_A2)
    worksheet5.write_column(1, 6, m13_A3)
    worksheet5.write_column(1, 7, m13_A4)
    worksheet5.write_column(1, 8, m13_B0)
    worksheet5.write_column(1, 9, m13_B1)
    worksheet5.write_column(1, 10, m13_B2)
    worksheet5.write_column(1, 11, m13_B3)
    worksheet5.write_column(1, 12, m13_B4)

    worksheet6.write('A1', '相互嗅探')
    worksheet6.write('B1', '两只靠近')
    worksheet6.write('C1', '相互打架')
    worksheet6.write('D1', '前者嗅探后者面部')
    worksheet6.write('E1', '前者嗅探后者身体')
    worksheet6.write('F1', '前者嗅探后者尾部')
    worksheet6.write('G1', '前者追逐后者')
    worksheet6.write('H1', '前者打后者')
    worksheet6.write('I1', '后者嗅探前者面部')
    worksheet6.write('J1', '后者嗅探前者身体')
    worksheet6.write('K1', '后者嗅探前者尾部')
    worksheet6.write('L1', '后者追逐前者')
    worksheet6.write('M1', '后者打前者')
    worksheet6.write_column(1, 0, m23_1)
    worksheet6.write_column(1, 1, m23_2)
    worksheet6.write_column(1, 2, m23_3)
    worksheet6.write_column(1, 3, m23_A0)
    worksheet6.write_column(1, 4, m23_A1)
    worksheet6.write_column(1, 5, m23_A2)
    worksheet6.write_column(1, 6, m23_A3)
    worksheet6.write_column(1, 7, m23_A4)
    worksheet6.write_column(1, 8, m23_B0)
    worksheet6.write_column(1, 9, m23_B1)
    worksheet6.write_column(1, 10, m23_B2)
    worksheet6.write_column(1, 11, m23_B3)
    worksheet6.write_column(1, 12, m23_B4)

    worksheet7.write('A1', '三只聚集')
    worksheet7.write_column(1, 0, m123)

    workbook.close()  # 关闭工作簿

def save_xlsx_1(mouse_one_geti,mouse_two_geti,mouse_three_geti,s_12,s_13,s_23,s_123,wanzhenglujing,star_frame,wenjian_star):


    m1_0, m1_1, m1_2, m1_3, m1_4, m1_5, m1_6, m1_7, m1_9 = [], [], [], [], [], [], [], [], []
    m1_0_time, m1_1_time, m1_2_time, m1_3_time, m1_4_time, m1_5_time, m1_6_time, m1_7_time, m1_9_time = [], [], [], [], [], [], [], [], []
    m2_0, m2_1, m2_2, m2_3, m2_4, m2_5, m2_6, m2_7, m2_9 = [], [], [], [], [], [], [], [], []
    m2_0_time, m2_1_time, m2_2_time, m2_3_time, m2_4_time, m2_5_time, m2_6_time, m2_7_time, m2_9_time = [], [], [], [], [], [], [], [], []
    m3_0, m3_1, m3_2, m3_3, m3_4, m3_5, m3_6, m3_7, m3_9 = [], [], [], [], [], [], [], [], []
    m3_0_time, m3_1_time, m3_2_time, m3_3_time, m3_4_time, m3_5_time, m3_6_time, m3_7_time, m3_9_time = [], [], [], [], [], [], [], [], []

    for number, counts in count_consecutive_numbers(mouse_one_geti).items():
        if number == 0:
            m1_0,m1_0_time = counts
        elif number == 1:
            m1_1,m1_1_time = counts
        elif number == 2:
            m1_2,m1_2_time= counts
        elif number == 3:
            m1_3,m1_3_time = counts
        elif number == 4:
            m1_4,m1_4_time= counts
        elif number == 5:
            m1_5,m1_5_time= counts
        elif number == 6:
            m1_6,m1_6_time = counts
        elif number == 7:
            m1_7,m1_7_time= counts
        elif number == 8:
            m1_8 = counts
        elif number == 9:
            m1_9,m1_9_time = counts

    for number, counts in count_consecutive_numbers(mouse_two_geti).items():
        if number == 0:
            m2_0,m2_0_time = counts
        elif number == 1:
            m2_1,m2_1_time = counts
        elif number == 2:
            m2_2,m2_2_time= counts
        elif number == 3:
            m2_3,m2_3_time = counts
        elif number == 4:
            m2_4,m2_4_time= counts
        elif number == 5:
            m2_5,m2_5_time= counts
        elif number == 6:
            m2_6,m2_6_time = counts
        elif number == 7:
            m2_7,m2_7_time= counts
        elif number == 8:
            m2_8 = counts
        elif number == 9:
            m2_9,m2_9_time = counts

    for number, counts in count_consecutive_numbers(mouse_three_geti).items():
        if number == 0:
            m3_0,m3_0_time = counts
        elif number == 1:
            m3_1,m3_1_time = counts
        elif number == 2:
            m3_2,m3_2_time= counts
        elif number == 3:
            m3_3,m3_3_time = counts
        elif number == 4:
            m3_4,m3_4_time= counts
        elif number == 5:
            m3_5,m3_5_time= counts
        elif number == 6:
            m3_6,m3_6_time = counts
        elif number == 7:
            m3_7,m3_7_time= counts
        elif number == 8:
            m3_8 = counts
        elif number == 9:
            m3_9,m3_9_time = counts
    m12_1 = []
    m12_2 = []
    m12_3 = []
    m12_A0 = []
    m12_A1 = []
    m12_A2 = []
    m12_A3 = []
    m12_A4 = []
    m12_B0 = []
    m12_B1 = []
    m12_B2 = []
    m12_B3 = []
    m12_B4 = []

    m12_time_1 = []
    m12_time_2 = []
    m12_time_3 = []
    m12_time_A0 = []
    m12_time_A1 = []
    m12_time_A2 = []
    m12_time_A3 = []
    m12_time_A4 = []
    m12_time_B0 = []
    m12_time_B1 = []
    m12_time_B2 = []
    m12_time_B3 = []
    m12_time_B4 = []


    for number, counts in count_consecutive_numbers(s_12).items():
        if number == 1:
            m12_1,m12_time_1= counts
        elif number == 2:
            m12_2,m12_time_2 = counts
        elif number == 3:
            m12_3,m12_time_3 = counts
        elif number == "A0":
            m12_A0,m12_time_A0 = counts
        elif number == "A1":
            m12_A1,m12_time_A1 = counts
        elif number == "A2":
            m12_A2,m12_time_A2 = counts
        elif number == "A3":
            m12_A3,m12_time_A3 = counts
        elif number == "A4":
            m12_A4,m12_time_A4 = counts
        elif number == "B0":
            m12_B0,m12_time_B0 = counts
        elif number == "B1":
            m12_B1,m12_time_B1 = counts
        elif number == "B2":
            m12_B2,m12_time_B2 = counts
        elif number == "B3":
            m12_B3,m12_time_B3 = counts
        elif number == "B4":
            m12_B4,m12_time_B4 = counts

    m13_1 = []
    m13_2 = []
    m13_3 = []
    m13_A0 = []
    m13_A1 = []
    m13_A2 = []
    m13_A3 = []
    m13_A4 = []
    m13_B0 = []
    m13_B1 = []
    m13_B2 = []
    m13_B3 = []
    m13_B4 = []

    m13_time_1 = []
    m13_time_2 = []
    m13_time_3 = []
    m13_time_A0 = []
    m13_time_A1 = []
    m13_time_A2 = []
    m13_time_A3 = []
    m13_time_A4 = []
    m13_time_B0 = []
    m13_time_B1 = []
    m13_time_B2 = []
    m13_time_B3 = []
    m13_time_B4 = []


    for number, counts in count_consecutive_numbers(s_13).items():
        if number == 1:
            m13_1,m13_time_1= counts
        elif number == 2:
            m13_2,m13_time_2 = counts
        elif number == 3:
            m13_3,m13_time_3 = counts
        elif number == "A0":
            m13_A0,m13_time_A0 = counts
        elif number == "A1":
            m13_A1,m13_time_A1 = counts
        elif number == "A2":
            m13_A2,m13_time_A2 = counts
        elif number == "A3":
            m13_A3,m13_time_A3 = counts
        elif number == "A4":
            m13_A4,m13_time_A4 = counts
        elif number == "B0":
            m13_B0,m13_time_B0 = counts
        elif number == "B1":
            m13_B1,m13_time_B1 = counts
        elif number == "B2":
            m13_B2,m13_time_B2 = counts
        elif number == "B3":
            m13_B3,m13_time_B3 = counts
        elif number == "B4":
            m13_B4,m13_time_B4 = counts

    m23_1 = []
    m23_2 = []
    m23_3 = []
    m23_A0 = []
    m23_A1 = []
    m23_A2 = []
    m23_A3 = []
    m23_A4 = []
    m23_B0 = []
    m23_B1 = []
    m23_B2 = []
    m23_B3 = []
    m23_B4 = []

    m23_time_1 = []
    m23_time_2 = []
    m23_time_3 = []
    m23_time_A0 = []
    m23_time_A1 = []
    m23_time_A2 = []
    m23_time_A3 = []
    m23_time_A4 = []
    m23_time_B0 = []
    m23_time_B1 = []
    m23_time_B2 = []
    m23_time_B3 = []
    m23_time_B4 = []


    for number, counts in count_consecutive_numbers(s_23).items():
        if number == 1:
            m23_1,m23_time_1= counts
        elif number == 2:
            m23_2,m23_time_2 = counts
        elif number == 3:
            m23_3,m23_time_3 = counts
        elif number == "A0":
            m23_A0,m23_time_A0 = counts
        elif number == "A1":
            m23_A1,m23_time_A1 = counts
        elif number == "A2":
            m23_A2,m23_time_A2 = counts
        elif number == "A3":
            m23_A3,m23_time_A3 = counts
        elif number == "A4":
            m23_A4,m23_time_A4 = counts
        elif number == "B0":
            m23_B0,m23_time_B0 = counts
        elif number == "B1":
            m23_B1,m23_time_B1 = counts
        elif number == "B2":
            m23_B2,m23_time_B2 = counts
        elif number == "B3":
            m23_B3,m23_time_B3 = counts
        elif number == "B4":
            m23_B4,m23_time_B4 = counts


    m123 = []
    m123_time = []
    for number, counts in count_consecutive_numbers(s_123).items():
        if number == 1:
            m123,m123_time = counts

    #将所有的时间转为为真实时间
    def frame2time(lst):
        from datetime import datetime,timedelta
        # 给定的时间字符串
        time_str = wenjian_star
        # 使用 datetime.strptime 方法解析时间字符串
        parsed_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")

        for i in range(len(lst)):
            shijian = round((lst[i]+star_frame)/15,3)
            delta = timedelta(seconds=shijian)
            result_time = parsed_time + delta
            lst[i] = result_time.strftime('%Y-%m-%d %H:%M:%S:%f')[:-3]

    frame2time(m1_0_time)
    frame2time(m1_1_time)
    frame2time(m1_2_time)
    frame2time(m1_3_time)
    frame2time(m1_4_time)
    frame2time(m1_5_time)
    frame2time(m1_6_time)
    frame2time(m1_7_time)
    frame2time(m1_9_time)
    frame2time(m2_0_time)
    frame2time(m2_1_time)
    frame2time(m2_2_time)
    frame2time(m2_3_time)
    frame2time(m2_4_time)
    frame2time(m2_5_time)
    frame2time(m2_6_time)
    frame2time(m2_7_time)
    frame2time(m2_9_time)
    frame2time(m3_0_time)
    frame2time(m3_1_time)
    frame2time(m3_2_time)
    frame2time(m3_3_time)
    frame2time(m3_4_time)
    frame2time(m3_5_time)
    frame2time(m3_6_time)
    frame2time(m3_7_time)
    frame2time(m3_9_time)
    #社交
    frame2time(m12_time_1)
    frame2time(m12_time_2)
    frame2time(m12_time_3)
    frame2time(m12_time_A0)
    frame2time(m12_time_A1)
    frame2time(m12_time_A2)
    frame2time(m12_time_A3)
    frame2time(m12_time_A4)
    frame2time(m12_time_B0)
    frame2time(m12_time_B1)
    frame2time(m12_time_B2)
    frame2time(m12_time_B3)
    frame2time(m12_time_B4)
    frame2time(m13_time_1)
    frame2time(m13_time_2)
    frame2time(m13_time_3)
    frame2time(m13_time_A0)
    frame2time(m13_time_A1)
    frame2time(m13_time_A2)
    frame2time(m13_time_A3)
    frame2time(m13_time_A4)
    frame2time(m13_time_B0)
    frame2time(m13_time_B1)
    frame2time(m13_time_B2)
    frame2time(m13_time_B3)
    frame2time(m13_time_B4)
    frame2time(m23_time_1)
    frame2time(m23_time_2)
    frame2time(m23_time_3)
    frame2time(m23_time_A0)
    frame2time(m23_time_A1)
    frame2time(m23_time_A2)
    frame2time(m23_time_A3)
    frame2time(m23_time_A4)
    frame2time(m23_time_B0)
    frame2time(m23_time_B1)
    frame2time(m23_time_B2)
    frame2time(m23_time_B3)
    frame2time(m23_time_B4)
    frame2time(m123_time)



    # 保存行为为xslx文件
    workbook = xlsxwriter.Workbook(wanzhenglujing)  # 创建一个excel文件
    worksheet1 = workbook.add_worksheet("mouse1")
    worksheet2 = workbook.add_worksheet("mouse2")
    worksheet3 = workbook.add_worksheet("mouse3")
    worksheet4 = workbook.add_worksheet("mouse12")
    worksheet5 = workbook.add_worksheet("mouse13")
    worksheet6 = workbook.add_worksheet("mouse23")
    worksheet7 = workbook.add_worksheet("mouse123")

    def getibiaotou(sheet):

        sheet.write('A1', '进食开始时间')
        sheet.write('B1', '进食')
        sheet.write('C1', '饮水开始时间')
        sheet.write('D1', '饮水')
        sheet.write('E1', '行走开始时间')
        sheet.write('F1', '行走')
        sheet.write('G1', '梳理开始时间')
        sheet.write('H1', '梳理')
        sheet.write('I1', '攀爬开始时间')
        sheet.write('J1', '攀爬')
        sheet.write('K1', '嗅探开始时间')
        sheet.write('L1', '嗅探')
        sheet.write('M1', '静止开始时间')
        sheet.write('N1', '静止')
        sheet.write('O1', '支撑站立开始时间')
        sheet.write('P1', '支撑站立')
        sheet.write('Q1', '睡觉开始时间')
        sheet.write('R1', '睡觉')
    getibiaotou(worksheet1)
    getibiaotou(worksheet2)
    getibiaotou(worksheet3)



    worksheet1.write_column(1, 0, m1_0_time)
    worksheet1.write_column(1, 1, m1_0)
    worksheet1.write_column(1, 2, m1_1_time)
    worksheet1.write_column(1, 3, m1_1)
    worksheet1.write_column(1, 4, m1_2_time)
    worksheet1.write_column(1, 5, m1_2)
    worksheet1.write_column(1, 6, m1_3_time)
    worksheet1.write_column(1, 7, m1_3)
    worksheet1.write_column(1, 8, m1_4_time)
    worksheet1.write_column(1, 9, m1_4)
    worksheet1.write_column(1, 10, m1_5_time)
    worksheet1.write_column(1, 11, m1_5)
    worksheet1.write_column(1, 12, m1_6_time)
    worksheet1.write_column(1, 13, m1_6)
    worksheet1.write_column(1, 14, m1_7_time)
    worksheet1.write_column(1, 15, m1_7)
    # worksheet1.write_column(1, 0, m1_0_time)
    # worksheet1.write_column(1, 8, m1_8)
    worksheet1.write_column(1, 16, m1_9_time)
    worksheet1.write_column(1, 17, m1_9)

    worksheet2.write_column(1, 0, m2_0_time)
    worksheet2.write_column(1, 1, m2_0)
    worksheet2.write_column(1, 2, m2_1_time)
    worksheet2.write_column(1, 3, m2_1)
    worksheet2.write_column(1, 4, m2_2_time)
    worksheet2.write_column(1, 5, m2_2)
    worksheet2.write_column(1, 6, m2_3_time)
    worksheet2.write_column(1, 7, m2_3)
    worksheet2.write_column(1, 8, m2_4_time)
    worksheet2.write_column(1, 9, m2_4)
    worksheet2.write_column(1, 10, m2_5_time)
    worksheet2.write_column(1, 11, m2_5)
    worksheet2.write_column(1, 12, m2_6_time)
    worksheet2.write_column(1, 13, m2_6)
    worksheet2.write_column(1, 14, m2_7_time)
    worksheet2.write_column(1, 15, m2_7)
    # worksheet2.write_column(1, 0, m2_0_time)
    # worksheet2.write_column(1, 8, m2_8)
    worksheet2.write_column(1, 16, m2_9_time)
    worksheet2.write_column(1, 17, m2_9)

    worksheet3.write_column(1, 0, m3_0_time)
    worksheet3.write_column(1, 1, m3_0)
    worksheet3.write_column(1, 2, m3_1_time)
    worksheet3.write_column(1, 3, m3_1)
    worksheet3.write_column(1, 4, m3_2_time)
    worksheet3.write_column(1, 5, m3_2)
    worksheet3.write_column(1, 6, m3_3_time)
    worksheet3.write_column(1, 7, m3_3)
    worksheet3.write_column(1, 8, m3_4_time)
    worksheet3.write_column(1, 9, m3_4)
    worksheet3.write_column(1, 10, m3_5_time)
    worksheet3.write_column(1, 11, m3_5)
    worksheet3.write_column(1, 12, m3_6_time)
    worksheet3.write_column(1, 13, m3_6)
    worksheet3.write_column(1, 14, m3_7_time)
    worksheet3.write_column(1, 15, m3_7)
    # worksheet3.write_column(1, 0, m3_0_time)
    # worksheet3.write_column(1, 8, m3_8)
    worksheet3.write_column(1, 16, m3_9_time)
    worksheet3.write_column(1, 17, m3_9)

    def shejiaobiaotou(sheet):

        sheet.write('A1', "相互嗅探开始时间")
        sheet.write('B1', "相互嗅探")
        sheet.write('C1', '两只靠近开始时间')
        sheet.write('D1', '两只靠近')
        sheet.write('E1', '相互打架开始时间')
        sheet.write('F1', '相互打架')
        sheet.write('G1', '前者嗅探后者面部开始时间')
        sheet.write('H1', '前者嗅探后者面部')
        sheet.write('I1', '前者嗅探后者身体开始时间')
        sheet.write('J1', '前者嗅探后者身体')
        sheet.write('K1', '前者嗅探后者尾部开始时间')
        sheet.write('L1', '前者嗅探后者尾部')
        sheet.write('M1', '前者追逐后者开始时间')
        sheet.write('N1', '前者追逐后者')
        sheet.write('O1', '前者打后者开始时间')
        sheet.write('P1', '前者打后者')
        sheet.write('Q1', '后者嗅探前者面部开始时间')
        sheet.write('R1', '后者嗅探前者面部')
        sheet.write('S1', '后者嗅探前者身体开始时间')
        sheet.write('T1', '后者嗅探前者身体')
        sheet.write('U1', '后者嗅探前者尾部开始时间')
        sheet.write('V1', '后者嗅探前者尾部')
        sheet.write('W1', '后者追逐前者开始时间')
        sheet.write('X1', '后者追逐前者')
        sheet.write('Y1', '后者打前者开始时间')
        sheet.write('Z1', '后者打前者')

    shejiaobiaotou(sheet=worksheet4)
    shejiaobiaotou(sheet=worksheet5)
    shejiaobiaotou(sheet=worksheet6)

    worksheet4.write_column(1, 0, m12_time_1)
    worksheet4.write_column(1, 1, m12_1)
    worksheet4.write_column(1, 2, m12_time_2)
    worksheet4.write_column(1, 3, m12_2)
    worksheet4.write_column(1, 4, m12_time_3)
    worksheet4.write_column(1, 5, m12_3)
    worksheet4.write_column(1, 6, m12_time_A0)
    worksheet4.write_column(1, 7, m12_A0)
    worksheet4.write_column(1, 8, m12_time_A1)
    worksheet4.write_column(1, 9, m12_A1)
    worksheet4.write_column(1, 10, m12_time_A2)
    worksheet4.write_column(1, 11, m12_A2)
    worksheet4.write_column(1, 12, m12_time_A3)
    worksheet4.write_column(1, 13, m12_A3)
    worksheet4.write_column(1, 14, m12_time_A4)
    worksheet4.write_column(1, 15, m12_A4)
    worksheet4.write_column(1, 16, m12_time_B0)
    worksheet4.write_column(1, 17, m12_B0)
    worksheet4.write_column(1, 18, m12_time_B1)
    worksheet4.write_column(1, 19, m12_B1)
    worksheet4.write_column(1, 20, m12_time_B2)
    worksheet4.write_column(1, 21, m12_B2)
    worksheet4.write_column(1, 22, m12_time_B3)
    worksheet4.write_column(1, 23, m12_B3)
    worksheet4.write_column(1, 24, m12_time_B4)
    worksheet4.write_column(1, 25, m12_B4)

    worksheet5.write_column(1, 0, m13_time_1)
    worksheet5.write_column(1, 1, m13_1)
    worksheet5.write_column(1, 2, m13_time_2)
    worksheet5.write_column(1, 3, m13_2)
    worksheet5.write_column(1, 4, m13_time_3)
    worksheet5.write_column(1, 5, m13_3)
    worksheet5.write_column(1, 6, m13_time_A0)
    worksheet5.write_column(1, 7, m13_A0)
    worksheet5.write_column(1, 8, m13_time_A1)
    worksheet5.write_column(1, 9, m13_A1)
    worksheet5.write_column(1, 10, m13_time_A2)
    worksheet5.write_column(1, 11, m13_A2)
    worksheet5.write_column(1, 12, m13_time_A3)
    worksheet5.write_column(1, 13, m13_A3)
    worksheet5.write_column(1, 14, m13_time_A4)
    worksheet5.write_column(1, 15, m13_A4)
    worksheet5.write_column(1, 16, m13_time_B0)
    worksheet5.write_column(1, 17, m13_B0)
    worksheet5.write_column(1, 18, m13_time_B1)
    worksheet5.write_column(1, 19, m13_B1)
    worksheet5.write_column(1, 20, m13_time_B2)
    worksheet5.write_column(1, 21, m13_B2)
    worksheet5.write_column(1, 22, m13_time_B3)
    worksheet5.write_column(1, 23, m13_B3)
    worksheet5.write_column(1, 24, m13_time_B4)
    worksheet5.write_column(1, 25, m13_B4)

    worksheet6.write_column(1, 0, m23_time_1)
    worksheet6.write_column(1, 1, m23_1)
    worksheet6.write_column(1, 2, m23_time_2)
    worksheet6.write_column(1, 3, m23_2)
    worksheet6.write_column(1, 4, m23_time_3)
    worksheet6.write_column(1, 5, m23_3)
    worksheet6.write_column(1, 6, m23_time_A0)
    worksheet6.write_column(1, 7, m23_A0)
    worksheet6.write_column(1, 8, m23_time_A1)
    worksheet6.write_column(1, 9, m23_A1)
    worksheet6.write_column(1, 10, m23_time_A2)
    worksheet6.write_column(1, 11, m23_A2)
    worksheet6.write_column(1, 12, m23_time_A3)
    worksheet6.write_column(1, 13, m23_A3)
    worksheet6.write_column(1, 14, m23_time_A4)
    worksheet6.write_column(1, 15, m23_A4)
    worksheet6.write_column(1, 16, m23_time_B0)
    worksheet6.write_column(1, 17, m23_B0)
    worksheet6.write_column(1, 18, m23_time_B1)
    worksheet6.write_column(1, 19, m23_B1)
    worksheet6.write_column(1, 20, m23_time_B2)
    worksheet6.write_column(1, 21, m23_B2)
    worksheet6.write_column(1, 22, m23_time_B3)
    worksheet6.write_column(1, 23, m23_B3)
    worksheet6.write_column(1, 24, m23_time_B4)
    worksheet6.write_column(1, 25, m23_B4)

    worksheet7.write('A1', '三只聚集开始时间')
    worksheet7.write('B1', '三只聚集')

    worksheet7.write_column(1, 0, m123_time)
    worksheet7.write_column(1, 1, m123)

    workbook.close()  # 关闭工作簿