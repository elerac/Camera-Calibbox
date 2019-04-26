import cv2
import numpy as np
import sys
from scipy.linalg import lstsq

#2値化処理のための，閾値決定
def binarization(img_box, scale=0.5):
    #カラー画像の場合，グレイスケールに変換する
    c = len(img_box.shape)
    if c==3:
        img_boxg = cv2.cvtColor(img_box, cv2.COLOR_BGR2GRAY)
    else:
        img_boxg = img_box
    img_blur = cv2.GaussianBlur(img_boxg, (5, 5), 0)
    thresh = 71
    winbin = "d : thresh +1   a : thresh -1    q : finish"
    print("d : thresh +1 \na : thresh -1 \nq : finish")
    while True:
        print("\r{0: >3d}".format(thresh), end="")
        ret, img_bin = cv2.threshold(img_blur, thresh, 255, cv2.THRESH_BINARY)
        img_binresize = cv2.resize(img_bin, None, fx=scale, fy=scale)
        cv2.imshow(winbin, img_binresize)
        key = cv2.waitKey(30)
        #key = ord("q")
        if key==ord("d"):
            thresh += 1
            if thresh>255:
                thresh=255
        elif key==ord("a"):
            thresh -= 1
            if thresh<0:
                thresh=0
        elif key==ord("q"):
            cv2.destroyWindow(winbin)
            print("\n")
            break
    return img_bin

#2点間の直線式の算出と，2直線の交点の算出
def calc_crosspoints(clicked_index, cog):
    imgpoints = []
    for i in range(0, 12, 4):
        index1 = clicked_index[i]
        index2 = clicked_index[i+1]
        index3 = clicked_index[i+2]
        index4 = clicked_index[i+3]
        for j in range(7):
            x1, y1 = cog[index1[j]]
            x2, y2 = cog[index2[j]]
            a = (y2-y1)/(x2-x1)
            b = y1-a*x1
            for k in range(7):
                x3, y3 = cog[index3[k]]
                x4, y4 = cog[index4[k]]
                c = (y4-y3)/(x4-x3)
                d = y3-c*x3
                #cross point y=ax+b, y=cx+d
                cross_x = (d-b)/(a-c)
                cross_y = (a*d-b*c)/(a-c)
                imgpoints.append([cross_x, cross_y])

    imgpoints = np.reshape(np.array(imgpoints), (147, 2)).astype(np.float32)
    return imgpoints

def calibrate_camera(coords, worlds):
    # build matrices
    data_num = coords.shape[0]
    A = np.zeros((data_num * 2, 11))
    b = np.zeros((data_num * 2))
    C23 = 1
    for i in range(data_num):
        u, v = coords[i]
        X, Y, Z = worlds[i]

        A[2 * i] = [X, Y, Z, 1, 0, 0, 0, 0, -X * u, -Y * u, -Z * u]
        A[2 * i + 1] = [0, 0, 0, 0, X, Y, Z, 1, -X * v, -Y * v, -Z * v]
        b[2 * i] = u * C23
        b[2 * i + 1] = v * C23

    # solve
    result = lstsq(A, b)
    C = np.append(result[0], C23).reshape((3, 4))
    return C

def main():
    def callback(event, X, Y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            x, y = X, Y
            #市街地距離を算出して、一番近い円のフラグを立てる
            distance = []
            for c in cog:
                cx = c[0]
                cy = c[1]
                d = abs(cx - x/scale) + abs(cy - y/scale)
                distance.append(d)
            min_i = np.argmin(np.array(distance))
            if flag[min_i]==False:
                flag[min_i] = True
                clicked_index.append(min_i)
                cx = cog[min_i][0]
                cy = cog[min_i][1]
                print("({0}, {1})".format(cx, cy))
    
    #表示する画像のスケール（ディスプレイの大きさによって調節）
    scale = 1.0
    #マーカの数（固定）
    marker_number = 7*4*3
    #立方体を撮影した画像
    img_box = cv2.imread(args[1], 1)
    size = cv2.cvtColor(img_box, cv2.COLOR_BGR2GRAY).shape[::-1]

    #2値化処理
    img_bin = binarization(img_box, scale)

    #ラベリング処理
    img_bin = 255 - img_bin
    label = cv2.connectedComponentsWithStats(img_bin)
    n = label[0] - 1
    rec = np.delete(label[2], 0, 0)
    cog = np.delete(label[3], 0, 0)

    #ブロブが規定の数に収まるように外れ値を省く
    if n<marker_number:
        print("This program need {0} markers.\nThere are only {1}".format(marker_number, n))
        exit()
    area = rec[:,4]
    index_delete = np.argsort(np.abs(area - np.median(area)))[marker_number:]
    n = marker_number
    rec = np.delete(rec, index_delete.tolist(), axis=0)
    cog = np.delete(cog, index_delete.tolist(), axis=0)

    #クリックされた時のフラグ
    flag = [False]*n
    flag_old = [True]*n
    #クリックしたインデックス
    clicked_index = []
    #ウインドウの定義
    window_name = "window"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, callback, cog)
    #色の定義
    color_selected = (47, 255, 173) #greenyellow
    color_unselected = (255, 204, 0) #skyblue
    #マーカの選択
    print("q : finish program and save coordinates\nr : reset")
    while True:
        #変更があった時のみ描画
        if flag_old != flag:
            flag_old = flag.copy()
            #円の描画処理
            for i in range(n):
                r = max(rec[i][2], rec[i][3])/2
                cx = int(cog[i][0])
                cy = int(cog[i][1])
                color = color_selected if flag[i] else color_unselected
                cv2.circle(img_box, (cx, cy), int(r*1.2), color, int((size[0]*size[1]*0.000001)**0.5))
                cv2.circle(img_box, (cx, cy), int(r*0.1), color, -1)
            img_show = cv2.resize(img_box, None, fx=scale, fy=scale)
        
        cv2.imshow(window_name, img_show)
        key = cv2.waitKey(300)
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("reset")
            flag = [False]*n
            clicked_index = []
    clicked_index = np.reshape(np.array(clicked_index), (12, 7)).tolist()

    #画像中の交点を求める
    imgpoints = calc_crosspoints(clicked_index, cog)
    
    #実空間上の交点の座標(8x8x8cmのボックスを想定)
    xz = [[[x+1, 0, z+1] for z in range(7)] for x in range(7)]
    yz = [[[0, y+1, z+1] for z in range(7)] for y in range(7)]
    xy = [[[x+1, y+1, 8] for y in range(7)] for x in range(7)]
    objpoints = np.reshape(np.array(xz+yz+xy), (147, 3)).astype(np.float32)

    C = calibrate_camera(imgpoints, objpoints)
    print("------------")
    print("C")
    print(C)
   
    cv2.destroyAllWindows()

if __name__=="__main__":
    args = sys.argv
    if len(args)!=2:
        print("Usage : python {} CalibBoxImage".format(args[0]))
    else:
        main()
