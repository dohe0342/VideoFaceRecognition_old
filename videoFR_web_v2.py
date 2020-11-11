import argparse
import cv2
import pickle
import os
import numpy as np
import sys
import glob
import time
import dlib
import face_model
import boostdb
from xml.etree.ElementTree import parse, Element, dump, SubElement
from collections import deque

retinaFaceDir = "/mydata/insightface/RetinaFace"
retinaFaceModelDir = '/mydata/insightface/RetinaFace/model/R50/R50'
recordDir = "/home/dohe0342/videoFR.avi"
benchmarkGoldenDir = '/home/dohe0342/VideoFaceRecognition/videos_chokepoint/*'
#goldenDir = '/mydata/openface/makeDataset/liverpool_capture_2/*'
#goldenDir = '/home/dohe0342/VideoFaceRecognition/videos/*'
goldenDir = '/home/dohe0342/VideoFaceRecognition/dohee/*'
benchmarkVidListDir = "/mydata/openface/benchmark_video/video_eval/*"
vidListDir = '/mydata/openface/makeDataset/liverpool_vid/*'
#vidListDir = '/home/dohe0342/insightface/RetinaFace/test_vid/*'
#vidListDir = '/mydata/openface/benchmark_video/video_eval_manypeople/*'
groundTruthDir = "/mydata/openface/benchmark_video/new_groundtruth/"

sys.path.append(retinaFaceDir)
from retinaface import RetinaFace

width = 1920
height = 1080
codec = 1196444237.0

dh_face_num = 0

#Make argparse object
parser = argparse.ArgumentParser()
parser.add_argument("--benchmark", action='store_true', help='Performance measure')
parser.add_argument("--video", action='store_true', help='Process already taken video(s)')
parser.add_argument("--display", action='store_true', help='Display FR result')
parser.add_argument("--record", action='store_true')
parser.add_argument("--back", type=str)
parser.add_argument("--db", action='store', required=True,
        help='Face db file name. Extract feature if no such file exists.')
parser.add_argument("--gpu", type=int, help='set gpu')
args = parser.parse_args()

if args.back == 'r50':
    arcFaceModelDir = '/mydata/insightface/recognition/models/model-r50-am-lfw/model,0'
elif args.back == 'r100':
    arcFaceModelDir = '/mydata/insightface/recognition/models/model-r100-ii/model,0'

class My_args:
    def __init__(self, gpu):
        self.image_size = '112, 112'
        self.model = arcFaceModelDir
        self.ga_model = ''
        self.gpu = gpu
        self.det = 0
        self.flip = 0
        self.threshold = 1.24

my_args = My_args(args.gpu)
insightfaceModel = face_model.FaceModel(my_args)

fd_threshold = 0.4
#gpuid = 0

dnn_detector = RetinaFace(retinaFaceModelDir, 0, args.gpu, 'net3')

buffer_size = 100
iou_threshold = 0.10
bench_iou = 0.4
detect_scale = 0.5

freeze_on = True
freeze_threshold = 0.65

unknown_metric = False
unknown_predictor_num = 10
guess_threshold = 0.30

fr_batch = True
db_gpu = True
db_tiled = True
aug_test = 1 #test purpose only!
rectToSqr = True

print_bench_fr_res = False
fullscreen = False
print_landmarks = False
print_time = False
debug = False

identity_dic = {}

# time measure params
first_df = True
tot_detect_face_time = 0
tot_update_facelist_time = 0
tot_create_facelist_time = 0
tot_getRep_time = 0
tot_update_prediction_time = 0
total_face_num = 0
skipped_faces = 0

#resized_shape = [960, 960]

class FaceObject:
    def __init__(self, loc):
        self.rectangle = loc
        self.rep_history = deque(maxlen=buffer_size)
        self.conf_history = deque(maxlen=buffer_size)
        self.pred_history = deque(maxlen=buffer_size)
        self.current_prediction = "UNKNWON"
        self.unknown_predictor = unknown_predictor_num / 2
        self.unknown_flag = False
        self.face_rectangle = loc
        self.now = 1
        self.landmark = None
        self.remove_flag = 0
        self.found_flag = 0
        self.current_conf = 0
        self.freeze_prediction = False
        
    def update_rectangle(self, loc):
        if((loc.bottom() > loc.top()) and (loc.left() < loc.right())):
            self.rectangle = loc
            self.face_rectangle = loc
        else:
            print("unvalid rectangle!")

    def add_rep_history(self, rep):
        self.rep_history.append(rep)
        
    def current_buffer_size(self):
        return len(self.rep_history)


def find_face(img):
    global total_face_num

    #resized = cv2.resize(img, dsize=(resized_shape[1], resized_shape[0]), interpolation=cv2.INTER_AREA)
    faces, landmarks = dnn_detector.detect(img, fd_threshold, im_scale=detect_scale, do_flip=False)
    BB = list()
    if faces is not None:
        for i in range(faces.shape[0]):
            total_face_num += 1
            box = faces[i].astype(np.int)
            if rectToSqr == True:
                w_m = int((box[0] + box[2]) / 2)
                h_m = int((box[1] + box[3]) / 2)
                w = int((box[2] - box[0]) / 2.0)
                #print(w*2)
                
                BB.append((dlib.rectangle(w_m-w, h_m-w, w_m+w, h_m+w), landmarks[i]))

                print(w_m-w, h_m-w, w_m+w, h_m+w)

            else:
                BB.append((dlib.rectangle(box[0], box[1], box[2], box[3]), landmarks[i]))

    return BB


def update_facelist_hog(img, facelist, BB):

    if len(facelist) == 0:
        return facelist, BB

    BB_temp = deque([])
    for bb in BB:
        BB_temp.append(bb)

    iou = np.zeros((len(BB), len(facelist)), dtype='f')

    for i, bb in enumerate(BB):
        for j, face in enumerate(facelist):
            boxA = [bb[0].left(), bb[0].top(), bb[0].right(), bb[0].bottom()]
            boxB = [face.rectangle.left(), face.rectangle.top(), face.rectangle.right(), face.rectangle.bottom()]
            iou[i][j] = bb_intersection_over_union(boxA, boxB)

    while True:
        if len(iou) == 0:
            break
        maxrow = np.argwhere(iou.max() == iou)[0][0]
        maxcol = np.argwhere(iou.max() == iou)[0][1]
        if (iou[maxrow][maxcol] > iou_threshold):
            if facelist[maxcol].found_flag == 0:
                facelist[maxcol].update_rectangle(BB[maxrow][0])
                facelist[maxcol].landmark = BB[maxrow][1]
                facelist[maxcol].found_flag = 1
            BB_temp.remove(BB[maxrow])
            iou[maxrow,:] = 0
        else:
            break

    BB = deque([])
    for bb in BB_temp:
        BB.append(bb)

    for face in facelist:
        if face.found_flag == 0:
            face.now = 0
            face.remove_flag = 1
        else:
            face.now = 1

        facelist = [face for face in facelist if face.remove_flag == 0]

    if len(facelist) == 0:
        return facelist, BB
        
    for face in facelist:
        face.found_flag = 0
                
    return ([face for face in facelist if face.remove_flag == 0], BB)


def create_facelist(img, BB, facelist):
    for bb in BB:
        facelist.append(FaceObject(bb[0]))
        facelist[-1].landmark = bb[1]

    return facelist


def getRep(face):
    rep = insightfaceModel.get_feature(face)

    return rep


def getRepAll(img, facelist):
    for face in facelist:
        if face.freeze_prediction and freeze_on:
            continue
        align = insightfaceModel.get_input_retina(img, face.rectangle, face.landmark)
        face.add_rep_history(getRep(align))

    return facelist


def getRepBatch(img, facelist):
    toProcess = []
    embeddings = []
    face_num = 0
    batch_size = 4

    global dh_face_num

    for enum, face in enumerate(facelist):
        if face.freeze_prediction and freeze_on:
            continue
        align = insightfaceModel.get_input_retina(img, face.rectangle, face.landmark)
        #img_align = np.reshape(align, (112,112,3))
        #cv2.imwrite('./cctv_res/dh_low_light/%s.jpg' %str(dh_face_num).zfill(4), img_align)
        dh_face_num += 1
        toProcess.append(align)
        face_num = enum
    face_num += 1
    if len(toProcess) == 0:
        return facelist
    toProcess = np.array(toProcess)
    toProcess = np.squeeze(toProcess)
    
    if face_num < batch_size:
        temp = np.zeros((batch_size,3,112,112))
        temp[0:toProcess.shape[0],:,:,:] += toProcess
        embeddings.append(insightfaceModel.get_feature(temp))
    elif face_num >= batch_size:
        batch_iter = int((face_num+(face_num%batch_size+batch_size-1))/batch_size)*batch_size
        temp = np.zeros((batch_iter,3,112,112))
        temp[0:toProcess.shape[0],:,:,:] += toProcess
        temp2 = []
        for i in range(int(batch_iter/batch_size)):
            temp2.append(temp[i*batch_size : (i+1)*batch_size, :, :, :])
        for batch_img in temp2:
            embeddings.append(insightfaceModel.get_feature(batch_img))
    
    i = 0
    for face in facelist:
        if face.freeze_prediction and freeze_on:
            continue
        add1 = int(i/batch_size)
        add2 = int(i%batch_size)
        face.add_rep_history(embeddings[add1][add2])
        #face.add_rep_history(embeddings[i])
        i += 1
    return facelist


def update_prediction(facelist, golden):
    global skipped_faces
    if not len(facelist):
        return facelist

#make frame vector
    frame_vec = np.array([]).astype(np.float32)
    for face in facelist:
        frame_vec = np.append(frame_vec, face.rep_history[-1])

    max_matrix = None
    
    if db_gpu:
        if db_tiled:
            max_matrix = boostdb.search_db_tiled(frame_vec)
        else:
            max_matrix = boostdb.search_db(frame_vec)
    else:
        max_matrix = boostdb.search_cpu(frame_vec, golden)

    for fn, face in enumerate(facelist):
        if face.freeze_prediction and freeze_on:
            skipped_faces += 1
            continue

        prediction_num = int(max_matrix[2 * fn])
        sim = guess_threshold
        prediction = "UNKNOWN"

        tempSim = float(max_matrix[2 * fn + 1])
        if tempSim > sim:
            sim = tempSim
            prediction = identity_dic[prediction_num]

        face.conf_history.append(sim)
        face.pred_history.append(prediction)

        if prediction == "UNKNOWN":
            face.unknown_predictor = np.clip(face.unknown_predictor + int(sim / 0.3), 1, unknown_predictor_num)
            if face.unknown_predictor > unknown_predictor_num / 2:
                face.unknown_flag = True
        else:
            face.unknown_predictor = np.clip(face.unknown_predictor - 1, 1, unknown_predictor_num)
            if face.conf_history[-1] > freeze_threshold and not face.unknown_flag:
                face.freeze_prediction = True
            if face.unknown_predictor <= unknown_predictor_num / 2:
                face.unknown_flag = False

        if not face.unknown_flag or not unknown_metric:
            maxIdx = np.argmax(face.conf_history)
            face.current_conf = face.conf_history[maxIdx]
            face.current_prediction = face.pred_history[maxIdx]
        else:
            face.current_prediction = "UNKNOWN"
    return facelist
        

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = (xB - xA + 1) * (yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    if xA >= xB or yA >= yB:
        iou = 0
    else:
        iou = interArea / float(boxAArea + boxBArea - interArea + 0.1)
    
    return iou


def draw_bb(img, facelist, fps, frame_number, vidName):
    cv2.putText(img,
            "FPS " + str(int(np.mean(fps))),
            (img.shape[1] - 300, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            5)
    
    cv2.putText(img,
            "Frame ID " + str(frame_number),
            (35, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 0, 0),
            5)

    for face in facelist:
        left = face.rectangle.left()
        right = face.rectangle.right()
        top = face.rectangle.top()
        bottom = face.rectangle.bottom()

        if face.current_prediction.split('_')[0] == 'UNKNOWN' or (face.unknown_flag and unknown_metric):
            cv2.rectangle(img, 
                (face.rectangle.left(), face.rectangle.top()),
                (face.rectangle.right(), face.rectangle.bottom()), 
                (0, 0, 255), 
                5)
        else:
            cv2.rectangle(img, 
                (face.rectangle.left(), face.rectangle.top()),
                (face.rectangle.right(), face.rectangle.bottom()), 
                (0, 255, 0), 
                5)

        if print_landmarks:
            for loc in face.landmark:
                cv2.circle(img, (loc[0], loc[1]), 4, (0, 255, 0), -1)

        displayLab = face.current_prediction

        description = None
        udescription = None
        if debug == True:
            description = "{}, {:.2f}, {}".format(displayLab, face.current_conf, face.unknown_predictor)
            udescription = "{}, {}".format(displayLab, face.unknown_predictor)
        else:
            description = "{}, {:.2f}".format(displayLab, face.current_conf)
            udescription = "{}".format(displayLab)

        if face.current_prediction.split('_')[0] == 'UNKNOWN' or (face.unknown_flag and unknown_metric):
            displayLab = 'UNKNOWN'
            cv2.putText(img, 
                         udescription,
                         (face.rectangle.left()-50, face.rectangle.top()-30), 
                         cv2.FONT_HERSHEY_SIMPLEX, 
                         1.5, 
                         (0, 0, 255), 
                         4)

        else:
            cv2.putText(img, 
                        description,
                        (face.rectangle.left()-50, face.rectangle.top()-30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.5, 
                        (0, 255, 0), 
                        4)

    if args.record == True:
        if not os.path.exists('./cctv_res/%s' %vidName):
            os.makedirs('./cctv_res/%s' %vidName)

        cv2.imwrite('/home/dohe0342/VideoFaceRecognition/res/' + str(vidName)+'_2' + '/' + str(frame_number).zfill(4)+'.png', img)
    return img


def face_recognition_hog(frame, faces, goldenFeatures):
    global print_time
    global first_df
    global tot_detect_face_time
    global tot_update_facelist_time
    global tot_create_facelist_time
    global tot_getRep_time
    global tot_update_prediction_time

    start = time.time()
    BB = find_face(frame)

    if print_time:
        print("detect face = " + "{:.4f}".format(1000*(time.time() - start)))

    if first_df:    
        first_df = False
    else:
        tot_detect_face_time += (time.time() - start)

    start = time.time()
    faces, BB = update_facelist_hog(frame, faces, BB)
    if print_time:
        print("update_facelist = " + "{:.4f}".format(1000*(time.time() - start)))
    tot_update_facelist_time += (time.time() - start)

    start = time.time()
    faces = create_facelist(frame, BB, faces)
    if print_time:
        print("create facelist = " + "{:.4f}".format(1000*(time.time() - start)))
    tot_create_facelist_time += (time.time() - start)

    start = time.time()
    if fr_batch == True:
        faces = getRepBatch(frame, faces)
    if fr_batch == False:
        faces = getRepAll(frame, faces)
    if print_time:
        print("get representation = " + "{:.4f}".format(1000*(time.time() - start)))
    tot_getRep_time += (time.time() - start)

    start = time.time()
    faces = update_prediction(faces, goldenFeatures)
    if print_time:
        print("predict face = " + "{:.4f}".format(1000*(time.time() - start)))
    tot_update_prediction_time += (time.time() - start)
    return faces


def buffer_image_raw(imgpath):
    image_list = deque([])

    filenames = glob.glob(imgpath + "/*")
    filenames.sort()

    for filename in filenames:
        im = cv2.imread(filename)
        image_list.append(im)

    return image_list, len(filenames)


def buffer_image_chokepoint(imgpath):
    image_list = deque([])

    filenames = glob.glob(imgpath + "/*.jpg")
    filenames.sort()

    for filename in filenames:
        im = cv2.imread(filename)
        im = cv2.resize(im, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
        im = cv2.copyMakeBorder(im, 0, 0, 240, 240, cv2.BORDER_CONSTANT, value=0)
        #im = cv2.resize(im, (480, 480), interpolation=cv2.INTER_LINEAR)
        image_list.append(im)

    return image_list, len(filenames)


def check_answer_chokepoint(frame, tree, root, facelist, frame_number, correct, num_person, false_alarm, id_correct):
    facelist_temp = deque([])
    for face in facelist:
        facelist_temp.append(face)
    num=0
    for person in root[frame_number].iter("person"):
        for box in person.iter("box"):
            if box.attrib["left"] != "None":
                box_ans = [int(box.attrib["left"])*1.8 + 240, int(box.attrib["top"])*1.8, int(box.attrib["right"])*1.8 + 240, int(box.attrib["bottom"])*1.8]
                #cv2.rectangle(frame, (int(box_ans[0]), int(box_ans[1])), (int(box_ans[2]), int(box_ans[3])), (0, 0, 155), 4)
                break
            else:
                box_ans = [0, 0, 0, 0]
        iou_max = 0;
        for face in facelist_temp:
            box = [face.rectangle.left(), face.rectangle.top(), face.rectangle.right(), face.rectangle.bottom()]
            iou = bb_intersection_over_union(box_ans, box)
            if iou > iou_max:
                iou_max = iou
                max_face = face
                if print_bench_fr_res:
                    print(person.attrib['id'], 'prediction: ' + max_face.current_prediction[-4:]) 
        if iou_max >= bench_iou:
            correct += 1
            if max_face.current_prediction[-4:] == person.attrib["id"]:
                id_correct += 1
            facelist_temp.remove(max_face)
        num_person += 1
        num += 1
    if(num!=0):
        false_alarm += len(facelist_temp)

    return frame, correct, num_person, false_alarm, id_correct


if __name__ == '__main__':
    goldenFeatures = []
    goldenSize = None

    if not os.path.isfile(args.db + '.npy'):
        print('golden feature extracting')

        numberer = 0
        goldenFiles = None
        if args.benchmark:
            goldenFiles = glob.glob(benchmarkGoldenDir)
        else:
            goldenFiles = glob.glob(goldenDir)
        goldenFiles.sort()
        for files in goldenFiles:
            sys.stdout.write(files.split('/')[-1] + '/')
            sys.stdout.flush()
            direc = files + '/*'
            imgname = glob.glob(direc)
            imgname.sort()
            if not len(imgname):
                continue
            for fimg in imgname:
                tempImg = cv2.imread(fimg)
                tempImg = insightfaceModel.get_input_light(tempImg)
                tempImg = np.reshape(tempImg, (1,3,112,112))
                tempFeat = getRep(tempImg)
                name = imgname[0].split('/')[-1][:-9]
                for i in range(aug_test):
                    goldenFeatures.append(tempFeat)
                    identity_dic[numberer] = name
                    numberer += 1
        goldenFeatures = np.array(goldenFeatures).astype(np.float32)
        goldenSize = goldenFeatures.shape[0]
        print('\ndone for ' + str(goldenSize) + ' feature(s)')
        goldenFeatures = goldenFeatures.T
        goldenFeatures = goldenFeatures.flatten()
        dicFile = open(args.db + 'Dic.pkl', 'wb')
        pickle.dump(identity_dic, dicFile, pickle.HIGHEST_PROTOCOL)
        dicFile.close()
        np.save(args.db, goldenFeatures)
    else:
        print('loading golden feature')
        goldenFeatures = np.load(args.db + '.npy')
        goldenSize = goldenFeatures.shape[0] / 512
        dicFile = open(args.db + 'Dic.pkl', 'rb')
        identity_dic = pickle.load(dicFile)
        dicFile.close()
        print('loaded ' + str(goldenSize) + ' feature(s)')

#initialize boostdb
    boostdb.init(goldenSize, goldenFeatures, 10000, 32000000, db_tiled)

    if args.record:
        codec_rec = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        video = cv2.VideoWriter(recordDir, codec_rec, 30.0, (width, height))

#for real-time web-camera
    if not args.video:
        #cap = cv2.VideoCapture(0)
        cap = cv2.VideoCapture('/dohe0342/insightface/RetinaFace/mask_far1.mp4')
        if not cap.isOpened():
            print("ERROR: Missing Camera");
            sys.exit()

        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        #cap.set(cv2.CAP_PROP_FOURCC, codec)
        #cap.set(6, codec)


        facelist = []
        frame = None
        

        frameTimer = time.time()
        frameNum = 0

        while True:
            anchorTime = time.time()

            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            frameReadTime = time.time() - anchorTime
            if print_time:
                print("- read frame = " + "{:.4f}".format(frameReadTime))
            anchorTime = time.time()

            facelist = face_recognition_hog(frame, facelist, goldenFeatures)
            frTime = time.time() - anchorTime

            frameTime = time.time() - frameTimer
            frame = draw_bb(frame, facelist, 1/frameTime)

            frameTimer = time.time()
            
            height, width = frame.shape[:2]

            anchorTime = time.time()
            try: 
                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                if fullscreen:
                    cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow('image', frame)
                if cv2.waitKey(1) > 0: break
            except:
                print("Display Error")
                sys.exit()

            if print_time:
                print("- total fr time = " + "{:.4f}".format(frTime))
                print("- image show time = " + "{:.4f}".format(time.time() - anchorTime))
                print("-- total frame time = " + "{:.4f}".format(frameTime))
                print("----------------------------------------------------")

            if args.record:
                video.write(frame)

            if cv2.waitKey(1) == ord('q'): break
            frameNum += 1

        cap.release()
        cv2.destroyAllWindows()

#for video processing
    else:
        vidList = None
        print('loading video')
        if args.benchmark:
            vidList = glob.glob(benchmarkVidListDir)
        else:
            vidList = glob.glob(vidListDir)

        print(vidList)
        all_vidname = deque([])
        all_fd_correct = deque([])
        all_fr_correct = deque([])
        all_try = deque([])
        all_false = deque([])
        all_frame = deque([])
        all_frame_time = deque([])
        processed_frames = 0

        for vid in vidList:
            vidName = vid.split('/')[-1]

            gt_tree = None
            gt_root = None

            if args.benchmark: 
                gt_tree = parse(groundTruthDir + vidName + ".xml")
                gt_root = gt_tree.getroot()

            frame_buffer = None
            frame_end = None
            if args.benchmark:
                frame_buffer, frame_end = buffer_image_chokepoint(vid)
            else:
                frame_buffer, frame_end = buffer_image_raw(vid)
            processed_frames += frame_end
            print(processed_frames)
            
            fd_correct = 0
            try_total = 0
            false_alarm = 0
            fr_correct = 0
            frame_number = 0
            facelist = []

            print('processing ' + vidName)
            total_start = time.time()
            sub_time = 0.0
            for frame_number, frame in enumerate(frame_buffer):
                    frame_start = time.time()

                    print(frame_number)

                    facelist = face_recognition_hog(frame, facelist, goldenFeatures)
                    frame_time = time.time() - frame_start

                    start = time.time()
                    frame_temp = None

                    if args.benchmark:
                        frame_temp, fd_correct, try_total, false_alarm, fr_correct = check_answer_chokepoint(frame.copy(), gt_tree, gt_root, facelist, frame_number, fd_correct, try_total, false_alarm, fr_correct)

                    if args.display == True:
                        if args.benchmark:
                            frame_temp = draw_bb(frame_temp, facelist, 1/frame_time, frame_number, vidName)
                        else:
                            frame_temp = draw_bb(frame.copy(), facelist, 1/frame_time, frame_number, vidName)
                            cv2.imwrite("./out_vid/2_" + str(frame_number) + ".jpg", frame_temp)
                        try: 
                            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                            if fullscreen:
                                cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                            cv2.imshow('image', frame_temp)
                            if cv2.waitKey(1) > 0: break
                        except:
                            print("Display Error")
                            sys.exit()
                    
                    if args.record == True:
                        if args.display == False:
                            frame_temp = draw_bb(frame.copy(), facelist, 1/frame_time, frame_number, vidName)
                        #video.write(frame_temp)
                    #frame_temp = draw_bb(frame.copy(), facelist, 1/frame_time, frame_number, vidName)
                    # quit the program on the press of key 'q'
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    sub_time += (time.time() - start)

            total_end = time.time() - sub_time

            print(vidName)
            print("avg time per frame = " + str(1000*(total_end - total_start)/frame_end))

            cv2.destroyAllWindows()

            if args.benchmark:
                print("====================================================")
                print("Score")
                print("try_total " + str(try_total))
                print("fd correct " + str(fd_correct))
                print("fr correct " + str(fr_correct))
                print("flase alarm " + str(false_alarm))
                if try_total:
                    print("fd accuracy " + str(float(fd_correct)/try_total))
                    print("fr accuracy " + str(float(fr_correct)/try_total))
                
                all_vidname.append(vidName)
                all_frame.append(frame_end)
                all_try.append(try_total)
                all_fd_correct.append(fd_correct)
                all_fr_correct.append(fr_correct)
                all_false.append(false_alarm)
                all_frame_time.append(1000*(total_end - total_start)/frame_end)

                print("video frame try fd fr false timePerFrame")
                for i, vid in enumerate(all_vidname):
                    print('%s %d %d %d %d %d %f' %(vid, all_frame[i], all_try[i], all_fd_correct[i], all_fr_correct[i], all_false[i], all_frame_time[i]))

print("---statistics---")
print("avg detect face = " + "{:.4f}".format(1000*tot_detect_face_time / processed_frames) + "ms")
print("avg update facelist = " + "{:.4f}".format(1000*tot_update_facelist_time / processed_frames) + "ms")
print("avg create facelist = " + "{:.4f}".format(1000*tot_create_facelist_time / processed_frames) + "ms")
print("avg get representation = " + "{:.4f}".format(1000*tot_getRep_time / processed_frames) + "ms")
print("avg update prediction = " + "{:.4f}".format(1000*tot_update_prediction_time / processed_frames) + "ms")
print("\n" + "total face num = " + str(total_face_num))
print("total skipped face num = " + str(skipped_faces))
print("avg skipped ratio = " + str(100*float(skipped_faces)/float(total_face_num)) + "%")
