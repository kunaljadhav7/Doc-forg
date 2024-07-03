# importing libraries
from flask import Flask,request,render_template
import pickle
import os 
from werkzeug.utils import secure_filename

from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

app=Flask(__name__)


# loading the Digital Forgery model
DF = pickle.load(open(os.path.join('models', 'DF_2.pkl'),'rb'))

# loading the Overwriting model
OW = pickle.load(open(os.path.join("models", "OW_2.pkl"), "rb"))

# updating the directory for uploading and predicting folders
upload_folder=os.path.join('static','uploads')
predicted_folder=os.path.join('static','Predicted')
app.config['PREDICTED']=predicted_folder
app.config['UPLOAD']=upload_folder

# defining the routes
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=["POST"])

# The predict function will be called on clicking the Submit button
def predict():
    # print("Enterd Here")

    def bboxes(result, thres):
        '''
        arguments:
        result = coordinates of boxes
        thres = Minimum threshold to reduce false positives

        Returns:
        coords = list of final coordinates after combining boxes
        conf = list of confidence scores corresponding to the final coordinates
        '''
        # print("calculating boxes here")
        boxes = result[0].boxes.cpu().numpy()
        # print(first)
        qu = {}
        final = []
        small = []
        conf = []
        visited = [0] * len(boxes)
        for i, box in enumerate(boxes):
            if(box.conf[0] < thres):
                continue
            if(visited[i]):
                continue
            visited[i] = 1
            r = box.xyxy[0]
            prev = r
            qu[i] = r
            # print(r)
            small = []
            con = box.conf[0]
            small.append(r)
            while qu:
                index, val = list(qu.items())[0]
                del qu[index]
                visited[index] = 1
                prev = val
                # print(val)
                for j, b in enumerate(boxes):
                    # print(j)
                    if(b.conf[0] < thres):
                        continue
                    
                    if(index == j):
                        continue
                    if(visited[j]):
                        continue

                    if( con < b.conf[0]):
                        con = b.conf[0]

                    p = b.xyxy[0]
                    dist = np.sqrt(np.square(p[0] - prev[0]) + np.square(p[1] - prev[1]))
                    if(dist < 15):
                        qu[j] = p
                        small.append(p)

            if(con):
                conf.append(round(con, 2))
            if(len(small)):
                final.append(small)
        
        coord = []
        for lst in final:
            min_x, min_y, max_x, max_y = 10000.0, 10000.0, -1.0, -1.0
            for lt in lst:
                if(lt[0] < min_x):
                    min_x = lt[0]
                    min_y = lt[1]
                if(lt[2] > max_x):
                    max_x = lt[2]
                    max_y = lt[3]
            coord.append([min_x, min_y, max_x, max_y])
        # print(coord)
        # print("Calculate boxes")
        return coord, conf

    def plot(result_DF, result_OW, img, ):
        '''
        arguments:
        result_DF = values of bounding boxes from Digital Forgery model
        result_OW = values of bounding boxes from Overwriting model
        img = the raw image that was uploaded

        Returns:
        img = Final image with plotted bounding box and confidence score
        '''
        # print("Going to plot")
        final_DF_coord, final_DF_conf = [], []
        final_OW_coord, final_OW_conf = [], []
        if(len(result_DF)):
            final_DF_coord, final_DF_conf = bboxes(result_DF, 0.47)
        if(len(result_OW)):
            final_OW_coord, final_OW_conf = bboxes(result_OW, 0.55)

        if(len(final_DF_conf)):
            for i, box in enumerate(final_DF_coord):
                r = [v.astype(int) for v in box]
                cv2.rectangle(img, r[:2], r[2:], (0, 0, 255), 1)
                text = str(round(final_DF_conf[i] * 100, 2))
                # print(text)
                img = cv2.putText(img, text, (r[0], r[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.23, (0, 0, 255), 1)
        
        if(len(final_OW_conf)):
            for i, box in enumerate(final_OW_coord):
                r = [v.astype(int) for v in box]
                # print(r)                                             
                cv2.rectangle(img, r[:2], r[2:], (255, 0, 0), 1)
                text = str(round(final_OW_conf[i] * 100, 2))
                # print(text)
                img = cv2.putText(img, text, (r[0], r[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.23, (255, 0, 0), 1)
        # print("Exiting from plot fn")

        return img
    try:
        """
        Main code that will save the uploaded image and returns the final predicted image
        """
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img = os.path.join(app.config['UPLOAD'], filename)

        u_img = cv2.imread(os.path.join("static", "uploads", filename))
        imag_DF = DF.predict(img)
        imag_OW = OW.predict(img)
        imag = plot(imag_DF, imag_OW, u_img)
        img_arr=imag
        # print("1")

        pred=Image.fromarray(img_arr[...,::-1])
        pred = pred.resize((640, 640))
        # print("2")
        pred.save(os.path.join(app.config['PREDICTED'], filename))
        # print('3')
        img1 = os.path.join(app.config['PREDICTED'], filename)
        # print('4')
        return render_template("final.html",img=img1)
    except Exception as e:
        print(e)
        return render_template('index.html')




if(__name__=="__main__"):
    port = os.environ.get('PORT', 8080)
    app.run(host="0.0.0.0", port = port)
