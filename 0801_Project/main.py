from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from keras.models import load_model
# import cv2
from keras import backend as K
from itertools import islice
from more_itertools import windowed
from flask import Flask, jsonify, render_template, request
import pandas as pd
from json import dumps


app = Flask(__name__)
app.config['ENV'] = 'development'
app.config['DEBUG'] = True
app.config['TESTING'] = True

@app.route('/accound')
def accounts():
    init= pd.read_csv("C:/Users\PC\PycharmProjects\python_flak2/temp/test_csv.csv")
    test=init.iloc[-10: -1,0]
    # print("test",test)
    a=[]
    for i in test:
       a.append(i)
    print("a",a)
    return jsonify(result=dumps(a))

@app.route("/_test", methods=['GET', 'POST'])
def test():
    global outcome1, factor

    if request.method == 'GET':
        print("got request---------------------------------------------------------")
        return render_template('test_temp.html')
    if request.method == 'POST':
        print("got post---------------------------------------------------------")
        K.clear_session()
        up = request.form.get('myRange_upper')
        down = request.form.get('myRange_lower')
        requested_number = request.form.get('requested_number')
        requested_number = int(requested_number)
        requested_time = request.form.get('myRange_time')
        # -----------------------------------
        # file = request.files['file']
        # file.save("E:/Python_New_Project/0801_Project/test_0801.csv")
        # Load trained LSTM model
        model = load_model('C:/Users\PC\PycharmProjects/0801_Project/model_RUL_0801_ver2.h5')
        #  Read the testing data saved through upload operation
        Data = pd.read_csv("C:/Users\PC\PycharmProjects/0801_Project/test_0801.csv")
        data = Data.iloc[:,0]
        print('Data = ',Data )
        print('data =', data)
        print('data_array',np.array(data))


        temp = np.array(data)

        print('temp shape',temp.shape)
        # Reshaping data to fit in
        temp=temp [-201:-1]
        temp_2 = temp
        print('temp shape', temp.shape)
        x_csv = temp.reshape((1, 200, 1))  # <------------------------------------------------------
        # print(x_csv)
        y_pred = model.predict(x_csv)
        # a = np.argmax(y_pred, axis=1)
        guess = y_pred
        featuretemp = []
        print("guss", guess[0][0])
        featuretemp.append(guess[0][0])
        print("len(temp_2)", len(temp_2))
        # the predictoins of future timestep
        for number in range(requested_number - 1):
            temp_3 = []
            for i in range(1, len(temp_2)):
                # print(i)
                temp_3.append(temp_2[i])
                # print(i)
            temp_3.append(guess[0][0])
            temp_2 = temp_3
            temp_3 = np.array(temp_3)
            x_csv = temp_3.reshape((1, 200, 1))
            y_pred = model.predict(x_csv)
            guess = y_pred
            print("gussNEW", guess[0][0])
            featuretemp.append(guess[0][0])
        print(featuretemp)
        # ---------------------------------------------------------
        x_label = []
        for i in range(len(featuretemp)):
            x_label.append(str(i + 1))

        y_values = featuretemp
        # upline=[]
        # for i in range(len(featuretemp)):
        #     upline.append(float(up))
        up_float = float(up)
        down_float = float(down)
        print('up', up_float)
        print("down", down_float)
        # ----------------------------------------------------------
        for i in featuretemp:
            if i > up_float or i < down_float:
                outcome1 = True

            else:
                outcome1 = False
        print('outcome1', outcome1)
        if outcome1 == True:
            factor = 0
        if outcome1 == False:
            factor = 1

        K.clear_session()
        return jsonify( post=(featuretemp), up=up, down=down, values=y_values,
                               outcome=outcome1, factor=factor, requested_number=requested_number
                               )

def index():
    return render_template('test_temp.html')
if __name__ == '__main__':
    app.run(debug=True)
