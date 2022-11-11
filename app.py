from flask import Flask, render_template, request
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
app = Flask(__name__)#initializing a flask app

@app.route('/',methods=['GET'])
@cross_origin()
def homePage():
    return render_template("index.html")
# @app.route('/report.html',methods=['GET'])  # route to display the home page
# @cross_origin()
# def report():
#     return render_template("report.html")
@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':

        overall_rating = float(request.form["overall_rating"])
        defensive_work_rate = float(request.form["defensive_work_rate"])
        potential =float(request.form["potential"])
        crossing = float(request.form["crossing"])
        finishing = float(request.form["finishing"])
        heading_accuracy = float(request.form["heading_accuracy"])
        short_passing = float(request.form["short_passing"])
        volleys = float(request.form["volleys"])
        dribbling =float(request.form["dribbling"])
        curve = float(request.form["curve"])
        free_kick_accuracy = float(request.form["free_kick_accuracy"])
        long_passing = float(request.form["long_passing"])
        ball_control = float(request.form["ball_control"])
        acceleration = float(request.form["acceleration"])
        sprint_speed = float(request.form["sprint_speed"])
        agility = float(request.form["agility"])
        reactions = float(request.form["reactions"])
        balance = float(request.form["balance"])
        shot_power = float(request.form["shot_power"])
        jumping = float(request.form["jumping"])
        stamina = float(request.form["stamina"])
        strength = float(request.form["strength"])
        long_shots = float(request.form["long_shots"])
        aggression = float(request.form["aggression"])
        interceptions = float(request.form["interceptions"])
        positioning = float(request.form["positioning"])
        vision = float(request.form["vision"])
        penalties = float(request.form["penalties"])
        marking = float(request.form["marking"])
        standing_tackle = float(request.form["standing_tackle"])
        sliding_tackle = float(request.form["sliding_tackle"])
        gk_diving = float(request.form["gk_diving"])
        gk_handling = float(request.form["gk_handling"])
        gk_kicking = float(request.form["gk_kicking"])
        gk_positioning = float(request.form["gk_positioning"])
        gk_reflexes = float(request.form["gk_reflexes"])
        preferred_foot = float(request.form["preferred_foot"])
        attacking_work_rate = float(request.form["attacking_work_rate"])
        experience_no_of_days = float(request.form["experience_no_of_days"])

        filename = "cluster.pickle"
        loaded_model = pickle.load(open(filename, 'rb'))
        cluster_no = loaded_model.predict([[ potential, crossing, finishing,heading_accuracy,short_passing , volleys, dribbling, curve,free_kick_accuracy, long_passing, ball_control, acceleration,sprint_speed, agility, reactions, balance, shot_power,jumping, stamina, strength, long_shots, aggression,interceptions, positioning, vision, penalties, marking,standing_tackle, sliding_tackle, gk_diving, gk_handling,gk_kicking, gk_positioning, gk_reflexes, preferred_foot,attacking_work_rate, defensive_work_rate, experience_no_of_days]])

        if cluster_no == 0:
            filename = "randomforest_cluster_1.pickle"
            model = pickle.load(open(filename,'rb'))
            prediction = model.predict([[potential,crossing,finishing,heading_accuracy,short_passing,volleys,dribbling,curve,free_kick_accuracy,long_passing,acceleration,agility,reactions,balance,shot_power,jumping,stamina,strength,long_shots,aggression,interceptions,positioning,vision,penalties,gk_diving,preferred_foot,attacking_work_rate,defensive_work_rate,experience_no_of_days,]])

        elif cluster_no == 1:
            filename ="model_ridge_cluster_0.pickle"
            model = pickle.load(open(filename, 'rb'))
            prediction = model.predict([[potential,crossing,finishing,heading_accuracy,short_passing,volleys,dribbling,curve,free_kick_accuracy,long_passing,ball_control,acceleration,sprint_speed,agility,reactions,balance,shot_power,jumping,stamina,strength,long_shots,aggression,interceptions,positioning,vision,penalties,marking,gk_diving,gk_handling,gk_kicking,preferred_foot,attacking_work_rate,defensive_work_rate,experience_no_of_days]])
        elif cluster_no==2:
            filename="model_xgb_cluster_2.pickle"
            li = [potential, crossing, finishing, heading_accuracy, short_passing, volleys, dribbling,
                  curve, free_kick_accuracy, long_passing, acceleration, agility, reactions, balance, shot_power,
                  jumping, stamina, strength, long_shots, aggression, interceptions, positioning, vision, penalties,
                  gk_diving, preferred_foot, attacking_work_rate, defensive_work_rate, experience_no_of_days]
            df = pd.DataFrame(li).T
            model= pickle.load(open(filename, 'rb'))
            prediction=model.predict(df)

        else:
           prediction="Its not belongs to any cluster"

        return render_template('result.html', prediction=prediction,cluster_no=cluster_no)



    else:
        return render_template('index.html')


if __name__ == "__main__":
	app.run(debug=True) #