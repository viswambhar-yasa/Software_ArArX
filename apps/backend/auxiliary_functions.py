
from datetime import datetime as dt
import mariadb
import pandas as pd
import numpy as np
from sklearn import svm
from apps.backend.parameters_database import parameters
from scipy import stats
from sklearn.svm import SVR, LinearSVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, BayesianRidge
username = 'root'
password = 'dbpwx61'
database = 'arardb'
global conn 
conn = mariadb.connect(
    user=username,
    password=password,
    host="139.20.22.156",
    port=3306,
    database=database)
# Dashboard


def datenum(date):
    d = dt.strptime(date, '%Y-%m-%d %H:%M:%S')
    return 366 + d.toordinal() + (d - dt.fromordinal(d.toordinal())).total_seconds()/(24*60*60)


def extract_experiment_info(exp_nr, database):
    if database == 'arardb':

        query = f"SELECT DISTINCT material.exp_nr as Exp_No, auftrag.exp_type as Exp_type, proben_bez as Sample, material as Material, project_name as Project, sample_owner as Owner, material.irr_batch as Irradiation, irr_platenr as Plate, irr_holenr as Hole, irr_weight as Weight, j_value as Jvalue, j_value_error as J_Error, f, f_error, messung.cycle_count, messung.tuning_file, correction_factors.ca3637, correction_factors.ca3637_error, correction_factors.ca3937, correction_factors.ca3937_error, correction_factors.k3839, correction_factors.k3839_error, correction_factors.k4039, correction_factors.k4039_error FROM(SELECT * from material WHERE exp_nr={exp_nr}) material LEFT JOIN auftrag on material.exp_nr = auftrag.exp_nr LEFT JOIN  messung on material.exp_nr = messung.exp_nr LEFT JOIN f_values on material.exp_nr = f_values.ID LEFT JOIN correction_factors ON material.irr_batch = correction_factors.irr_batch order by messung.acq_datetime desc"
    else:
        query = f"SELECT DISTINCT A.id, A.batch, B.labbook, B.label, C.`type`, A.j_value, A.j_error, A.lambda40, A.lambda40_error, A.standard_age, A.standard_age_error  FROM standards A LEFT JOIN series B ON A.id = B.series_id LEFT JOIN series_type C ON B.series_type_id = C.series_type_id ORDER BY A.date desc"
    data = pd.read_sql(query, conn)
    experiment_info = data.to_dict()
    return experiment_info


def extract_intensities(conn, experiment_info):
    experiments = experiment_info['Exp_No']
    experiments_type = experiment_info['Exp_type']
    n = len(experiments)
    for i in range(n):
        expr_nr = experiments[i]
        expr_ty = experiments_type[i]
        if expr_nr > 3832:
            query = f"SELECT mess.serial_nr, mess.device, mess.acq_datetime,scan, mess.weight, mess.inlet_file, mess.tuning_file, mess.method_name,method_time_zeros.time AS d_method_name,inlet_method_equil_times.time AS d_inlet_file,CASE mess.device WHEN 'ir' THEN power.power WHEN 's' THEN ROUND((-0.00015 * temperatur.temp*temperatur.temp) + (1.478*temperatur.temp-547.6)) ELSE 0 END AS value,messwerte.scan, messwerte.mass, messwerte.start, messwerte.v36, messwerte.v37, messwerte.v38, messwerte.v39, messwerte.v40 FROM(SELECT * from messung WHERE exp_nr='{expr_nr}') mess LEFT JOIN POWER ON power.serial_nr = mess.serial_nr LEFT JOIN temperatur ON temperatur.serial_nr = mess.serial_nr LEFT JOIN method_time_zeros ON mess.method_name = method_time_zeros.method LEFT JOIN inlet_method_equil_times ON mess.inlet_file = inlet_method_equil_times.method LEFT JOIN messwerte ON mess.serial_nr = messwerte.serial_nr ORDER BY serial_nr, scan"
        else:
            query = f"SELECT mess.serial_nr, mess.device, mess.acq_datetime,scan, mess.weight, mess.inlet_file, mess.tuning_file, mess.method_name,method_time_zeros.time AS d_method_name,inlet_method_equil_times.time AS d_inlet_file,CASE mess.device WHEN 'ir' THEN power.power WHEN 's' THEN temperatur.temp ELSE 0 END AS value,messwerte.scan, messwerte.mass, messwerte.start, messwerte.v36, messwerte.v37, messwerte.v38, messwerte.v39, messwerte.v40 FROM(SELECT * from messung WHERE exp_nr='{expr_nr}') mess LEFT JOIN POWER ON power.serial_nr = mess.serial_nr LEFT JOIN temperatur ON temperatur.serial_nr = mess.serial_nr LEFT JOIN method_time_zeros ON mess.method_name = method_time_zeros.method LEFT JOIN inlet_method_equil_times ON mess.inlet_file = inlet_method_equil_times.method LEFT JOIN messwerte ON mess.serial_nr = messwerte.serial_nr ORDER BY serial_nr, scan"
        data = pd.read_sql(query, conn)
        if expr_ty == 'wait':
            data['device'] = 'w'
        elif expr_ty == 'airshot':
            data['device'] = 'as'
        elif expr_ty == 'blank':
            data['device'] = 'b'
        data['time_zero'] = data['start'] - \
            data['d_method_name']-data['d_inlet_file']
        data['acq_datetime_num'] = data['acq_datetime'].apply(str)
        data['acq_datetime_num'] = data['acq_datetime_num'].apply(datenum)
    return data


def extract_blank_data(conn, blank_exps):
    query = f"SELECT mess.exp_nr,mess.serial_nr, mess.device, mess.acq_datetime,scan, mess.weight, mess.inlet_file, mess.tuning_file, mess.method_name,method_time_zeros.time AS d_method_name,inlet_method_equil_times.time AS d_inlet_file,CASE mess.device WHEN 'ir' THEN power.power WHEN 's' THEN ROUND((-0.00015 * temperatur.temp*temperatur.temp) + (1.478*temperatur.temp-547.6)) ELSE 0 END AS value,messwerte.scan, messwerte.mass, messwerte.start, messwerte.v36, messwerte.v37, messwerte.v38, messwerte.v39, messwerte.v40 FROM(SELECT * from messung WHERE exp_nr in {blank_exps}) mess LEFT JOIN POWER ON power.serial_nr = mess.serial_nr LEFT JOIN temperatur ON temperatur.serial_nr = mess.serial_nr LEFT JOIN method_time_zeros ON mess.method_name = method_time_zeros.method LEFT JOIN inlet_method_equil_times ON mess.inlet_file = inlet_method_equil_times.method LEFT JOIN messwerte ON mess.serial_nr = messwerte.serial_nr ORDER BY serial_nr, scan"
    blank_data = pd.read_sql(query, conn)
    blank_data['time_zero'] = blank_data['start'] - \
        blank_data['d_method_name']-blank_data['d_inlet_file']
    blank_data['acq_datetime_num'] = blank_data['acq_datetime'].apply(str)
    blank_data['acq_datetime_num'] = blank_data['acq_datetime_num'].apply(
        datenum)
    return blank_data


def extracting_senstivities(conn, experiment_info):
    experiments = experiment_info['Exp_No'][0]
    query = f" WITH tab AS(     SELECT DISTINCT mess.serial_nr, mess.acq_datetime     FROM(SELECT * from messung WHERE exp_nr={experiments}) mess     LEFT JOIN POWER ON power.serial_nr=mess.serial_nr     LEFT JOIN temperatur ON temperatur.serial_nr=mess.serial_nr     LEFT JOIN method_time_zeros ON mess.method_name=method_time_zeros.method     LEFT JOIN inlet_method_equil_times ON mess.inlet_file=inlet_method_equil_times.method     LEFT JOIN messwerte ON mess.serial_nr=messwerte.serial_nr ) SELECT serial_nr, sensitivity FROM(     SELECT serial_nr, sensitivity, acq_datetime, DATE, H, ROW_NUMBER() over(partition by serial_nr order by H) AS row FROM(         SELECT t.serial_nr, s.sensitivity, t.acq_datetime, s.date, abs(t.acq_datetime-s.date) AS H FROM tab t CROSS JOIN sensitivities s     )G )D WHERE ROW = 1"
    sensitivities_data = pd.read_sql(query, conn)
    return sensitivities_data


def extracting_blanks(conn, start_date, end_date, exp_list=('blank', 'coldblank', 'airshot')):
    """

    Args:
        conn (_type_): _description_
        start_date (_type_): _description_
        end_date (_type_): _description_
        exp_list (tuple, optional): _description_. Defaults to ('blank', 'coldblank', 'airshot').

    Returns:
        _type_: _description_
    """
    query = f"SELECT * FROM(SELECT auftrag.exp_nr, exp_type, messung.acq_datetime FROM(select * from auftrag WHERE exp_type IN {exp_list}) auftrag INNER JOIN messung ON auftrag.exp_nr=messung.exp_nr WHERE acq_datetime <= '{start_date}' Order BY acq_datetime desc LIMIT 1 )A UNION SELECT * FROM(     SELECT auftrag.exp_nr, exp_type, messung.acq_datetime     FROM(select * from auftrag WHERE exp_type IN {exp_list}) auftrag INNER JOIN messung     ON auftrag.exp_nr=messung.exp_nr     WHERE acq_datetime >= '{start_date}' AND acq_datetime <= '{end_date}'     Order BY acq_datetime )B "
    blank_data = pd.read_sql(query, conn)
    return blank_data


def extracting_all_blanks(conn, exp_list=('blank', 'coldblank', 'airshot', 'hotshot')):
    query = f"SELECT auftrag.exp_nr, exp_type, messung.acq_datetime FROM(select * from auftrag WHERE exp_type IN {exp_list}) auftrag INNER JOIN messung ON auftrag.exp_nr = messung.exp_nr Order BY acq_datetime desc"
    all_blank_data = pd.read_sql(query, conn)
    return all_blank_data


def automatic_blank_assigment(blank_data, subset_intensities):
    blank_data = blank_data.to_numpy()
    if len(blank_data) == 1:
        subset_intensities['blank_experiment_no'] = blank_data[0, 0]
        subset_intensities['blank_experiment_type'] = blank_data[0, 1]
        return subset_intensities[['serial_nr', 'blank_experiment_no', 'blank_experiment_type']]
    else:
        subset_intensities = subset_intensities.to_numpy()
        output_array = [[]*int(subset_intensities.shape[1]+1)
                        ]*subset_intensities.shape[0]
        for i in range(len(subset_intensities)):
            for j in range(1, len(blank_data)):
                if subset_intensities[i, 1] > blank_data[j-1, 2]:

                    if subset_intensities[i, 1] <= blank_data[j, 2]:
                        output_array[i] = [int(subset_intensities[i, 0]),
                                           int(blank_data[j-1, 0]), blank_data[j-1, 1]]
                        break
                    else:
                        output_array[i] = [int(subset_intensities[i, 0]),
                                           int(blank_data[j, 0]), blank_data[j, 1]]

        output_array = pd.DataFrame(output_array, columns=[
                                    'serial_nr', 'blank_experiment_no', 'blank_experiment_type'])
        return output_array


def extract_irradiation(conn, experiment_info):
    irradiation = experiment_info['Irradiation'][0]
    query = f"SELECT irr_enddatetime,irr_duration FROM irr_batch WHERE irr_batch='{irradiation}'"
    irradiation_data = pd.read_sql(query, conn)
    irradiation_data['irr_enddatetime_num'] = irradiation_data['irr_enddatetime'].apply(
        str)
    irradiation_data['irr_enddatetime_num'] = irradiation_data['irr_enddatetime_num'].apply(
        datenum)
    experiment_info.update(irradiation_data.to_dict())
    return experiment_info


def regression_models(X, Y, modeltype='Linear'):
    
    if modeltype == 'Linear':
        X = np.array(X).reshape(-1, 1)
        Y = np.array(Y).reshape(-1, 1)
        model = LinearRegression()
        model.fit(X,Y)
        return model
    elif modeltype == 'Quadratic':
        X = np.array(X).reshape(-1, 1)
        Y = np.array(Y).reshape(-1, 1)
        nb_degree = 2
        polynomial_features = PolynomialFeatures(degree=nb_degree)
        X_TRANSF = polynomial_features.fit_transform(X)
        model = LinearRegression()
        model.fit(X_TRANSF, Y)
        return model
    elif modeltype == 'Power':
        X = np.array(X).reshape(-1, 1)
        Y = np.array(Y).reshape(-1, 1)
        model = LinearSVC()
        model.fit(X, Y)
        return model
    elif modeltype == 'Lasso':
        X = np.array(X).reshape(-1, 1)
        Y = np.array(Y).reshape(-1, 1)
        model = Lasso(alpha=1.0)
        model.fit(X, Y)
        return model
    elif modeltype == 'ElasticNet':
        X = np.array(X).reshape(-1, 1)
        Y = np.array(Y).reshape(-1, 1)
        model = ElasticNet(alpha=1.0)
        model.fit(X, Y)
        return model
    elif modeltype == 'Bayesian':
        X = np.array(X).reshape(-1, 1)
        Y = np.array(Y).reshape(-1, 1)
        model = BayesianRidge()
        model.fit(X, np.ravel(Y))
        return model
    elif modeltype == 'Automatic':
        loss=np.inf
        models_list = ['Linear', 'Bayesian', 'ElasticNet']
        for i in models_list:
            model=regression_models(X, Y, i)
            Y_pred = model.predict(np.array(X).reshape(-1, 1))
            cal_loss = mean_squared_error(Y_pred, np.array(Y))
            if cal_loss<loss:

                best_model = model
                loss = cal_loss
        return best_model

def check_steps(intensities_data, parameters):
    serial_numbers = intensities_data[[
        'serial_nr']].drop_duplicates().to_numpy()
    rol = parameters['rol']
    offset_max = parameters['offset_max']
    step_diff = pd.DataFrame()
    for i in serial_numbers:
        sub_intensities = intensities_data.loc[(
            intensities_data['serial_nr'] == i[0])]
        step_duration = int(len(sub_intensities))
        min_size = step_duration * rol / 100
        sub_intensities = sub_intensities[['v36', 'v37', 'v38', 'v39', 'v40']]
        step_diff = abs(sub_intensities.diff().dropna())
        mean_step_diff = step_diff.mean()
        for j in ['v39', 'v40']:
            offset = 1
            offset_max = mean_step_diff[j] * offset_max
            index = 0
            for index, value in enumerate(step_diff[j]):
                if value > offset:
                    offset = [offset, index]
