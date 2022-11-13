from dash import Dash, html, dcc, Input, Output, State, dash_table, no_update, register_page
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from apps.backend.auxiliary_functions import *
from apps.dash_navbar.navigation_bar import Navbar
import pandas as pd
import json
import numpy as np
from sqlalchemy import create_engine

# blank_intercept=[ ]


def age_dashboard(server):


    external_stylesheets = [
        'https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP]
    age_app = Dash(__name__, server=server, url_base_pathname='/age_analysis/',
                   external_stylesheets=external_stylesheets)
    
    register_page(
    __name__,
    path='/age_analysis/')
    
    def age_layout():
        global conn,software_conn
        url = r"mariadb+mariadbconnector://root:dbpwx61@139.20.22.156:3306/arardb"
        engine = create_engine(url)
        softurl = r"mariadb+mariadbconnector://root:dbpwx61@139.20.22.156:3306/ararsoftware"
        soft_engine = create_engine(softurl)
        software_conn = soft_engine.connect()
        conn = engine.connect()
        exp_index = np.squeeze(np.array(pd.read_sql(
            "SELECT exp_index  FROM experiments where exp_key='1'", software_conn)))
        global exp_nr
        dashboard_arardb = pd.read_sql(
            "SELECT exp_nr,material,proben_bez,project_name,sample_owner,irr_batch  FROM material order by exp_nr desc", conn)
        dashboard_arardb["id"] = dashboard_arardb.index
        exp_nr = str(dashboard_arardb.iloc[exp_index][0])
        experiment_info = extract_experiment_info(exp_nr, database)
        experiment_info = extract_irradiation(conn, experiment_info)
        pd_exp = pd.DataFrame.from_dict(experiment_info)
        pd_exp = pd_exp[['Exp_No', 'Exp_type', 'Sample', 'Material', 'Project', 'Owner',
                         'Irradiation', 'Plate', 'Hole', 'Weight', 'Jvalue', 'J_Error', 'f',
                        'f_error', 'cycle_count', 'tuning_file', 'irr_enddatetime', 'irr_duration',
                         'irr_enddatetime_num']]
        intensities = extract_intensities(conn, experiment_info)
        sensitivities = extracting_senstivities(conn, experiment_info)
        intensities = pd.merge(intensities, sensitivities,
                               how='left', on='serial_nr')
        blank_assignments_data = pd.read_sql(
            f"SELECT serial_nr,blank_experiment_nr,blank_experiment_type FROM blank_assignments where exp_nr ={exp_nr}", software_conn)
        blank_assignments_data['assignment'] = 'Manual'
        if blank_assignments_data.empty:
            blank_data = extracting_blanks(conn, intensities[['acq_datetime']].min()[
                                           0], intensities[['acq_datetime']].max()[0])
            subset_intensities = intensities[[
                'serial_nr', 'acq_datetime']].drop_duplicates()
            blank_data = automatic_blank_assigment(
                blank_data, subset_intensities)
            blank_list = tuple(blank_data['blank_experiment_no'].unique())+(0,)
            blank_intercept = pd.read_sql(
                f"SELECT exp_nr,serial_nr,ar36_intercept,ar36_standard_error,ar37_intercept,ar37_standard_error,ar37_standard_error,ar38_intercept,ar38_standard_error,ar39_intercept,ar39_standard_error,ar40_intercept,ar40_standard_error FROM experiment_intercepts where exp_nr in {blank_list}", software_conn)
            blank_intercept['blank_assignment'] = 'Automatic'
        else:
            blank_list = tuple(
                blank_assignments_data['blank_experiment_no'].unique())+(0,)
            blank_intercept = pd.read_sql(
                f"SELECT exp_nr,serial_nr,ar36_intercept,ar36_standard_error,ar37_intercept,ar37_standard_error,ar37_standard_error,ar38_intercept,ar38_standard_error,ar39_intercept,ar39_standard_error,ar40_intercept,ar40_standard_error FROM experiment_intercepts where exp_nr in {blank_list}", software_conn)
            blank_intercept['blank_assignment'] = 'Manual'

        blank_data1 = extract_blank_data(conn, blank_list)
        blank_info = blank_data1[['exp_nr', 'serial_nr', 'device', 'weight',
                                  'inlet_file', 'tuning_file', 'd_inlet_file']].drop_duplicates()

        blank_info = pd.merge(blank_info, blank_intercept,
                              how='outer', on='exp_nr')
        experiment_intercepts = pd.read_sql(
            f"Select exp_nr, serial_nr, ar36_intercept, ar36_standard_error, ar37_intercept, ar37_standard_error, ar37_standard_error, ar38_intercept, ar38_standard_error, ar39_intercept, ar39_standard_error, ar40_intercept, ar40_standard_error FROM experiment_intercepts where exp_nr in {blank_list}", software_conn)
        assigned_blank_intensities = pd.merge(intensities[['serial_nr', 'device', 'weight', 'method_name']].drop_duplicates(), blank_data,
                                              how='left', on='serial_nr')
        intensities_intercepts = pd.merge(
            assigned_blank_intensities, experiment_intercepts, how='left', on=['serial_nr'])
        #f_interpolation = parameters['f_value_interpolation']
        #ar4036= input from user
        #f_values = extract_f_value(conn, exp_nr, f_interpolation, 298.6)
        corrected_experiment_intercepts = correcting_intercepts(
            blank_info, intensities_intercepts)
        print(corrected_experiment_intercepts)
        header=Navbar()
        layout=html.Div([header])
        return layout
    age_app.layout = age_layout

    
