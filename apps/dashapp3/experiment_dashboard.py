from dash import Dash, html, dcc, Input, Output, State, dash_table, no_update, register_page
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from apps.backend.auxiliary_functions import *
from apps.dash_navbar.navigation_bar import Navbar
import pandas as pd
import json
import numpy as np

# blank_intercept=[ ]


def experiment_dashboard(server):
    with open(r'.\apps\private_data\account_details.json', 'r') as f:
        try:
            account_details = json.load(f)
        except:
            account_details = account_details = {"username": 'root',
                                                 "database": 'arardb',
                                                 "password": 'dbpwx61'}
    f.close()
    username = account_details['username']
    password = account_details['password']
    database = account_details['database']
    global conn
    conn = mariadb.connect(
        user=username,
        password=password,
        host="139.20.22.156",
        port=3306,
        database=database)

    global software_conn
    software_conn = mariadb.connect(
            user=username,
            password=password,
            host="139.20.22.156",
            port=3306,
            database="ararsoftware")

    external_stylesheets = [
        'https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP]
    exp_app = Dash(__name__, server=server, url_base_pathname='/experiment/',
                     external_stylesheets=external_stylesheets)

    register_page(
    __name__,
    path='/experiment/')
    
    def exp_layout():
        global software_conn
        software_conn = mariadb.connect(
            user=username,
            password=password,
            host="139.20.22.156",
            port=3306,
            database="ararsoftware")
        global pd_exp, blank_info, intensities_intercepts, intensities
        exp_index = np.squeeze(np.array(pd.read_sql(
            "SELECT exp_index  FROM experiments where exp_key='1'", software_conn)))
        dashboard_arardb = pd.read_sql(
            "SELECT exp_nr,material,proben_bez,project_name,sample_owner,irr_batch  FROM material order by exp_nr desc", conn)
        dashboard_arardb["id"] = dashboard_arardb.index
        global exp_nr
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
                f"SELECT *  FROM blank_intercepts where exp_nr in {blank_list}", software_conn)
            blank_intercept['blank_assignment'] = 'Automatic'
        else:
            blank_list = tuple(
                blank_assignments_data['blank_experiment_no'].unique())+(0,)
            blank_intercept['blank_assignment'] = 'Manual'

        blank_data1 = extract_blank_data(conn, blank_list)
        blank_info = blank_data1[['exp_nr', 'serial_nr', 'device', 'weight',
                                  'inlet_file', 'tuning_file', 'd_inlet_file']].drop_duplicates()
        blank_info = pd.merge(blank_info, blank_intercept,how='outer', on='exp_nr')
        experiment_intercepts = pd.read_sql(
            f"SELECT serial_nr,Ar36_intercept,Ar37_intercept,Ar38_intercept,Ar39_intercept,Ar40_intercept FROM experiment_intercepts where exp_nr ='{exp_nr}'", software_conn)
        assigned_blank_intensities = pd.merge(intensities[['serial_nr', 'device', 'weight', 'method_name']].drop_duplicates(), blank_data,
                                              how='left', on='serial_nr')
        intensities_intercepts = pd.merge(
            assigned_blank_intensities, experiment_intercepts, how='left', on='serial_nr')
        header = Navbar()
        layout = html.Div([
            header,
            dcc.Tabs(id="exp-example-graph", value='exp-1-example-graph', children=[
                dcc.Tab(label='Experiment Assignment', value='exp-1-example-graph'),
                dcc.Tab(label='Experiment Intercepts', value='exp-2-example-graph'),
            ]),
            html.Div(id='exp-content-example-graph')
        ])
        return layout
    
    exp_app.layout = exp_layout

    def render_exp_table():
        return html.Div([html.Label("Experiments Details", 
                            style=dict(fontWeight='bold')),
                        dash_table.DataTable(
                        id="exp_table",
                        data=pd_exp.to_dict("records"),
                        columns=[{"id": x, "name": x}
                                 for x in pd_exp.columns],
                        style_cell=dict(textAlign='left'),
                        style_header=dict(fontWeight='bold'))]
                        ,style={'display': 'inline-block'})

    def render_blank_info_table():
            return html.Div([html.Label("Blank Intercept Info", style=dict(fontWeight='bold')),
                      dash_table.DataTable(
                data=blank_info.to_dict("records"),
                columns=[{"id": x, "name": x}
                         for x in blank_info.columns],
                page_current=0,
                page_size=8,
                page_action='native',
                style_cell=dict(textAlign='left'),
                style_header=dict(fontWeight='bold'),
            )
            ], style={'display': 'block'})

    def render_intensities_info_table():
        return html.Div([html.Label("Experiment Intercept Info", style=dict(fontWeight='bold')),
                  dash_table.DataTable(
            data=intensities_intercepts.to_dict("records"),
            columns=[{"id": x, "name": x}
                     for x in intensities_intercepts.columns],
            page_current=0,
            page_size=5,
            page_action='native',
            style_cell=dict(textAlign='left'),
            style_header=dict(fontWeight='bold'),
        )
        ], style={'display': 'block'})

    @exp_app.callback(Output('exp-content-example-graph', 'children'),
                      Input('exp-example-graph', 'value'))
    def render_content(tab):
        if tab == 'exp-1-example-graph':
            global pd_exp, blank_info, intensities_intercepts, intensities
            exp_index = np.squeeze(np.array(pd.read_sql(
                "SELECT exp_index  FROM experiments where exp_key='1'", software_conn)))
            dashboard_arardb = pd.read_sql(
                "SELECT exp_nr,material,proben_bez,project_name,sample_owner,irr_batch  FROM material order by exp_nr desc", conn)
            dashboard_arardb["id"] = dashboard_arardb.index
            global exp_nr
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
                    f"SELECT *  FROM blank_intercepts where exp_nr in {blank_list}", software_conn)
                blank_intercept['blank_assignment'] = 'Automatic'
            else:
                blank_list = tuple(
                    blank_assignments_data['blank_experiment_no'].unique())+(0,)
                blank_intercept['blank_assignment'] = 'Manual'

            blank_data1 = extract_blank_data(conn, blank_list)
            blank_info = blank_data1[['exp_nr', 'serial_nr', 'device', 'weight',
                                      'inlet_file', 'tuning_file', 'd_inlet_file']].drop_duplicates()
            blank_info = pd.merge(blank_info, blank_intercept,how='left', on='exp_nr')
            experiment_intercepts = pd.read_sql(
                f"SELECT serial_nr,Ar36_intercept,Ar37_intercept,Ar38_intercept,Ar39_intercept,Ar40_intercept FROM experiment_intercepts where exp_nr ='{exp_nr}'", software_conn)
            assigned_blank_intensities = pd.merge(intensities[['serial_nr', 'device', 'weight', 'method_name']].drop_duplicates(), blank_data,
                                                  how='left', on='serial_nr')
            intensities_intercepts = pd.merge(
            assigned_blank_intensities, experiment_intercepts, how='left', on='serial_nr')
            exp_info = render_exp_table()
            blank_info=render_blank_info_table()
            intensities_info = render_intensities_info_table()
            tab1_layout= html.Div(
                [html.Div([exp_info],
                style={'display': 'flex', 'justifyContent': 'center', "padding-top": '10px', "padding-bottom": '20px'}),
                html.Div([blank_info],
                         style={'display': 'flex', 'justifyContent': 'center', "padding-bottom": '20px'}),
                html.Div([intensities_info],
                         style={'display': 'flex', 'justifyContent': 'center', "padding-bottom": '20px'})
                ])
            return tab1_layout
        else:
            tab2_layout =  html.Div([
                html.Div([
                    html.Div([
                        html.Table([
                            html.Tr([html.Th(['Isotopes']), html.Th(['Ar36']), html.Th(
                                ['Ar37']), html.Th(['Ar38']), html.Th(['Ar39']), html.Th(['Ar40'])]),
                            html.Tr([html.Th(['Regression Intercepts']),
                                     html.Td(id='intercept_A36'), html.Td(
                                id='intercept_A37'), html.Td(
                                id='intercept_A38'), html.Td(
                                id='intercept_A39'), html.Td(
                                id='intercept_A40')]),
                            html.Tr([html.Th(['Regression MSE']),
                                     html.Td(id='loss_A36'), html.Td(
                                id='loss_A37'), html.Td(
                                id='loss_A38'), html.Td(
                                id='loss_A39'), html.Td(
                                id='loss_A40')])
                        ])], style={'display': 'inline-block',
                                    'vertical-align': 'top',
                                    'width': '70%', "padding-left": '15px', "padding-bottom": '5px'}),
                    html.Div([dcc.Dropdown(
                        intensities['serial_nr'].unique(),
                        intensities['serial_nr'].unique()[0],
                        id='blank_experiment_selection'),
                    ], style={'width': '10%', 'vertical-align': 'top', 'display': 'inline-block', "padding-right": '10px'}),
                    html.Div([html.Button('Save Intercepts', id='save-val', n_clicks=0)], style={'display': 'inline-block',
                                                                                                 'vertical-align': 'top',
                                                                                                 'width': '5%', "padding-right": '10px'}),
                    dcc.ConfirmDialog(id='confirm-danger',
                                      message='Are you sure you want to save?',
                                      ),
                    html.Div(id='container-button-basic')
                ],
                ),
                html.Div([
                    html.Div([
                        html.Div([
                            dcc.RadioItems(
                                ['Linear', 'Quadratic',
                                 'ElasticNet', 'Bayesian', 'Automatic'],
                                'Automatic',
                                id='regression_model_A36',
                                labelStyle={'display': 'inline-block', 'marginTop': '5px'})
                        ], style=dict(display='flex', justifyContent='center')),
                        dcc.RangeSlider(0, 100,
                                        step=2,
                                        id='quantile_slider_A36',
                                        value=[10, 90],
                                        marks={str(marker): str(marker) +
                                               '%' for marker in range(0, 100, 10)},
                                        allowCross=False
                                        ),
                        dcc.Graph(
                            id='figure_A36',
                        )
                    ], style={'width': '33%', 'display': 'inline-block', 'justifyContent': 'center'}),

                    html.Div([
                        html.Div([
                            dcc.RadioItems(
                                ['Linear', 'Quadratic',
                                 'ElasticNet', 'Bayesian', 'Automatic'],
                                'Automatic',
                                id='regression_model_A37',
                                labelStyle={'display': 'inline-block', 'marginTop': '5px'})
                        ], style=dict(display='flex', justifyContent='center')),
                        dcc.RangeSlider(0, 100,
                                        step=2,
                                        id='quantile_slider_A37',
                                        value=[10, 90],
                                        marks={str(marker): str(marker) +
                                               '%' for marker in range(0, 100, 10)},
                                        allowCross=False
                                        ),
                        dcc.Graph(
                            id='figure_A37',
                        ),
                    ], style={'width': '33%', 'display': 'inline-block'}),

                    html.Div([
                        html.Div([
                            dcc.RadioItems(
                                ['Linear', 'Quadratic',
                                 'ElasticNet', 'Bayesian', 'Automatic'],
                                'Automatic',
                                id='regression_model_A38',
                                labelStyle={'display': 'inline-block', 'marginTop': '5px'})
                        ], style=dict(display='flex', justifyContent='center')),
                        dcc.RangeSlider(0, 100,
                                        step=2,
                                        id='quantile_slider_A38',
                                        value=[10, 90],
                                        marks={str(marker): str(marker) +
                                               '%' for marker in range(0, 100, 10)},
                                        allowCross=False
                                        ),
                        dcc.Graph(
                            id='figure_A38',
                        )
                    ], style={'width': '33%', 'display': 'inline-block'})
                ], style={'display': 'block', "padding-bottom": '5px'}),

                html.Div([
                    html.Div([
                        html.Div([
                            dcc.RadioItems(
                                ['Linear', 'Quadratic',
                                    'ElasticNet', 'Bayesian', 'Automatic'],
                                'Automatic',
                                id='regression_model_A39',
                                labelStyle={'display': 'inline-block', 'marginTop': '5px'})
                        ], style=dict(display='flex', justifyContent='center')),
                        dcc.RangeSlider(0, 100,
                                        step=2,
                                        id='quantile_slider_A39',
                                        value=[10, 90],
                                        marks={str(marker): str(marker) +
                                               '%' for marker in range(0, 100, 10)},
                                        allowCross=False
                                        ),
                        dcc.Graph(
                            id='figure_A39',
                        )
                    ], style={'width': '33%', 'display': 'inline-block'}),

                    html.Div([
                        html.Div([
                            dcc.RadioItems(
                                ['Linear', 'Quadratic',
                                    'ElasticNet', 'Bayesian', 'Automatic'],
                                'Quadratic',
                                id='regression_model_A40',
                                labelStyle={'display': 'inline-block', 'marginTop': '5px'})
                        ], style=dict(display='flex', justifyContent='center')),

                        dcc.RangeSlider(0, 100,
                                        step=2,
                                        id='quantile_slider_A40',
                                        value=[5, 95],
                                        marks={str(marker): str(marker) +
                                               '%' for marker in range(0, 100, 10)},
                                        allowCross=False
                                        ),
                        dcc.Graph(
                            id='figure_A40',)
                    ], style={'width': '33%', 'display': 'inline-block'})
                ], style={'display': 'block'}),
                dcc.Store(id='intermediate-value')
            ], style={'width': '100%', 'display': 'inline-block'})
            
            return tab2_layout

    @exp_app.callback(
            Output('figure_A36', 'figure'),
            Output('intercept_A36', 'children'),
            Output('loss_A36', 'children'),
            Input('blank_experiment_selection', 'value'),
            Input('quantile_slider_A36', 'value'),
            Input('regression_model_A36', 'value'))
    def updated_Ar36(exp_no, quantile_range, regression_model):
        Ar_blank = intensities[intensities['serial_nr'] == exp_no]
        x = Ar_blank['start']
        v = Ar_blank['v36']
        low_percentile = quantile_range[0]/100
        high_percentile = quantile_range[1]/100
        q_low = v.quantile(low_percentile)
        q_hi = v.quantile(high_percentile)
        data = [go.Scatter(x=x[(v < q_hi) & (v > q_low)].to_list(),
                           y=v[(v < q_hi) & (v > q_low)].to_list(), mode='markers', name='not Outliers'),
                go.Scatter(x=x[(v > q_hi)].to_list(),
                           y=v[(v > q_hi)].to_list(), mode='markers', name='Upper Outliers'),
                go.Scatter(x=x[(v < q_low)].to_list(),
                           y=v[(v < q_low)].to_list(), mode='markers', name='Lower Outliers')]
        X_quan = x[(v < q_hi) & (v > q_low)]
        Y_quan = v[(v < q_hi) & (v > q_low)]
        model = regression_models(X_quan, Y_quan, regression_model)
        if regression_model == 'Linear':
            intercept = model.predict([[np.array(x)[0]]])
            Y_pred = model.predict(np.array(X_quan).reshape(-1, 1))
            loss = mean_squared_error(Y_pred, np.array(Y_quan))
            line_model = np.squeeze(model.predict(np.array(x).reshape(-1, 1)))
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        name=regression_model+' regression'))
        elif regression_model == 'Quadratic':
            polynomial_features = PolynomialFeatures(degree=2)
            x_all = polynomial_features.fit_transform(
                np.array(x).reshape(-1, 1))
            X_TRANSF = polynomial_features.fit_transform(
                np.array(X_quan).reshape(-1, 1))
            intercept = model.predict([x_all[0, :]])
            Y_pred = model.predict(X_TRANSF)
            loss = mean_squared_error(Y_pred, np.array(Y_quan))
            line_model = np.squeeze(model.predict(x_all))
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        name=regression_model+' regression'))
        elif regression_model == 'SupportVectors':
            intercept = model.predict([[np.array(x)[0]]])
            Y_pred = model.predict(np.array(X_quan).reshape(-1, 1))
            loss = mean_squared_error(Y_pred, np.array(Y_quan))
            line_model = np.squeeze(model.predict(np.array(x).reshape(-1, 1)))
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        name=regression_model+' regression'))
        elif regression_model == 'ElasticNet':
            intercept = model.predict([[np.array(x)[0]]])
            Y_pred = model.predict(np.array(X_quan).reshape(-1, 1))
            loss = mean_squared_error(Y_pred, np.array(Y_quan))
            line_model = np.squeeze(model.predict(np.array(x).reshape(-1, 1)))
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        name=regression_model+' regression'))
        elif regression_model == 'Bayesian':
            intercept = model.predict([[np.array(x)[0]]])
            Y_pred = model.predict(np.array(X_quan).reshape(-1, 1))
            loss = mean_squared_error(Y_pred, np.array(Y_quan))
            line_model = np.squeeze(model.predict(np.array(x).reshape(-1, 1)))
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        name=regression_model+' regression'))
        elif regression_model == 'Automatic':
            intercept = model.predict([[np.array(x)[0]]])
            Y_pred = model.predict(np.array(X_quan).reshape(-1, 1))
            loss = mean_squared_error(Y_pred, np.array(Y_quan))
            line_model = np.squeeze(model.predict(np.array(x).reshape(-1, 1)))
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                                   name=regression_model+' regression'))

        figure = go.Figure(data=data)
        figure.update_layout(title="Ar36", showlegend=False)

        figure.update_xaxes(title_text='Time (sec)',
                            linewidth=1)

        figure.update_yaxes(title_text='Intensity (V)',
                            linewidth=1)
        if model is None:
            return figure, 0, 0
        return figure, np.squeeze(intercept), loss

    @exp_app.callback(
        Output('figure_A37', 'figure'),
        Output('intercept_A37', 'children'),
        Output('loss_A37', 'children'),
        Input('blank_experiment_selection', 'value'),
        Input('quantile_slider_A37', 'value'),
        Input('regression_model_A37', 'value'))
    def updated_A37(exp_no, quantile_range, regression_model):
        Ar_blank = intensities[intensities['serial_nr'] == exp_no]
        x = Ar_blank['start']
        v = Ar_blank['v37']
        low_percentile = quantile_range[0]/100
        high_percentile = quantile_range[1]/100
        q_low = v.quantile(low_percentile)
        q_hi = v.quantile(high_percentile)
        data = [go.Scatter(x=x[(v < q_hi) & (v > q_low)].to_list(),
                           y=v[(v < q_hi) & (v > q_low)].to_list(), mode='markers', name='not Outliers'),
                go.Scatter(x=x[(v > q_hi)].to_list(),
                           y=v[(v > q_hi)].to_list(), mode='markers', name='Upper Outliers'),
                go.Scatter(x=x[(v < q_low)].to_list(),
                           y=v[(v < q_low)].to_list(), mode='markers', name='Lower Outliers')]
        model = regression_models(x[(v < q_hi) & (v > q_low)],
                                  v[(v < q_hi) & (v > q_low)], regression_model)
        X_quan = x[(v < q_hi) & (v > q_low)]
        Y_quan = v[(v < q_hi) & (v > q_low)]
        model = regression_models(X_quan, Y_quan, regression_model)
        if regression_model == 'Linear':
            intercept = model.predict([[np.array(x)[0]]])
            Y_pred = model.predict(np.array(X_quan).reshape(-1, 1))
            loss = mean_squared_error(Y_pred, np.array(Y_quan))
            line_model = np.squeeze(model.predict(np.array(x).reshape(-1, 1)))
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        name=regression_model+' regression'))
        elif regression_model == 'Quadratic':
            polynomial_features = PolynomialFeatures(degree=2)
            x_all = polynomial_features.fit_transform(
                np.array(x).reshape(-1, 1))
            X_TRANSF = polynomial_features.fit_transform(
                np.array(X_quan).reshape(-1, 1))
            intercept = model.predict([x_all[0, :]])
            Y_pred = model.predict(X_TRANSF)
            loss = mean_squared_error(Y_pred, np.array(Y_quan))
            line_model = np.squeeze(model.predict(x_all))
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        name=regression_model+' regression'))
        elif regression_model == 'SupportVectors':
            intercept = model.predict([[np.array(x)[0]]])
            Y_pred = model.predict(np.array(X_quan).reshape(-1, 1))
            loss = mean_squared_error(Y_pred, np.array(Y_quan))
            line_model = np.squeeze(model.predict(np.array(x).reshape(-1, 1)))
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        name=regression_model+' regression'))
        elif regression_model == 'ElasticNet':
            intercept = model.predict([[np.array(x)[0]]])
            Y_pred = model.predict(np.array(X_quan).reshape(-1, 1))
            loss = mean_squared_error(Y_pred, np.array(Y_quan))
            line_model = np.squeeze(model.predict(np.array(x).reshape(-1, 1)))
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        name=regression_model+' regression'))
        elif regression_model == 'Bayesian':
            intercept = model.predict([[np.array(x)[0]]])
            Y_pred = model.predict(np.array(X_quan).reshape(-1, 1))
            loss = mean_squared_error(Y_pred, np.array(Y_quan))
            line_model = np.squeeze(model.predict(np.array(x).reshape(-1, 1)))
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        name=regression_model+' regression'))
        elif regression_model == 'Automatic':
            intercept = model.predict([[np.array(x)[0]]])
            Y_pred = model.predict(np.array(X_quan).reshape(-1, 1))
            loss = mean_squared_error(Y_pred, np.array(Y_quan))
            line_model = np.squeeze(model.predict(np.array(x).reshape(-1, 1)))
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                                   name=regression_model+' regression'))

        figure = go.Figure(data=data)
        figure.update_layout(title="Ar37", showlegend=False)
        figure.update_xaxes(title_text='Time (sec)',
                            linewidth=1)

        figure.update_yaxes(title_text='Intensity (V)',
                            linewidth=1)
        if model is None:
            return figure, 0, 0
        return figure, np.squeeze(intercept), loss

    @exp_app.callback(
        Output('figure_A38', 'figure'),
        Output('intercept_A38', 'children'),
        Output('loss_A38', 'children'),
        Input('blank_experiment_selection', 'value'),
        Input('quantile_slider_A38', 'value'),
        Input('regression_model_A38', 'value'))
    def updated_A38(exp_no, quantile_range, regression_model):
        Ar_blank = intensities[intensities['serial_nr'] == exp_no]
        x = Ar_blank['start']
        v = Ar_blank['v38']
        low_percentile = quantile_range[0]/100
        high_percentile = quantile_range[1]/100
        q_low = v.quantile(low_percentile)
        q_hi = v.quantile(high_percentile)
        data = [go.Scatter(x=x[(v < q_hi) & (v > q_low)].to_list(),
                           y=v[(v < q_hi) & (v > q_low)].to_list(), mode='markers', name='not Outliers'),
                go.Scatter(x=x[(v > q_hi)].to_list(),
                           y=v[(v > q_hi)].to_list(), mode='markers', name='Upper Outliers'),
                go.Scatter(x=x[(v < q_low)].to_list(),
                           y=v[(v < q_low)].to_list(), mode='markers', name='Lower Outliers')]
        X_quan = x[(v < q_hi) & (v > q_low)]
        Y_quan = v[(v < q_hi) & (v > q_low)]
        model = regression_models(X_quan, Y_quan, regression_model)
        if regression_model == 'Linear':
            intercept = model.predict([[np.array(x)[0]]])
            Y_pred = model.predict(np.array(X_quan).reshape(-1, 1))
            loss = mean_squared_error(Y_pred, np.array(Y_quan))
            line_model = np.squeeze(model.predict(np.array(x).reshape(-1, 1)))
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        name=regression_model+' regression'))
        elif regression_model == 'Quadratic':
            polynomial_features = PolynomialFeatures(degree=2)
            x_all = polynomial_features.fit_transform(
                np.array(x).reshape(-1, 1))
            X_TRANSF = polynomial_features.fit_transform(
                np.array(X_quan).reshape(-1, 1))
            intercept = model.predict([x_all[0, :]])
            Y_pred = model.predict(X_TRANSF)
            loss = mean_squared_error(Y_pred, np.array(Y_quan))
            line_model = np.squeeze(model.predict(x_all))
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        name=regression_model+' regression'))
        elif regression_model == 'SupportVectors':
            intercept = model.predict([[np.array(x)[0]]])
            Y_pred = model.predict(np.array(X_quan).reshape(-1, 1))
            loss = mean_squared_error(Y_pred, np.array(Y_quan))
            line_model = np.squeeze(model.predict(np.array(x).reshape(-1, 1)))
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        name=regression_model+' regression'))
        elif regression_model == 'ElasticNet':
            intercept = model.predict([[np.array(x)[0]]])
            Y_pred = model.predict(np.array(X_quan).reshape(-1, 1))
            loss = mean_squared_error(Y_pred, np.array(Y_quan))
            line_model = np.squeeze(model.predict(np.array(x).reshape(-1, 1)))
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        name=regression_model+' regression'))
        elif regression_model == 'Bayesian':
            intercept = model.predict([[np.array(x)[0]]])
            Y_pred = model.predict(np.array(X_quan).reshape(-1, 1))
            loss = mean_squared_error(Y_pred, np.array(Y_quan))
            line_model = np.squeeze(model.predict(np.array(x).reshape(-1, 1)))
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        name=regression_model+' regression'))
        elif regression_model == 'Automatic':
            intercept = model.predict([[np.array(x)[0]]])
            Y_pred = model.predict(np.array(X_quan).reshape(-1, 1))
            loss = mean_squared_error(Y_pred, np.array(Y_quan))
            line_model = np.squeeze(model.predict(np.array(x).reshape(-1, 1)))
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                                   name=regression_model+' regression'))

        figure = go.Figure(data=data)
        figure.update_layout(title="Ar38", showlegend=False)
        figure.update_xaxes(title_text='Time (sec)',
                            linewidth=1)

        figure.update_yaxes(title_text='Intensity (V)',
                            linewidth=1)
        if model is None:
            return figure, 0, 0
        return figure, np.squeeze(intercept), loss

    @exp_app.callback(
        Output('figure_A39', 'figure'),
        Output('intercept_A39', 'children'),
        Output('loss_A39', 'children'),
        Input('blank_experiment_selection', 'value'),
        Input('quantile_slider_A39', 'value'),
        Input('regression_model_A39', 'value'))
    def updated_A39(exp_no, quantile_range, regression_model):
        Ar_blank = intensities[intensities['serial_nr'] == exp_no]
        x = Ar_blank['start']
        v = Ar_blank['v39']
        low_percentile = quantile_range[0]/100
        high_percentile = quantile_range[1]/100
        q_low = v.quantile(low_percentile)
        q_hi = v.quantile(high_percentile)
        data = [go.Scatter(x=x[(v < q_hi) & (v > q_low)].to_list(),
                           y=v[(v < q_hi) & (v > q_low)].to_list(), mode='markers', name='not Outliers'),
                go.Scatter(x=x[(v > q_hi)].to_list(),
                           y=v[(v > q_hi)].to_list(), mode='markers', name='Upper Outliers'),
                go.Scatter(x=x[(v < q_low)].to_list(),
                           y=v[(v < q_low)].to_list(), mode='markers', name='Lower Outliers')]
        X_quan = x[(v < q_hi) & (v > q_low)]
        Y_quan = v[(v < q_hi) & (v > q_low)]
        model = regression_models(X_quan, Y_quan, regression_model)
        if regression_model == 'Linear':
            intercept = model.predict([[np.array(x)[0]]])
            Y_pred = model.predict(np.array(X_quan).reshape(-1, 1))
            loss = mean_squared_error(Y_pred, np.array(Y_quan))
            line_model = np.squeeze(model.predict(np.array(x).reshape(-1, 1)))
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        name=regression_model+' regression'))
        elif regression_model == 'Quadratic':
            polynomial_features = PolynomialFeatures(degree=2)
            x_all = polynomial_features.fit_transform(
                np.array(x).reshape(-1, 1))
            X_TRANSF = polynomial_features.fit_transform(
                np.array(X_quan).reshape(-1, 1))
            intercept = model.predict([x_all[0, :]])
            Y_pred = model.predict(X_TRANSF)
            loss = mean_squared_error(Y_pred, np.array(Y_quan))
            line_model = np.squeeze(model.predict(x_all))
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        name=regression_model+' regression'))
        elif regression_model == 'SupportVectors':
            intercept = model.predict([[np.array(x)[0]]])
            Y_pred = model.predict(np.array(X_quan).reshape(-1, 1))
            loss = mean_squared_error(Y_pred, np.array(Y_quan))
            line_model = np.squeeze(model.predict(np.array(x).reshape(-1, 1)))
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        name=regression_model+' regression'))
        elif regression_model == 'ElasticNet':
            intercept = model.predict([[np.array(x)[0]]])
            Y_pred = model.predict(np.array(X_quan).reshape(-1, 1))
            loss = mean_squared_error(Y_pred, np.array(Y_quan))
            line_model = np.squeeze(model.predict(np.array(x).reshape(-1, 1)))
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        name=regression_model+' regression'))
        elif regression_model == 'Bayesian':
            intercept = model.predict([[np.array(x)[0]]])
            Y_pred = model.predict(np.array(X_quan).reshape(-1, 1))
            loss = mean_squared_error(Y_pred, np.array(Y_quan))
            line_model = np.squeeze(model.predict(np.array(x).reshape(-1, 1)))
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        name=regression_model+' regression'))
        elif regression_model == 'Automatic':
            intercept = model.predict([[np.array(x)[0]]])
            Y_pred = model.predict(np.array(X_quan).reshape(-1, 1))
            loss = mean_squared_error(Y_pred, np.array(Y_quan))
            line_model = np.squeeze(model.predict(np.array(x).reshape(-1, 1)))
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                                   name=regression_model+' regression'))

        figure = go.Figure(data=data)
        figure.update_layout(title="Ar39", showlegend=False)
        figure.update_xaxes(title_text='Time (sec)',
                            linewidth=1)

        figure.update_yaxes(title_text='Intensity (V)',
                            linewidth=1)
        if model is None:
            return figure, 0, 0
        return figure, np.squeeze(intercept), loss

    @exp_app.callback(
        Output('figure_A40', 'figure'),
        Output('intercept_A40', 'children'),
        Output('loss_A40', 'children'),
        Input('blank_experiment_selection', 'value'),
        Input('quantile_slider_A40', 'value'),
        Input('regression_model_A40', 'value'))
    def updated_A40(exp_no, quantile_range, regression_model):
        Ar_blank = intensities[intensities['serial_nr'] == exp_no]
        x = Ar_blank['start']
        v = Ar_blank['v40']
        low_percentile = quantile_range[0]/100
        high_percentile = quantile_range[1]/100
        q_low = v.quantile(low_percentile)
        q_hi = v.quantile(high_percentile)
        data = [go.Scatter(x=x[(v < q_hi) & (v > q_low)].to_list(),
                           y=v[(v < q_hi) & (v > q_low)].to_list(), mode='markers', name='not Outliers'),
                go.Scatter(x=x[(v > q_hi)].to_list(),
                           y=v[(v > q_hi)].to_list(), mode='markers', name='Upper Outliers'),
                go.Scatter(x=x[(v < q_low)].to_list(),
                           y=v[(v < q_low)].to_list(), mode='markers', name='Lower Outliers')]
        X_quan = x[(v < q_hi) & (v > q_low)]
        Y_quan = v[(v < q_hi) & (v > q_low)]
        model = regression_models(X_quan, Y_quan, regression_model)
        if regression_model == 'Linear':
            intercept = model.predict([[np.array(x)[0]]])
            Y_pred = model.predict(np.array(X_quan).reshape(-1, 1))
            loss = mean_squared_error(Y_pred, np.array(Y_quan))
            line_model = np.squeeze(model.predict(np.array(x).reshape(-1, 1)))
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        name=regression_model+' regression'))
        elif regression_model == 'Quadratic':
            polynomial_features = PolynomialFeatures(degree=2)
            x_all = polynomial_features.fit_transform(
                np.array(x).reshape(-1, 1))
            X_TRANSF = polynomial_features.fit_transform(
                np.array(X_quan).reshape(-1, 1))
            intercept = model.predict([x_all[0, :]])
            Y_pred = model.predict(X_TRANSF)
            loss = mean_squared_error(Y_pred, np.array(Y_quan))
            line_model = np.squeeze(model.predict(x_all))
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        name=regression_model+' regression'))
        elif regression_model == 'SupportVectors':
            intercept = model.predict([[np.array(x)[0]]])
            Y_pred = model.predict(np.array(X_quan).reshape(-1, 1))
            loss = mean_squared_error(Y_pred, np.array(Y_quan))
            line_model = np.squeeze(model.predict(np.array(x).reshape(-1, 1)))
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        name=regression_model+' regression'))
        elif regression_model == 'ElasticNet':
            intercept = model.predict([[np.array(x)[0]]])
            Y_pred = model.predict(np.array(X_quan).reshape(-1, 1))
            loss = mean_squared_error(Y_pred, np.array(Y_quan))
            line_model = np.squeeze(model.predict(np.array(x).reshape(-1, 1)))
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        name=regression_model+' regression'))
        elif regression_model == 'Bayesian':
            intercept = model.predict([[np.array(x)[0]]])
            Y_pred = model.predict(np.array(X_quan).reshape(-1, 1))
            loss = mean_squared_error(Y_pred, np.array(Y_quan))
            line_model = np.squeeze(model.predict(np.array(x).reshape(-1, 1)))
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        name=regression_model+' regression'))
        elif regression_model == 'Automatic':
            intercept = model.predict([[np.array(x)[0]]])
            Y_pred = model.predict(np.array(X_quan).reshape(-1, 1))
            loss = mean_squared_error(Y_pred, np.array(Y_quan))
            line_model = np.squeeze(model.predict(np.array(x).reshape(-1, 1)))
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                                   name=regression_model+' regression'))

        figure = go.Figure(data=data)
        figure.update_layout(title="Ar40", showlegend=False)
        figure.update_xaxes(title_text='Time (sec)',
                            linewidth=1)

        figure.update_yaxes(title_text='Intensity (V)',
                            linewidth=1)
        if model is None:
            return figure, 0, 0
        return figure, np.squeeze(intercept), loss

    @exp_app.callback(Output('confirm-danger', 'displayed'),
                        Input('save-val', 'n_clicks'))
    def display_confirm(n):
        if n >= 1:
            return True
        return False

    def update_blank_intercepts(serial_nr, A36, A37, A38, A39, A40):
        cur = software_conn.cursor()
        cur.execute(
            f"INSERT INTO experiment_intercepts (exp_nr,serial_nr, Ar36_intercept, Ar37_intercept, Ar38_intercept, Ar39_intercept, Ar40_intercept) VALUES('{exp_nr}','{serial_nr}', '{A36}', '{A37}', '{A38}', '{A39}', '{A40}') ON DUPLICATE KEY UPDATE Ar36_intercept = '{A36}', Ar37_intercept ='{A37}', Ar38_intercept ='{A38}', Ar39_intercept= '{A39}', Ar40_intercept ='{A40}'")
        software_conn.commit()
        pass

    @exp_app.callback(
        Output('container-button-basic', 'children'),
        Output('confirm-danger', 'submit_n_clicks'),
        Input('confirm-danger', 'submit_n_clicks'),
        Input('blank_experiment_selection', 'value'),
        Input('intercept_A36', 'children'),
        Input('intercept_A37', 'children'),
        Input('intercept_A38', 'children'),
        Input('intercept_A39', 'children'),
        Input('intercept_A40', 'children')
    )
    def save_intercepts(n, serial_nr, A36, A37, A38, A39, A40):
        if n is not None and n >= 1:
            update_blank_intercepts(serial_nr, np.squeeze(A36), np.squeeze(
                A37), np.squeeze(A38), np.squeeze(A39), np.squeeze(A40))
        return 0, 0
    return exp_app
# if __name__ == '__main__':
#     app.run_server(debug=True)
