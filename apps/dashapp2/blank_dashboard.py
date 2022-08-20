from dash import Dash, html, dcc, Input, Output, State, dash_table, no_update
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from apps.backend.auxiliary_functions import *
from apps.dash_navbar.navigation_bar import Navbar
import pandas as pd
import json
import numpy as np
from sqlalchemy import create_engine
import dash

def blank_dashboard(server):
    external_stylesheets = [
        'https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP]
    blank_app = Dash(__name__, server=server, url_base_pathname='/blanks/',
                     external_stylesheets=external_stylesheets)
    dash.register_page(__name__,path='/blanks/')
    def update_layout():
        global software_conn
        softurl = r"mariadb+mariadbconnector://root:dbpwx61@139.20.22.156:3306/ararsoftware"
        soft_engine = create_engine(softurl)
        global conn
        url = r"mariadb+mariadbconnector://root:dbpwx61@139.20.22.156:3306/arardb"
        engine = create_engine(url)
        software_conn = soft_engine.connect()
        exp_index = np.squeeze(np.array(pd.read_sql(
                "SELECT exp_index  FROM experiments where exp_key=1", software_conn)))
        conn=engine.connect()    
        dashboard_arardb = pd.read_sql(
            "SELECT exp_nr,material,proben_bez,project_name,sample_owner,irr_batch  FROM material order by exp_nr desc", conn)
        dashboard_arardb["id"] = dashboard_arardb.index
        global exp_nr
        exp_nr = str(dashboard_arardb.iloc[exp_index][0])
        global pd_exp, experiment_info
        experiment_info = extract_experiment_info(exp_nr, database)
        experiment_info = extract_irradiation(conn, experiment_info)
        pd_exp = pd.DataFrame.from_dict(experiment_info)
        pd_exp = pd_exp[['Exp_No', 'Exp_type', 'Sample', 'Material', 'Project', 'Owner',
                         'Irradiation', 'Plate', 'Hole', 'Weight', 'Jvalue', 'J_Error', 'f',
                        'f_error', 'cycle_count', 'tuning_file', 'irr_enddatetime', 'irr_duration',
                         'irr_enddatetime_num']]
        global intensities
        intensities = extract_intensities(conn, experiment_info)
        sensitivities = extracting_senstivities(conn, experiment_info)
        intensities = pd.merge(intensities, sensitivities,
                               how='left', on='serial_nr')
        global all_blanks
        dashboard_arardb = pd.read_sql(
            "SELECT exp_nr,material,proben_bez,project_name,sample_owner,irr_batch  FROM material order by exp_nr desc", conn)
    
        all_blanks = extracting_all_blanks(conn)
        global blank_data
        blank_data = extracting_blanks(
            conn, intensities[[
                'acq_datetime']].min()[0], intensities[[
                    'acq_datetime']].max()[0])
        global subset_intensities
        subset_intensities = intensities[[
            'serial_nr', 'acq_datetime']].drop_duplicates()
        blank_data = automatic_blank_assigment(
            blank_data, subset_intensities)
        blank_list = tuple(blank_data['blank_experiment_no'].unique())+(0,)
        global blank_data1
        blank_data1 = extract_blank_data(conn, blank_list)
        global manual_intensities
        manual_intensities = intensities
        intensities = pd.merge(intensities, blank_data,
                               how='left', on='serial_nr')
        global blank_intercept
        blank_intercept = pd.read_sql(
            f"SELECT *  FROM experiment_intercepts where exp_nr in {blank_list}", software_conn)
        global blanks
        blanks = blank_data1[['serial_nr', 'exp_nr', 'start', 'v36', 'v37', 'v38',
                              'v39', 'v40', 'time_zero']]
        header = Navbar()
        layout = html.Div([
            header,
            dcc.Tabs(id="tabs-example-graph", value='tab-1-example-graph', children=[
                dcc.Tab(label='Blank Assignment', value='tab-1-example-graph'),
                dcc.Tab(label='Blank Intercepts', value='tab-2-example-graph'),
            ]),
            html.Div(id='tabs-content-example-graph')
        ])
        return layout

    blank_app.layout = update_layout

    @blank_app.callback(Output('tabs-content-example-graph', 'children'),
                        Input('tabs-example-graph', 'value'))
    def render_content(tab):
        if tab == 'tab-1-example-graph':
            return html.Div([
                html.Div([html.Div([html.Label("Experiments Details", style=dict(fontWeight='bold')),
                          dash_table.DataTable(
                    id="exp_table",
                    data=pd_exp.to_dict("records"),
                    columns=[{"id": x, "name": x}
                             for x in pd_exp.columns],
                    style_cell=dict(textAlign='left'),
                    style_header=dict(fontWeight='bold'),
                )], style={'display': 'inline-block'})], style={'display': 'flex', 'justifyContent': 'center', "padding-top": '10px', "padding-bottom": '20px'}),
            html.Div([
                    dcc.RadioItems(
                         ['Manual', 'Automatic'],
                         'Manual',
                         id='blank_selection_id',
                         labelStyle={'display': 'inline-block', "padding-left": '25px', 'marginTop': '5px'})
                ], style=dict(display='flex', justifyContent='left', padding_bottom='10px')),  # blanktable

                html.Div([html.Div(id='blanktable')]), ])
        elif tab == 'tab-2-example-graph':
            layout = html.Div([
                html.Div([
                    html.Div([
                        html.Table([
                            html.Tr([html.Th(['Isotopes']), html.Th(['Ar36']), html.Th(
                                ['Ar37']), html.Th(['Ar38']), html.Th(['Ar39']), html.Th(['Ar40'])]),
                            html.Tr([html.Th(['Regression Intercepts ']),
                                     html.Td(id='intercept_offset_A36'), html.Td(
                                id='intercept_offset_A37'), html.Td(
                                id='intercept_offset_A38'), html.Td(
                                id='intercept_offset_A39'), html.Td(
                                id='intercept_offset_A40')]),
                            html.Tr([html.Th(['Intercepts Coefficient']),
                                     html.Td(id='intercept_A36'), html.Td(
                                id='intercept_A37'), html.Td(
                                id='intercept_A38'), html.Td(
                                id='intercept_A39'), html.Td(
                                id='intercept_A40')]),
                            html.Tr([html.Th(['Intercept Standard error \u00B1']),
                                     html.Td(id='loss_A36'), html.Td(
                                id='loss_A37'), html.Td(
                                id='loss_A38'), html.Td(
                                id='loss_A39'), html.Td(
                                id='loss_A40')])
                        ])], style={'display': 'inline-block',
                                    'vertical-align': 'top',
                                    'width': '70%', "padding-left": '15px', "padding-bottom": '5px'}),
                    html.Div([dcc.Dropdown(
                        blanks['exp_nr'].unique(),
                        blanks['exp_nr'].unique()[0],
                        id='blank_experiment_selection'),
                    ], style={'width': '10%', 'vertical-align': 'top', 'display': 'inline-block', "padding-right": '10px'}),
                    html.Div([html.Button('Save Intercepts', id='save-val', n_clicks=0)], style={'display': 'inline-block',
                                                                                                 'vertical-align': 'top',
                                                                                                 'width': '5%', "padding-right": '10px'}),
                    dcc.ConfirmDialog(id='confirm-danger',
                                      message='Are you sure you want to save?',
                                      ),
                    html.Div(id='container-button-basic',
                             children='')
                ],
                ), html.Div(id='intercept_tab',
                            children='')])
            return layout

    @blank_app.callback(Output('blanktable', 'children'),
                        Input('blank_selection_id', 'value'))
    def blank_assignment(value):
        if value == 'Automatic':
            global intensities
            intensities = extract_intensities(conn, experiment_info)
            sensitivities = extracting_senstivities(conn, experiment_info)
            intensities = pd.merge(intensities, sensitivities,
                                   how='left', on='serial_nr')
            global all_blanks
            all_blanks = extracting_all_blanks(conn)
            global blank_data
            blank_data = extracting_blanks(
                conn, intensities[[
                    'acq_datetime']].min()[0], intensities[[
                        'acq_datetime']].max()[0])

            subset_intensities = intensities[[
                'serial_nr', 'acq_datetime']].drop_duplicates()
            global blanks
            blank_data = automatic_blank_assigment(
                blank_data, subset_intensities)
            blank_list = tuple(blank_data['blank_experiment_no'].unique())+(0,)
            global blank_intercept
            blank_intercept = pd.read_sql(
                f"SELECT *  FROM experiment_intercepts where exp_nr in {blank_list}", software_conn)
            global blank_data1
            blank_data1 = extract_blank_data(conn, blank_list)
            global manual_intensities
            manual_intensities = intensities
            intensities = pd.merge(intensities, blank_data,
                                   how='left', on='serial_nr')
            global blanks
            blanks = blank_data1[['serial_nr', 'exp_nr', 'start', 'v36', 'v37', 'v38',
                                  'v39', 'v40', 'time_zero']]
            #print(blanks)
            blank_info = blank_data1[["exp_nr", "serial_nr", "device", "acq_datetime", "inlet_file", 'd_inlet_file', 'method_name', 'd_method_name',
                                      ]].drop_duplicates()
            blank_info = blank_info.rename(
                columns={"d_inlet_file": "inlet_time", "d_method_name": "method_time"})
            blank_layout = html.Div([html.Div([html.Div([
                html.Label("Blank experiments Information table",
                           style=dict(fontWeight='bold', display='inline-block')),
                dash_table.DataTable(
                    data=blank_info.to_dict("records"),
                    columns=[{"id": x, "name": x}
                             for x in blank_info.columns],
                    style_cell=dict(textAlign='left'),
                    style_header=dict(fontWeight='bold'),
                )], style={
                    'width': '75%', 'display': 'inline-block', 'justifyContent': 'center', "padding-left": '15px', "padding-top": '10px'})],
                style={
                'display': 'flex', 'justifyContent': 'center', "padding-left": '5px', "padding-right": '15px', "padding-top": '10px', "padding-bottom": '10px'}
            ),
                html.Div([html.Div([html.Label("Blank Intercepts table", style=dict(fontWeight='bold', display='inline-block')),
                          dash_table.DataTable(
                    data=blank_intercept.to_dict("records"),
                    columns=[{"id": x, "name": x}
                             for x in blank_intercept.columns],
                    style_cell=dict(textAlign='left'),
                    style_header=dict(fontWeight='bold'),

                )
                ], style={
                     'display': 'inline-block', 'justifyContent': 'center', "padding-top": '10px'}
                )], style={
                    'display': 'flex', 'justifyContent': 'center', "padding-left": '5px', "padding-right": '15px', "padding-top": '10px', "padding-bottom": '20px'}
            ),
                html.Div(
                html.Div([html.Label("Blank Assignment table", style=dict(fontWeight='bold')),
                          dash_table.DataTable(
                    data=blank_data.to_dict("records"),
                    columns=[{"id": x, "name": x}
                             for x in blank_data.columns],
                    page_current=0,
                    page_size=8,
                    page_action='native',
                    style_cell=dict(textAlign='left'),
                    style_header=dict(fontWeight='bold'),
                )

                ], style={'display': 'block'}), style={
                    'display': 'flex', 'justifyContent': 'center'})])
            return blank_layout
        else:
            blank_layout = html.Div([html.Div([html.Div([html.Label("Blank experiments list", style=dict(fontWeight='bold')),
                                               dash_table.DataTable(
                id="all_blank_table",
                data=all_blanks.to_dict("records"),
                columns=[{"name": i, "id": i}
                         for i in all_blanks.columns if i != "id"],
                sort_action='native',
                filter_action='native',
                row_selectable="multiple",
                page_current=0,
                page_size=5,
                page_action='native',
                style_cell=dict(textAlign='left'),
                style_header=dict(fontWeight='bold'),
            )

            ], style={'display': 'block'})], style={
                'display': 'flex', 'justifyContent': 'center'}), html.Div([html.Div(id='blankassign_table')])])

            return blank_layout

    @blank_app.callback(Output('intercept_tab', 'children'),
                        Input('blank_experiment_selection', 'value'))
    def blank_intercept(val):
        blank_intercept = pd.read_sql_query(f"select * from experiments_analysis where exp_nr={val}",software_conn)
        if blank_intercept.empty :
            blank_dic = {'exp_nr': {0: val}, 'serial_nr': {0: 0}, 'ar36_intercept': {0: 0.0}, 'ar36_intercept_error': {0: 0.0}, 'ar36_regression_model': {0: 'Automatic'}, 'ar36_outlier_model': {0: 0}, 'ar36_upper_quantile': {0: 100}, 'ar36_lower_quantile': {0: 0}, 'ar37_intercept': {0: 0.0}, 'ar37_intercept_error': {0: 0.0}, 'ar37_regression_model': {0: 'Automatic'}, 'ar37_outlier_model': {0: 0}, 'ar37_upper_quantile': {0: 100}, 'ar37_lower_quantile': {0: 0}, 'ar38_intercept': {0: 0.0}, 'ar38_intercept_error': {
                0: 0.0}, 'ar38_regression_model': {0: 'Automatic'}, 'ar38_outlier_model': {0: 0}, 'ar38_upper_quantile': {0: 100}, 'ar38_lower_quantile': {0: 0}, 'ar39_intercept': {0: 0.0}, 'ar39_intercept_error': {0: 0.0}, 'ar39_regression_model': {0: 'Automatic'}, 'ar39_outlier_model': {0: 0}, 'ar39_upper_quantile': {0: 100}, 'ar39_lower_quantile': {0: 0}, 'ar40_intercept': {0: 0.0}, 'ar40_intercept_error': {0: 0.0}, 'ar40_regression_model': {0: 'Automatic'}, 'ar40_outlier_model': {0: 0}, 'ar40_upper_quantile': {0: 100}, 'ar40_lower_quantile': {0: 0}}
            blank_intercept=pd.DataFrame(blank_dic)
        
        layout=html.Div([html.Div([
                    html.Div([
                        html.Div([
                            dcc.RadioItems(
                                ['Linear', 'Quadratic',
                                 'ElasticNet', 'Bayesian', 'Automatic'],
                                blank_intercept['ar36_regression_model'].values[0],
                                id='regression_model_A36',
                                labelStyle={'display': 'inline-block', 'marginTop': '5px'})
                        ], style=dict(display='flex', justifyContent='center')),
                        dcc.RangeSlider(0, 100,
                                        step=2,
                                        id='quantile_slider_A36',
                                        value=[
                                            blank_intercept['ar36_lower_quantile'].values[0], blank_intercept['ar36_upper_quantile'].values[0]],
                                        marks={str(marker): str(marker) +
                                               '%' for marker in range(0, 100, 10)},
                                        allowCross=False
                                        ),
                        dcc.RadioItems(
                                [0, 1,
                                 2, 3],
                            blank_intercept['ar36_outlier_model'].values[0],id='regression_outlier_A36', labelStyle={'display': 'inline-block', 'marginTop': '5px'}, style=dict(display='flex', justifyContent='center')),
                        dcc.Graph(
                            id='figure_A36',
                        )
                    ], style={'width': '33%', 'display': 'inline-block', 'justifyContent': 'center'}),

                    html.Div([
                        html.Div([
                            dcc.RadioItems(
                                ['Linear', 'Quadratic',
                                 'ElasticNet', 'Bayesian', 'Automatic'],
                                blank_intercept['ar37_regression_model'].values[0],
                                id='regression_model_A37',
                                labelStyle={'display': 'inline-block', 'marginTop': '5px'})
                        ], style=dict(display='flex', justifyContent='center')),
                        dcc.RangeSlider(0, 100,
                                        step=2,
                                        id='quantile_slider_A37',
                                        value=[blank_intercept['ar37_lower_quantile'].values[0],
                                               blank_intercept['ar37_upper_quantile'].values[0]],
                                        marks={str(marker): str(marker) +
                                               '%' for marker in range(0, 100, 10)},
                                        allowCross=False
                                        ),
                        dcc.RadioItems(
                            [0, 1,
                             2, 3],
                            blank_intercept['ar37_outlier_model'].values[0], id='regression_outlier_A37', labelStyle={'display': 'inline-block', 'marginTop': '5px'}, style=dict(display='flex', justifyContent='center')),
                        dcc.Graph(
                            id='figure_A37',
                        ),
                    ], style={'width': '33%', 'display': 'inline-block'}),

                    html.Div([
                        html.Div([
                            dcc.RadioItems(
                                ['Linear', 'Quadratic',
                                 'ElasticNet', 'Bayesian', 'Automatic'],
                                blank_intercept['ar38_regression_model'].values[0],
                                id='regression_model_A38',
                                labelStyle={'display': 'inline-block', 'marginTop': '5px'})
                        ], style=dict(display='flex', justifyContent='center')),
                        dcc.RangeSlider(0, 100,
                                        step=2,
                                        id='quantile_slider_A38',
                                        value=[blank_intercept['ar38_lower_quantile'].values[0], blank_intercept['ar38_upper_quantile'].values[0]],
                                        marks={str(marker): str(marker) +
                                               '%' for marker in range(0, 100, 10)},
                                        allowCross=False
                                        ),
                        dcc.RadioItems(
                            [0, 1,
                             2, 3],
                            blank_intercept['ar38_outlier_model'].values[0], id='regression_outlier_A38', labelStyle={'display': 'inline-block', 'marginTop': '5px'}, style=dict(display='flex', justifyContent='center')),
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
                                blank_intercept['ar39_regression_model'].values[0],
                                id='regression_model_A39',
                                labelStyle={'display': 'inline-block', 'marginTop': '5px'})
                        ], style=dict(display='flex', justifyContent='center')),
                        dcc.RangeSlider(0, 100,
                                        step=2,
                                        id='quantile_slider_A39',
                                        value=[blank_intercept['ar39_lower_quantile'].values[0], blank_intercept['ar39_upper_quantile'].values[0]],
                                        marks={str(marker): str(marker) +
                                               '%' for marker in range(0, 100, 10)},
                                        allowCross=False
                                        ),
                        dcc.RadioItems(
                            [0, 1,
                             2, 3],
                            blank_intercept['ar39_outlier_model'].values[0], id='regression_outlier_A39', labelStyle={'display': 'inline-block', 'marginTop': '5px'}, style=dict(display='flex', justifyContent='center')),
                        dcc.Graph(
                            id='figure_A39',
                        )
                    ], style={'width': '33%', 'display': 'inline-block'}),

                    html.Div([
                        html.Div([
                                  dcc.RadioItems(
                            ['Linear', 'Quadratic',
                             'ElasticNet', 'Bayesian', 'Automatic'],
                            blank_intercept['ar40_regression_model'].values[0],
                            id='regression_model_A40',
                            labelStyle={'display': 'inline-block', 'marginTop': '5px'})
                        ], style=dict(display='flex', justifyContent='center')),

                        dcc.RangeSlider(0, 100,
                                        step=2,
                                        id='quantile_slider_A40',
                                        value=[blank_intercept['ar40_lower_quantile'].values[0], blank_intercept['ar40_upper_quantile'].values[0]],
                                        marks={str(marker): str(marker) +
                                               '%' for marker in range(0, 100, 10)},
                                        allowCross=False
                                        ),
                        dcc.RadioItems(
                            [0, 1,
                             2, 3],
                            blank_intercept['ar40_outlier_model'].values[0], id='regression_outlier_A40', labelStyle={'display': 'inline-block', 'marginTop': '5px'}, style=dict(display='flex', justifyContent='center')),
                        dcc.Graph(
                            id='figure_A40',)
                    ], style={'width': '33%', 'display': 'inline-block'})
                ], style={'display': 'block'}),
                dcc.Store(id='intermediate-value')
            ], style={'width': '100%', 'display': 'inline-block'})
        return layout

    @blank_app.callback(Output('blankassign_table', 'children'),
                            Input("all_blank_table", "selected_rows"))
    def blank_exp_assigment(row):
        blank_exp = []
        blank_type = []
        global blank_data
        if row is not None:
            for i in row:
                blank_exp.append(all_blanks.iloc[i][0])
                blank_type.append(all_blanks.iloc[i][1])

            blank_data_db = pd.read_sql(
                f"SELECT serial_nr,serial_acq_datetime,blank_experiment_nr as blank_experiment_no,blank_experiment_type  FROM blank_assignments where exp_nr = {exp_nr}", software_conn)

            if len(row) == 1 and blank_data_db.empty:
                blank_data['blank_experiment_no'] = blank_exp[0]
                blank_data['blank_experiment_type'] = blank_type[0]
            else:
                blank_data['blank_experiment_no'] = 0
                blank_data['blank_experiment_type'] = '-'

            blank_data = pd.merge(
                blank_data, subset_intensities[['serial_nr', 'acq_datetime']], on='serial_nr')
            blank_data = blank_data.iloc[:, 0:4]
            blank_data.columns.values[3] = "serial_acq_datetime"
            if not blank_data_db.empty:
                blank_data = blank_data_db.copy()
                blank_exp = list(blank_exp) + \
                    list(blank_data['blank_experiment_no'].unique())

            #blank_data['acq_datetime'] = blank_time
            blank_list = tuple(blank_exp)+(0,)
            global blank_intercept
            blank_intercept = pd.read_sql(
                f"SELECT *  FROM experiment_intercepts where exp_nr in {blank_list}", software_conn)
            blank_data1 = extract_blank_data(conn, blank_list)
            global intensities
            intensities = pd.merge(manual_intensities, blank_data,
                                   how='left', on='serial_nr')
            global blanks
            blanks = blank_data1[['serial_nr', 'exp_nr', 'start', 'v36', 'v37', 'v38',
                                  'v39', 'v40', 'time_zero']]
            blank_info = blank_data1[["exp_nr", "serial_nr", "device", "acq_datetime", "inlet_file", 'd_inlet_file', 'method_name', 'd_method_name',
                                      ]].drop_duplicates()
            blank_info = blank_info.rename(
                columns={"d_inlet_file": "inlet_time", "d_method_name": "method_time"})
            blank_layout = html.Div([html.Div([html.Div([
                html.Label("Blank experiments Information table",
                           style=dict(fontWeight='bold', display='inline-block')),
                dash_table.DataTable(
                    data=blank_info.to_dict("records"),
                    columns=[{"id": x, "name": x}
                             for x in blank_info.columns],
                    style_cell=dict(textAlign='left'),
                    style_header=dict(fontWeight='bold'),
                )], style={
                    'width': '75%', 'display': 'inline-block', 'justifyContent': 'center', "padding-left": '15px', "padding-top": '10px'})],
                style={
                'display': 'flex', 'justifyContent': 'center', "padding-left": '5px', "padding-right": '15px', "padding-top": '10px', "padding-bottom": '10px'}
            ),
                html.Div([html.Div([html.Label("Blank Intercepts table", style=dict(fontWeight='bold', display='inline-block')),
                          dash_table.DataTable(
                    data=blank_intercept.to_dict("records"),
                    columns=[{"id": x, "name": x}
                             for x in blank_intercept.columns],
                    style_cell=dict(textAlign='left'),
                    style_header=dict(fontWeight='bold'),

                )
                ], style={
                     'display': 'inline-block', 'justifyContent': 'center', "padding-top": '10px'}
                )], style={
                    'display': 'flex', 'justifyContent': 'center', "padding-left": '5px', "padding-bottom": '20px'}
            ),
                html.Div(
                html.Div([html.Label("Blank Assignment table", style=dict(fontWeight='bold')),
                         html.Div(dash_table.DataTable(
                             id='editable_blank_table',
                             data=blank_data.to_dict("records"),
                             columns=[{"id": x, "name": x, "presentation": 'dropdown'}
                                      for x in blank_data.columns],
                             editable=True,
                             css=[{"selector": ".Select-menu-outer",
                                   "rule": 'display : block !important'}],
                             dropdown={
                                 'blank_experiment_no': {
                                     'options': [
                                         {'label': i, 'value': i}
                                         for i in blank_exp
                                     ]
                                 }},
                             page_current=0,
                             page_size=5,
                             page_action='native',
                             style_cell=dict(textAlign='left'),
                             style_header=dict(fontWeight='bold'),
                         ))

                ], style={'display ': 'block !important'}), style={
                    'display': 'flex', 'justifyContent': 'center'}),
                html.Div([html.Div([html.Button('Save Blank Assigments', id='save-intercept', n_clicks=0)], style={'display': 'flex', 'justifyContent': 'center',
                                                                                                                   'vertical-align': 'top',
                                                                                                                   'width': '5%', "padding-right": '10px'})], style={'display': 'flex', 'justifyContent': 'center'}),
                dcc.ConfirmDialog(id='confirmdanger1',
                                  message='Do you want to save the blank assignments?',
                                  ), html.Div(id='container-button-basic1')])
            return blank_layout

    @blank_app.callback(
        Output('figure_A36', 'figure'),
        Output('intercept_A36', 'children'),
        Output('loss_A36', 'children'),
        Output('intercept_offset_A36', 'children'),
        Input('blank_experiment_selection', 'value'),
        Input('quantile_slider_A36', 'value'),
        Input('regression_model_A36', 'value'),
        Input('regression_outlier_A36', 'value'))
    def updated_Ar36(exp_no, quantile_range, regression_model,outlier_option):
        Ar_blank = blanks[blanks['exp_nr'] == exp_no]
        global sv_serial_nr
        sv_serial_nr = Ar_blank['serial_nr'].drop_duplicates()
        x = Ar_blank['start']
        v = Ar_blank['v36']
        x_offset = Ar_blank['time_zero']
        rm = Regression(x.values, v.values,x_offset.values, quantile_range,
                        outlier_option, regression_model)
        rm.plot()
        figure = go.Figure(data=rm.plotdata)
        figure.update_layout(title="Ar36", showlegend=False)
        figure.update_xaxes(title_text='Time (sec)',
                            linewidth=1)
        figure.update_yaxes(title_text='Intensity (V)',
                            linewidth=1)
        intercept = rm.opt_model_parameters[-1]
        standard_error = rm.model_standard_errors[-1]
        offset_intercept = np.squeeze(rm.y_offset_pred[0])
        return figure, intercept, standard_error, offset_intercept

    @blank_app.callback(
        Output('figure_A37', 'figure'),
        Output('intercept_A37', 'children'),
        Output('loss_A37', 'children'),
        Output('intercept_offset_A37', 'children'),
        Input('blank_experiment_selection', 'value'),
        Input('quantile_slider_A37', 'value'),
        Input('regression_model_A37', 'value'),
        Input('regression_outlier_A37', 'value'))
    def updated_A37(exp_no, quantile_range, regression_model,outlier_option):
        Ar_blank = blanks[blanks['exp_nr'] == exp_no]
        x = Ar_blank['start']
        v = Ar_blank['v37']
        x_offset = Ar_blank['time_zero']
        rm37 = Regression(x.values, v.values,x_offset.values, quantile_range,
                        outlier_option, regression_model)
        rm37.plot()
        figure = go.Figure(data=rm37.plotdata)
        figure.update_layout(title="Ar37", showlegend=False)
        figure.update_xaxes(title_text='Time (sec)',
                            linewidth=1)
        figure.update_yaxes(title_text='Intensity (V)',
                            linewidth=1)
        intercept = rm37.opt_model_parameters[-1]
        standard_error = rm37.model_standard_errors[-1]
        offset_intercept=np.squeeze(rm37.y_offset_pred[0])
        return figure, intercept, standard_error, offset_intercept
    
    @blank_app.callback(
        Output('figure_A38', 'figure'),
        Output('intercept_A38', 'children'),
        Output('loss_A38', 'children'),
        Output('intercept_offset_A38', 'children'),
        Input('blank_experiment_selection', 'value'),
        Input('quantile_slider_A38', 'value'),
        Input('regression_model_A38', 'value'),
        Input('regression_outlier_A38', 'value'))
    def updated_A38(exp_no, quantile_range, regression_model, outlier_option):
        Ar_blank = blanks[blanks['exp_nr'] == exp_no]
        x = Ar_blank['start']
        v = Ar_blank['v38']
        x_offset = Ar_blank['time_zero']
        rm38 = Regression(x.values, v.values,x_offset.values, quantile_range,
                        outlier_option, regression_model)
        rm38.plot()
        figure = go.Figure(data=rm38.plotdata)
        figure.update_layout(title="Ar38", showlegend=False)
        figure.update_xaxes(title_text='Time (sec)',
                            linewidth=1)
        figure.update_yaxes(title_text='Intensity (V)',
                            linewidth=1)
        intercept = rm38.opt_model_parameters[-1]
        standard_error = rm38.model_standard_errors[-1]
        offset_intercept = np.squeeze(rm38.y_offset_pred[0])
        return figure, intercept, standard_error, offset_intercept

    @blank_app.callback(
        Output('figure_A39', 'figure'),
        Output('intercept_A39', 'children'),
        Output('loss_A39', 'children'),
        Output('intercept_offset_A39', 'children'),
        Input('blank_experiment_selection', 'value'),
        Input('quantile_slider_A39', 'value'),
        Input('regression_model_A39', 'value'),
        Input('regression_outlier_A39', 'value'))
    def updated_A39(exp_no, quantile_range, regression_model, outlier_option):
        Ar_blank = blanks[blanks['exp_nr'] == exp_no]
        x = Ar_blank['start']
        v = Ar_blank['v39']
        x_offset = Ar_blank['time_zero']
        rm39 = Regression(x.values, v.values,x_offset.values, quantile_range,
                        outlier_option, regression_model)
        rm39.plot()
        figure = go.Figure(data=rm39.plotdata)
        figure.update_layout(title="Ar39", showlegend=False)
        figure.update_xaxes(title_text='Time (sec)',
                            linewidth=1)
        figure.update_yaxes(title_text='Intensity (V)',
                            linewidth=1)
        intercept = rm39.opt_model_parameters[-1]
        standard_error = rm39.model_standard_errors[-1]
        offset_intercept = np.squeeze(rm39.y_offset_pred[0])
        return figure, intercept, standard_error, offset_intercept

    @blank_app.callback(
        Output('figure_A40', 'figure'),
        Output('intercept_A40', 'children'),
        Output('loss_A40', 'children'),
        Output('intercept_offset_A40', 'children'),
        Input('blank_experiment_selection', 'value'),
        Input('quantile_slider_A40', 'value'),
        Input('regression_model_A40', 'value'),
        Input('regression_outlier_A40', 'value'))
    def updated_A40(exp_no, quantile_range, regression_model, outlier_option):
        Ar_blank = blanks[blanks['exp_nr'] == exp_no]
        x = Ar_blank['start']
        v = Ar_blank['v40']
        x_offset = Ar_blank['time_zero']
        rm40 = Regression(x.values, v.values,x_offset.values, quantile_range,
                        outlier_option, regression_model)
        rm40.plot()
        figure = go.Figure(data=rm40.plotdata)
        figure.update_layout(title="Ar40", showlegend=False)
        figure.update_xaxes(title_text='Time (sec)',
                            linewidth=1)
        figure.update_yaxes(title_text='Intensity (V)',
                            linewidth=1)
        intercept = rm40.opt_model_parameters[-1]
        standard_error = rm40.model_standard_errors[-1]
        offset_intercept = np.squeeze(rm40.y_offset_pred[0])
        return figure, intercept, standard_error, offset_intercept

    @blank_app.callback(Output('confirmdanger1', 'displayed'),
                        Input('save-intercept', 'n_clicks'))
    def display_confirm_assigments(n):
        if n is not None and n >= 1:
            return True
        return False

    def update_blank_assigments(table):
        cursor = software_conn.cursor()
# Insert Dataframe into SQL Server:
        for index, row in table.iterrows():
            cursor.execute(
                f"INSERT INTO blank_assignments (exp_nr, serial_nr,serial_acq_datetime,blank_experiment_nr,blank_experiment_type) VALUES('{row.exp_nr}', '{row.serial_nr}', '{row.serial_acq_datetime}', '{row.blank_experiment_nr}', '{row.blank_experiment_type}') ON DUPLICATE KEY UPDATE blank_experiment_nr = '{row.blank_experiment_nr}', blank_experiment_type ='{row.blank_experiment_type}'")
        software_conn.commit()
        pass

    @blank_app.callback(
        Output('container-button-basic1', 'children'),
        Output('confirmdanger1', 'submit_n_clicks'),
        Input('confirmdanger1', 'submit_n_clicks'),
        Input('editable_blank_table', 'data'),
        Input('editable_blank_table', 'columns')
    )
    def save_blank_assigments(n, rows, columns):
        assigned_blanks = pd.DataFrame(
            rows, columns=[c['name'] for c in columns])
        final_blanks = pd.merge(assigned_blanks[['serial_nr', 'blank_experiment_no', 'serial_acq_datetime']], all_blanks[[
                                'exp_nr', 'exp_type']], left_on='blank_experiment_no', right_on='exp_nr')
        final_blanks['exp_nr'] = exp_nr
        if n is not None and n >= 1:
            final_blanks = final_blanks[[
                'exp_nr', 'serial_nr', 'serial_acq_datetime', 'blank_experiment_no', 'exp_type']]
            final_blanks.columns.values[3] = "blank_experiment_nr"
            final_blanks.columns.values[4] = "blank_experiment_type"
            update_blank_assigments(final_blanks)
        return 0, 0

    @blank_app.callback(Output('confirm-danger', 'displayed'),
                        Input('save-val', 'n_clicks'))
    def display_confirm(n):
        if n >= 1:
            return True
        return False

    def update_blank_intercepts(exp, intercept_a36, error_a36, rm_36, ro_36, quantile_lower_36, quantile_upper_36,
                                intercept_a37, error_a37, rm_37, ro_37, quantile_lower_37, quantile_upper_37,
                                intercept_a38, error_a38, rm_38, ro_38, quantile_lower_38, quantile_upper_38,
                                intercept_a39, error_a39, rm_39, ro_39, quantile_lower_39, quantile_upper_39,
                                intercept_a40, error_a40, rm_40, ro_40, quantile_lower_40, quantile_upper_40):
    
        softurl = r"mariadb+mariadbconnector://root:dbpwx61@139.20.22.156:3306/ararsoftware"
        soft_engine = create_engine(softurl)
        serial_nr = int(sv_serial_nr)
        soft_engine.execute(f"INSERT INTO experiments_analysis (exp_nr,serial_nr, ar36_intercept, ar36_intercept_error, ar36_regression_model, ar36_outlier_model,ar36_upper_quantile, ar36_lower_quantile,ar37_intercept, ar37_intercept_error, ar37_regression_model, ar37_outlier_model,ar37_upper_quantile, ar37_lower_quantile ,ar38_intercept, ar38_intercept_error, ar38_regression_model, ar38_outlier_model,ar38_upper_quantile, ar38_lower_quantile,ar39_intercept, ar39_intercept_error, ar39_regression_model, ar39_outlier_model,ar39_upper_quantile, ar39_lower_quantile,ar40_intercept, ar40_intercept_error, ar40_regression_model, ar40_outlier_model,ar40_upper_quantile, ar40_lower_quantile) VALUES('{exp}',{serial_nr},'{intercept_a36}', '{error_a36}', '{rm_36}', '{ro_36}', '{quantile_upper_36}', '{quantile_lower_36}','{intercept_a37}', '{error_a37}', '{rm_37}', '{ro_37}','{quantile_upper_37}', '{quantile_lower_37}','{intercept_a38}', '{error_a38}', '{rm_38}', '{ro_38}','{quantile_upper_38}', '{quantile_lower_38}','{intercept_a39}', '{error_a39}', '{rm_39}', '{ro_39}','{quantile_upper_39}', '{quantile_lower_39}', '{intercept_a40}', '{error_a40}', '{rm_40}', '{ro_40}','{quantile_upper_40}', '{quantile_lower_40}') ON DUPLICATE KEY UPDATE ar36_intercept='{intercept_a36}', ar36_intercept_error='{error_a36}', ar36_regression_model='{rm_36}',ar36_outlier_model= '{ro_36}', ar36_lower_quantile='{quantile_lower_36}', ar36_upper_quantile='{quantile_upper_36}',ar37_intercept='{intercept_a37}', ar37_intercept_error='{error_a37}', ar37_regression_model='{rm_37}',ar37_outlier_model= '{ro_37}', ar37_lower_quantile='{quantile_lower_37}', ar37_upper_quantile='{quantile_upper_37}',ar38_intercept='{intercept_a38}', ar38_intercept_error='{error_a38}', ar38_regression_model='{rm_38}',ar38_outlier_model= '{ro_38}', ar38_lower_quantile='{quantile_lower_38}', ar38_upper_quantile='{quantile_upper_38}',ar39_intercept='{intercept_a39}', ar39_intercept_error='{error_a39}', ar39_regression_model='{rm_39}',ar39_outlier_model= '{ro_39}', ar39_lower_quantile='{quantile_lower_39}', ar39_upper_quantile='{quantile_upper_39}',ar40_intercept='{intercept_a40}', ar40_intercept_error='{error_a40}', ar40_regression_model='{rm_40}',ar40_outlier_model= '{ro_40}', ar40_lower_quantile='{quantile_lower_40}', ar40_upper_quantile='{quantile_upper_40}'"
            )
        
        pass
    
    def update_time_intercepts(exp,offset_36,error_a36,offset_37,error_a37,offset_38,error_a38,offset_39,error_a39,offset_40,error_a40):
        softurl = r"mariadb+mariadbconnector://root:dbpwx61@139.20.22.156:3306/ararsoftware"
        soft_engine = create_engine(softurl)
        serial_nr = int(sv_serial_nr)
        soft_engine.execute(f"INSERT INTO experiment_intercepts (exp_nr,serial_nr, ar36_intercept, ar36_standard_error,ar37_intercept, ar37_standard_error,ar38_intercept, ar38_standard_error,ar39_intercept, ar39_standard_error,ar40_intercept, ar40_standard_error) VALUES ('{exp}','{serial_nr}','{offset_36}','{error_a36}','{offset_37}','{error_a37}','{offset_38}','{error_a38}','{offset_39}','{error_a39}','{offset_40}','{error_a40}') ON DUPLICATE KEY UPDATE ar36_intercept='{offset_36}',ar36_standard_error='{error_a36}',ar37_intercept='{offset_37}',ar37_standard_error='{error_a37}',ar38_intercept='{offset_38}',ar38_standard_error='{error_a38}',ar39_intercept='{offset_39}',ar39_standard_error='{error_a39}',ar40_intercept='{offset_40}',ar40_standard_error='{error_a40}'"
                            )
        pass

    @blank_app.callback(
        Output('container-button-basic', 'children'),
        Output('confirm-danger', 'submit_n_clicks'),
        Input('confirm-danger', 'submit_n_clicks'),
        Input('blank_experiment_selection', 'value'),
   
        Input('intercept_A36', 'children'),
        Input('loss_A36', 'children'),
        Input('regression_model_A36', 'value'),
        Input('regression_outlier_A36', 'value'),
        Input('quantile_slider_A36', 'value'),

        Input('intercept_A37', 'children'),
        Input('loss_A37', 'children'),
        Input('regression_model_A37', 'value'),
        Input('regression_outlier_A37', 'value'),
        Input('quantile_slider_A37', 'value'),

        Input('intercept_A38', 'children'),
        Input('loss_A38', 'children'),
        Input('regression_model_A38', 'value'),
        Input('regression_outlier_A38', 'value'),
        Input('quantile_slider_A38', 'value'),

        Input('intercept_A39', 'children'),
        Input('loss_A39', 'children'),
        Input('regression_model_A39', 'value'),
        Input('regression_outlier_A39', 'value'),
        Input('quantile_slider_A39', 'value'),

        Input('intercept_A40', 'children'),
        Input('loss_A40', 'children'),
        Input('regression_model_A40', 'value'),
        Input('regression_outlier_A40', 'value'),
        Input('quantile_slider_A40', 'value'),

        Input('intercept_offset_A36', 'children'),
        Input('intercept_offset_A37', 'children'),
        Input('intercept_offset_A38', 'children'),
        Input('intercept_offset_A39', 'children'),
        Input('intercept_offset_A40', 'children'),
    )
    def save_intercepts(n, exp, intercept_a36,error_a36,rm_36,ro_36,quantile_36,
                        intercept_a37,error_a37,rm_37,ro_37,quantile_37,
                        intercept_a38, error_a38, rm_38, ro_38, quantile_38,
                        intercept_a39,error_a39,rm_39,ro_39,quantile_39,
                        intercept_a40,error_a40,rm_40,ro_40,quantile_40,offset_36,offset_37,offset_38,offset_39,offset_40):

        if n is not None and n >= 1:
            update_blank_intercepts(exp, np.squeeze(intercept_a36), 
                np.squeeze(error_a36), np.squeeze(rm_36), np.squeeze(ro_36), np.squeeze(quantile_36[0]), np.squeeze(quantile_36[1]),
                np.squeeze(intercept_a37), np.squeeze(
                error_a37), np.squeeze(rm_37), np.squeeze(ro_37), np.squeeze(quantile_37[0]), np.squeeze(quantile_37[1]),
                np.squeeze(intercept_a38), np.squeeze(
                error_a38), np.squeeze(rm_38), np.squeeze(ro_38), np.squeeze(quantile_38[0]), np.squeeze(quantile_38[1]),
                np.squeeze(intercept_a39), np.squeeze(
                error_a39), np.squeeze(rm_39), np.squeeze(ro_39), np.squeeze(quantile_39[0]), np.squeeze(quantile_39[1]),
                np.squeeze(intercept_a40), np.squeeze(
                error_a40), np.squeeze(rm_40), np.squeeze(ro_40), np.squeeze(quantile_40[0]), np.squeeze(quantile_40[1]))
            update_time_intercepts(exp, np.squeeze(offset_36), np.squeeze(error_a36), np.squeeze(offset_37), np.squeeze(error_a37),
                                   np.squeeze(offset_38), np.squeeze(error_a38), np.squeeze(offset_39), np.squeeze(error_a39), np.squeeze(offset_40), np.squeeze(error_a40))
        return 0, 0
    return blank_app
