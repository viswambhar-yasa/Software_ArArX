from dash import Dash, html, dcc, Input, Output, State
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from apps.backend.auxiliary_functions import *
from apps.dash_navbar.navigation_bar import Navbar
import pandas as pd
import json


# blank_intercept=[ ]
def blank_dashboard(server):
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
    conn = mariadb.connect(
        user=username,
        password=password,
        host="139.20.22.156",
        port=3306,
        database=database)
    
    external_stylesheets = [
        'https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP]
    blank_app = Dash(__name__, server=server, url_base_pathname='/blanks/',
                     external_stylesheets=external_stylesheets)
    def update_layout():
        software_conn = mariadb.connect(
            user=username,
            password=password,
            host="139.20.22.156",
            port=3306,
            database="ararsoftware")
        exp_index = np.squeeze(np.array(pd.read_sql(
            "SELECT exp_index FROM experiments where exp_key=1", software_conn)))
        print(exp_index)
        dashboard_arardb = pd.read_sql(
            "SELECT exp_nr,material,proben_bez,project_name,sample_owner,irr_batch  FROM material order by exp_nr desc", conn)
        dashboard_arardb["id"] = dashboard_arardb.index
        exp_nr = str(dashboard_arardb.iloc[exp_index][0])
        print(exp_nr)
        experiment_info = extract_experiment_info(exp_nr, database)
        experiment_info = extract_irradiation(conn, experiment_info)
        intensities = extract_intensities(conn, experiment_info)
        sensitivities = extracting_senstivities(conn, experiment_info)
        intensities = pd.merge(intensities, sensitivities,
                               how='left', on='serial_nr')
        blank_data = extracting_blanks(
            conn, intensities[[
                'acq_datetime']].min()[0], intensities[[
                    'acq_datetime']].max()[0])
        subset_intensities = intensities[[
            'serial_nr', 'acq_datetime']].drop_duplicates()
        global blanks
        blank_data = automatic_blank_assigment(blank_data, subset_intensities)
        print(blank_data.columns)
        blank_list = tuple(blank_data['blank_experiment_no'].unique())+(0,)
        blank_data1 = extract_blank_data(conn, blank_list)
        intensities = pd.merge(intensities, blank_data, how='left', on='serial_nr')
        blanks = blank_data1[['serial_nr', 'exp_nr', 'start', 'v36', 'v37', 'v38',
                              'v39', 'v40']]
        header = Navbar()
        layout = html.Div([header,
                                     html.Div([
                                         dcc.Dropdown(
                                             blanks['exp_nr'].unique(),
                                             blanks['exp_nr'].unique()[0],
                                             id='blank_experiment_selection'),
                                     ], style={'width': '33%', 'display': 'block', "padding-bottom": '10px'}),

                                     html.Div([
                                         html.Div([html.Table([
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

                                         ])])], style={'width': '100%', 'display': 'flex', 'justifyContent': 'center', "padding-bottom": '5px'}),
                                     html.Div([
                                         html.Div([
                                             html.Div([
                                                 dcc.RadioItems(
                                                     ['Linear', 'Quadratic',
                                                         'Power', 'Automatic'],
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
                                                         'Power', 'Automatic'],
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
                                                         'Power', 'Automatic'],
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
                                                         'Power', 'Automatic'],
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
                                                         'Power', 'Automatic'],
                                                     'Automatic',
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
                                     dcc.Store(id='intermediate-value'), dcc.Interval(id='interval',
                                                                                      interval=1000, n_intervals=0)
                                     ], style={'width': '100%', 'display': 'inline-block'},)
        return layout
    blank_app.layout = update_layout
    @blank_app.callback(
        Output('figure_A36', 'figure'),
        Output('intercept_A36', 'children'),
        Output('loss_A36', 'children'),
        Input('blank_experiment_selection', 'value'),
        Input('quantile_slider_A36', 'value'),
        Input('regression_model_A36', 'value'))
    def updated_Ar36(exp_no, quantile_range, regression_model):
        Ar_blank = blanks[blanks['exp_nr'] == exp_no]
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
        model = regression_models(x[(v < q_hi) & (v > q_low)],
                                  v[(v < q_hi) & (v > q_low)], regression_model)
        if regression_model == 'Linear':
            line_model = model[0]*x+model[1]
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        mode='lines', name=regression_model+' regression'))
        elif regression_model == 'Quadratic':
            line_model = model[0]*x**2+model[1]*x+model[2]
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        mode='lines', name=regression_model+' regression'))

        figure = go.Figure(data=data)
        figure.update_layout(title="Ar36", showlegend=False)

        figure.update_xaxes(title_text='Time (sec)',
                            linewidth=1)

        figure.update_yaxes(title_text='Intensity (V)',
                            linewidth=1)
        if model is None:
            return figure, 0, 0
        return figure, model[-1], 0

    @blank_app.callback(
        Output('figure_A37', 'figure'),
        Output('intercept_A37', 'children'),
        Output('loss_A37', 'children'),
        Input('blank_experiment_selection', 'value'),
        Input('quantile_slider_A37', 'value'),
        Input('regression_model_A37', 'value'))
    def updated_A37(exp_no, quantile_range, regression_model):
        Ar_blank = blanks[blanks['exp_nr'] == exp_no]
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
        if regression_model == 'Linear':
            line_model = model[0]*x+model[1]
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        mode='lines', name=regression_model+' regression'))
        elif regression_model == 'quadratic':
            line_model = model[0]*x**2+model[1]*x+model[2]
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        mode='lines', name=regression_model+' regression'))

        figure = go.Figure(data=data)
        figure.update_layout(title="Ar37", showlegend=False)
        figure.update_xaxes(title_text='Time (sec)',
                            linewidth=1)

        figure.update_yaxes(title_text='Intensity (V)',
                            linewidth=1)
        if model is None:
            return figure, 0, 0
        return figure, model[-1], 0

    @blank_app.callback(
        Output('figure_A38', 'figure'),
        Output('intercept_A38', 'children'),
        Output('loss_A38', 'children'),
        Input('blank_experiment_selection', 'value'),
        Input('quantile_slider_A38', 'value'),
        Input('regression_model_A38', 'value'))
    def updated_A38(exp_no, quantile_range, regression_model):
        Ar_blank = blanks[blanks['exp_nr'] == exp_no]
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
        model = regression_models(x[(v < q_hi) & (v > q_low)],
                                  v[(v < q_hi) & (v > q_low)], regression_model)
        if regression_model == 'Linear':
            line_model = model[0]*x+model[1]
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        mode='lines', name=regression_model+' regression'))
        elif regression_model == 'quadratic':
            line_model = model[0]*x**2+model[1]*x+model[2]
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        mode='lines', name=regression_model+' regression'))

        figure = go.Figure(data=data)
        figure.update_layout(title="Ar38", showlegend=False)
        figure.update_xaxes(title_text='Time (sec)',
                            linewidth=1)

        figure.update_yaxes(title_text='Intensity (V)',
                            linewidth=1)
        if model is None:
            return figure, 0, 0
        return figure, model[-1], 0

    @blank_app.callback(
        Output('figure_A39', 'figure'),
        Output('intercept_A39', 'children'),
        Output('loss_A39', 'children'),
        Input('blank_experiment_selection', 'value'),
        Input('quantile_slider_A39', 'value'),
        Input('regression_model_A39', 'value'))
    def updated_A38(exp_no, quantile_range, regression_model):
        Ar_blank = blanks[blanks['exp_nr'] == exp_no]
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
        model = regression_models(x[(v < q_hi) & (v > q_low)],
                                  v[(v < q_hi) & (v > q_low)], regression_model)
        if regression_model == 'Linear':
            line_model = model[0]*x+model[1]
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        mode='lines', name=regression_model+' regression'))
        elif regression_model == 'Quadratic':
            line_model = model[0]*x**2+model[1]*x+model[2]
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        mode='lines', name=regression_model+' regression'))

        figure = go.Figure(data=data)
        figure.update_layout(title="Ar39", showlegend=False)
        figure.update_xaxes(title_text='Time (sec)',
                            linewidth=1)

        figure.update_yaxes(title_text='Intensity (V)',
                            linewidth=1)
        if model is None:
            return figure, 0, 0
        return figure, model[-1], 0

    @blank_app.callback(
        Output('figure_A40', 'figure'),
        Output('intercept_A40', 'children'),
        Output('loss_A40', 'children'),
        Input('blank_experiment_selection', 'value'),
        Input('quantile_slider_A40', 'value'),
        Input('regression_model_A40', 'value'))
    def updated_A38(exp_no, quantile_range, regression_model):
        Ar_blank = blanks[blanks['exp_nr'] == exp_no]
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
        model = regression_models(x[(v < q_hi) & (v > q_low)],
                                  v[(v < q_hi) & (v > q_low)], regression_model)
        if regression_model == 'Linear':
            line_model = model[0]*x+model[1]
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        mode='lines', name=regression_model+' regression'))
        elif regression_model == 'Quadratic':
            line_model = model[0]*x**2+model[1]*x+model[2]
            data.append(go.Scatter(x=x.to_list(), y=line_model,
                        mode='lines', name=regression_model+' regression'))

        figure = go.Figure(data=data)
        figure.update_layout(title="Ar40", showlegend=False)
        figure.update_xaxes(title_text='Time (sec)',
                            linewidth=1)

        figure.update_yaxes(title_text='Intensity (V)',
                            linewidth=1)
        if model is None:
            return figure, 0, 0
        return figure, model[-1], 0
    return blank_app

# if __name__ == '__main__':
#     app.run_server(debug=True)
