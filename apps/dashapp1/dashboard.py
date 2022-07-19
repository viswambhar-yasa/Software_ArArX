import mariadb
import pandas as pd
from dash import html, Dash, Input, Output, dash_table, no_update, dcc
from apps.dash_navbar.navigation_bar import Navbar
import dash_bootstrap_components as dbc
import json
import numpy as np

def dashboard(server):
    external_stylesheets = [
        'https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP]
    dash_app = Dash(__name__, server=server, url_base_pathname='/dashboard/',
                    external_stylesheets=external_stylesheets)
    with open(r'.\apps\private_data\account_details.json', 'r') as f:
        try:
            account_details=json.load(f)
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
    global software_conn
    software_conn=mariadb.connect(
    user=username,
    password=password,
    host="139.20.22.156",
    port=3306,
    database="ararsoftware")
    dashboard_arardb = pd.read_sql(
        "SELECT exp_nr,material,proben_bez,project_name,sample_owner,irr_batch  FROM material order by exp_nr desc", conn)
    dashboard_arardb["id"] = dashboard_arardb.index
    header = Navbar()
    dash_app.layout = html.Div([header,
                            html.Div([  
                                dash_table.DataTable(
                                    id="table",
                                    row_selectable="single",
                                    columns=[{"name": i, "id": i}
                                             for i in dashboard_arardb.columns if i != "id"],
                                    data=dashboard_arardb.to_dict("records"),
                                    sort_action='native',
                                    filter_action='native',
                                    page_current=0,
                                    page_size=10,
                                    page_action='native',
                                    style_cell=dict(textAlign='left'),
                                    style_header=dict(fontWeight='bold'),
                                ), dcc.Store(id="derived_viewport_selected_row_ids")], style={'width': '100%', 'display': 'flex', 'justifyContent': 'center', "padding-top": '70px'})])
    dashboard_callbacks(dash_app)
    return dash_app


def update(selRows):
    if len(selRows) > 0:
        exp_index=np.squeeze(np.array(selRows))
        print(exp_index)
        cur = software_conn.cursor()
        key=1
        cur.execute(
            f"UPDATE experiments SET exp_index = '{exp_index}' WHERE exp_key = {key}")
        software_conn.commit()
    pass

def dashboard_callbacks(dash_app):
    @dash_app.callback(
        Output("table", "style_data_conditional"),
        Input("table", "derived_viewport_selected_row_ids"))
    def style_selected_rows(selRows):
        if selRows is None:
            return no_update
        else:
            index=selRows
            update(selRows)
            with open(r'.\apps\private_data\exp_nr.txt', 'w') as f:
                f.write(str(index))
            f.close()
            return [ 
            {"if": {"filter_query": "{{id}} ={}".format(
                i)}, "backgroundColor": "grey", }
            for i in selRows]
    # @dash_app.callback(
    #     Output("table", "style_data_conditional"),
    #     Input("table", "derived_viewport_selected_row_ids"))
    # def experiment_details(selRows):
    #     if selRows is None:
    #         return no_update
    #     else:
    #         
    #         exp_data=extract_experiment_info(exp_nr, database='arardb')

# username = 'root'
# password = 'dbpwx61'
# database = 'arardb'
# global db_server
# db_server = mariadb.connect(
#     user=username,
#     password=password,
#     host="139.20.22.156",
#     port=3306,
#     database=database)

# x=10
# dashboard().run_server(debug=True)
