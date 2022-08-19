import mariadb
import pandas as pd
from dash import html, Dash, Input, Output, dash_table, no_update, dcc
from apps.dash_navbar.navigation_bar import Navbar
import dash_bootstrap_components as dbc
import json
import numpy as np
from flask import session
from sqlalchemy import create_engine
#def dashboard(server):

def dashboard(server):
    external_stylesheets = [
        'https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP]
    dash_app = Dash(__name__, server=server, url_base_pathname='/dashboard/',
                    external_stylesheets=external_stylesheets)
    
    header = Navbar()
    def layout():
            global url
            url =r"mariadb+mariadbconnector://root:dbpwx61@139.20.22.156:3306/arardb"
            engine = create_engine(url)
            with engine.connect() as conn:
                global dashboard_arardb
                dashboard_arardb = pd.read_sql_query(
                    "SELECT exp_nr,material,proben_bez,project_name,sample_owner,irr_batch  FROM material order by exp_nr desc", conn)
            dashboard_arardb["id"] = dashboard_arardb.index
            dash_layout = dash_app.layout = html.Div([header,
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
                                                                                      ), dcc.Store(id="derived_viewport_selected_row_ids")],
                                                      style={'width': '100%', 'display': 'flex', 'justifyContent': 'center', "padding-top": '70px'})])
            dashboard_callbacks(dash_app)
            return dash_layout
    dash_app.layout = layout()


def update(selRows):
    if len(selRows) > 0:
        exp_index=np.squeeze(np.array(selRows))
        software_url = 'mariadb+mariadbconnector://root:dbpwx61@139.20.22.156:3306/ararsoftware'
        exp_nr = str(dashboard_arardb.iloc[exp_index][0])
        dcc.Store(id='exp_nr', data=exp_nr)
        soft_engine = create_engine(software_url)
        with soft_engine.connect() as conn:
            key=1
            conn.execute(
                f"UPDATE experiments SET exp_index = '{exp_index}' WHERE exp_key = {key}")
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
  
