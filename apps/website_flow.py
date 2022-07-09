from flask import Blueprint, render_template, request, flash, redirect, url_for,session
import mariadb
import pandas as pd
from dash import html, Dash, Input, Output, dash_table, no_update, dcc
import json
website = Blueprint("website", __name__)


@website.route('/', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        username = request.form.get('Username')
        database = request.form.get('Database')
        password = request.form.get('Password')
        account_details={"username": username,
              "database": database,
              "password": password}
        session['username'] = username
        session['password'] = password
        session['Database'] = database
        try:
            db_server = mariadb.connect(
                user=username,
                password=password,
                host="139.20.22.156",
                port=3306,
                database=database)
            with open(r'.\apps\private_data\account_details.json', 'w') as f:
                json.dump(account_details, f)
            f.close()
            print("redirect")
            return redirect('/dashboard')
        except mariadb.Error as e:
            flash(
                f"Error connecting to MariaDB Platform: {e}", category='error')
    return render_template("login.html")


@website.route('/dashboard')
def render_dashboard():
    if request.method == 'POST':
        print()
    return redirect(url_for('/dashboard'))


@website.route('/blanks')
def render_blank():
    return redirect(url_for('/blanks'))

# @website_flow.route('/dashboard', methods=['POST', 'GET'])
# def dashboard():
#     dashboard_arardb = pd.read_sql(
#         "SELECT exp_nr,material,proben_bez,project_name,sample_owner,irr_batch  FROM material order by exp_nr ", conn)
#     dashboard_arardb_column_names = [
#         'expr no', 'material', 'probe', 'project name', 'sample owner', 'irr batch']
#     if request.method == 'POST':
#         experiemnt_name = request.form.get('experiment_name')
#         dashboard_arardb = pd.read_sql(
#             "SELECT exp_nr,material,proben_bez,project_name,sample_owner,irr_batch  FROM material order by exp_nr ", conn)

#     return render_template("dashboard_copy.html", title=database, column_names=dashboard_arardb_column_names, row_data=list(dashboard_arardb.values.tolist()), zip=zip)


# @website.route('/dashboard', methods=['POST', 'GET'])
# def my_dash_app():
#     from dashboard import dashboard
#     return dashboard(website)


