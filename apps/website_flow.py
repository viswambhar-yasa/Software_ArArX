from flask import Blueprint, render_template, request, flash, redirect, url_for,session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker,scoped_session
from dash import dcc
website = Blueprint("website", __name__)

@website.route('/', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        username = request.form.get('Username')
        database = request.form.get('Database')
        password = request.form.get('Password')
        session['username'] = username
        session['password'] = password
        session['Database'] = database
        try:
            global url
            url = f"mariadb+mariadbconnector://{username}:{password}@139.20.22.156:3306/{database}"
            engine = create_engine(url)
            session['SQLALCHEMY_DATABASE_URI']=url
            #scoped_session(Session())
            dcc.Store(id='db-url', data=url)
            return redirect('/dashboard')
        except Exception as e:
            dcc.Store(id='db-url', data=None)
            flash(
                f"Error connecting to MariaDB Platform: {e}", category='error')
    return render_template("login.html")


@website.route('/dashboard')
def render_dashboard():
    if request.method == 'POST':
        print()
    return redirect(url_for('/dashboard'))


# @website.route('/blanks')
# def render_blank():
#     return redirect(url_for('/blanks'))

# @website.route('/experiment')
# def render_experiment():
#     return redirect(url_for('/experiment'))

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


