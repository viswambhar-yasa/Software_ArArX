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
            return redirect('/dashboard/')
        except Exception as e:
            dcc.Store(id='db-url', data=None)
            flash(
                f"Error connecting to MariaDB Platform: {e}", category='error')
    return render_template("login.html")


@website.route('/dashboard/')
def render_dashboard():
    return redirect(url_for('/dashboard/'))


@website.route('/blanks')
def render_blank():
    return redirect(url_for('/blanks'))

@website.route('/experiment')
def render_experiment():
    return redirect(url_for('/experiment'))


