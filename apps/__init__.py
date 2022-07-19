from flask import Flask
from apps.dashapp1.dashboard import dashboard
from apps.dashapp2.blank_dashboard import blank_dashboard
from apps.dashapp3.experiment_dashboard import experiment_dashboard
def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "ArArlab"
    dashboard(app)
    blank_dashboard(app)
    experiment_dashboard(app)
    from .website_flow import website
    app.register_blueprint(website, url_prefix='/')
    return app
