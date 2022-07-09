from flask import Flask

def create_app():
    app=Flask(__name__)
    app.config["SECRET_KEY"]="ArArlab"
    
    from .website_flow import website_flow
    app.register_blueprint(website_flow, url_prefix='/')
    
    return app