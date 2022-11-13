import dash_bootstrap_components as dbc
from dash import html


def Navbar():
    layout = html.Div([
        dbc.NavbarSimple(
            
            children=[
                dbc.NavItem(dbc.NavLink(
                    "Dashbaord", href="/dashboard/", active='exact',external_link=True)),
                dbc.NavItem(dbc.NavLink(
                    "Blanks", href="/blanks/", active='exact', external_link=True)),
                dbc.NavItem(dbc.NavLink(
                    "Experiments", href="/experiment/", active='exact', external_link=True)),
                dbc.NavItem(dbc.NavLink("Age Analysis",
                            href="/age_analysis/", active='exact', external_link=True)),
            ],
            brand="Thermochronology",
            brand_href="/",
            brand_external_link=True,
            color="dark",
            dark=True,
        )
    ])

    return layout

