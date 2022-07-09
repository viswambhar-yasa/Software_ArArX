import dash_bootstrap_components as dbc
from dash import html


def Navbar():
    layout = html.Div([
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink(
                    "Dashbaord", href="/dashboard/", active='exact')),
                dbc.NavItem(dbc.NavLink(
                    "Blanks", href="/blanks/", active='exact')),
                dbc.NavItem(dbc.NavLink("Experiments", href="/experiment/",active='exact')),
                dbc.NavItem(dbc.NavLink("Age Analysis",
                            href="/experiment/", active='exact')),
            ],
            brand="Thermochronology",
            brand_href="/",
            color="dark",
            dark=True,
        ),
    ])

    return layout

