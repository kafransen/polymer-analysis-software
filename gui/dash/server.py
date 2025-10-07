from flask import Flask
from dash import Dash

server = Flask('chem_analysis')
app = Dash(__name__, server=server)

app.config.suppress_callback_exceptions = True
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True
