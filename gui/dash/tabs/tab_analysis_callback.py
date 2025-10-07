from dash.dependencies import Input, Output

from gui.dash.server import app


# Tab 2 callback
@app.callback(Output('page-3-content', 'children'),
              [Input('page-3-radios', 'value')])
def page_2_radios(value):
    return 'You have selected "{}"'.format(value)
