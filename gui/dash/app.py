from gui.dash.server import app

from dash import html, dcc
from dash.dependencies import Input, Output

# layouts
app.layout = html.Div([
    html.H1('Size Exclusion Analysis'),
    dcc.Tabs(
        children=[
            dcc.Tab(label='Load', value='tab-load'),
            dcc.Tab(label='Process', value='tab-process'),
            dcc.Tab(label='Analysis', value='tab-analysis')
        ],
        value='tab-Load',
        id='tabs'
    ),
    html.Div(id='tabs-content'),

    dcc.Store(id='dataframe'),
    dcc.Store(id='tab_load_reload'),
])

from gui.dash.tabs.tab_load import tab_load_layout
from gui.dash.tabs.tab_process import tab_process_layout
from gui.dash.tabs.tab_analysis import tab_analysis_layout


# callbacks
@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-load':
        return tab_load_layout
    elif tab == 'tab-process':
        return tab_process_layout
    elif tab == 'tab-analysis':
        return tab_analysis_layout

    return tab_load_layout

