
from dash import html, dcc

tab_analysis_layout = html.Div([
    html.H1('Page 3'),
    dcc.RadioItems(
        id='page-3-radios',
        options=[{'label': i, 'value': i} for i in ['Orange', 'Blue', 'Red']],
        value='Orange'
    ),
    html.Div(id='page-3-content')
])
