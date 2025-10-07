
import pandas as pd
from dash.dependencies import Input, Output, State
from dash.dash_table.Format import Format, Scheme
from dash.exceptions import PreventUpdate

from gui.dash.server import app
from chem_analysis.utils.load_csv import load_csv


@app.callback([Output('output-data-upload', 'children'), Output('dataframe', 'data')],
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])
def upload_file(list_of_contents, filename):
    if list_of_contents is not None:
        result, df = load_csv(list_of_contents, filename)
        if df is not None:
            print(df.head())
            df = df.to_json(orient='split')
        return result, df
    return None, {}


@app.callback([Output('data_table', 'data'), Output('data_table', 'columns')],
              [Input('dataframe', 'data')])
def set_table(json_data):
    if json_data:
        print("hi")
        df = pd.read_json(json_data, orient='split')
        data = df.iloc[:10, :].to_dict('records')
        columns = []
        for i in df.columns:
            columns.append({
                "name": i,
                "id": i,
                "renamable": True,
                "deletable": True,
                "type": 'numeric',
                "format": Format(precision=2, scheme=Scheme.exponent) #precision=2, scheme=Scheme.fixed  https://dash.plotly.com/datatable/data-formatting
            })

        return data, columns
    raise PreventUpdate


@app.callback(Output("ri_time", "options"),
              Input('dataframe', 'data'))
def update_radio(json_data):
    if json_data:
        df = pd.read_json(json_data, orient='split')
        options = []
        for col in df.columns:
            options.append({"label": col, "value": col})

        return options

    raise PreventUpdate
