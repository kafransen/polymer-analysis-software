import base64
import io
import pandas as pd

from dash import html


def load_csv(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return html.Div(['Invalid file type. Only csv.']), None
    except Exception as e:
        print(e)
        return html.Div(['There was an error processing this file.']), None

    df = reduce_data(df)
    return html.Div(["Data successfully loaded."]), df


def reduce_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[::10, :]
