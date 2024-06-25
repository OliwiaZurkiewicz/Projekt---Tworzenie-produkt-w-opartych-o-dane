from typing import Any

import os
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objs_figure.Figure as Figure
from dash import Dash, Input, Output, callback, dcc, html
from dash.dependencies import Input, Output
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

app = Dash(__name__, assets_folder="../assets")

dir = os.path.dirname(__file__)
path = os.path.join(dir, "raw_sales.csv")


def import_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["datesold"] = pd.to_datetime(df["datesold"], yearfirst=True)
    df["Year"] = df.datesold.dt.year
    df.drop_duplicates(subset=["datesold"], inplace=True)
    df = df[df["price"] <= 2000000]
    df = df.sort_values(by=["datesold"])
    return df


df = import_data(path)

colors = {"background": "#F5CCB0", "text": "#F57C00"}


def preprocessing_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(df[["propertyType"]])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(["propertyType"]))
    df = df.reset_index(drop=True)
    df = pd.concat([df, one_hot_df], axis=1)
    df = df.drop(["propertyType"], axis=1)
    df["datesold"] = pd.to_datetime(df["datesold"])
    df = df.copy()
    df["dayofweek"] = df.datesold.dt.weekday
    df["quarter"] = df.datesold.dt.quarter
    df["month"] = df.datesold.dt.month
    df["year"] = df.datesold.dt.year
    df["dayofyear"] = df.datesold.dt.dayofyear
    df["dayofmonth"] = df.datesold.dt.day
    df["weekofyear"] = df.datesold.dt.isocalendar().week
    train_len = int(0.9 * len(df))
    train = df[:train_len]
    test = df[train_len:]
    X_train = train[
        [
            "dayofweek",
            "quarter",
            "month",
            "year",
            "dayofyear",
            "dayofmonth",
            "weekofyear",
            "postcode",
            "bedrooms",
            "propertyType_house",
            "propertyType_unit",
        ]
    ]
    Y_train = train[["price"]]
    X_test = test[
        [
            "dayofweek",
            "quarter",
            "month",
            "year",
            "dayofyear",
            "dayofmonth",
            "weekofyear",
            "postcode",
            "bedrooms",
            "propertyType_house",
            "propertyType_unit",
        ]
    ]
    Y_test = test[["price"]]
    return X_train, Y_train, X_test, Y_test, train_len


X_train, Y_train, X_test, Y_test, train_len = preprocessing_data(df)


def Decision_Tree_predict(name: str, X_test: pd.DataFrame, Y_test: pd.DataFrame) -> tuple[np.ndarray, float]:
    pipe = pickle.load(open(name, "rb"))
    predictions = pipe.predict(X_test)
    rmse = float(format(np.sqrt(mean_squared_error(Y_test, predictions)), ".3f"))
    return predictions, rmse


def Random_Forest_predict(name: str, X_test: pd.DataFrame, Y_test: pd.DataFrame) -> tuple[np.ndarray, float]:
    pipe = pickle.load(open(name, "rb"))
    predictions = pipe.predict(X_test)
    rmse = float(format(np.sqrt(mean_squared_error(Y_test, predictions)), ".3f"))
    return predictions, rmse


def plot_predict(
    data: pd.DataFrame, train_len: int, pred: np.ndarray, output: pd.core.series.Series, input: pd.core.series.Series
) -> Figure:
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=input[train_len : len(data)],
            y=output[train_len : len(data)],
            name="test",
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=input[:train_len],
            y=output[:train_len],
            name="train",
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=input[train_len : len(data)].reset_index(drop=True),
            y=pred,
            name="prediction",
        )
    )

    return fig2


methods_array = np.array(["Decision tree", "Random Forest"])

app.layout = html.Div(
    style={"backgroundColor": colors["background"]},
    children=[
        html.H1(
            children="Przewidywanie cen mieszkań",
            style={"textAlign": "center", "color": colors["text"]},
        ),
        html.Div(
            [
                html.H3(
                    children="Wybierz liczbę sypialni",
                    style={"textAlign": "center", "color": colors["text"]},
                    className="card",
                ),
                dcc.Dropdown(
                    df["bedrooms"].unique(),
                    df["bedrooms"].unique(),
                    id="bedrooms-selection",
                    multi=True,
                ),
            ],
            style={"width": "48%", "display": "inline-block"},
        ),
        html.Div(
            [
                html.H3(
                    children="Wybierz typ własności nieruchomości",
                    style={"textAlign": "center", "color": colors["text"]},
                    className="card",
                ),
                dcc.Dropdown(
                    df["propertyType"].unique(),
                    df["propertyType"].unique(),
                    id="type-selection",
                    multi=True,
                ),
            ],
            style={"width": "48%", "float": "right", "display": "inline-block"},
        ),
        # html.Br(),
        html.Div(
            [
                html.H3(
                    children="Wybierz zakres lat sprzedaży nieruchomości",
                    style={"textAlign": "center", "color": colors["text"]},
                    className="card",
                ),
                dcc.RangeSlider(
                    df["Year"].min(),
                    df["Year"].max(),
                    step=None,
                    id="date-selection",
                    value=[df["Year"].min(), df["Year"].max()],
                    marks={str(year): str(year) for year in df["Year"].unique()},
                ),
            ]
        ),
        html.Div(dcc.Graph(id="chart")),
        html.Div(
            [
                html.H3(
                    children="Wybierz metodę przewidywania, aby zobaczyć jej skuteczność",
                    style={"textAlign": "center", "color": colors["text"]},
                    className="card",
                ),
                dcc.Dropdown(np.unique(methods_array), None, id="method-selection"),
            ]
        ),
        html.Div(dcc.Graph(id="chart2")),
    ],
)


@callback(
    Output("chart", "figure"),
    Output("chart2", "figure"),
    Input("bedrooms-selection", "value"),
    Input("type-selection", "value"),
    Input("date-selection", "value"),
    Input("method-selection", "value"),
)
def update_graph(
    selected_bedrooms_value: str,
    selected_type_value: str,
    dates_selection_value: str,
    selected_method_value: str,
) -> Any:
    tmp = df.loc[df.loc[:, "bedrooms"].isin(selected_bedrooms_value), :]
    tmp = tmp.loc[tmp.loc[:, "propertyType"].isin(selected_type_value), :]
    tmp = tmp[tmp.loc[:, "Year"] <= dates_selection_value[1]]
    tmp = tmp[tmp.loc[:, "Year"] >= dates_selection_value[0]]

    fig = px.line(
        tmp,
        x="datesold",
        y="price",
        labels={
            "datesold": "Data sprzedaży",
            "price": "Cena nieruchomości",
        },
    )

    if selected_method_value == "Decision tree":
        prediction_DT, rmse_DT = Decision_Tree_predict(
            os.path.join(dir, "model_DT.pkl"),
            X_test,
            Y_test,
        )
        fig2 = plot_predict(df, train_len, prediction_DT, df.price, df.datesold)
    elif selected_method_value == "Random Forest":
        prediction_RF, rmse_RF = Random_Forest_predict(
            os.path.join(dir, "model_FR.pkl"),
            X_test,
            Y_test,
        )
        fig2 = plot_predict(df, train_len, prediction_RF, df.price, df.datesold)
    else:
        fig2 = fig

    annotations = []
    annotations.append(
        dict(
            xref="paper",
            yref="paper",
            x=0.0,
            y=1.05,
            xanchor="left",
            yanchor="bottom",
            text="Ceny sprzedaży nieruchomości",
            font=dict(family="Arial", size=30, color=colors["text"]),
            showarrow=False,
        )
    )

    fig.update_layout(plot_bgcolor=colors["background"], annotations=annotations)
    return fig, fig2


if __name__ == "__main__":
    app.run_server(debug=True)
