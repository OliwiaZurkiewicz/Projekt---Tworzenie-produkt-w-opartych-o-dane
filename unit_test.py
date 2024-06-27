import os

import numpy as np
import pandas as pd
import pytest
from Application import (
    Decision_Tree_predict,
    Random_Forest_predict,
    import_data,
    plot_predict,
    preprocessing_data,
    update_graph
)
from plotly.graph_objs._figure import Figure

df: pd.DataFrame
X_test: pd.DataFrame
Y_test: pd.DataFrame
prediction_DT: np.ndarray
prediction_FR: np.ndarray
prediction_DT_len: float
prediction_FR_len: float


@pytest.mark.parametrize("mock_path", ["test_raw_sales.csv"])
def test_import_data(mock_path: str) -> None:
    mock_df: pd.DataFrame = import_data(mock_path)
    assert isinstance(mock_df, pd.DataFrame)
    global df
    df = mock_df


def test_preprocessing_data() -> None:
    global df
    result_tuple = preprocessing_data(df)
    for el in result_tuple[:-1]:
        assert isinstance(el, pd.DataFrame)

    global X_test, Y_test
    X_test = result_tuple[2]
    Y_test = result_tuple[3]
    assert type(result_tuple[-1]) == int


@pytest.mark.parametrize("name", [os.path.join(os.path.dirname(__file__), "model_DT.pkl")])
def test_Decision_Tree_predict(name: str) -> None:
    global X_test, Y_test
    result_tuple: tuple[np.ndarray, float] = Decision_Tree_predict(name, X_test, Y_test)

    assert isinstance(result_tuple[0], np.ndarray)
    assert type(result_tuple[1]) == float

    global prediction_DT, prediction_DT_len
    prediction_DT = result_tuple[0]
    prediction_DT_len = result_tuple[1]


@pytest.mark.parametrize("name", [os.path.join(os.path.dirname(__file__), "model_FR.pkl")])
def test_Random_Forest_predict(name: str) -> None:
    global X_test, Y_test
    result_tuple: tuple[np.ndarray, float] = Random_Forest_predict(name, X_test, Y_test)

    assert isinstance(result_tuple[0], np.ndarray)
    assert type(result_tuple[1]) == float

    global prediction_FR, prediction_FR_len
    prediction_FR = result_tuple[0]
    prediction_FR_len = result_tuple[1]


def test_plot_predict_Decision_Tree() -> None:
    global df, prediction_DT
    train_len: int = int(0.9 * len(df))
    fig: Figure = plot_predict(df, train_len, prediction_DT, df["price"], df["datesold"])

    assert isinstance(fig, Figure)


def test_plot_predict_Random_Forest() -> None:
    global df, prediction_FR
    train_len: int = int(0.9 * len(df))
    fig: Figure = plot_predict(df, train_len, prediction_FR, df["price"], df["datesold"])

    assert isinstance(fig, Figure)


@pytest.mark.parametrize("mock_selected_bedrooms_value", [[4, 3, 5, 2, 1, 0]])
@pytest.mark.parametrize("mock_selected_dates", [[2007, 2019]])
@pytest.mark.parametrize("mock_selected_type_value", [["house", "unit"]])
@pytest.mark.parametrize("mock_selected_method_value", [None, "Random Forest", "Decision Tree"])
def test_update_graph(
    mock_selected_bedrooms_value: list[int],
    mock_selected_dates: list[int],
    mock_selected_type_value: list[str],
    mock_selected_method_value: str | None,
) -> None:
    fig1, fig2 = update_graph(
        mock_selected_bedrooms_value, mock_selected_type_value, mock_selected_dates, mock_selected_method_value
    )
    assert isinstance(fig1, Figure)
    assert isinstance(fig2, Figure)
