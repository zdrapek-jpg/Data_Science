def  fill_with_mean_value_for_numeric_consept(col_data,class_col):
    return col_data.fillna(col_data.group_by(class_col))

def fill_with_most_appeared_in_consept(col_data,class_col):
    return col_data.fillna(col_data.group_by(class_col))

def delete_whitespace_if_exist(data_frame):
    return data_frame.applymap(lambda x: x.strip() if isinstance(x,str) else x )

def delete_where_class_val_is_missing(data_frame,class_col):
    """

    :param class_col: zmienna decyzyjna z któej usuwamy brakujące dane
    :return:
    """
    return data_frame.dropna(subset=[class_col] ,inplace=True)
def looking_for_missing_values(data_frame,decision_col):
    from numpy import nan
    ## zamiana braków na nan
    data_frame[decision_col].replace(["", " "],nan, inplace=True)
    ### usuwanie weirszy z nan



    #### wartości None i np.nan zjadziemy za pomocą funkcji frame.isnull() lub frame.isna(), isnan
    str_cols =  data_frame.select_dtypes(include=["string","object","category"])
    digits_col =data_frame.select_dtypes(include=["number", "int", "float"])
    # uzupełnianie danych na podstawie klasy decyzyjnej dla danych numerycznych to mean dla klasy decyzyjnej dla ciągłych najczęściej występująca
    data1 = fill_with_mean_value_for_numeric_consept(digits_col,decision_col)
    data2 = fill_with_most_appeared_in_consept(str_cols,decision_col)

    pass