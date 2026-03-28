import pandas as pd

def clean_v(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = dummy_clean(df)
    # df = kolona_klean()

    return df
    

def dummy_clean(df : pd.DataFrame) -> pd.DataFrame:
    """
        odvoji po kolonama ove dummy funk...
    """
    df = df.copy()

    subset_without_first = df.columns[1:]

    df = df.drop_duplicates(subset=subset_without_first, keep='first')
    ###############################################################################################
    # num_last_cols = 22
    # last22_cols = df.columns[-num_last_cols:]
    # other_cols = df.columns[1:-num_last_cols] 

    # df['group_id'] = pd.factorize(
    #     df[other_cols].fillna('NaN').astype(str).agg('-'.join, axis=1)
    # )[0]

    # df[last22_cols] = df.groupby('group_id')[last22_cols].transform('mean')


    # df_unique = df.drop_duplicates(subset=other_cols, keep='first').copy()

    # df_unique = df_unique.drop(columns=['group_id'])
    
    return df