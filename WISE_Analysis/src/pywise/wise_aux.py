def standardize_column_names(df):
    """
    """
    df = df.rename({col: "_".join([word.lower() for word in col.split()]) 
                    for col in df.columns if "capital" not in col.lower()
                    }, 
                   axis=1
                  )
    df = df.rename({col: col.lower() for col in df.columns if "capital" in col.lower()}, 
                   axis=1
                  )
    
    return df

def pretty_print(dfdict):
    """
    Create pretty printout of a
    dictionary with pd.DataFrames
    as values.
    """
    for k, v in dfdict.items():
        print(k + ":")
        print("-"*(len(k)+1))
        display(v)
        print()