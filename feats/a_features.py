import pandas as pd
import numpy as np
def features_a(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = drop_all_null_columns(df)
    df = drop_almost_constant_columns(df, threshold=0.99)
    df = create_urban_risk_score(df)
    df['log_vehicle_value_new'] = np.log1p(df['vehicle_value_new'])

    price_cols = [col for col in df.columns if col.endswith('_price')]

# Primeni funkciju
    df, removed = remove_high_correlation_features(
        df, 
        threshold=0.97, 
        keep_targets=price_cols,  # Cene se ne izbacuju
        verbose=True
    )
    
    return df


def dummy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering za 'a' deo
    """
    return df



def drop_all_null_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Brise kolone koje imaju samo null vrednosti.
    """
    return df.dropna(axis=1, how="all")


def drop_almost_constant_columns(df: pd.DataFrame, threshold: float = 0.99) -> pd.DataFrame:
    cols_to_drop = []

    for col in df.columns:
        non_null = df[col].dropna()

        if non_null.empty:
            continue

        most_common_ratio = non_null.value_counts(normalize=True, dropna=True).iloc[0]

        if most_common_ratio >= threshold:
            cols_to_drop.append(col)

    return df.drop(columns=cols_to_drop)


def remove_high_correlation_features(data, threshold=0.95, keep_targets=None, verbose=True):
    """
    Uklanja jedan feature iz svakog para sa korelacijom > threshold
    
    Parameters:
    -----------
    data : DataFrame
    threshold : float
        Prag korelacije (default 0.95)
    keep_targets : list
        Liste kolona koje MORAJU biti zadržane (npr. cene)
    verbose : bool
        Da li štampati detalje
    
    Returns:
    --------
    data_clean : DataFrame
        DataFrame bez redundantnih kolona
    removed_cols : list
        Lista izbačenih kolona
    """
    
    if keep_targets is None:
        keep_targets = []
    
    data_clean = data.copy()
    removed_cols = []
    
    # Izaberi samo numeričke kolone
    numeric_cols = data_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    # Ukloni target kolone iz razmatranja (ne smeju se izbaciti)
    cols_to_check = [col for col in numeric_cols if col not in keep_targets]
    
    if len(cols_to_check) < 2:
        print("Nema dovoljno numeričkih kolona za proveru")
        return data_clean, removed_cols
    
    if verbose:
        print("="*80)
        print(f"UKLANJANJE VISOKO KORELISANIH FEATURE-A (prag: {threshold})")
        print("="*80)
        print(f"Broj numeričkih kolona za proveru: {len(cols_to_check)}")
    
    # Izračunaj korelacionu matricu
    corr_matrix = data_clean[cols_to_check].corr().abs()
    
    # Gornji trougao matrice (bez dijagonale)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Pronađi kolone za izbacivanje
    to_drop = []
    
    for column in upper.columns:
        if column in to_drop:
            continue
        
        # Pronađi kolone sa korelacijom > threshold
        high_corr_cols = upper.index[upper[column] > threshold].tolist()
        
        for corr_col in high_corr_cols:
            if corr_col not in to_drop:
                corr_value = corr_matrix.loc[column, corr_col]
                to_drop.append(corr_col)
                removed_cols.append(corr_col)
                
                if verbose:
                    print(f"   ✗ IZBACUJEM: {corr_col}")
                    print(f"     (korelacija {corr_value:.4f} sa {column})")
    
    # Izbaci kolone
    if to_drop:
        data_clean = data_clean.drop(columns=to_drop, errors='ignore')
        
        if verbose:
            print("\n" + "="*80)
            print("STATISTIKA")
            print("="*80)
            print(f"Pre izbacivanja: {len(data.columns)} kolona")
            print(f"Posle izbacivanja: {len(data_clean.columns)} kolona")
            print(f"Izbačeno: {len(to_drop)} kolona")
    else:
        if verbose:
            print("\n✅ Nema kolona za izbacivanje (sve korelacije ispod praga)")
    
    return data_clean, removed_cols



def create_urban_risk_score(data):
    """
    Kreira risk score koji kombinuje pozitivne i negativne feature-e
    """
    
    positive_features = [
        'postal_code_department_stores_within_10_km',
        'postal_code_primary_schools_within_3_km',
        'postal_code_supermarkets_within_3_km'
    ]
    
    negative_features = [
        'postal_code_urban_category'
    ]
    
    data['urban_amenities_score'] = 0
    for feat in positive_features:
        if feat in data.columns:
            norm = (data[feat] - data[feat].min()) / (data[feat].max() - data[feat].min())
            data['urban_amenities_score'] += norm
    
    for feat in negative_features:
        if feat in data.columns:
            norm = 1 - (data[feat] - data[feat].min()) / (data[feat].max() - data[feat].min())
            data['urban_amenities_score'] += norm
    
    data['urban_amenities_score'] = data['urban_amenities_score'] / (len(positive_features) + len(negative_features))
    
    print("✅ Kreiran: urban_amenities_score")
    return data