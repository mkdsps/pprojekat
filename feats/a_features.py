import pandas as pd
import numpy as np
def features_a(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = drop_all_null_columns(df)
    df = drop_almost_constant_columns(df, threshold=0.99)
    df = create_urban_risk_score(df)
    df = add_vehicle_score_selected_insurers(df)
    df['log_vehicle_value_new'] = np.log1p(df['vehicle_value_new'])
    df["driver_age"] = (
        2026 - pd.to_datetime(df["contractor_birthdate"], dayfirst=True, errors="coerce").dt.year
    ).astype("Int64")
    df["driver_age_band"] = pd.cut(
        df["driver_age"],
        bins=[17, 24, 30, 40, 50, 60, 70, 80, 90, 120],
        labels=["18-24", "25-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90", "90+"]
    )
    df.drop(columns=["driver_age"], inplace=True, errors="ignore")
    df = add_dependency_features(df)
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


def add_vehicle_score_selected_insurers(df: pd.DataFrame, selected_prices=None) -> pd.DataFrame:
    df = df.copy()

    vehicle_features = [
        "vehicle_ownership_duration",
        "vehicle_engine_size",
        "vehicle_power",
        "vehicle_net_weight",
        "vehicle_gross_weight",
        "vehicle_length",
        "vehicle_width",
        "vehicle_height",
        "vehicle_number_of_cylinders",
        "vehicle_number_of_doors",
        "vehicle_number_of_seats",
        "vehicle_value_new",
        "vehicle_net_max_power",
        "vehicle_net_max_power_electric",
        "vehicle_nominal_continuous_max_power",
        "vehicle_power_to_net_weight_ratio",
        "vehicle_age",
        "vehicle_years_since_country_first_registration",
        "vehicle_odometer_verdict_code",
        "vehicle_planned_annual_mileage",
    ]

    if selected_prices is None:
        selected_prices = [
            "Insurer_C_price",
            "Insurer_H_price",
            "Insurer_G_price",
            "Insurer_J_price",
        ]

    selected_prices = [c for c in selected_prices if c in df.columns]
    existing_features = [c for c in vehicle_features if c in df.columns]

    corr_matrix = df[existing_features + selected_prices].corr(numeric_only=True).loc[existing_features, selected_prices]
    feature_weights = corr_matrix.mean(axis=1)

    z = pd.DataFrame(index=df.index)
    for col in existing_features:
        s = pd.to_numeric(df[col], errors="coerce")
        std = s.std()
        if pd.notna(std) and std != 0:
            z[col] = (s - s.mean()) / std
        else:
            z[col] = 0.0

    raw_score = pd.Series(0.0, index=df.index)
    for col in existing_features:
        raw_score = raw_score + z[col].fillna(0) * feature_weights[col]

    min_val = raw_score.min()
    max_val = raw_score.max()

    if pd.notna(max_val) and pd.notna(min_val) and max_val != min_val:
        df["vehicle_score"] = (raw_score - min_val) / (max_val - min_val)
    else:
        df["vehicle_score"] = 0.5

    weight_table = (
        feature_weights
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .rename("weight")
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    return df


def add_dependency_features(data):
    coverage_map = {'mtpl': 1, 'limited_casco': 2, 'casco': 3}
    data['coverage_num'] = data['coverage'].map(coverage_map)
    
    auto_dep_map = {1: 0.2, 2: 0.6, 3: 0.9}
    data['auto_dependency'] = data['coverage_num'].map(auto_dep_map)
    
    urban_dep_map = {1: 0.1, 2: 0.5, 3: 0.8}
    data['urban_dependency'] = data['coverage_num'].map(urban_dep_map)
    
    data['auto_dependency'] = data['auto_dependency'].fillna(data['auto_dependency'].median())
    data['urban_dependency'] = data['urban_dependency'].fillna(data['urban_dependency'].median())
    
    return data