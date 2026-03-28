import pandas as pd
import numpy as np
def features_a(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = drop_all_null_columns(df)
    df = drop_almost_constant_columns(df, threshold=0.99)
    df = create_urban_risk_score(df)
    df = add_vehicle_score_selected_insurers(df)
    df['log_vehicle_value_new'] = np.log1p(pd.to_numeric(df['vehicle_value_new'], errors='coerce').fillna(0))
    df["driver_age"] = (
        2026 - pd.to_datetime(df["contractor_birthdate"], dayfirst=True, errors="coerce").dt.year
    ).astype("Int64")
    df["driver_age_band"] = pd.cut(
        df["driver_age"],
        bins=[17, 24, 30, 40, 50, 60, 70, 80, 90, 120],
        labels=["18-24", "25-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90", "90+"]
    )
    df.drop(columns=["driver_age"], inplace=True, errors="ignore")
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
    Kreira risk score koji kombinuje pozitivne i negativne feature-e.
    Popravljeno da radi sa stringovima i izbegava deljenje nulom.
    """
    data = data.copy()
    
    positive_features = [
        'postal_code_department_stores_within_10_km',
        'postal_code_primary_schools_within_3_km',
        'postal_code_supermarkets_within_3_km'
    ]
    
    negative_features = [
        'postal_code_urban_category'
    ]
    
    data['urban_amenities_score'] = 0.0
    valid_features_count = 0

    all_features = positive_features + negative_features
    
    for feat in all_features:
        if feat in data.columns:
            # KLJUČNI FIX: Konverzija u broj. Ako je string, postaće broj ili NaN
            col_numeric = pd.to_numeric(data[feat], errors='coerce').fillna(0)
            
            f_min = col_numeric.min()
            f_max = col_numeric.max()
            
            # Provera da se izbegne deljenje nulom
            if f_max - f_min > 0:
                if feat in positive_features:
                    norm = (col_numeric - f_min) / (f_max - f_min)
                else:
                    norm = 1 - (col_numeric - f_min) / (f_max - f_min)
                
                data['urban_amenities_score'] += norm
                valid_features_count += 1
    
    # Prosek svih skorova
    if valid_features_count > 0:
        data['urban_amenities_score'] /= valid_features_count
    
    print("✅ Kreiran: urban_amenities_score")
    return data


def add_vehicle_score_selected_insurers(df: pd.DataFrame, selected_prices=None) -> pd.DataFrame:
    df = df.copy()

    vehicle_features = [
        "vehicle_ownership_duration", "vehicle_engine_size", "vehicle_power",
        "vehicle_net_weight", "vehicle_gross_weight", "vehicle_length",
        "vehicle_width", "vehicle_height", "vehicle_number_of_cylinders",
        "vehicle_number_of_doors", "vehicle_number_of_seats",
        "vehicle_value_new", "vehicle_net_max_power",
        "vehicle_net_max_power_electric", "vehicle_nominal_continuous_max_power",
        "vehicle_power_to_net_weight_ratio", "vehicle_age",
        "vehicle_years_since_country_first_registration",
        "vehicle_odometer_verdict_code", "vehicle_planned_annual_mileage",
    ]

    if selected_prices is None:
        selected_prices = ["Insurer_C_price", "Insurer_H_price", "Insurer_G_price", "Insurer_J_price"]

    # 1. Filtriraj samo ono što stvarno postoji u tabeli
    existing_features = [c for c in vehicle_features if c in df.columns]
    existing_prices = [c for c in selected_prices if c in df.columns]

    if not existing_features or not existing_prices:
        df["vehicle_score"] = 0.5
        return df

    # 2. KLJUČNI FIX: Pretvori kolone u numerički tip pre korelacije
    # Kreiramo privremeni DF samo za proračun tegova
    temp_corr_df = df[existing_features + existing_prices].copy()
    for col in temp_corr_df.columns:
        temp_corr_df[col] = pd.to_numeric(temp_corr_df[col], errors='coerce').fillna(0)

    # 3. Sad računaj korelaciju - numeric_only više nije problem jer smo ih pretvorili
    corr_matrix = temp_corr_df.corr().loc[existing_features, existing_prices]
    
    # Izračunaj težine (mean korelacije sa izabranim osiguravačima)
    feature_weights = corr_matrix.mean(axis=1).fillna(0)

    # 4. Izračunaj Z-score na originalnom DF-u (uz konverziju u float)
    z = pd.DataFrame(index=df.index)
    for col in existing_features:
        s = pd.to_numeric(df[col], errors="coerce").astype(float)
        std = s.std()
        if pd.notna(std) and std != 0:
            z[col] = (s - s.mean()) / std
        else:
            z[col] = 0.0

    # Pomnoži z-score sa težinama
    raw_score = (z[existing_features] * feature_weights).sum(axis=1)

    # 5. Normalizacija finalnog skora na 0-1
    min_val, max_val = raw_score.min(), raw_score.max()
    if pd.notna(max_val) and max_val != min_val:
        df["vehicle_score"] = (raw_score - min_val) / (max_val - min_val)
    else:
        df["vehicle_score"] = 0.5

    print("✅ Kreiran: vehicle_score")
    return df