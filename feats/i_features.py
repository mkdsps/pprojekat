import pandas as pd
import numpy as np

def features_i(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = dummy_features(df)
    df = create_insurer_e_i_features(df)
    return df

def dummy_features(df: pd.DataFrame) -> pd.DataFrame:
    return df

def create_insurer_e_i_features(data):
    data = data.copy()
    
    # 1. Konverzija u brojeve i rešavanje Arrow/String problema
    for col in ['claim_free_years', 'Insurer_E_price']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col].astype(str).replace('nan', np.nan), errors='coerce').fillna(0)
    
    # 2. Bezbedna konverzija provincije (radi i ako je tekst i ako je broj)
    if 'province' in data.columns:
        if data['province'].dtype == 'object' or data['province'].dtype.name == 'string':
            data['province_code'] = data['province'].astype('category').cat.codes
        else:
            data['province_code'] = pd.to_numeric(data['province'], errors='coerce').fillna(0)
    else:
        raise ValueError("❌ Kolona 'province' ne postoji!")

    # --- FEATURE 1: cfy_province_interaction ---
    data['cfy_province_interaction'] = data['claim_free_years'] * data['province_code']
    
    # --- FEATURE 2: cfy_standardized_by_province ---
    data['cfy_standardized_by_province'] = data.groupby('province')['claim_free_years'].transform(
        lambda x: (x - x.mean()) / x.std() if len(x) > 1 and x.std() > 0 else 0
    ).fillna(0)
    
    # --- FEATURE 3: combined_risk_score ---
    # Normalizacija CFY (manje godina = veći rizik)
    cfy_risk = -data['claim_free_years']
    c_min, c_max = cfy_risk.min(), cfy_risk.max()
    data['cfy_risk_norm'] = (cfy_risk - c_min) / (c_max - c_min) if c_max > c_min else 0
    
    # Normalizacija rizika provincije
    if 'Insurer_E_price' in data.columns:
        province_map = data.groupby('province')['Insurer_E_price'].mean()
        data['prov_risk'] = data['province'].map(province_map)
        p_min, p_max = province_map.min(), province_map.max()
        data['prov_risk_norm'] = (data['prov_risk'] - p_min) / (p_max - p_min) if p_max > p_min else 0
    else:
        p_min, p_max = data['province_code'].min(), data['province_code'].max()
        data['prov_risk_norm'] = (data['province_code'] - p_min) / (p_max - p_min) if p_max > p_min else 0
        
    data['combined_risk_score'] = data['cfy_risk_norm'] * 0.6 + data['prov_risk_norm'] * 0.4
    
    # Finalno čišćenje privremenih kolona i NaN vrednosti
    temp_cols = ['cfy_risk', 'cfy_risk_norm', 'prov_risk', 'prov_risk_norm', 'province_code']
    data = data.drop(columns=[c for c in temp_cols if c in data.columns])
    
    cols_to_fill = ['cfy_province_interaction', 'cfy_standardized_by_province', 'combined_risk_score']
    for col in cols_to_fill:
        data[col] = data[col].fillna(data[col].median() if not data[col].isna().all() else 0)
        
    return data