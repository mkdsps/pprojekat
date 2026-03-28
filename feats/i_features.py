import pandas as pd

def features_i(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = dummy_features(df)
    df = create_insurer_e_i_features(df)

    return df


def dummy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering za 'i' deo
    """
    return df

def create_insurer_e_i_features(data):
    """
    Kreira 3 ključna feature-a za Insurer_E i I:
    1. cfy_province_interaction - interakcija claim_free_years i province
    2. cfy_standardized_by_province - standardizovan claim_free_years unutar svake province
    3. combined_risk_score - kombinovani risk score (CFY + province)
    
    Parameters:
    -----------
    data : pandas DataFrame
        Ulazni podaci koji moraju sadržati kolone:
        - 'claim_free_years'
        - 'province'
        - 'Insurer_E_price' (za normalizaciju province risk)
    
    Returns:
    --------
    data : pandas DataFrame
        Podaci sa dodatim novim kolonama
    """
    
    # Napravi kopiju da ne bi menjao original
    data = data.copy()
    
    print("="*80)
    print("KREIRANJE FEATURE-A ZA INSURER_E I I")
    print("="*80)
    
    # -------------------------------------------------------------------------
    # 1. Provera i konverzija ulaznih kolona
    # -------------------------------------------------------------------------
    
    # Proveri claim_free_years
    if 'claim_free_years' not in data.columns:
        raise ValueError("❌ Kolona 'claim_free_years' ne postoji!")
    
    # Konvertuj claim_free_years u numeric ako nije
    if data['claim_free_years'].dtype == 'object':
        data['claim_free_years'] = pd.to_numeric(data['claim_free_years'], errors='coerce')
        print("✅ Konvertovan 'claim_free_years' u numeric")
    
    # Proveri province
    if 'province' not in data.columns:
        raise ValueError("❌ Kolona 'province' ne postoji!")
    
    # Konvertuj province u numeric ako nije
    if data['province'].dtype == 'object':
        data['province'] = pd.to_numeric(data['province'], errors='coerce')
        print("✅ Konvertovan 'province' u numeric")
    
    # -------------------------------------------------------------------------
    # 2. FEATURE 1: cfy_province_interaction
    # -------------------------------------------------------------------------
    print("\n1. Kreiranje: cfy_province_interaction")
    
    data['cfy_province_interaction'] = data['claim_free_years'] * data['province']
    print("   ✅ cfy_province_interaction = claim_free_years × province")
    
    # Provera
    print(f"   Opseg: {data['cfy_province_interaction'].min():.2f} - {data['cfy_province_interaction'].max():.2f}")
    print(f"   NaN: {data['cfy_province_interaction'].isna().sum()}")
    
    # -------------------------------------------------------------------------
    # 3. FEATURE 2: cfy_standardized_by_province
    # -------------------------------------------------------------------------
    print("\n2. Kreiranje: cfy_standardized_by_province")
    
    # Standardizacija unutar svake province
    data['cfy_standardized_by_province'] = data.groupby('province')['claim_free_years'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x
    )
    print("   ✅ cfy_standardized_by_province = (CFY - mean_province) / std_province")
    
    # Provera
    print(f"   Opseg: {data['cfy_standardized_by_province'].min():.2f} - {data['cfy_standardized_by_province'].max():.2f}")
    print(f"   NaN: {data['cfy_standardized_by_province'].isna().sum()}")
    
    # -------------------------------------------------------------------------
    # 4. FEATURE 3: combined_risk_score
    # -------------------------------------------------------------------------
    print("\n3. Kreiranje: combined_risk_score")
    
    # 4a. Normalizuj claim_free_years (što manje godina = veći rizik)
    # Invertuj: negativne i male vrednosti daju veći rizik
    data['cfy_risk'] = -data['claim_free_years']  # negativno postaje pozitivno
    cfy_min = data['cfy_risk'].min()
    cfy_max = data['cfy_risk'].max()
    
    if cfy_max > cfy_min:
        data['cfy_risk_norm'] = (data['cfy_risk'] - cfy_min) / (cfy_max - cfy_min)
    else:
        data['cfy_risk_norm'] = 0
    
    print("   ✅ cfy_risk_norm = normalizovan rizik od claim_free_years")
    
    # 4b. Normalizuj province risk (na osnovu Insurer_E cene)
    # Koristi Insurer_E kao referencu za rizik po provinciji
    if 'Insurer_E_price' in data.columns:
        province_avg_risk = data.groupby('province')['Insurer_E_price'].mean()
        data['province_risk'] = data['province'].map(province_avg_risk)
        
        province_min = province_avg_risk.min()
        province_max = province_avg_risk.max()
        
        if province_max > province_min:
            data['province_risk_norm'] = (data['province_risk'] - province_min) / (province_max - province_min)
        else:
            data['province_risk_norm'] = 0
        
        print("   ✅ province_risk_norm = normalizovan rizik po provinciji (na osnovu Insurer_E)")
    else:
        # Ako nema Insurer_E, koristi prostu normalizaciju province broja
        print("   ⚠️ Insurer_E_price ne postoji, koristim prostu normalizaciju province")
        province_min = data['province'].min()
        province_max = data['province'].max()
        if province_max > province_min:
            data['province_risk_norm'] = (data['province'] - province_min) / (province_max - province_min)
        else:
            data['province_risk_norm'] = 0
    
    # 4c. Kombinovani score (60% CFY, 40% Province)
    data['combined_risk_score'] = data['cfy_risk_norm'] * 0.6 + data['province_risk_norm'] * 0.4
    print("   ✅ combined_risk_score = 0.6 × cfy_risk_norm + 0.4 × province_risk_norm")
    
    # Provera
    print(f"   Opseg: {data['combined_risk_score'].min():.4f} - {data['combined_risk_score'].max():.4f}")
    print(f"   NaN: {data['combined_risk_score'].isna().sum()}")
    
    # -------------------------------------------------------------------------
    # 5. Popuni NaN vrednosti (ako ih ima)
    # -------------------------------------------------------------------------
    print("\n4. Popunjavanje NaN vrednosti:")
    
    for col in ['cfy_province_interaction', 'cfy_standardized_by_province', 'combined_risk_score']:
        if data[col].isna().sum() > 0:
            median_val = data[col].median()
            data[col] = data[col].fillna(median_val)
            print(f"   ✅ {col}: popunjeno sa medijanom ({median_val:.4f})")
    
    # -------------------------------------------------------------------------
    # 6. Provera korelacija (opciono)
    # -------------------------------------------------------------------------
    print("\n5. Provera korelacija sa Insurer_E i Insurer_I:")
    
    for insurer in ['Insurer_E', 'Insurer_I']:
        price_col = f'{insurer}_price'
        if price_col in data.columns:
            print(f"\n   {insurer}:")
            for col in ['cfy_province_interaction', 'cfy_standardized_by_province', 'combined_risk_score']:
                valid = data[[col, price_col]].dropna()
                if len(valid) > 100:
                    corr = valid[col].corr(valid[price_col])
                    print(f"      {col:35} {corr:8.4f}")
    
    # -------------------------------------------------------------------------
    # 7. Sažetak
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("SAŽETAK KREIRANIH FEATURE-A")
    print("="*80)
    print(f"""
    ✅ Kreirana 3 nova feature-a:
    
    1. cfy_province_interaction
       - Tip: float64
       - Formula: claim_free_years × province
       - Koristi: hvata interakciju između bonusa/malusa i lokacije
    
    2. cfy_standardized_by_province
       - Tip: float64
       - Formula: (CFY - mean_province) / std_province
       - Koristi: relativni položaj vozača unutar svoje provincije
    
    3. combined_risk_score
       - Tip: float64
       - Formula: 0.6 × cfy_risk_norm + 0.4 × province_risk_norm
       - Koristi: jedinstvena mera rizika (veći score = veći rizik)
    
    Ukupno novih kolona: 3
    """)
    
    return data