def engineer_features(df):
    df['kd_ratio'] = df['kills'] / df['deaths'].replace(0, 1)
    df['total_rounds'] = df['team_a_rounds'] + df['team_b_rounds']
    return df