
def create_interaction_features(X_df):
    """Create engineered features for personality prediction"""
    X_new = X_df.copy()

    # Social Engagement Score
    X_new['Social_Engagement'] = (
        X_df['Social_event_attendance'] + 
        X_df['Going_outside'] + 
        X_df['Friends_circle_size'] + 
        X_df['Post_frequency']
    ) / 4

    # Introversion Tendency
    X_new['Introversion_Tendency'] = (
        X_df['Time_spent_Alone'] + 
        X_df['Stage_fear_Yes'] + 
        X_df['Drained_after_socializing_Yes']
    ) / 3

    # Social vs Solitary Balance
    X_new['Social_Solitary_Balance'] = X_new['Social_Engagement'] - X_new['Introversion_Tendency']

    return X_new
