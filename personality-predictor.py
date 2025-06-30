import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page config
st.set_page_config(
    page_title="Personality Predictor",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .extrovert-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border: 3px solid #2196f3;
        color: #000
    }
    .introvert-box {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border: 3px solid #f44336;
        color: #000
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all saved model components"""
    try:
        model = joblib.load('models/personality_model.pkl')
        scaler = joblib.load('models/personality_scaler.pkl')
        label_encoder = joblib.load('models/personality_label_encoder.pkl')
        return model, scaler, label_encoder, True
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, False

def create_interaction_features(X_df):
    """Create engineered features for personality prediction"""
    X_new = X_df.copy()
    
    X_new['Social_Engagement'] = (
        X_df['Social_event_attendance'] + 
        X_df['Going_outside'] + 
        X_df['Friends_circle_size'] + 
        X_df['Post_frequency']
    ) / 4
    
    X_new['Introversion_Tendency'] = (
        X_df['Time_spent_Alone'] + 
        X_df['Stage_fear_Yes'] + 
        X_df['Drained_after_socializing_Yes']
    ) / 3
    
    X_new['Social_Solitary_Balance'] = X_new['Social_Engagement'] - X_new['Introversion_Tendency']
    
    return X_new

def predict_personality(model, scaler, label_encoder, inputs):
    """Make personality prediction"""
    try:
        input_data = pd.DataFrame({
            'Time_spent_Alone': [inputs[0]],
            'Social_event_attendance': [inputs[1]],
            'Going_outside': [inputs[2]],
            'Friends_circle_size': [inputs[3]],
            'Post_frequency': [inputs[4]],
            'Stage_fear_Yes': [1 if inputs[5] else 0],
            'Drained_after_socializing_Yes': [1 if inputs[6] else 0]
        })
        
        input_scaled = scaler.transform(input_data)
        input_scaled = pd.DataFrame(input_scaled, columns=input_data.columns)
        input_engineered = create_interaction_features(input_scaled)
        
        prediction = model.predict(input_engineered)[0]
        prediction_proba = model.predict_proba(input_engineered)[0]
        personality = label_encoder.inverse_transform([prediction])[0]
        confidence = max(prediction_proba)
        
        return {
            'prediction': personality,
            'confidence': confidence,
            'probabilities': {
                'Extrovert': prediction_proba[0],
                'Introvert': prediction_proba[1]
            },
            'features': {
                'Social_Engagement': input_engineered['Social_Engagement'].iloc[0],
                'Introversion_Tendency': input_engineered['Introversion_Tendency'].iloc[0],
                'Social_Solitary_Balance': input_engineered['Social_Solitary_Balance'].iloc[0]
            },
            'success': True
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Personality Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Discover if you\'re an Extrovert or Introvert with 91% accuracy!</p>', unsafe_allow_html=True)
    
    # Load models
    model, scaler, label_encoder, models_loaded = load_models()
    
    if not models_loaded:
        st.error("❌ Could not load the trained models. Please ensure model files are in the 'models/' directory.")
        st.stop()
    
    st.markdown("### 📋 Answer these 7 quick questions about yourself:")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="question-box">', unsafe_allow_html=True)
        st.markdown("**⏰ Daily Alone Time**")
        time_alone = st.slider(
            "How many hours do you spend alone each day?",
            min_value=0.0, max_value=11.0, value=5.0, step=0.5,
            format="%.1f hours"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="question-box">', unsafe_allow_html=True)
        st.markdown("**🎉 Social Events**")
        social_events = st.slider(
            "How many social events do you attend monthly?",
            min_value=0, max_value=10, value=4,
            format="%d events" if st.session_state.get('social_events', 4) < 10 else "10+ events"
        )
        if social_events == 10:
            st.caption("📍 10+ events per month")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="question-box">', unsafe_allow_html=True)
        st.markdown("**🚶 Going Outside**")
        going_outside = st.slider(
            "How often do you go outside for activities weekly?",
            min_value=0, max_value=7, value=4,
            format="%d times" if st.session_state.get('going_outside', 4) < 7 else "Daily"
        )
        if going_outside == 7:
            st.caption("📍 Daily outdoor activities")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="question-box">', unsafe_allow_html=True)
        st.markdown("**👥 Close Friends**")
        friends_circle = st.slider(
            "How many people do you consider close friends?",
            min_value=0, max_value=15, value=6,
            format="%d friends" if st.session_state.get('friends_circle', 6) < 15 else "15+ friends"
        )
        if friends_circle == 15:
            st.caption("📍 15+ close friends")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="question-box">', unsafe_allow_html=True)
        st.markdown("**📱 Social Media Activity**")
        post_frequency = st.slider(
            "How often do you post on social media weekly?",
            min_value=0, max_value=10, value=3,
            format="%d posts" if st.session_state.get('post_frequency', 3) < 10 else "10+ posts"
        )
        if post_frequency == 10:
            st.caption("📍 10+ posts per week")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="question-box">', unsafe_allow_html=True)
        st.markdown("**🎭 Public Speaking**")
        stage_fear = st.radio(
            "Do you feel anxious when speaking publicly?",
            options=[False, True],
            format_func=lambda x: "😌 No, I'm comfortable" if not x else "😰 Yes, I get nervous",
            horizontal=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="question-box">', unsafe_allow_html=True)
        st.markdown("**😴 After Socializing**")
        drained_after_social = st.radio(
            "Do you feel tired after socializing and need alone time?",
            options=[False, True],
            format_func=lambda x: "⚡ No, I feel energized" if not x else "🔋 Yes, I need to recharge",
            horizontal=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Center the predict button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        predict_clicked = st.button("🔮 Discover My Personality!", type="primary", use_container_width=True)
    
    if predict_clicked:
        inputs = [time_alone, social_events, going_outside, friends_circle, 
                 post_frequency, stage_fear, drained_after_social]
        
        with st.spinner("🧠 Analyzing your personality..."):
            result = predict_personality(model, scaler, label_encoder, inputs)
        
        if result['success']:
            personality = result['prediction']
            confidence = result['confidence']
            probabilities = result['probabilities']
            features = result['features']
            
            # Results section
            st.markdown("---")
            
            # Main prediction with better styling
            box_class = "extrovert-box" if personality == "Extrovert" else "introvert-box"
            icon = "🌟" if personality == "Extrovert" else "🌙"
            
            st.markdown(f"""
            <div class="prediction-box {box_class}">
                <h1>{icon} You are an {personality}! {icon}</h1>
                <h2>Confidence: {confidence:.0%}</h2>
                <p style="font-size: 1.1rem; margin-top: 1rem;">
                    Based on your behavioral patterns, our AI model is {confidence:.0%} confident in this prediction.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed analysis
            st.markdown("### 📊 Detailed Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Personality Breakdown:**")
                st.progress(probabilities['Extrovert'], text=f"🌟 Extrovert: {probabilities['Extrovert']:.1%}")
                st.progress(probabilities['Introvert'], text=f"🌙 Introvert: {probabilities['Introvert']:.1%}")
            
            with col2:
                st.markdown("**What This Means:**")
                if personality == "Extrovert":
                    st.success("""
                    **You tend to:**
                    • Gain energy from social interaction
                    • Feel comfortable in groups
                    • Think out loud and process externally
                    • Seek stimulation from the outside world
                    """)
                else:
                    st.info("""
                    **You tend to:**
                    • Recharge through alone time
                    • Prefer deeper, smaller social circles
                    • Think before speaking
                    • Find fulfillment in internal reflection
                    """)
            
            # Behavioral scores
            st.markdown("### 🎯 Your Behavioral Scores")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("🤝 Social Engagement", f"{features['Social_Engagement']:.2f}")
                st.caption("Your outward social activity level")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("🏠 Introversion Tendency", f"{features['Introversion_Tendency']:.2f}")
                st.caption("Your preference for solitary activities")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                balance = features['Social_Solitary_Balance']
                st.metric("⚖️ Social-Solitary Balance", f"{balance:.2f}")
                st.caption("The key factor in your personality type")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional insights
            if abs(balance) < 0.3:
                st.info("🎭 **Interesting!** You show balanced traits between both personality types. You might be an Ambivert - someone who has both extroverted and introverted tendencies!")
            
            if confidence < 0.7:
                st.warning("🤔 **Note:** The confidence is moderate. You might want to retake the test or consider that you have a mix of both personality traits.")
        
        else:
            st.error(f"❌ Prediction failed: {result['error']}")
    
    # Footer information
    st.markdown("---")
    st.markdown("### ℹ️ About This Predictor")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎯 Accuracy**
        - 91% accurate on test data
        - Based on 2,512 real behavioral records
        - Uses Random Forest algorithm
        """)
    
    with col2:
        st.markdown("""
        **🔍 Key Insights**
        - Social-Solitary Balance is most important (54.6%)
        - Social Engagement patterns matter (43.5%)
        - Friend circle size influences prediction (0.6%)
        """)
    
    with col3:
        st.markdown("""
        **⚠️ Important**
        - Results are for self-reflection
        - Personality exists on a spectrum
        - Everyone has mixed traits
        - Consider retaking if confidence < 70%
        """)

if __name__ == "__main__":
    main()
