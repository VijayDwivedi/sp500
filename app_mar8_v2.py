import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Configure the page settings for full width
st.set_page_config(
    page_title="SP500 Flag Pattern Analysis",
    page_icon="📊",
    layout="wide",  # Set layout to wide for full-width coverage
)



# Title Section with Enhanced Styling
st.markdown("""
    <div style='text-align: center;'>
        <h1>📊 SP500 Flag Pattern Analysis 📈</h1>
        <h4>🎯 Enhancing Stock Market Insights with S&P 500 Bullish Pattern</h4>
    </div>
    """, unsafe_allow_html=True)

# Sidebar Styling
st.sidebar.markdown("""
    <div style='text-align: center;'>
        <h3>🎯 S&P 500 Analysis</h3>
        <h4>📝 Fill Out the Details Below</h4>
    </div>
    """, unsafe_allow_html=True)


def get_sp500_data():
    # Add slider for days in sidebar
#    st.sidebar.markdown("### **S&P 500 Analysis**")
#    st.sidebar.markdown("### **Fill Out the side Bar Detail**")
    days = st.sidebar.slider('📅 Select Years', 
                           min_value=1, 
                           max_value=20, 
                           value=1,
                           help='Number of years of historical data to analyze')
    
 
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*days)
    sp500 = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
    return sp500


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)

def is_strong_flag(data, flag_end_date):
    rsi = compute_rsi(data['Close'])
    if flag_end_date in rsi.index:
        return rsi.loc[flag_end_date].iloc[0] > 50
    return False

def is_above_ma50(data, flag_end_date):
    ma50 = data['Close'].rolling(window=50).mean()
    if flag_end_date in ma50.index:
        return data.loc[flag_end_date, 'Close'].iloc[0] > ma50.loc[flag_end_date].iloc[0]
    return False

def detect_flag_patterns(data, pole_threshold=0.015, flag_threshold=0.025, consolidation_days=10):
    flags = []
    
    for i in range(10, len(data) - consolidation_days - 1):  
        # Calculate price change over the pole period
        current_close = data['Close'].iloc[i].item()
        past_close = data['Close'].iloc[i-10].item()
        price_change = (current_close - past_close) / past_close
        
        # Identify a flagpole (bullish or bearish)
        if abs(price_change) > pole_threshold:
            trend_direction = 'bullish' if price_change > 0 else 'bearish'
            
            consolidation_period = data.iloc[i:i + consolidation_days]
            high_price = consolidation_period['High'].max().item()
            low_price = consolidation_period['Low'].min().item()
            price_range = (high_price - low_price) / low_price
            
            # Ensure consolidation range is within threshold
            if price_range < flag_threshold:
                consolidation_prices = consolidation_period['Close'].values.astype(float)
                flag_slope = np.polyfit(range(len(consolidation_prices)), consolidation_prices, 1)[0]
                
                # Different criteria for bullish and bearish patterns
                if trend_direction == 'bullish':
                    # Stricter criteria for bullish patterns
                    if flag_slope <= 0:
                        flag_end_date = data.index[i + consolidation_days]
                        if is_strong_flag(data, flag_end_date) and is_above_ma50(data, flag_end_date):
                            flags.append({
                                'date': data.index[i],
                                'pole_start': data.index[i-10],
                                'flag_end': flag_end_date,
                                'pole_price_change': price_change,
                                'consolidation_range': price_range,
                                'trend_direction': trend_direction
                            })
                else:
                    # More relaxed criteria for bearish patterns
                    if flag_slope > 0:
                        flag_end_date = data.index[i + consolidation_days]
                        # Remove RSI and MA50 constraints for bearish patterns
                        flags.append({
                            'date': data.index[i],
                            'pole_start': data.index[i-10],
                            'flag_end': flag_end_date,
                            'pole_price_change': price_change,
                            'consolidation_range': price_range,
                            'trend_direction': trend_direction
                        })
    
    return flags

# [Keep all other functions the same until calculate_success_probability]


def calculate_success_probability(data, flags, threshold=0.0001):
    st.sidebar.markdown("### 📈 Profit Analysis post Bullish Flag formation")
    days_after = st.sidebar.selectbox('📊 Select Analysis Period', 
                             options=[1, 7, 14, 21, 30],
                             help='Number of days after pattern to analyze success')
    if not flags:
        st.warning("No flag patterns detected.")
        return 0.0, 0.0, pd.DataFrame()

    bullish_success = 0
    bearish_success = 0
    valid_bullish = 0
    valid_bearish = 0
    pattern_details = []

    # List to store pattern details
    pattern_details = []

    for flag in flags:
        flag_end_date = flag['flag_end']
        if flag_end_date not in data.index:
            continue

        flag_end_idx = data.index.get_loc(flag_end_date)
        if flag_end_idx + days_after >= len(data):
            continue

        future_price = data['Close'].iloc[flag_end_idx + days_after].iloc[0]
        flag_end_price = data.loc[flag_end_date, 'Close'].iloc[0]
        
        price_change = (future_price - flag_end_price) / flag_end_price
        
        # Determine if pattern was successful
        success = False
        if flag['trend_direction'] == 'bullish':
            if price_change >= threshold:
                bullish_success += 1
                success = True
            valid_bullish += 1
        else:  # bearish
            if price_change <= -threshold:
                bearish_success += 1
                success = True
            valid_bearish += 1

        # Add pattern details to list
        pattern_details.append({
            'Pattern Start Date': flag['pole_start'].strftime('%Y-%m-%d'),
            'Pole End/Flag Start Date': flag['date'].strftime('%Y-%m-%d'),
            'Flag End Date': flag['flag_end'].strftime('%Y-%m-%d'),
            'Flag Type': flag['trend_direction'].capitalize(),
            'Price Change (%)': round(price_change * 100, 2),
            'Success': success,
            'Days After': days_after,
            'Threshold (%)': round(threshold * 100, 2)
        })

    bullish_prob = bullish_success / valid_bullish if valid_bullish > 0 else 0
    bearish_prob = bearish_success / valid_bearish if valid_bearish > 0 else 0


    # Create DataFrame from pattern details
    df_patterns = pd.DataFrame(pattern_details)
    gain = df_patterns['Price Change (%)'].sum()
    
 
# Alternative version with different styling
    st.markdown(f"""
        <div style='padding: 20px; border-left: 5px solid #1f77b4; background-color: white; border-radius: 5px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);'>
            <h1 style='color: #1f77b4; text-align: center;'>Pattern Performance 📈</h1>
            <div style='text-align: center;'>
                <p style='font-size: 28px; margin: 10px 0;'>
                    Bullish Flag Success Rate after <span style='color: #1f77b4;'>{days_after}</span> days
                </p>
                <p style='font-size: 48px; font-weight: bold; color: #2e7d32; margin: 20px 0;'>
                    {bullish_prob*100:.2f}%
                </p>
                <p style='font-size: 24px; color: #666;'>
                    {bullish_success} successful patterns out of {valid_bullish} total
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # You can also add a metrics display below
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Success Rate",
            value=f"{bullish_prob*100:.2f}%",
            delta=f"{bullish_success} successful patterns"
        )
    with col2:
        st.metric(
            label="Total Patterns",
            value=f"{valid_bullish}",
            delta="analyzed patterns"
        )
    with col3:
        st.metric(
            label="Analysis Period",
            value=f"{days_after} days",
            delta="after pattern formation"
        )





    # Enhanced Summary Section
    st.markdown("""
        <div style='background-color: #ffffff; padding: 25px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin: 20px 0;'>
            <h1 style='color: #1f77b4; text-align: center; margin-bottom: 20px;'>
                📊 Investment Summary
            </h1>
            <div style='text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 8px;'>
                <p style='font-size: 20px; color: #444; line-height: 1.6;'>
                    Investment Analysis in S&P 500 stocks following Bullish Flag Pattern identification:
                </p>
                <p style='font-size: 18px; color: #666; margin-top: 10px;'>
                    Holding Period: <span style='color: #1f77b4; font-weight: bold;'>{} Days</span>
                </p>
            </div>
        </div>
    """.format(days_after), unsafe_allow_html=True)

    # Create columns for metrics
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        # Determine color and icon based on gain value
        color = "#2e7d32" if gain > 0 else "#d32f2f" if gain < 0 else "#666666"
        icon = "📈" if gain > 0 else "📉" if gain < 0 else "➡️"
        trend = "Profit" if gain > 0 else "Loss" if gain < 0 else "Neutral"
        
        # Display enhanced metric
        st.markdown(f"""
            <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); text-align: center;'>
                <h3 style='color: #666; margin-bottom: 10px;'>Overall Performance</h3>
                <div style='font-size: 48px; font-weight: bold; color: {color}; margin: 20px 0;'>
                    {icon} {gain:.2f}%
                </div>
                <p style='font-size: 20px; color: {color};'>
                    {trend}
                </p>
            </div>
        """, unsafe_allow_html=True)

    # Additional Analysis Details
    st.markdown("""
        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-top: 20px;'>
            <h4 style='color: #666; text-align: center;'>Analysis Details</h4>
            <div style='display: flex; justify-content: center; gap: 20px; margin-top: 15px;'>
                <div style='text-align: center;'>
                    <p style='font-size: 16px; color: #666;'>Pattern Type</p>
                    <p style='font-size: 18px; color: #1f77b4;'>Bullish Flag</p>
                </div>
                <div style='text-align: center;'>
                    <p style='font-size: 16px; color: #666;'>Analysis Period</p>
                    <p style='font-size: 18px; color: #1f77b4;'>{} Days</p>
                </div>
                <div style='text-align: center;'>
                    <p style='font-size: 16px; color: #666;'>Market Index</p>
                    <p style='font-size: 18px; color: #1f77b4;'>S&P 500</p>
                </div>
            </div>
        </div>
    """.format(days_after), unsafe_allow_html=True)

    # Add performance insights based on gain value
    if gain > 0:
        st.success(f"📈 Positive return of {gain:.2f}% observed over {days_after} days holding period")
    elif gain < 0:
        st.error(f"📉 Negative return of {gain:.2f}% observed over {days_after} days holding period")
    else:
        st.info("➡️ No significant change in value observed")

    # Optional: Add a performance breakdown
    st.markdown("""
        <div style='margin-top: 20px; padding: 15px; background-color: white; border-radius: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);'>
            <h4 style='color: #666; text-align: center;'>Performance Insights</h4>
            <ul style='list-style-type: none; padding: 0;'>
                <li style='margin: 10px 0; text-align: center;'>
                    🎯 Pattern identification success rate indicates market trend reliability
                </li>
                <li style='margin: 10px 0; text-align: center;'>
                    ⏱️ {days_after}-day holding period shows optimal pattern completion
                </li>
                <li style='margin: 10px 0; text-align: center;'>
                    📊 Results based on historical S&P 500 price action
                </li>
            </ul>
        </div>
    """, unsafe_allow_html=True)





















    
    #st.write(f"Bearish Success Rate: {bearish_prob*100:.2f}% ({bearish_success}/{valid_bearish})")
    
    return bullish_prob, bearish_prob, df_patterns

def main():
    sp500_data = get_sp500_data()
    flags = detect_flag_patterns(sp500_data)

    # Calculate and display success probability
    bullish_prob, bearish_prob, pattern_df = calculate_success_probability(sp500_data, flags)

    # Display pattern details in expander
    with st.expander("View Pattern Details and downloading the data"):
        st.write("All Patterns:")
        pattern_df['Pattern Start Date'] = pd.to_datetime(pattern_df['Pattern Start Date']).dt.strftime('%d-%m-%Y')
        pattern_df['Flag End Date'] = pd.to_datetime(pattern_df['Flag End Date']).dt.strftime('%d-%m-%Y')
        pattern_df['Pole End/Flag Start Date'] = pd.to_datetime(pattern_df['Pole End/Flag Start Date']).dt.strftime('%d-%m-%Y')
        st.dataframe(pattern_df)
        # Add download button
        csv = pattern_df.to_csv(index=False)
        st.download_button(
            label="Download Pattern Data as CSV",
            data=csv,
            file_name="flag_patterns.csv",
            mime="text/csv"
        )
   
    
    st.markdown("## 📈 Flag Pattern Visualization")
    flags = detect_flag_patterns(sp500_data)

    # Plot flags
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(sp500_data.index, sp500_data['Close'], label='S&P 500', color='blue', alpha=0.6)

    # Format x-axis dates
    import matplotlib.dates as mdates
    date_format = mdates.DateFormatter('%d-%m-%Y')
    ax.xaxis.set_major_formatter(date_format)

    # Rotate and align the tick labels so they look better
    plt.gcf().autofmt_xdate()  # Auto-rotate and align the tick labels

    for flag in flags:
        pole_period = sp500_data.loc[flag['pole_start']:flag['date']]
        flag_period = sp500_data.loc[flag['date']:flag['flag_end']]
        
        color = 'g-' if flag['trend_direction'] == 'bullish' else 'r-'
        ax.plot(pole_period.index, pole_period['Close'], color, linewidth=2, 
                label=f"{flag['trend_direction'].capitalize()} Pole")
        ax.plot(flag_period.index, flag_period['Close'], color.replace("-", "--"), 
                linewidth=2, label=f"{flag['trend_direction'].capitalize()} Flag")

    plt.title('S&P 500 with Flag Patterns')
    plt.xlabel('Date')
    plt.ylabel('Price, $')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.grid(True)

    # Adjust layout to prevent date labels from being cut off
    plt.tight_layout()

    st.pyplot(fig)
        

if __name__ == "__main__":
    main()




# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h3>⚠️ Disclaimer</h3>
        <p>This application is for informational purposes only and does not constitute financial advice.</p>
        <p>🔍 Use the data to complement your own research and analysis.</p>
        <p>⚖️ No guarantees are made regarding trading performance.</p>
        <br>
        <p>🏢 Powered by Advanced Financial Analytics</p>
        <p style='font-size: 0.8em;'>📊 Data sourced from Yahoo Finance</p>
        <p style='font-size: 0.8em;'>© 2023 SP500 Flag Pattern Analysis</p>
    </div>
    """, unsafe_allow_html=True)

