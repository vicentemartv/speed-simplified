import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
import json
from datetime import datetime, timedelta, timezone
import pytz
import matplotlib.dates as mdates
import time
import calendar

# Set page configuration
st.set_page_config(
    page_title="Machine Speed Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stMetric {
        background-color: #EEF2FF;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .stMetric label {
        color: #1E3A8A;
        font-weight: bold;
    }
    /* Style for the metrics in top/bottom listings */
    .metric-card {
        background-color: white; 
        padding: 10px; 
        border-radius: 5px; 
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    /* Stats card for insights */
    .stats-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stats-card h4 {
        margin-top: 0;
        color: #1E3A8A;
        font-size: 16px;
    }
    .stats-value {
        font-size: 24px;
        font-weight: bold;
        color: #1E40AF;
    }
    .stats-label {
        font-size: 14px;
        color: #6B7280;
    }
    .fastest {
        color: #047857;  /* Green for fastest */
    }
    .slowest {
        color: #B91C1C;  /* Red for slowest */
    }
    .stProgress > div > div > div {
        height: 10px;
    }
    .insight-title {
        font-size: 18px;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 10px;
    }
    .loaded-day {
        background-color: #D1FAE5;
        border-radius: 4px;
        padding: 5px 10px;
        margin-right: 5px;
        margin-bottom: 5px;
        font-weight: bold;
        display: inline-block;
    }
    .pending-day {
        background-color: #E5E7EB;
        border-radius: 4px;
        padding: 5px 10px;
        margin-right: 5px;
        margin-bottom: 5px;
        display: inline-block;
    }
    .failed-day {
        background-color: #FEE2E2;
        border-radius: 4px;
        padding: 5px 10px;
        margin-right: 5px;
        margin-bottom: 5px;
        display: inline-block;
    }
    .progress-container {
        border: 1px solid #E5E7EB;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 15px;
    }
    .progress-title {
        font-weight: bold;
        margin-bottom: 5px;
    }
    .day-status {
        margin-bottom: 10px;
    }
    /* Continue button styling */
    .continue-button {
        background-color: #0EA5E9;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 20px;
        font-weight: bold;
        cursor: pointer;
    }
    /* Loading indicator styling */
    .loading-indicator {
        display: flex;
        align-items: center;
        margin-top: 10px;
        padding: 10px;
        background-color: #F0F9FF;
        border-radius: 5px;
        border-left: 4px solid #0EA5E9;
    }
    .loading-spinner {
        margin-right: 10px;
        color: #0EA5E9;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("Machine Speed Analyzer")
st.markdown("### Daily Data Collection & Speed Analysis")

# Define Houston timezone
houston_tz = pytz.timezone('America/Chicago')  # Central Time (Houston)

# =============== Helper Functions ===============

def format_datetime(dt):
    """Format datetime in DD-MMM HH:MM:SS format using Houston timezone"""
    if pd.isna(dt):
        return "N/A"
    
    # Convert to Houston time if it's timezone aware
    if dt.tzinfo is not None:
        dt = dt.astimezone(houston_tz)
    
    return dt.strftime("%d-%b %H:%M:%S")

def format_date(dt):
    """Format date in DD-MMM format"""
    if pd.isna(dt):
        return "N/A"
    
    # Handle datetime or date objects
    if hasattr(dt, 'strftime'):
        return dt.strftime("%d-%b")
    return str(dt)

def determine_shift(timestamp):
    """
    Determine shift based on time of day:
    1st shift: 6AM to 4PM
    2nd shift: 8PM to 6AM
    Returns "1st", "2nd", or "None"
    """
    # Convert to Houston time
    if timestamp.tzinfo is not None:
        time_local = timestamp.astimezone(houston_tz)
    else:
        time_local = timestamp
    
    hour = time_local.hour
    
    if 6 <= hour < 16:  # 6AM to 4PM
        return "1st"
    elif hour >= 20 or hour < 6:  # 8PM to 6AM
        return "2nd"
    else:
        return "None"

def get_week_number(date):
    """Get week number for a date in ISO format (year-week)"""
    return date.strftime('%Y-W%U')

def get_month_name(date):
    """Get month name for a date"""
    return date.strftime('%b %Y')

# Function to get list of days in a month, excluding weekends
def get_days_in_month(year, month):
    """Get list of dates for all weekdays in the specified month (Mon-Fri only)"""
    # Get the number of days in the month
    _, num_days = calendar.monthrange(year, month)
    
    # Generate list of date objects, excluding weekends
    days = []
    for day in range(1, num_days + 1):
        date = datetime(year, month, day, tzinfo=houston_tz)
        # Check if it's a weekday (0=Monday, 6=Sunday)
        if date.weekday() < 5:  # Only weekdays (Mon-Fri)
            days.append(date)
    
    return days

# Function to get weekdays in a week
def get_days_in_week(start_date):
    """Get list of weekdays (Mon-Fri) starting from start_date for up to 7 days"""
    days = []
    # Check up to 7 days to find weekdays
    for i in range(7):
        day = start_date + timedelta(days=i)
        if day.weekday() < 5:  # Only weekdays (Mon-Fri)
            days.append(day)
    return days

# Function to check if a timestamp is during a valid shift
def is_valid_shift_time(timestamp):
    """
    Check if the timestamp falls within defined shift hours:
    1st shift: 6AM to 4PM
    2nd shift: 8PM to 6AM
    Returns True if within shift hours, False otherwise
    """
    # Convert to Houston time
    if timestamp.tzinfo is not None:
        time_local = timestamp.astimezone(houston_tz)
    else:
        time_local = timestamp
    
    # Extract hour
    hour = time_local.hour
    
    # 1st shift: 6AM to 4PM (6-16)
    if 6 <= hour < 16:
        return True
    
    # 2nd shift: 8PM to 6AM (20-23, 0-6)
    if hour >= 20 or hour < 6:
        return True
    
    # Not in a shift
    return False

# Function to get start and end datetime for a given day
def get_day_range(day_date):
    """Return start and end datetime for a single day"""
    # Start of day
    start_date = day_date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # End of day
    end_date = day_date.replace(hour=23, minute=59, second=59, microsecond=999999)
    
    return start_date, end_date

# =============== GraphQL Query Functions ===============

def fetch_machines(api_endpoint, api_key):
    """Fetch all available machines with their names and refs"""
    query = """
    query GetMachines {
        machines(where: {decommissionedAt: {_is_null: true}}) {
            name
            machineRef
        }
    }
    """
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        response = requests.post(
            api_endpoint,
            headers=headers,
            json={"query": query}
        )
        
        if response.status_code != 200:
            st.error(f"Error fetching machines: {response.text}")
            return []
            
        data = response.json()
        
        if "data" in data and "machines" in data["data"]:
            return data["data"]["machines"]
        else:
            st.error("No machine data returned from the API")
            return []
    except Exception as e:
        st.error(f"Error fetching machines: {str(e)}")
        return []

def fetch_speed_data(api_endpoint, api_key, machine_ref, start_time, end_time, metric_name="speed"):
    """Fetch speed data for a specific machine and time window"""
    query = """
    query SpeedQuery {
        quantities(
            args: {machineRef: %s, metricKey: "%s", windowEndAt: "%s", windowStartAt: "%s"}
        ) {
            eventTime
            machineRef
            metricKey
            sequence
            value
        }
    }
    """ % (machine_ref, metric_name, end_time, start_time)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        response = requests.post(
            api_endpoint,
            headers=headers,
            json={"query": query}
        )
        
        if response.status_code != 200:
            return pd.DataFrame(), f"Error: {response.text}"
            
        data = response.json()
        
        if "data" in data and "quantities" in data["data"]:
            # Convert to DataFrame
            df = pd.DataFrame(data["data"]["quantities"])
            
            if len(df) > 0:
                # Convert to proper types - using 'coerce' to handle various ISO formats
                df['eventTime'] = pd.to_datetime(df['eventTime'], format='ISO8601')
                df['value_numeric'] = pd.to_numeric(df['value'], errors='coerce')
                
                # Add date components for grouping - convert to Houston time first
                df['eventTime_local'] = df['eventTime'].dt.tz_convert(houston_tz)
                df['date'] = df['eventTime_local'].dt.date
                df['date_str'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
                df['hour'] = df['eventTime_local'].dt.hour
                df['day_of_week'] = df['eventTime_local'].dt.day_name()
                df['week'] = df['eventTime_local'].dt.isocalendar().week
                df['year_week'] = df['eventTime_local'].dt.strftime('%Y-W%U')
                df['month'] = df['eventTime_local'].dt.strftime('%B %Y')
                
                # Determine shift
                df['shift'] = df['eventTime_local'].apply(determine_shift)
            
            return df, f"Found {len(df)} speed records"
        else:
            return pd.DataFrame(), "No speed data returned from API"
    except Exception as e:
        return pd.DataFrame(), f"Error: {str(e)}"

def fetch_execution_data(api_endpoint, api_key, machine_ref, start_time, end_time):
    """Fetch execution state intervals for a specific machine and time window"""
    query = """
    query ExecutionQuery {
        stateIntervals(
            args: {machineRef: %s, metricKey: "execution", windowEndAt: "%s", windowStartAt: "%s"}
        ) {
            startAt
            endAt
            machineRef
            metricKey
            value
        }
    }
    """ % (machine_ref, end_time, start_time)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        response = requests.post(
            api_endpoint,
            headers=headers,
            json={"query": query}
        )
        
        if response.status_code != 200:
            return pd.DataFrame(), f"Error: {response.text}"
            
        data = response.json()
        
        if "data" in data and "stateIntervals" in data["data"]:
            # Convert to DataFrame
            df = pd.DataFrame(data["data"]["stateIntervals"])
            
            if len(df) > 0:
                # Convert to proper types - using 'coerce' to handle various ISO formats
                df['startAt'] = pd.to_datetime(df['startAt'], format='ISO8601')
                df['endAt'] = pd.to_datetime(df['endAt'], format='ISO8601')
                
                # Convert to local time for display
                df['startAt_local'] = df['startAt'].dt.tz_convert(houston_tz)
                df['endAt_local'] = df['endAt'].dt.tz_convert(houston_tz)
                
                # Calculate duration
                df['duration_seconds'] = (df['endAt'] - df['startAt']).dt.total_seconds()
                
                # Add month info
                df['month'] = df['startAt_local'].dt.strftime('%B %Y')
            
            return df, f"Found {len(df)} execution intervals"
        else:
            return pd.DataFrame(), "No execution data returned from API"
    except Exception as e:
        return pd.DataFrame(), f"Error: {str(e)}"

def fetch_day_data(api_endpoint, api_key, machine_ref, day_date, metric_name="speed"):
    """
    Fetch data for a specific day
    Returns processed and filtered dataframe of speed data during ACTIVE periods
    Only includes data during valid shift hours (1st: 6AM-4PM, 2nd: 8PM-6AM)
    """
    try:
        # Check if this is a weekend
        if day_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
            return pd.DataFrame(columns=['eventTime', 'eventTime_local', 'value', 'value_numeric', 'period_id']), pd.DataFrame(), f"Skipping weekend: {day_date.strftime('%Y-%m-%d')}"
        
        # Get start and end datetimes for the day
        start_date, end_date = get_day_range(day_date)
        
        # Convert to ISO format
        start_iso = start_date.isoformat()
        end_iso = end_date.isoformat()
        
        # Format day for display
        day_display = day_date.strftime('%Y-%m-%d')
        
        # Fetch execution data
        execution_df, exec_message = fetch_execution_data(api_endpoint, api_key, machine_ref, start_iso, end_iso)
        
        if len(execution_df) == 0:
            return pd.DataFrame(columns=['eventTime', 'eventTime_local', 'value', 'value_numeric', 'period_id']), pd.DataFrame(), f"No execution data for {day_display}"
        
        # Fetch speed data with the specified metric name
        speed_df, speed_message = fetch_speed_data(api_endpoint, api_key, machine_ref, start_iso, end_iso, metric_name)
        
        if len(speed_df) == 0:
            return pd.DataFrame(columns=['eventTime', 'eventTime_local', 'value', 'value_numeric', 'period_id']), execution_df, f"No speed data for {day_display}"
        
        # Ensure necessary columns exist in speed_df
        if 'value' not in speed_df.columns:
            speed_df['value'] = None
            
        if 'value_numeric' not in speed_df.columns:
            try:
                speed_df['value_numeric'] = pd.to_numeric(speed_df['value'], errors='coerce')
            except Exception as e:
                speed_df['value_numeric'] = np.nan
                st.warning(f"Error converting speed values to numeric for {day_display}: {str(e)}")
        
        # ----- ACTIVE PERIOD IDENTIFICATION AND SHIFT FILTERING -----
        
        # Step 1: Make a clean copy of our execution intervals for ACTIVE periods only
        active_periods = execution_df[execution_df['value'] == 'ACTIVE'].copy()
        
        if len(active_periods) == 0:
            return pd.DataFrame(columns=['eventTime', 'eventTime_local', 'value', 'value_numeric', 'period_id']), execution_df, f"No ACTIVE periods for {day_display}"
        
        # Ensure timestamps are standardized
        active_periods['startAt'] = pd.to_datetime(active_periods['startAt'])
        active_periods['endAt'] = pd.to_datetime(active_periods['endAt'])
        speed_df['eventTime'] = pd.to_datetime(speed_df['eventTime'])
        
        # Step 2: Filter active periods to only include those during valid shift hours
        valid_active_periods = []
        for idx, period in active_periods.iterrows():
            start_time = period['startAt']
            end_time = period['endAt']
            
            # Check if the period is at least partially during a valid shift
            start_in_shift = is_valid_shift_time(start_time)
            end_in_shift = is_valid_shift_time(end_time)
            
            # If either start or end is in a shift, include the period but trim it to shift hours
            if start_in_shift or end_in_shift:
                # For now, include the whole period
                valid_active_periods.append(period)
        
        # Use filtered periods if any, otherwise return empty
        if not valid_active_periods:
            return pd.DataFrame(columns=['eventTime', 'eventTime_local', 'value', 'value_numeric', 'period_id']), execution_df, f"No ACTIVE periods during shift hours for {day_display}"
        
        # Convert valid periods back to DataFrame
        valid_active_periods_df = pd.DataFrame(valid_active_periods)
        
        # Initialize debug info
        st.session_state.debug_info = {
            'active_periods_count': len(valid_active_periods_df),
            'total_speed_measurements': len(speed_df),
            'active_periods': [],
            'period_stats': []
        }
        
        # Step 3: Collect all speed measurements that occurred during ACTIVE periods in valid shift hours
        active_speeds = pd.DataFrame()
        
        for idx, period in valid_active_periods_df.iterrows():
            try:
                # Extract start and end times for this ACTIVE period
                start_time = period['startAt']
                end_time = period['endAt']
                
                # Calculate period duration in minutes for debugging
                duration_mins = (end_time - start_time).total_seconds() / 60
                
                # Find speed measurements within this ACTIVE period
                period_speeds = speed_df[
                    (speed_df['eventTime'] >= start_time) & 
                    (speed_df['eventTime'] <= end_time)
                ].copy()
                
                # Additional filter: Only include measurements during valid shift hours
                period_speeds = period_speeds[period_speeds['eventTime'].apply(is_valid_shift_time)]
                
                # Ensure necessary columns exist in period_speeds
                for col in ['value', 'value_numeric']:
                    if col not in period_speeds.columns and col in speed_df.columns:
                        period_speeds[col] = speed_df[col]
                    elif col not in period_speeds.columns:
                        period_speeds[col] = None
                
                # Create value_numeric if it doesn't exist but value does
                if 'value_numeric' not in period_speeds.columns and 'value' in period_speeds.columns:
                    try:
                        period_speeds['value_numeric'] = pd.to_numeric(period_speeds['value'], errors='coerce')
                    except:
                        period_speeds['value_numeric'] = None
                
                # Count valid measurements (non-zero, non-null) - safely handle columns
                valid_speeds = pd.DataFrame()
                if len(period_speeds) > 0 and 'value_numeric' in period_speeds.columns:
                    try:
                        valid_speeds = period_speeds[
                            (period_speeds['value_numeric'] > 0) & 
                            (period_speeds['value_numeric'].notna())
                        ]
                        period_avg = valid_speeds['value_numeric'].mean() if len(valid_speeds) > 0 else 0
                    except Exception as e:
                        st.warning(f"Error filtering valid speeds: {str(e)}")
                        period_avg = 0
                else:
                    period_avg = 0
                
                # Add period info to debug data
                period_info = {
                    'start': start_time.strftime('%H:%M:%S'),
                    'end': end_time.strftime('%H:%M:%S'),
                    'duration_mins': round(duration_mins, 2),
                    'total_measurements': len(period_speeds),
                    'valid_measurements': len(valid_speeds),
                    'avg_speed': round(period_avg, 2) if not pd.isna(period_avg) else 0
                }
                
                st.session_state.debug_info['active_periods'].append({
                    'start': start_time.strftime('%H:%M:%S'),
                    'end': end_time.strftime('%H:%M:%S'),
                    'duration_mins': round(duration_mins, 2)
                })
                
                st.session_state.debug_info['period_stats'].append(period_info)
                
                # Only use periods with valid speed measurements
                if len(valid_speeds) > 0:
                    # Add a period identifier for debugging
                    valid_speeds['period_id'] = idx
                    active_speeds = pd.concat([active_speeds, valid_speeds])
            except Exception as e:
                st.warning(f"Error processing period {idx}: {str(e)}")
        
        # Ensure unique records and necessary columns
        result_df = pd.DataFrame(columns=['eventTime', 'eventTime_local', 'value', 'value_numeric', 'period_id'])
        
        if len(active_speeds) > 0:
            try:
                # Make sure all required columns exist
                for col in ['eventTime', 'eventTime_local', 'value', 'value_numeric', 'period_id']:
                    if col not in active_speeds.columns:
                        if col == 'eventTime_local' and 'eventTime' in active_speeds.columns:
                            active_speeds[col] = active_speeds['eventTime'].dt.tz_convert(houston_tz)
                        elif col == 'value_numeric' and 'value' in active_speeds.columns:
                            active_speeds[col] = pd.to_numeric(active_speeds['value'], errors='coerce')
                        else:
                            active_speeds[col] = None
                
                # Drop duplicates and update the result_df
                active_speeds = active_speeds.drop_duplicates(subset=['eventTime', 'value'])
                result_df = active_speeds.copy()
                
                # Update debug info
                st.session_state.debug_info['active_speed_measurements'] = len(active_speeds)
                
                # Calculate average speed for debugging - safely check columns
                if 'value_numeric' in active_speeds.columns:
                    try:
                        valid_speeds = active_speeds[
                            (active_speeds['value_numeric'] > 0) & 
                            (active_speeds['value_numeric'].notna())
                        ]
                        
                        if len(valid_speeds) > 0:
                            avg_speed = valid_speeds['value_numeric'].mean()
                            st.session_state.debug_info['valid_speed_count'] = len(valid_speeds)
                            st.session_state.debug_info['avg_valid_speed'] = round(avg_speed, 2)
                    except Exception as e:
                        st.warning(f"Error calculating average speed: {str(e)}")
                
                # Add a note about shift filtering
                shift_msg = " (filtered by shift hours: 6AM-4PM, 8PM-6AM)"
                return result_df, valid_active_periods_df, f"Success: {len(result_df)} speed measurements during ACTIVE periods for {day_display}{shift_msg}"
            except Exception as e:
                st.warning(f"Error in final processing: {str(e)}")
                return result_df, valid_active_periods_df, f"Error processing data for {day_display}: {str(e)}"
        else:
            # Return empty DataFrame with necessary columns
            return result_df, valid_active_periods_df, f"No valid speed measurements during ACTIVE periods in shift hours for {day_display}"
    except Exception as e:
        # Global error handling
        st.error(f"Unexpected error processing {day_date.strftime('%Y-%m-%d')}: {str(e)}")
        # Return empty DataFrame with required columns
        return pd.DataFrame(columns=['eventTime', 'eventTime_local', 'value', 'value_numeric', 'period_id']), pd.DataFrame(), f"Error: {str(e)}"

def calculate_avg_speeds(speed_df):
    """
    Calculate average speeds by day, week, month, shift, and hour
    Returns a tuple of dataframes (daily, weekly, monthly, shift, hourly)
    """
    try:
        if len(speed_df) == 0:
            # Create empty DataFrames with required columns to avoid errors downstream
            empty_df = pd.DataFrame(columns=['date', 'avg_speed', 'count', 'weighted_avg', 'date_str', 'date_formatted'])
            empty_hour_df = pd.DataFrame(columns=['hour', 'avg_speed', 'count', 'weighted_avg', 'hour_str'])
            return empty_df, empty_df, empty_df, empty_df, empty_hour_df
        
        # Filter out zero, null, and negative values for more accurate statistics
        try:
            valid_speed_df = speed_df[
                (speed_df['value_numeric'] > 0) & 
                (speed_df['value_numeric'].notna())
            ].copy()
        except:
            # If filtering fails, create an empty DataFrame
            st.warning("Error filtering valid speed values. Check data format.")
            empty_df = pd.DataFrame(columns=['date', 'avg_speed', 'count', 'weighted_avg', 'date_str', 'date_formatted'])
            empty_hour_df = pd.DataFrame(columns=['hour', 'avg_speed', 'count', 'weighted_avg', 'hour_str'])
            return empty_df, empty_df, empty_df, empty_df, empty_hour_df
        
        # If nothing remains after filtering, return empty DataFrames
        if len(valid_speed_df) == 0:
            empty_df = pd.DataFrame(columns=['date', 'avg_speed', 'count', 'weighted_avg', 'date_str', 'date_formatted'])
            empty_hour_df = pd.DataFrame(columns=['hour', 'avg_speed', 'count', 'weighted_avg', 'hour_str'])
            return empty_df, empty_df, empty_df, empty_df, empty_hour_df
        
        # Add hour column for hourly analysis (ensures it's an integer)
        try:
            if 'eventTime_local' in valid_speed_df.columns:
                valid_speed_df['hour'] = valid_speed_df['eventTime_local'].dt.hour
                # Ensure hour_str is correctly formatted - convert to int first to avoid float formatting issues
                valid_speed_df['hour_str'] = valid_speed_df['hour'].apply(
                    lambda x: f"{int(x):02d}:00" if pd.notna(x) else "00:00"
                )
            else:
                # If eventTime_local is missing, add placeholder columns
                valid_speed_df['hour'] = 0
                valid_speed_df['hour_str'] = "00:00"
        except Exception as e:
            st.warning(f"Error creating hour columns: {str(e)}")
            valid_speed_df['hour'] = 0
            valid_speed_df['hour_str'] = "00:00"
        
        # Calculate weighted average (for debug purposes)
        try:
            # Assume each measurement is valid until the next measurement
            valid_speed_df = valid_speed_df.sort_values('eventTime')
            valid_speed_df['next_time'] = valid_speed_df['eventTime'].shift(-1)
            
            # For the last row, assume a duration of 1 minute
            # Using iloc to avoid ambiguity with loc when duplicate indices exist
            if len(valid_speed_df) > 0:
                valid_speed_df.iloc[-1, valid_speed_df.columns.get_loc('next_time')] = valid_speed_df.iloc[-1]['eventTime'] + pd.Timedelta(minutes=1)
            
            # Calculate duration in seconds
            valid_speed_df['duration_seconds'] = (valid_speed_df['next_time'] - valid_speed_df['eventTime']).dt.total_seconds()
            
            # Cap durations at 5 minutes to prevent extreme values
            valid_speed_df['duration_seconds'] = valid_speed_df['duration_seconds'].clip(upper=300)
            
            # Calculate weighted value
            valid_speed_df['weighted_value'] = valid_speed_df['value_numeric'] * valid_speed_df['duration_seconds']
            
            # Update debug info with weighted average
            total_duration = valid_speed_df['duration_seconds'].sum()
            weighted_avg = valid_speed_df['weighted_value'].sum() / total_duration if total_duration > 0 else 0
            
            if 'debug_info' in st.session_state:
                st.session_state.debug_info['weighted_avg_speed'] = round(weighted_avg, 2)
                st.session_state.debug_info['total_duration_seconds'] = round(total_duration, 2)
        except Exception as e:
            st.warning(f"Error calculating weighted values: {str(e)}")
            valid_speed_df['duration_seconds'] = 60  # Default 1-minute duration
            valid_speed_df['weighted_value'] = valid_speed_df['value_numeric'] * valid_speed_df['duration_seconds']
        
        # Compute aggregated statistics with error handling
        try:
            # Daily averages
            daily_avg = valid_speed_df.groupby('date').agg({
                'value_numeric': ['mean', 'count'],
                'weighted_value': ['sum'],
                'duration_seconds': ['sum']
            })
            daily_avg.columns = ['avg_speed', 'count', 'weighted_value_sum', 'duration_sum']
            daily_avg['weighted_avg'] = daily_avg['weighted_value_sum'] / daily_avg['duration_sum']
            daily_avg = daily_avg.reset_index()
            daily_avg['date_str'] = daily_avg['date'].apply(lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x))
            daily_avg['date_formatted'] = daily_avg['date'].apply(lambda x: format_date(x) if hasattr(x, 'strftime') else str(x))
        except Exception as e:
            st.warning(f"Error calculating daily averages: {str(e)}")
            daily_avg = pd.DataFrame(columns=['date', 'avg_speed', 'count', 'weighted_avg', 'date_str', 'date_formatted'])
        
        try:
            # Weekly averages
            weekly_avg = valid_speed_df.groupby('year_week').agg({
                'value_numeric': ['mean', 'count'],
                'weighted_value': ['sum'],
                'duration_seconds': ['sum']
            })
            weekly_avg.columns = ['avg_speed', 'count', 'weighted_value_sum', 'duration_sum']
            weekly_avg['weighted_avg'] = weekly_avg['weighted_value_sum'] / weekly_avg['duration_sum']
            weekly_avg = weekly_avg.reset_index()
            
            # Extract week number for display - with error handling
            weekly_avg['week_num'] = weekly_avg['year_week'].apply(
                lambda x: x.split('W')[1] if isinstance(x, str) and 'W' in x else "0"
            )
            weekly_avg['week_display'] = 'Week ' + weekly_avg['week_num']
        except Exception as e:
            st.warning(f"Error calculating weekly averages: {str(e)}")
            weekly_avg = pd.DataFrame(columns=['year_week', 'avg_speed', 'count', 'weighted_avg', 'week_num', 'week_display'])
        
        try:
            # Monthly averages
            monthly_avg = valid_speed_df.groupby('month').agg({
                'value_numeric': ['mean', 'count'],
                'weighted_value': ['sum'],
                'duration_seconds': ['sum']
            })
            monthly_avg.columns = ['avg_speed', 'count', 'weighted_value_sum', 'duration_sum']
            monthly_avg['weighted_avg'] = monthly_avg['weighted_value_sum'] / monthly_avg['duration_sum']
            monthly_avg = monthly_avg.reset_index()
        except Exception as e:
            st.warning(f"Error calculating monthly averages: {str(e)}")
            monthly_avg = pd.DataFrame(columns=['month', 'avg_speed', 'count', 'weighted_avg'])
        
        try:
            # Shift averages
            shift_avg = valid_speed_df.groupby(['date', 'shift']).agg({
                'value_numeric': ['mean', 'count'],
                'weighted_value': ['sum'],
                'duration_seconds': ['sum']
            })
            shift_avg.columns = ['avg_speed', 'count', 'weighted_value_sum', 'duration_sum']
            shift_avg['weighted_avg'] = shift_avg['weighted_value_sum'] / shift_avg['duration_sum']
            shift_avg = shift_avg.reset_index()
            
            # Filter out "None" shift
            shift_avg = shift_avg[shift_avg['shift'] != 'None']
            
            # Add date_formatted
            shift_avg['date_formatted'] = shift_avg['date'].apply(lambda x: format_date(x) if hasattr(x, 'strftime') else str(x))
            shift_avg['shift_date'] = shift_avg['date_formatted'] + ' (' + shift_avg['shift'] + ')'
        except Exception as e:
            st.warning(f"Error calculating shift averages: {str(e)}")
            shift_avg = pd.DataFrame(columns=['date', 'shift', 'avg_speed', 'count', 'weighted_avg', 'date_formatted', 'shift_date'])
        
        try:
            # Hourly averages
            hourly_avg = valid_speed_df.groupby('hour').agg({
                'value_numeric': ['mean', 'count'],
                'weighted_value': ['sum'],
                'duration_seconds': ['sum']
            })
            hourly_avg.columns = ['avg_speed', 'count', 'weighted_value_sum', 'duration_sum']
            hourly_avg['weighted_avg'] = hourly_avg['weighted_value_sum'] / hourly_avg['duration_sum']
            hourly_avg = hourly_avg.reset_index()
            
            # FIX: Safe conversion of hour to string format to avoid float formatting issues
            hourly_avg['hour_str'] = hourly_avg['hour'].apply(
                lambda x: f"{int(x):02d}:00" if pd.notna(x) and not pd.isna(x) else "00:00"
            )
        except Exception as e:
            st.warning(f"Error calculating hourly averages: {str(e)}")
            hourly_avg = pd.DataFrame(columns=['hour', 'avg_speed', 'count', 'weighted_avg', 'hour_str'])
        
        return daily_avg, weekly_avg, monthly_avg, shift_avg, hourly_avg
    except Exception as e:
        # Global error handling
        st.error(f"Unexpected error in calculate_avg_speeds: {str(e)}")
        # Return empty DataFrames with required columns
        empty_df = pd.DataFrame(columns=['date', 'avg_speed', 'count', 'weighted_avg', 'date_str', 'date_formatted'])
        empty_hour_df = pd.DataFrame(columns=['hour', 'avg_speed', 'count', 'weighted_avg', 'hour_str'])
        return empty_df, empty_df, empty_df, empty_df, empty_hour_df

# =============== Visualization Functions ===============

def plot_time_series(daily_avg, weekly_avg, monthly_avg, timeframe='daily'):
    """Create a time series plot based on selected timeframe"""
    if timeframe == 'daily':
        if daily_avg is None or len(daily_avg) == 0:
            return None, "No daily data available"
        
        df = daily_avg.sort_values('date')
        x_values = df['date_formatted']
        title = 'Daily Average Speed'
        x_label = 'Date'
        y_values = df['weighted_avg'] if 'weighted_avg' in df.columns else df['avg_speed']
    elif timeframe == 'weekly':
        if weekly_avg is None or len(weekly_avg) == 0:
            return None, "No weekly data available"
        
        df = weekly_avg.sort_values('year_week')
        x_values = df['week_display']
        title = 'Weekly Average Speed'
        x_label = 'Week'
        y_values = df['weighted_avg'] if 'weighted_avg' in df.columns else df['avg_speed']
    elif timeframe == 'monthly':
        if monthly_avg is None or len(monthly_avg) == 0:
            return None, "No monthly data available"
        
        df = monthly_avg
        x_values = df['month']
        title = 'Monthly Average Speed'
        x_label = 'Month'
        y_values = df['weighted_avg'] if 'weighted_avg' in df.columns else df['avg_speed']
    else:
        return None, "Invalid timeframe selected"
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bar chart
    bars = ax.bar(x_values, y_values, color='#4361EE')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}',
                ha='center', va='bottom', rotation=0, fontsize=8)
    
    # Set chart properties
    ax.set_xlabel(x_label)
    ax.set_ylabel('Average Speed')
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels if needed
    if timeframe in ["daily", "weekly"]:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    return fig, None

def plot_shift_comparison(shift_avg):
    """Create a bar chart comparing 1st and 2nd shifts"""
    if shift_avg is None or len(shift_avg) == 0:
        return None, "No shift data available"
    
    # Calculate overall shift averages
    if 'weighted_avg' in shift_avg.columns:
        overall_shift_avg = shift_avg.groupby('shift').agg({
            'weighted_avg': 'mean',
            'count': 'sum'
        }).reset_index()
        speed_column = 'weighted_avg'
    else:
        overall_shift_avg = shift_avg.groupby('shift').agg({
            'avg_speed': 'mean',
            'count': 'sum'
        }).reset_index()
        speed_column = 'avg_speed'
    
    # Sort by shift
    overall_shift_avg = overall_shift_avg.sort_values('shift')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart
    bars = ax.bar(overall_shift_avg['shift'], overall_shift_avg[speed_column], 
                  color=['#4361EE', '#3B82F6'])
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}',
                ha='center', va='bottom', rotation=0, fontsize=10)
    
    # Set chart properties
    ax.set_xlabel('Shift')
    ax.set_ylabel('Average Speed')
    ax.set_title('Average Speed by Shift')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    return fig, None

def create_insights_board(daily_avg, weekly_avg, monthly_avg, shift_avg):
    """Create an insights board with fastest and slowest periods"""
    insights = {}
    
    # Check if we have data
    if daily_avg is None or len(daily_avg) == 0:
        return None
    
    # Filter to only include days with sufficient measurements
    min_daily_measurements = 5  # Arbitrary threshold
    valid_daily = daily_avg[daily_avg['count'] >= min_daily_measurements].copy()
    
    if len(valid_daily) > 0:
        # Use weighted average if available
        if 'weighted_avg' in valid_daily.columns:
            # Fastest days
            fastest_days = valid_daily.sort_values('weighted_avg', ascending=False).head(5)
            insights['fastest_days'] = fastest_days
            
            # Slowest days
            slowest_days = valid_daily.sort_values('weighted_avg').head(5)
            insights['slowest_days'] = slowest_days
        else:
            # Fastest days
            fastest_days = valid_daily.sort_values('avg_speed', ascending=False).head(5)
            insights['fastest_days'] = fastest_days
            
            # Slowest days
            slowest_days = valid_daily.sort_values('avg_speed').head(5)
            insights['slowest_days'] = slowest_days
    
    # Weekly insights
    if weekly_avg is not None and len(weekly_avg) > 0:
        # Filter to only include weeks with sufficient measurements
        min_weekly_measurements = 20  # Arbitrary threshold
        valid_weekly = weekly_avg[weekly_avg['count'] >= min_weekly_measurements].copy()
        
        if len(valid_weekly) > 0:
            # Use weighted average if available
            if 'weighted_avg' in valid_weekly.columns:
                # Fastest weeks
                fastest_weeks = valid_weekly.sort_values('weighted_avg', ascending=False).head(5)
                insights['fastest_weeks'] = fastest_weeks
                
                # Slowest weeks
                slowest_weeks = valid_weekly.sort_values('weighted_avg').head(5)
                insights['slowest_weeks'] = slowest_weeks
            else:
                # Fastest weeks
                fastest_weeks = valid_weekly.sort_values('avg_speed', ascending=False).head(5)
                insights['fastest_weeks'] = fastest_weeks
                
                # Slowest weeks
                slowest_weeks = valid_weekly.sort_values('avg_speed').head(5)
                insights['slowest_weeks'] = slowest_weeks
    
    # Shift insights
    if shift_avg is not None and len(shift_avg) > 0:
        # Calculate which shift is faster on average
        if 'weighted_avg' in shift_avg.columns:
            shift_comparison = shift_avg.groupby('shift').agg({
                'weighted_avg': 'mean'
            }).reset_index()
        else:
            shift_comparison = shift_avg.groupby('shift').agg({
                'avg_speed': 'mean'
            }).reset_index()
        
        if len(shift_comparison) > 1:  # We have both shifts
            insights['shift_comparison'] = shift_comparison
    
    return insights

# =============== Session State Management ===============

# Initialize session state variables
if 'machines' not in st.session_state:
    st.session_state.machines = []

if 'active_speeds' not in st.session_state:
    st.session_state.active_speeds = pd.DataFrame()
    
if 'filtered_execution' not in st.session_state:
    st.session_state.filtered_execution = pd.DataFrame()
    
if 'loaded_days' not in st.session_state:
    st.session_state.loaded_days = set()
    
if 'failed_days' not in st.session_state:
    st.session_state.failed_days = set()
    
if 'current_progress' not in st.session_state:
    st.session_state.current_progress = {"days": [], "current_idx": 0, "complete": True}
    
if 'status_message' not in st.session_state:
    st.session_state.status_message = ""
    
if 'metric_name' not in st.session_state:
    st.session_state.metric_name = "speed"
    
if 'paused' not in st.session_state:
    st.session_state.paused = False

if 'loading_next_week' not in st.session_state:
    st.session_state.loading_next_week = False
    
if 'debug_info' not in st.session_state:
    st.session_state.debug_info = {}

# =============== Sidebar Configuration ===============

st.sidebar.title("Configuration")

# API Configuration
with st.sidebar.expander("API Settings", expanded=True):
    # Fixed API endpoint
    api_endpoint = "https://api.machinemetrics.com/proxy/graphql"
    api_key = st.text_input("API Key", type="password")
    
    # Metric name selection
    st.session_state.metric_name = st.selectbox(
        "Metric name for speed data:",
        ["speed", "velocity", "actual_speed", "rpm", "spindle_speed", "feed_rate"],
        index=0
    )
    
    # Button to fetch machines list
    if st.button("Load Machines"):
        if not api_key:
            st.error("Please enter your API key to continue")
        else:
            with st.spinner("Loading machines..."):
                machines = fetch_machines(api_endpoint, api_key)
                if machines:
                    st.session_state.machines = machines
                    st.success(f"Loaded {len(machines)} machines")
    
    # Machine selection dropdown
    machine_options = {m["name"]: m["machineRef"] for m in st.session_state.machines} if st.session_state.machines else {}
    
    if not machine_options:
        st.warning("Load machines first or enter a machine ID manually")
        machine_id = st.number_input("Machine ID", value=14343, min_value=1)
        machine_name = "Unknown Machine"
    else:
        machine_name = st.selectbox("Select Machine", list(machine_options.keys()))
        machine_id = machine_options[machine_name]
        st.session_state.machine_name = machine_name

# Date Selection
with st.sidebar.expander("Data Selection", expanded=True):
    # Option to select date range type
    range_type = st.radio(
        "Select data to load:",
        ["Single Day", "Week", "Month"], 
        horizontal=True
    )
    
    # Month selection
    if range_type == "Month":
        current_month = datetime.now(houston_tz).month
        current_year = datetime.now(houston_tz).year
        
        month = st.selectbox(
            "Select month:",
            range(1, current_month + 1),
            format_func=lambda m: datetime(current_year, m, 1).strftime('%B')
        )
        
        selected_days = get_days_in_month(current_year, month)
        
    # Week selection
    elif range_type == "Week":
        # Get first day of current month to start week selection
        current_date = datetime.now(houston_tz)
        month_start = datetime(current_date.year, current_date.month, 1, tzinfo=houston_tz)
        
        # Get weeks starting from beginning of month
        weeks = []
        start_date = month_start
        while start_date.month == current_date.month or start_date.month == current_date.month - 1:
            end_date = start_date + timedelta(days=6)
            weeks.append((start_date, end_date, f"{start_date.strftime('%b %d')} - {end_date.strftime('%b %d')}"))
            start_date = start_date + timedelta(days=7)
        
        # Let user select a week
        week_options = [w[2] for w in weeks]
        selected_week = st.selectbox("Select week:", week_options)
        
        # Get the start date for the selected week
        week_start = weeks[week_options.index(selected_week)][0]
        
        # Get days in the selected week
        selected_days = get_days_in_week(week_start)
        
    # Single day selection
    else:
        selected_date = st.date_input(
            "Select day:",
            datetime.now(houston_tz).date()
        )
        
        # Convert to datetime with timezone
        selected_days = [datetime.combine(selected_date, datetime.min.time(), tzinfo=houston_tz)]
    
    # Load Data Button
    start_load = st.button("Load Selected Data", disabled=not api_key)
    
    if start_load:
        # Reset any pause state
        st.session_state.paused = False
        st.session_state.loading_next_week = False
        
        # Setup the progress tracker
        st.session_state.current_progress = {
            "days": selected_days,
            "current_idx": 0,
            "complete": False,
            "requested_type": range_type
        }
        
        # Reset status message
        st.session_state.status_message = f"Starting to load {len(selected_days)} day(s)..."
    
    # Clear Data Button
    if st.button("Clear All Data", disabled=len(st.session_state.loaded_days) == 0):
        st.session_state.active_speeds = pd.DataFrame()
        st.session_state.filtered_execution = pd.DataFrame()
        st.session_state.loaded_days = set()
        st.session_state.failed_days = set()
        st.session_state.current_progress = {"days": [], "current_idx": 0, "complete": True}
        st.session_state.paused = False
        st.session_state.loading_next_week = False
        st.session_state.status_message = "All data cleared"
        st.success("All data cleared")

# Analysis Options
with st.sidebar.expander("Analysis Options", expanded=True):
    timeframe = st.radio("Select timeframe for analysis:", 
                       ["Daily", "Weekly", "Monthly"], 
                       index=0,
                       horizontal=True)

# Instructions in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Instructions")
st.sidebar.markdown("""
1. Enter your API key and click "Load Machines"
2. Select a machine from the dropdown
3. Choose a day, week, or month to load
4. The app will load data one day at a time
5. After each week completes, you can choose to continue or stop
6. Use the Analysis Options to change the timeframe view
""")

# =============== Main Content Area ===============

# Create dedicated section for the continue button (always visible)
continue_section = st.empty()
progress_section = st.empty()
debug_section = st.empty()

# Initialize debug info if not present
if 'debug_info' not in st.session_state:
    st.session_state.debug_info = {}

# Display debug info in development mode
if st.sidebar.checkbox("Show Debug Info", value=True):
    with debug_section.container():
        st.subheader("Debug Information")
        if st.session_state.debug_info:
            # Display main statistics
            debug_stats = {k: v for k, v in st.session_state.debug_info.items() if k not in ['active_periods', 'period_stats']}
            st.json(debug_stats)
            
            # Display shift hours information
            st.subheader("Shift Hours Settings")
            st.markdown("""
            * **1st Shift:** 6:00 AM - 4:00 PM (Weekdays only)
            * **2nd Shift:** 8:00 PM - 6:00 AM (Weekdays only)
            * **Weekends:** Excluded completely
            """)
            
            # Display active period statistics in a formatted table
            if 'period_stats' in st.session_state.debug_info and st.session_state.debug_info['period_stats']:
                st.subheader("Active Period Statistics")
                
                # Create a DataFrame from period stats for better display
                period_stats_df = pd.DataFrame(st.session_state.debug_info['period_stats'])
                
                # Highlight periods with no valid measurements
                def highlight_low_speed(row):
                    if row['valid_measurements'] == 0 or row['avg_speed'] == 0:
                        return ['background-color: #ffe6e6'] * len(row)
                    elif row['avg_speed'] >= 7:
                        return ['background-color: #e6ffe6'] * len(row)
                    else:
                        return [''] * len(row)
                
                st.dataframe(period_stats_df.style.apply(highlight_low_speed, axis=1))
                
                # Show overall weighted average calculation
                if period_stats_df['valid_measurements'].sum() > 0:
                    st.subheader("Speed by Duration Analysis")
                    
                    # Calculate weighted average based on period duration
                    period_stats_df['speed_x_duration'] = period_stats_df['avg_speed'] * period_stats_df['duration_mins']
                    weighted_avg = period_stats_df['speed_x_duration'].sum() / period_stats_df['duration_mins'].sum()
                    
                    # Create a summary table
                    summary_data = {
                        'Measurement Type': ['Simple Average', 'Duration-Weighted Average'],
                        'Value': [period_stats_df['avg_speed'].mean(), weighted_avg]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df['Value'] = summary_df['Value'].round(2)
                    
                    # Display the summary
                    st.table(summary_df)
            
            # Display sample of active speed values
            if 'active_speeds' in st.session_state and len(st.session_state.active_speeds) > 0:
                st.subheader("Sample of Speed Values During ACTIVE Periods")
                # Create a filtered view with non-zero and non-null speeds
                valid_speeds = st.session_state.active_speeds[
                    (st.session_state.active_speeds['value_numeric'] > 0) & 
                    (st.session_state.active_speeds['value_numeric'].notna())
                ].copy()
                
                if len(valid_speeds) > 0:
                    # Create a histogram of speed values
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.hist(valid_speeds['value_numeric'], bins=20, color='#4361EE', alpha=0.7)
                    ax.set_xlabel('Speed Value')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Distribution of Speed Values During ACTIVE Periods')
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    st.pyplot(fig)
                    
                    # Show sample data
                    sample_size = min(10, len(valid_speeds))
                    valid_speeds['local_time'] = valid_speeds['eventTime_local'].dt.strftime('%H:%M:%S')
                    st.write(valid_speeds[['local_time', 'value_numeric', 'period_id']].sample(sample_size))
                    
                    # Show summary statistics
                    st.subheader("Speed Value Statistics")
                    stats_df = pd.DataFrame({
                        'Statistic': ['Count', 'Mean', 'Median', 'Min', 'Max', 'Std Dev'],
                        'Value': [
                            len(valid_speeds),
                            valid_speeds['value_numeric'].mean(),
                            valid_speeds['value_numeric'].median(),
                            valid_speeds['value_numeric'].min(),
                            valid_speeds['value_numeric'].max(),
                            valid_speeds['value_numeric'].std()
                        ]
                    })
                    
                    # Format numeric values
                    stats_df['Value'] = stats_df['Value'].apply(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
                    st.table(stats_df)
                else:
                    st.warning("No valid (non-zero, non-null) speed values during ACTIVE periods")
        else:
            st.info("No debug information available yet")

# Display progress bar for current operation if applicable
if not st.session_state.current_progress["complete"] and not st.session_state.paused:
    current_idx = st.session_state.current_progress["current_idx"]
    total_days = len(st.session_state.current_progress["days"])
    if total_days > 0:
        with progress_section.container():
            progress_percent = current_idx / total_days
            st.progress(progress_percent)
            st.write(f"Processing day {current_idx + 1} of {total_days}...")

# Check if loading is paused after a week or we're in the middle of loading the next week
if st.session_state.paused:
    with continue_section.container():
        st.info("Loading paused after completing a week of data.")
        
        if st.button("➡️ Continue Loading Next Week", key="continue_next_week"):
            with st.spinner("Preparing to load next week of data..."):
                st.session_state.paused = False
                st.session_state.loading_next_week = True
                st.session_state.status_message = "Continuing to load more days..."
                # Add a small delay to show the spinner
                time.sleep(0.5)
            # Rerun to continue processing
            st.experimental_rerun()
elif st.session_state.loading_next_week:
    with continue_section.container():
        st.markdown("""
        <div class="loading-indicator">
            <div class="loading-spinner">⟳</div>
            <div>Loading next batch of data... Please wait.</div>
        </div>
        """, unsafe_allow_html=True)

# Check if we need to process the next day in the queue
if not st.session_state.current_progress["complete"] and not st.session_state.paused and len(st.session_state.current_progress["days"]) > 0:
    # Get the current day to process
    current_idx = st.session_state.current_progress["current_idx"]
    
    if current_idx < len(st.session_state.current_progress["days"]):
        current_day = st.session_state.current_progress["days"][current_idx]
        day_str = current_day.strftime('%Y-%m-%d')
        
        # Display progress
        progress_message = st.empty()
        progress_message.info(f"Loading data for {day_str} ({current_idx + 1} of {len(st.session_state.current_progress['days'])})")
        
        # Fetch data for the current day
        with st.spinner(f"Processing {day_str}..."):
            active_speeds, filtered_execution, message = fetch_day_data(
                api_endpoint, api_key, machine_id, current_day, 
                metric_name=st.session_state.metric_name
            )
            
            # Update session state with results
            if len(active_speeds) > 0:
                # Add to existing data if any
                if len(st.session_state.active_speeds) > 0:
                    st.session_state.active_speeds = pd.concat([st.session_state.active_speeds, active_speeds])
                    st.session_state.filtered_execution = pd.concat([st.session_state.filtered_execution, filtered_execution])
                else:
                    st.session_state.active_speeds = active_speeds
                    st.session_state.filtered_execution = filtered_execution
                
                # Add to loaded days
                st.session_state.loaded_days.add(day_str)
                st.session_state.status_message = message
                progress_message.success(message)
            else:
                # Record as failed
                st.session_state.failed_days.add(day_str)
                st.session_state.status_message = message
                progress_message.warning(message)
            
            # Small delay to ensure message is visible
            time.sleep(1)
        
        # Reset loading_next_week flag if it was set
        if st.session_state.loading_next_week and current_idx % 7 == 0:
            st.session_state.loading_next_week = False
        
        # Update progress
        st.session_state.current_progress["current_idx"] += 1
        
        # Check if we've completed all days or need to ask for confirmation to continue
        if st.session_state.current_progress["current_idx"] >= len(st.session_state.current_progress["days"]):
            st.session_state.current_progress["complete"] = True
            st.session_state.loading_next_week = False
            st.success(f"Completed loading {range_type} data!")
        elif st.session_state.current_progress["current_idx"] % 7 == 0 and st.session_state.current_progress["requested_type"] != "Week":
            # After loading a week worth of data, pause and ask if user wants to continue
            st.session_state.paused = True
            st.session_state.loading_next_week = False
            st.experimental_rerun()
        else:
            # Continue to the next day
            st.experimental_rerun()

# Display loaded days status
if len(st.session_state.loaded_days) > 0 or len(st.session_state.failed_days) > 0:
    st.markdown("### Data Loading Status")
    
    # Get all days that were attempted
    all_attempted_days = set(day.strftime('%Y-%m-%d') for day in st.session_state.current_progress.get("days", []))
    if not all_attempted_days:
        all_attempted_days = st.session_state.loaded_days.union(st.session_state.failed_days)
    
    all_attempted_days = sorted(list(all_attempted_days))
    
    # Display days as colored boxes
    day_html = "<div style='line-height: 2.5;'>"
    
    for day in all_attempted_days:
        if day in st.session_state.loaded_days:
            day_html += f'<span class="loaded-day">{day}</span>'
        elif day in st.session_state.failed_days:
            day_html += f'<span class="failed-day">{day}</span>'
        else:
            day_html += f'<span class="pending-day">{day}</span>'
    
    day_html += "</div>"
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(day_html, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='padding: 10px; border: 1px solid #E5E7EB; border-radius: 5px;'>
            <p><span style='background-color: #D1FAE5; padding: 3px 8px; border-radius: 4px;'>■</span> Loaded: {len(st.session_state.loaded_days)}</p>
            <p><span style='background-color: #FEE2E2; padding: 3px 8px; border-radius: 4px;'>■</span> Failed: {len(st.session_state.failed_days)}</p>
            <p><span style='background-color: #E5E7EB; padding: 3px 8px; border-radius: 4px;'>■</span> Pending: {len(all_attempted_days) - len(st.session_state.loaded_days) - len(st.session_state.failed_days)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Show last status message
    if st.session_state.status_message:
        st.info(f"Status: {st.session_state.status_message}")

# Check if we have data to display
has_data = len(st.session_state.active_speeds) > 0

# Main layout
if not has_data:
    st.info("👈 Configure the API settings, select a machine, and start loading data")
else:
    try:
        # Calculate aggregated data
        daily_avg, weekly_avg, monthly_avg, shift_avg, hourly_avg = calculate_avg_speeds(st.session_state.active_speeds)
        
        # Calculate insights
        insights = create_insights_board(daily_avg, weekly_avg, monthly_avg, shift_avg)
        
        # Get the selected timeframe
        timeframe_lower = timeframe.lower()
        
        # Display machine stats
        machine_name = st.session_state.get('machine_name', 'Machine')
        st.subheader(f"Speed Analysis: {machine_name}")
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_days = len(daily_avg) if daily_avg is not None and 'date' in daily_avg.columns else 0
            st.metric("Total Days with Data", total_days)
        
        with col2:
            # Use weighted average if available
            if 'debug_info' in st.session_state and 'weighted_avg_speed' in st.session_state.debug_info:
                overall_avg = st.session_state.debug_info['weighted_avg_speed']
            else:
                try:
                    valid_speeds = st.session_state.active_speeds[
                        (st.session_state.active_speeds['value_numeric'] > 0) & 
                        (st.session_state.active_speeds['value_numeric'].notna())
                    ]
                    overall_avg = valid_speeds['value_numeric'].mean() if len(valid_speeds) > 0 else 0
                except:
                    overall_avg = 0
            st.metric("Overall Avg Speed", f"{overall_avg:.2f}")
        
        with col3:
            total_measurements = len(st.session_state.active_speeds)
            try:
                valid_measurements = len(st.session_state.active_speeds[
                    (st.session_state.active_speeds['value_numeric'] > 0) & 
                    (st.session_state.active_speeds['value_numeric'].notna())
                ])
            except:
                valid_measurements = 0
            st.metric("Valid Measurements", f"{valid_measurements:,}")
        
        with col4:
            # Calculate which shift is faster
            if shift_avg is not None and len(shift_avg) > 0 and 'shift' in shift_avg.columns:
                try:
                    speed_column = 'weighted_avg' if 'weighted_avg' in shift_avg.columns else 'avg_speed'
                    shift_comparison = shift_avg.groupby('shift').agg({
                        speed_column: 'mean'  # Use weighted average if available
                    }).reset_index()
                    
                    if len(shift_comparison) > 1:  # We have both shifts
                        first_shift_avg = shift_comparison[shift_comparison['shift'] == '1st'][speed_column].iloc[0]
                        second_shift_avg = shift_comparison[shift_comparison['shift'] == '2nd'][speed_column].iloc[0]
                        
                        faster_shift = "1st" if first_shift_avg > second_shift_avg else "2nd"
                        st.metric("Faster Shift", faster_shift)
                except Exception as e:
                    st.warning(f"Could not calculate faster shift: {str(e)}")
        
        # Create tabs for different visualizations
        tabs = st.tabs(["Time Series", "Shift Comparison", "Hourly Analysis", "Speed Data Table"])
        
        with tabs[0]:
            # Plot the time series
            st.subheader(f"{timeframe} Average Speed")
            
            try:
                fig, error_msg = plot_time_series(daily_avg, weekly_avg, monthly_avg, timeframe_lower)
                
                if error_msg:
                    st.warning(error_msg)
                elif fig:
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Error plotting time series: {str(e)}")
        
        with tabs[1]:
            # Plot shift comparison
            st.subheader("Shift Comparison")
            
            try:
                fig, error_msg = plot_shift_comparison(shift_avg)
                
                if error_msg:
                    st.warning(error_msg)
                elif fig:
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Error plotting shift comparison: {str(e)}")
        
        with tabs[2]:
            # Hourly analysis
            st.subheader("Hourly Speed Analysis")
            
            try:
                if hourly_avg is not None and len(hourly_avg) > 0 and 'hour' in hourly_avg.columns:
                    # Create two columns for the hourly analysis
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Create hourly plot
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Plot both simple and weighted averages
                        ax.bar(hourly_avg['hour_str'], hourly_avg['avg_speed'], alpha=0.7, label='Simple Average')
                        ax.plot(hourly_avg['hour_str'], hourly_avg['weighted_avg'], color='red', marker='o', label='Weighted Average')
                        
                        # Add value labels
                        for i, value in enumerate(hourly_avg['weighted_avg']):
                            if pd.notna(value):
                                ax.text(i, value + 0.2, f'{value:.1f}', ha='center', fontsize=9)
                        
                        # Set chart properties
                        ax.set_xlabel('Hour of Day')
                        ax.set_ylabel('Average Speed')
                        ax.set_title('Average Speed by Hour')
                        ax.grid(axis='y', linestyle='--', alpha=0.7)
                        ax.legend()
                        
                        # Rotate x-axis labels
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        
                        st.pyplot(fig)
                    
                    with col2:
                        # Create hourly table
                        st.write("Hourly Speed Data")
                        hourly_table = hourly_avg[['hour_str', 'avg_speed', 'weighted_avg', 'count']].copy()
                        hourly_table.columns = ['Hour', 'Simple Avg', 'Weighted Avg', 'Measurements']
                        hourly_table['Simple Avg'] = hourly_table['Simple Avg'].round(2)
                        hourly_table['Weighted Avg'] = hourly_table['Weighted Avg'].round(2)
                        
                        # Style the dataframe for better visualization
                        st.dataframe(hourly_table.style.highlight_max(subset=['Weighted Avg'], axis=0, color='lightgreen')
                                     .highlight_min(subset=['Weighted Avg'], axis=0, color='#ffcccb'))
                else:
                    st.warning("No hourly data available")
            except Exception as e:
                st.error(f"Error displaying hourly analysis: {str(e)}")
        
        with tabs[3]:
            # Raw speed data table
            st.subheader("Speed Data During ACTIVE Periods")
            
            try:
                if 'value_numeric' in st.session_state.active_speeds.columns:
                    valid_speeds = st.session_state.active_speeds[
                        (st.session_state.active_speeds['value_numeric'] > 0) & 
                        (st.session_state.active_speeds['value_numeric'].notna())
                    ].copy()
                    
                    if len(valid_speeds) > 0:
                        # Format datetime for display
                        if 'eventTime_local' in valid_speeds.columns:
                            valid_speeds['time'] = valid_speeds['eventTime_local'].dt.strftime('%H:%M:%S')
                            valid_speeds['date'] = valid_speeds['eventTime_local'].dt.strftime('%Y-%m-%d')
                        else:
                            valid_speeds['time'] = "Unknown"
                            valid_speeds['date'] = "Unknown"
                        
                        # Select relevant columns that are guaranteed to exist
                        display_cols = ['date', 'time']
                        if 'value_numeric' in valid_speeds.columns:
                            display_cols.append('value_numeric')
                        
                        display_df = valid_speeds[display_cols].copy()
                        display_df.columns = ['Date', 'Time', 'Speed'] if len(display_cols) == 3 else ['Date', 'Time']
                        
                        # Show the data
                        st.dataframe(display_df.sort_values(['Date', 'Time']), height=400)
                    else:
                        st.warning("No valid speed data available")
                else:
                    st.warning("Speed data is missing required columns")
            except Exception as e:
                st.error(f"Error displaying speed data table: {str(e)}")
        
        # Right column - insights
        st.sidebar.subheader("Speed Insights")
        
        if insights is None:
            st.sidebar.warning("No insights available yet - need more data")
        else:
            try:
                # Fastest Days
                if 'fastest_days' in insights and len(insights['fastest_days']) > 0:
                    st.sidebar.markdown('<p class="insight-title fastest">5 Fastest Days</p>', unsafe_allow_html=True)
                    
                    for _, row in insights['fastest_days'].iterrows():
                        date = row['date_formatted']
                        speed = row['weighted_avg'] if 'weighted_avg' in row else row['avg_speed']
                        count = row['count']
                        
                        st.sidebar.markdown(
                            f"""<div class="metric-card">
                                <strong>{date}</strong><br>
                                Average Speed: <span class="fastest">{speed:.2f}</span><br>
                                Measurements: {count}
                            </div>""", 
                            unsafe_allow_html=True
                        )
                
                # Slowest Days
                if 'slowest_days' in insights and len(insights['slowest_days']) > 0:
                    st.sidebar.markdown('<p class="insight-title slowest">5 Slowest Days</p>', unsafe_allow_html=True)
                    
                    for _, row in insights['slowest_days'].iterrows():
                        date = row['date_formatted']
                        speed = row['weighted_avg'] if 'weighted_avg' in row else row['avg_speed']
                        count = row['count']
                        
                        st.sidebar.markdown(
                            f"""<div class="metric-card">
                                <strong>{date}</strong><br>
                                Average Speed: <span class="slowest">{speed:.2f}</span><br>
                                Measurements: {count}
                            </div>""", 
                            unsafe_allow_html=True
                        )
            except Exception as e:
                st.sidebar.error(f"Error displaying insights: {str(e)}")
        
        # Data download section
        with st.expander("Download Data"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.session_state.active_speeds is not None and len(st.session_state.active_speeds) > 0:
                    try:
                        speed_csv = st.session_state.active_speeds.to_csv(index=False)
                        st.download_button(
                            label="Download Raw Speed Data",
                            data=speed_csv,
                            file_name="active_speed_data.csv",
                            mime="text/csv"
                        )
                    except:
                        st.warning("Error preparing speed data for download")
            
            with col2:
                if daily_avg is not None and len(daily_avg) > 0:
                    try:
                        daily_csv = daily_avg.to_csv(index=False)
                        st.download_button(
                            label="Download Daily Averages",
                            data=daily_csv,
                            file_name="daily_avg_speeds.csv",
                            mime="text/csv"
                        )
                    except:
                        st.warning("Error preparing daily data for download")
            
            with col3:
                if hourly_avg is not None and len(hourly_avg) > 0:
                    try:
                        hourly_csv = hourly_avg.to_csv(index=False)
                        st.download_button(
                            label="Download Hourly Analysis",
                            data=hourly_csv,
                            file_name="hourly_avg_speeds.csv",
                            mime="text/csv"
                        )
                    except:
                        st.warning("Error preparing hourly data for download")
    except Exception as e:
        st.error(f"An error occurred while processing the data: {str(e)}")
        st.info("Try clearing the data and loading again with a smaller date range.")

# Footer
st.sidebar.markdown('---')
st.sidebar.markdown('Machine Speed Analyzer v6.1')
st.sidebar.markdown('© 2025 Machine Metrics Tool')