import streamlit as st
from googleapiclient.discovery import build
import pytz
import matplotlib.pyplot as plt
from datetime import datetime
from google_auth_oauthlib.flow import InstalledAppFlow
import pandas as pd
import requests
from pytrends.request import TrendReq
from pytrends.exceptions import ResponseError
from openai import OpenAI
from serpapi import GoogleSearch
from datetime import datetime, timedelta
from fpdf import FPDF
import requests
from PIL import Image
import cv2
from io import BytesIO
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation



pytrends = TrendReq(hl='en-US', tz=360)


# BASIC VIDEO ANALYSIS 

# Utility functions to perform each type of analysis
def get_video_metadata(video_id):
    request = youtube.videos().list(part="snippet,statistics", id=video_id)
    response = request.execute()
    
    if not response["items"]:
        return {"error": "Video not found"}

    video = response["items"][0]["snippet"]
    statistics = response["items"][0]["statistics"]
    
    metadata = {
        "title": video["title"],
        "description": video.get("description"),
        "published_at": video["publishedAt"],
        "channel_title": video["channelTitle"],
        "view_count": statistics.get("viewCount"),
        "like_count": statistics.get("likeCount"),
        "comment_count": statistics.get("commentCount"),
    }
    return metadata

def show_video_tags(video_id):
    request = youtube.videos().list(part="snippet", id=video_id)
    response = request.execute()

    if not response["items"]:
        return {"error": "Video not found"}

    tags = response["items"][0]["snippet"].get("tags", [])
    return {"tags": tags}

def extract_video_keywords(video_id):
    # Extract video keywords based on tags and description
    metadata = get_video_metadata(video_id)
    tags = show_video_tags(video_id).get("tags", [])
    keywords = tags + metadata.get("description", "").split()
    return {"keywords": list(set(keywords))}

def get_video_performance_metrics(video_id):
    statistics = get_video_metadata(video_id)
    return {
        "view_count": statistics["view_count"],
        "like_count": statistics["like_count"],
        "comment_count": statistics["comment_count"]
    }

def summarize_video_statistics(video_id):
    statistics = get_video_performance_metrics(video_id)
    summary = (
        f"Views: {statistics['view_count']}, "
        f"Likes: {statistics['like_count']}, "
        f"Comments: {statistics['comment_count']}"
    )
    return {"summary": summary}

def show_video_engagement_metrics(video_id):
    statistics = get_video_metadata(video_id)
    if not statistics.get("view_count") or not statistics.get("like_count"):
        return {"error": "Insufficient data for engagement calculation"}

    engagement_rate = (
        (int(statistics["like_count"]) + int(statistics["comment_count"]))
        / int(statistics["view_count"]) * 100
    )
    return {"engagement_rate": f"{engagement_rate:.2f}%"}

def compare_metrics_between_videos(video_id_1, video_id_2):
    video_1_stats = get_video_performance_metrics(video_id_1)
    video_2_stats = get_video_performance_metrics(video_id_2)
    
    comparison = {
        "video_1": video_1_stats,
        "video_2": video_2_stats,
        "difference": {
            "views_diff": int(video_1_stats["view_count"]) - int(video_2_stats["view_count"]),
            "likes_diff": int(video_1_stats["like_count"]) - int(video_2_stats["like_count"]),
            "comments_diff": int(video_1_stats["comment_count"]) - int(video_2_stats["comment_count"]),
        }
    }
    return comparison



# CHANNEL ANALYSIS
# Utility functions to perform each type of channel analysis
def analyze_channel(channel_id):
    request = youtube.channels().list(
        part="snippet,statistics,brandingSettings",
        id=channel_id
    )
    response = request.execute()

    if not response["items"]:
        return {"error": "Channel not found"}

    channel = response["items"][0]
    channel_info = {
        "title": channel["snippet"]["title"],
        "description": channel["snippet"].get("description", ""),
        "country": channel["snippet"].get("country", "N/A"),
        "published_at": channel["snippet"]["publishedAt"],
        "view_count": channel["statistics"].get("viewCount"),
        "subscriber_count": channel["statistics"].get("subscriberCount"),
        "video_count": channel["statistics"].get("videoCount"),
        "branding_title": channel.get("brandingSettings", {}).get("channel", {}).get("title"),
    }
    return channel_info

def show_top_performing_videos(channel_id, max_results=5):
    request = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        order="viewCount",
        maxResults=max_results
    )
    response = request.execute()

    top_videos = [
        {
            "title": item["snippet"]["title"],
            "video_id": item["id"]["videoId"],
            "description": item["snippet"]["description"],
            "published_at": item["snippet"]["publishedAt"]
        }
        for item in response["items"] if item["id"]["kind"] == "youtube#video"
    ]
    return top_videos

def get_channel_growth_metrics(channel_id):
    channel_info = analyze_channel(channel_id)
    if "error" in channel_info:
        return channel_info

    growth_metrics = {
        "view_count": channel_info.get("view_count"),
        "subscriber_count": channel_info.get("subscriber_count"),
        "video_count": channel_info.get("video_count"),
        "published_at": channel_info.get("published_at")
    }
    return growth_metrics

def compare_channels(channel_id_1, channel_id_2):
    channel_1_info = analyze_channel(channel_id_1)
    channel_2_info = analyze_channel(channel_id_2)
    
    if "error" in channel_1_info or "error" in channel_2_info:
        return {"error": "One or both channels not found"}

    comparison = {
        "channel_1": channel_1_info,
        "channel_2": channel_2_info,
        "differences": {
            "view_diff": int(channel_1_info["view_count"]) - int(channel_2_info["view_count"]),
            "subscriber_diff": int(channel_1_info["subscriber_count"]) - int(channel_2_info["subscriber_count"]),
            "video_diff": int(channel_1_info["video_count"]) - int(channel_2_info["video_count"]),
        }
    }
    return comparison

def show_channel_engagement_trends(channel_id):
    # Assuming "likes", "comments", and "views" are available for engagement analysis
    top_videos = show_top_performing_videos(channel_id, max_results=10)
    if not top_videos:
        return {"error": "No top videos found"}

    trends = []
    for video in top_videos:
        video_id = video["video_id"]
        stats = get_video_performance_metrics(video_id)  # Reusing the function defined previously
        trends.append({
            "video_title": video["title"],
            "likes": stats.get("like_count"),
            "views": stats.get("view_count"),
            "comments": stats.get("comment_count")
        })

    return trends

def analyze_upload_schedule(channel_id):
    request = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        order="date",
        maxResults=50
    )
    response = request.execute()

    upload_dates = [item["snippet"]["publishedAt"] for item in response["items"] if item["id"]["kind"] == "youtube#video"]
    upload_days = pd.to_datetime(upload_dates).day_name()
    
    day_distribution = upload_days.value_counts().to_dict()
    return {"upload_schedule": day_distribution}

def get_subscriber_growth_rate(channel_id):
    """
    Fetches the subscriber growth rate for the given YouTube channel ID.
    Returns the growth rate as a percentage over the total days active.
    """
    # Fetch channel details
    channel_response = youtube.channels().list(
        part="snippet,statistics",
        id=channel_id
    ).execute()
    
    if not channel_response['items']:
        return {"error": "Channel not found."}

    # Extract relevant data
    channel_info = channel_response['items'][0]
    published_at = pd.to_datetime(channel_info['snippet']['publishedAt'])  # Channel creation date
    subscriber_count_start = int(channel_info['statistics']['subscriberCount'])

    # Fetch subscriber count now
    current_stats = youtube.channels().list(
        part="statistics",
        id=channel_id
    ).execute()

    subscriber_count_now = int(current_stats['items'][0]['statistics']['subscriberCount'])

    # Ensure both timestamps are timezone-aware
    today = pd.Timestamp.now(tz='UTC')
    if published_at.tzinfo is None:
        published_at = published_at.tz_localize('UTC')  # Convert to UTC if naive

    # Calculate days active
    days_active = (today - published_at).days

    # Calculate subscriber growth rate
    growth_rate = ((subscriber_count_now - subscriber_count_start) / days_active) * 100 if days_active > 0 else 0

    return {
        "Subscriber Count Start": subscriber_count_start,
        "Subscriber Count Now": subscriber_count_now,
        "Days Active": days_active,
        "Growth Rate (%)": growth_rate
    }

# CONTENT STRATEGY
# Function to suggest video ideas
def suggest_video_ideas(related_topic, geo="US"):
    pytrends.build_payload([related_topic], cat=0, timeframe="today 3-m", geo=geo, gprop="")
    related_queries = pytrends.related_queries().get(related_topic, {}).get("top", pd.DataFrame())
    
    if related_queries.empty:
        return f"No video ideas found for the topic {related_topic}"
    
    video_ideas = related_queries["query"].head(5).tolist()
    return video_ideas

# Function to generate title ideas
def generate_title_ideas(related_topic):
    prompt = f"Generate 5 catchy YouTube video title ideas about {related_topic}:"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    # Extract title ideas from response
    title_ideas = response.choices[0].message.content.strip().split("\n")
    
    return title_ideas

# Function to optimize description
def optimize_description(related_topic):
    prompt = f"Write an engaging YouTube video description about {related_topic} that includes tips and trends."
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Function to get thumbnail suggestions
def get_thumbnail_suggestions(related_topic):
    prompt = f"Suggest 3 engaging thumbnail ideas for a YouTube video about {related_topic}."
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Function to recommend upload time
def recommend_upload_time(related_topic):
    prompt = f"Recommend the best days and times to upload YouTube videos about {related_topic} for maximum audience engagement"
    response = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# Function to get content calendar suggestions
def get_content_calendar_suggestions(related_topic, duration="1 week"):
    prompt = f"Create a YouTube content calendar for {duration} focused on '{related_topic}', including video titles, descriptions, and thumbnails."
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        content_calendar = response.choices[0].message.content.strip()
        return content_calendar
    except Exception as e:
        print(f"Error: {e}")
        return None

# Function to analyze best posting times
def analyze_best_posting_times(related_topic):
    prompt = f"Analyze and suggest the best days and times to post YouTube videos about {related_topic} based on audience engagement, considering factors like viewer demographics, timezone, and watch history."
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# TRENDING ANALYSIS
def get_trending_topics(region_code="US", max_results=5):
    trending_searches_df = pytrends.trending_searches(pn=region_code)
    trending_topics = trending_searches_df.head(max_results).values.flatten().tolist()
    return trending_topics

def get_trending_hashtags(region_code="US", max_results=5):
    trending_videos = get_viral_videos(region_code, max_results)
    hashtags = []
    for video in trending_videos:
        tags = video.get("tags", [])
        hashtags.extend([f"#{tag}" for tag in tags if tag])
    
    unique_hashtags = list(set(hashtags))  # Remove duplicates
    return unique_hashtags[:max_results]

def get_viral_videos(region_code="US", max_results=5):
    request = youtube.videos().list(
        part="snippet,statistics",
        chart="mostPopular",
        regionCode=region_code,
        maxResults=max_results
    )
    response = request.execute()
    trending_videos = [
        {
            "title": item["snippet"]["title"],
            "video_id": item["id"],
            "description": item["snippet"]["description"],
            "tags": item.get("snippet", {}).get("tags", []),
            "view_count": item["statistics"].get("viewCount"),
            "like_count": item["statistics"].get("likeCount"),
        }
        for item in response["items"]
    ]
    return trending_videos

def analyze_trends(keyword, timeframe="now 7-d", geo="united_states"):
    pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo=geo, gprop="")
    interest_over_time_df = pytrends.interest_over_time()
    if interest_over_time_df.empty:
        return {"error": f"No trend data found for '{keyword}'"}
    
    return interest_over_time_df.reset_index().to_dict(orient="records")

def compare_trend_data(keyword1, keyword2, geo="united_states"):
    pytrends.build_payload([keyword1, keyword2], cat=0, timeframe="now 7-d", geo=geo, gprop="")
    comparison_data = pytrends.interest_over_time()
    if comparison_data.empty:
        return {"error": "No comparison data found"}
    
    return comparison_data.reset_index().to_dict(orient="records")

def extract_categories_info(categories_dict):
    """
    Recursively extracts the category information and flattens it into a list of dictionaries.
    """
    categories = []
    
    def recursive_extract(children, parent_name=""):
        for child in children:
            category = {
                'Parent': parent_name,
                'Category': child.get('name'),
                'ID': child.get('id')
            }
            categories.append(category)
            if 'children' in child:
                recursive_extract(child['children'], child.get('name'))
    
    # Start recursive extraction
    if 'children' in categories_dict:
        recursive_extract(categories_dict['children'])
    
    return pd.DataFrame(categories)

def show_rising_trends(pn=None):
    """
    Fetches trending categories using pytrends, extracts nested details, and returns a DataFrame.
    """
    try:
        # Initialize pytrends request
        pytrends = TrendReq(hl='en-US', tz=360)
        
        # Get trending categories (assuming this fetches a nested structure)
        trending_searches = pytrends.categories()  # Returns a nested dictionary
        
        # Extract information into a structured DataFrame
        extracted_df = extract_categories_info(trending_searches)
        
        return extracted_df
    except ResponseError as e:
        print(f"Error: {e}")
        return None

def get_weekly_trend_report(region_code="US", keyword="youtube"):
    """
    Fetch weekly trend report for a given keyword and region from Google Trends.

    Parameters:
    - region_code (str): The geographical region code (default is "US").
    - keyword (str): The search term to get trends for (default is "youtube").

    Returns:
    - DataFrame containing the interest over time for the keyword or a message indicating no data was found.
    """
    try:
        # Create a TrendReq object
        pytrends = TrendReq(hl='en-US', tz=360)
        
        # Build the payload with the user-provided keyword
        pytrends.build_payload([keyword], cat=0, timeframe="now 7-d", geo=region_code, gprop="")
        
        # Get interest over time
        weekly_trends = pytrends.interest_over_time()
        
        # Check if the DataFrame is empty
        if weekly_trends.empty:
            return "No data found for the provided region or keyword."

        return weekly_trends

    except ResponseError as e:
        return f"An error occurred: {str(e)}"
    except KeyError as e:
        return f"An error occurred: Missing key in response data: {str(e)}"
    
    
# KEYWORD RESEARCH
def research_keywords(q, location, api_key):
    try:
        # Format location
        location = location.replace(" ", "+")  # Replace spaces with +
        search = GoogleSearch({
            "q": q, 
            "location": location,
            "api_key": api_key
        })
        result = search.get_dict()
        
        # Extract specific data
        related_searches_df = pd.json_normalize(result['related_searches'])
        
        return related_searches_df
    
    except Exception as e:
        return {"error": str(e)}


def get_search_volume(keyword, location):
    try:
        pytrends.build_payload([keyword], cat=0, timeframe="now 7-d", geo=location, gprop="")
        search_volume = pytrends.interest_over_time()
        
        if search_volume.empty:
            return pd.DataFrame()  # Return empty DataFrame
        
        # Convert to DataFrame and reset index
        df = search_volume[[keyword]].reset_index()
        
        # Convert 'date' column to datetime format
        df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

def show_related_keywords(keyword, location):
    try:
        # Build payload for Google Trends
        pytrends.build_payload([keyword], cat=0, timeframe="now 7-d", geo=location, gprop="")
        
        # Get rising related queries
        related_keywords = pytrends.suggestions(keyword=keyword)
        
        # Return related queries as DataFrame
        return pd.DataFrame(related_keywords)
    
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

def analyze_keyword_competition(keyword, location):
    try:
        # Build payload for Google Trends
        pytrends.build_payload([keyword], cat=0, timeframe="now 7-d", geo=location, gprop="")
        
        # Get interest by region
        interest_by_region = pytrends.interest_by_region()
        
        # Check if interest_by_region is empty
        if interest_by_region.empty:
            return pd.DataFrame()  # Return empty DataFrame
        
        # Convert to DataFrame and reset index
        df = interest_by_region[[keyword]].reset_index()
        
        return df
    
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})


def get_best_tags(keyword, location):
    try:
        # Build payload for Google Trends
        pytrends.build_payload([keyword], cat=0, timeframe="now 7-d", geo=location, gprop="")
        
        # Get suggestions
        # Get suggestions
        suggestions = pytrends.suggestions(keyword=keyword)
        
        # Check if suggestions are empty
        if not suggestions:
            return pd.DataFrame({"Message": ["No tags found"]})
        
        # Extract titles from suggestions
        titles = [suggestion["title"] for suggestion in suggestions]
        
        # Create DataFrame with top suggestions
        df = pd.DataFrame(titles, columns=["Tag"])
        
        # Add hashtag to each tag
        df["Tag"] = "#" + df["Tag"]
        
        return df.head(5)  # Return top 5 suggestions
    
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})


def show_keyword_trends(keyword, location):
    try:
        pytrends.build_payload([keyword], cat=0, timeframe="now 7-d", geo=location, gprop="")
        trends = pytrends.interest_over_time()
        
        # Check if trends are empty
        if trends.empty:
            return pd.DataFrame({"error": [f"No trends found for '{keyword}'"]})
        
        # Reset index
        trends.reset_index(inplace=True)
        
        # Plot trends
        plt.figure(figsize=(10, 6))
        plt.plot(trends['date'], trends[keyword])
        plt.title(f"{keyword} Trends over Time")
        plt.xlabel("Date")
        plt.ylabel("Interest")
        plt.grid(True)
        plt.tight_layout()
        # Use Streamlit to display the plot
        st.pyplot(plt)
        
        return trends

    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

def compare_keywords(keyword1, keyword2, location):
    try:
        pytrends.build_payload([keyword1, keyword2], cat=0, timeframe="now 7-d", geo=location, gprop="")
        comparison_data = pytrends.interest_over_time()
    
    # Check if the DataFrame is empty
        if comparison_data.empty:
            return pd.DataFrame({"error": [f"No comparison data found for '{keyword1}' and '{keyword2}'"]})
        
        # Reset index to turn 'date' into a column
        comparison_data.reset_index(inplace=True)

        # Plot the comparison data
        plt.figure(figsize=(12, 6))
        plt.plot(comparison_data['date'], comparison_data[keyword1], marker='o', label=keyword1)
        plt.plot(comparison_data['date'], comparison_data[keyword2], marker='s', label=keyword2)
        plt.title(f"Interest Over Time: {keyword1} vs {keyword2}")
        plt.xlabel("Date")
        plt.ylabel("Interest")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # Use Streamlit to display the plot
        st.pyplot(plt)

        return comparison_data

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame({"error": [str(e)]})

def generate_tags(keyword, location):
    # Define prompt for tag generation
    prompt = f"Generate 10 relevant tags for '{keyword} that is mostly used in {location}'"

    # Use OpenAI API to generate tags
    response = OpenAI.Completion.create(
        model="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=60,
        n=1,
        stop=None
    )

    # Extract generated tags
    generated_tags = response.choices[0].text.splitlines()

    # Remove duplicates and add hashtag
    generated_tags = list(set([f"#{tag.replace(' ', '')}" for tag in generated_tags]))

    # Create DataFrame with generated tags
    df = pd.DataFrame(generated_tags, columns=["Tag"])

    # Return DataFrame
    return df

# PERFORMANCE INSIGHT
# Function to authenticate and return the service
def show_traffic_sources(channel_id, start_date, end_date):
    try:
        youtubeAnalytics = get_service()
        result = execute_api_request(
            youtubeAnalytics.reports().query,
            ids=f"channel=={channel_id}",
            startDate=start_date,
            endDate=end_date,
            metrics="views,estimatedMinutesWatched",
            dimensions="insightTrafficSourceType",
            sort="-views"
        )
        
        if result is None:
            return {"error": "API request failed"}
        
        # Extract column names from API response
        columns = [item["name"] for item in result.get("columnHeaders", [])]
        
        return pd.DataFrame(result.get("rows", []), columns=columns)
    
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

def get_audience_retention_data(channel_id, video_id, start_date, end_date):
    youtubeAnalytics = get_service()
    if youtubeAnalytics is None:
        return {"error": "Failed to authenticate with YouTube Analytics API"}
    
    try:
        result = execute_api_request(
            youtubeAnalytics.reports().query,
            ids=f"channel=={channel_id}",  # or "contentOwner==CONTENT_OWNER_ID"
            startDate=start_date,
            endDate=end_date,
            dimensions="elapsedVideoTimeRatio",
            metrics="audienceWatchRatio,relativeRetentionPerformance",
            filters=f"video=={video_id};audienceType==ORGANIC",
            sort="elapsedVideoTimeRatio"
        )
        
        if result is None:
            return {"error": "API request failed"}
        
        columns = [item["name"] for item in result.get("columnHeaders", [])]
        
        return pd.DataFrame(result.get("rows", []), columns=columns)
    
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

def analyze_click_through_rate(channel_id, start_date, end_date):
    youtubeAnalytics = get_service()
    result = execute_api_request(
        youtubeAnalytics.reports().query,
        ids=f"channel=={channel_id}",
        startDate=start_date,
        endDate=end_date,
        metrics="cardClickRate,cardTeaserClickRate",
        dimensions="cardType",
        sort="cardType"
    )
    return pd.DataFrame(result.get("rows", []), columns=["cardType", "cardClickRate", "cardTeaserClickRate"])

def show_viewer_demographics(channel_id, start_date, end_date):
    youtubeAnalytics = get_service()
    result = execute_api_request(
        youtubeAnalytics.reports().query,
        ids=f"channel=={channel_id}",
        startDate=start_date,
        endDate=end_date,
        metrics="viewerPercentage",
        dimensions="ageGroup,gender"
    )
    return pd.DataFrame(result.get("rows", []), columns=["ageGroup", "gender", "viewerPercentage"])

def get_watch_time_analytics(channel_id, start_date, end_date):
    youtubeAnalytics = get_service()
    result = execute_api_request(
        youtubeAnalytics.reports().query,
        ids=f"channel=={channel_id}",
        startDate=start_date,
        endDate=end_date,
        metrics="estimatedMinutesWatched,averageViewDuration",
        dimensions="day",
        sort="day"
    )
    return pd.DataFrame(result.get("rows", []), columns=["day", "estimatedMinutesWatched", "averageViewDuration"])

def show_engagement_patterns(channel_id, start_date, end_date):
    youtubeAnalytics = get_service()
    result = execute_api_request(
        youtubeAnalytics.reports().query,
        ids=f"channel=={channel_id}",
        startDate=start_date,
        endDate=end_date,
        metrics="likes,dislikes,comments,shares",
        dimensions="day",
        sort="day"
    )
    return pd.DataFrame(result.get("rows", []), columns=["day", "likes", "dislikes", "comments", "shares"])

def compare_performance_metrics(channel_id, start_date, end_date, metrics1, metrics2):
    youtubeAnalytics = get_service()
    result = execute_api_request(
        youtubeAnalytics.reports().query,
        ids=f"channel=={channel_id}",
        startDate=start_date,
        endDate=end_date,
        metrics="views,likes,subscribersGained,shares",
        dimensions="day",
        sort="day"
    )
    return pd.DataFrame(result.get("rows", []), columns=["day", "views", "likes", "subscribersGained", "shares"])

def get_viewer_behavior_insights(channel_id, start_date, end_date):
    youtubeAnalytics = get_service()
    result = execute_api_request(
        youtubeAnalytics.reports().query,
        ids=f"channel=={channel_id}",
        startDate=start_date,
        endDate=end_date,
        metrics="views,averageViewDuration,audienceWatchRatio",
        dimensions="elapsedVideoTimeRatio",
        sort="elapsedVideoTimeRatio"
    )
    return pd.DataFrame(result.get("rows", []), columns=["elapsedVideoTimeRatio", "views", "averageViewDuration", "audienceWatchRatio"])

# BASIC EARNINGS ESTIMATES FUNCTIONS
def channel_earnings_estimator(channel_id, start_date, end_date):
    youtube_analytics = get_service()
    result = execute_api_request(
        youtube_analytics.reports().query,
        ids=f"channel=={channel_id}",
        startDate=start_date,
        endDate=end_date,
        dimensions="",
        metrics="estimatedRevenue"
    )
    return result.get("rows", [{}])[0].get("estimatedRevenue")

def revenue_analysis_report(channel_id, start_date, end_date):
    youtube_analytics = get_service()
    result = execute_api_request(
        youtube_analytics.reports().query,
        ids=f"channel=={channel_id}",
        startDate=start_date,
        endDate=end_date,
        dimensions="",
        metrics="estimatedRevenue"
    )
    return pd.DataFrame(result.get("rows", []), columns=["estimatedRevenue"])

def url_earnings_potential(video_id, start_date, end_date):
    youtube_analytics = get_service()
    result = execute_api_request(
        youtube_analytics.reports().query,
        ids=f"video=={video_id}",
        startDate=start_date,
        endDate=end_date,
        dimensions="",
        metrics="estimatedRevenue"
    )
    return result.get("rows", [{}])[0].get("estimatedRevenue")

def monthly_channel_earnings_forecast(channel_id, start_date, end_date):
    youtube_analytics = get_service()
    result = execute_api_request(
        youtube_analytics.reports().query,
        ids=f"channel=={channel_id}",
        startDate=start_date,
        endDate=end_date,
        metrics="estimatedEarnings",
        dimensions="month"
    )
    return pd.DataFrame(result.get("rows", []), columns=["month", "estimatedEarnings"])

def detailed_earnings_breakdown(channel_id, start_date, end_date):
    youtube_analytics = get_service()
    result = execute_api_request(
        youtube_analytics.reports().query,
        ids=f"channel=={channel_id}",
        startDate=start_date,
        endDate=end_date,
        metrics="estimatedEarnings,estimatedRevenue,monetizedPlaybacks",
        dimensions="day"
    )
    return pd.DataFrame(result.get("rows", []), columns=["day", "estimatedEarnings", "estimatedRevenue", "monetizedPlaybacks"])

def url_revenue_potential_analysis(video_id, start_date, end_date):
    youtube_analytics = get_service()
    result = execute_api_request(
        youtube_analytics.reports().query,
        ids=f"video=={video_id}",
        startDate=start_date,
        endDate=end_date,
        metrics="estimatedRevenue",
        dimensions=""
    )
    return result.get("rows", [{}])[0].get("estimatedRevenue")

def channel_earnings_projections(channel_id, start_date, end_date):
    youtube_analytics = get_service()
    result = execute_api_request(
        youtube_analytics.reports().query,
        ids=f"channel=={channel_id}",
        startDate=start_date,
        endDate=end_date,
        metrics="estimatedEarnings",
        dimensions="month"
    )
    return pd.DataFrame(result.get("rows", []), columns=["month", "estimatedEarnings"])

def cpm_estimator(channel_id, start_date, end_date):
    youtube_analytics = get_service()
    result = execute_api_request(
        youtube_analytics.reports().query,
        ids=f"channel=={channel_id}",
        startDate=start_date,
        endDate=end_date,
        metrics="estimatedRevenue, monetizedPlaybacks"
    )
    earnings = result.get("rows", [{}])[0].get("estimatedRevenue")
    playbacks = result.get("rows", [{}])[0].get("monetizedPlaybacks")
    cpm = (float(earnings) / float(playbacks)) * 1000
    return cpm

# Revenue Stream Analysis functions
def channel_ad_revenue_report(channel_id, start_date, end_date):
    youtube_analytics = get_service()
    result = execute_api_request(
        youtube_analytics.reports().query,
        ids=f"channel=={channel_id}",
        startDate=start_date,
        endDate=end_date,
        metrics="estimatedAdRevenue",
        dimensions=""
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "Estimated Earnings from Ad Views": row["metrics"][0]["estimatedAdRevenue"]
        })
    return pd.DataFrame(data)

# def super_chat_earnings_tracker(channel_id, start_date, end_date):
#     youtube_analytics = get_service()
#     result = execute_api_request(
#         youtube_analytics.reports().query,
#         ids=f"channel=={channel_id}",
#         startDate=start_date,
#         endDate=end_date,
#         metrics="superChatEarnings"
#     )
#     rows = result.get("rows", [])
#     data = []
#     for row in rows:
#         data.append({
#             "Date": row["dimensions"][0],
#             "Super Chat Earnings": row["metrics"][0]["superChatEarnings"]
#         })
#     return pd.DataFrame(data)

# def memberships_revenue_analysis(channel_id, start_date, end_date):
#     youtube_analytics = get_service()
#     result = execute_api_request(
#         youtube_analytics.reports().query,
#         ids=f"channel=={channel_id}",
#         startDate=start_date,
#         endDate=end_date,
#         metrics="membershipsRevenue"
#     )
#     rows = result.get("rows", [])
#     data = []
#     for row in rows:
#         data.append({
#             "Date": row["dimensions"][0],
#             "Memberships Revenue": row["metrics"][0]["membershipsRevenue"]
#         })
#     return pd.DataFrame(data)

# def merchandise_sales_insights(channel_id, start_date, end_date):
#     youtube_analytics = get_service()
#     result = execute_api_request(
#         youtube_analytics.reports().query,
#         ids=f"channel=={channel_id}",
#         startDate=start_date,
#         endDate=end_date,
#         metrics="merchandiseSales"
#     )
#     rows = result.get("rows", [])
#     data = []
#     for row in rows:
#         data.append({
#             "Date": row["dimensions"][0],
#             "Merchandise Sales": row["metrics"][0]["merchandiseSales"]
#         })
#     return pd.DataFrame(data)

# def super_thanks_revenue_breakdown(channel_id, start_date, end_date):
#     youtube_analytics = get_service()
#     result = execute_api_request(
#         youtube_analytics.reports().query,
#         ids=f"channel=={channel_id}",
#         startDate=start_date,
#         endDate=end_date,
#         metrics="superThanksRevenue"
#     )
#     rows = result.get("rows", [])
#     data = []
#     for row in rows:
#         data.append({
#             "Date": row["dimensions"][0],
#             "Super Thanks Revenue": row["metrics"][0]["superThanksRevenue"]
#         })
#     return pd.DataFrame(data)

def youtube_premium_revenue_share_calculator(channel_id, start_date, end_date):
    youtube_analytics = get_service()
    result = execute_api_request(
        youtube_analytics.reports().query,
        ids=f"channel=={channel_id}",
        startDate=start_date,
        endDate=end_date,
        metrics="estimatedRedPartnerRevenue",
        dimensions=""
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "YouTube Premium Revenue": row["metrics"][0]["estimatedRedPartnerRevenue"]
        })
    return pd.DataFrame(data)

# def super_stickers_earnings_estimator(channel_id, start_date, end_date):
#     youtube_analytics = get_service()
#     result = execute_api_request(
#         youtube_analytics.reports().query,
#         ids=f"channel=={channel_id}",
#         startDate=start_date,
#         endDate=end_date,
#         metrics="superStickersEarnings"
#     )
#     rows = result.get("rows", [])
#     data = []
#     for row in rows:
#         data.append({
#             "Date": row["dimensions"][0],
#             "Super Stickers Earnings": row["metrics"][0]["superStickersEarnings"]
#         })
#     return pd.DataFrame(data)

# EARNINGS BY TIME PERIOD FUNCTIONS
def thirty_day_revenue_snapshot(channel_id, start_date, end_date):
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')
    youtube_analytics = get_service()
    result = execute_api_request(
        youtube_analytics.reports().query,
        ids=f"channel=={channel_id}",
        startDate=start_date,
        endDate=end_date,
        metrics="estimatedRevenue",
        dimensions=""
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "Estimated Revenue": row["metrics"][0]["estimatedRevenue"]
        })
    return pd.DataFrame(data)

def annualized_revenue_estimate(channel_id, start_date, end_date):
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    youtube_analytics = get_service()
    result = execute_api_request(
        youtube_analytics.reports().query,
        ids=f"channel=={channel_id}",
        startDate=start_date,
        endDate=end_date,
        metrics="estimatedRevenue",
        dimensions=""
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "Estimated Revenue": row["metrics"][0]["estimatedRevenue"]
        })
    return pd.DataFrame(data)

def quarterly_earnings_analysis(channel_id, quarter, start_date, end_date):
    if quarter == 1:
        start_date = f"{datetime.today().year}-01-01"
        end_date = f"{datetime.today().year}-03-31"
    elif quarter == 2:
        start_date = f"{datetime.today().year}-04-01"
        end_date = f"{datetime.today().year}-06-30"
    elif quarter == 3:
        start_date = f"{datetime.today().year}-07-01"
        end_date = f"{datetime.today().year}-09-30"
    else:
        start_date = f"{datetime.today().year}-10-01"
        end_date = f"{datetime.today().year}-12-31"
    youtube_analytics = get_service()
    result = execute_api_request(
        youtube_analytics.reports().query,
        ids=f"channel=={channel_id}",
        metrics="estimatedRevenue"
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "Estimated Earnings": row["metrics"][0]["estimatedEarnings"]
        })
    return pd.DataFrame(data)

def monthly_growth_tracker(channel_id, start_date, end_date):
    youtube_analytics = get_service()
    result = execute_api_request(
        youtube_analytics.reports().query,
        ids=f"channel=={channel_id}",
        startDate=start_date,
        endDate=end_date,
        metrics="estimatedRevenue",
        dimensions=""
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "Estimated Revenue": row["metrics"][0]["estimatedRevenue"]
        })
    return pd.DataFrame(data)

def weekly_earnings_summary(channel_id, start_date, end_date):
    youtube_analytics = get_service()
    result = execute_api_request(
        youtube_analytics.reports().query,
        ids=f"channel=={channel_id}",
        startDate=start_date,
        endDate=end_date,
        dimensions="",
        metrics="estimatedRevenue"
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "Estimated Revenue": row["metrics"][0]["estimatedRevenue"]
        })
    return pd.DataFrame(data)

def earnings_comparison_tool(channel_id, start_date, end_date):
    youtube_analytics = get_service()
    result = execute_api_request(
        youtube_analytics.reports().query,
        ids=f"channel=={channel_id}",
        startDate=start_date,
        endDate=end_date,
        dimensions="",
        metrics="estimatedRevenue"
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "Estimated Revenue": row["metrics"][0]["estimatedRevenue"]
        })
    return pd.DataFrame(data)

def year_over_year_growth_rate(channel_id, start_date, end_date):
    youtube_analytics = get_service()
    result = execute_api_request(
        youtube_analytics.reports().query,
        ids=f"channel=={channel_id}",
        startDate=start_date,
        endDate=end_date,
        dimensions="",
        metrics="estimatedRevenue"
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "Estimated Revenue": row["metrics"][0]["estimatedRevenue"]
        })
    return pd.DataFrame(data)

def six_month_revenue_outlook(channel_id, start_date, end_date):
    youtube_analytics = get_service()
    result = execute_api_request(
        youtube_analytics.reports().query,
        ids=f"channel=={channel_id}",
        startDate=start_date,
        endDate=end_date,
        dimensions="",
        metrics="estimatedRevenue"
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "Estimated Revenue": row["metrics"][0]["estimatedRevenue"]
        })
    return pd.DataFrame(data)

# REGIONAL COMPLIANCE CHECKS
def content_restrictions_by_country(channel_id, country):
    youtube = get_service()
    result = execute_api_request(
        youtube.videos().list,
        part="contentDetails",
        id=channel_id,
        regionCode=country
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Video ID": row["id"],
            "Content Restrictions": row["contentDetails"]["contentRating"]
        })
    return pd.DataFrame(data)

def age_rating_requirements_by_region(channel_id, region):
    youtube = get_service()
    result = execute_api_request(
        youtube.videos().list,
        part="contentDetails",
        id=channel_id,
        regionCode=region
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Video ID": row["id"],
            "Age Rating": row["contentDetails"]["contentRating"]["ytRating"]
        })
    return pd.DataFrame(data)

def monetization_policies_by_country(channel_id, country):
    youtube = get_service()
    result = execute_api_request(
        youtube.videos().list,
        part="monetizationDetails",
        id=channel_id,
        regionCode=country
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Video ID": row["id"],
            "Monetization Policy": row["monetizationDetails"]["access"]
        })
    return pd.DataFrame(data)

def content_guidelines_analysis_by_region(channel_id, region):
    youtube = get_service()
    result = execute_api_request(
        youtube.videos().list,
        part="contentDetails",
        id=channel_id,
        regionCode=region
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Video ID": row["id"],
            "Content Guidelines": row["contentDetails"]["contentRating"]["reasons"]
        })
    return pd.DataFrame(data)

def restricted_keywords_by_country(channel_id, country):
    youtube = get_service()
    result = execute_api_request(
        youtube.videos().list,
        part="snippet",
        id=channel_id,
        regionCode=country
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Video ID": row["id"],
            "Restricted Keywords": row["snippet"]["tags"]
        })
    return pd.DataFrame(data)

def copyright_rules_by_region(channel_id, region):
    youtube = get_service()
    result = execute_api_request(
        youtube.videos().list,
        part="contentDetails",
        id=channel_id,
        regionCode=region
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Video ID": row["id"],
            "Copyright Rules": row["contentDetails"]["copyright"]
        })
    return pd.DataFrame(data)

def advertising_restrictions_check(channel_id):
    youtube = get_service()
    result = execute_api_request(
        youtube.videos().list,
        part="monetizationDetails",
        id=channel_id
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Video ID": row["id"],
            "Advertising Restrictions": row["monetizationDetails"]["access"]
        })
    return pd.DataFrame(data)

def regional_policy_compliance(channel_id, region):
    youtube = get_service()
    result = execute_api_request(
        youtube.videos().list,
        part="contentDetails,monetizationDetails,snippet",
        id=channel_id,
        regionCode=region
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Video ID": row["id"],
            "Content Restrictions": row["contentDetails"]["contentRating"],
            "Monetization Policy": row["monetizationDetails"]["access"],
            "Restricted Keywords": row["snippet"]["tags"],
            "Copyright Rules": row["contentDetails"]["copyright"]
        })
    return pd.DataFrame(data)

# LOCALIZATION FUNCTIONS
def translation_quality_by_language(video_id, language):
    youtube = get_service()
    result = execute_api_request(
        youtube.videos().list,
        part="id,snippet",
        id=video_id,
        hl=language
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Video ID": row["id"],
            "Translation Quality": row["snippet"]["title"],
            "Language": language
        })
    return pd.DataFrame(data)

def caption_performance_in_countries(video_id, country):
    youtube = get_service()
    result = execute_api_request(
        youtube.videos().list,
        part="id,snippet",
        id=video_id,
        regionCode=country
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Video ID": row["id"],
            "Caption Performance": row["snippet"]["captions"],
            "Country": country
        })
    return pd.DataFrame(data)

def subtitle_recommendations(video_id):
    youtube = get_service()
    result = execute_api_request(
        youtube.videos().list,
        part="id,snippet",
        id=video_id
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Video ID": row["id"],
            "Subtitle Recommendations": row["snippet"]["tags"]
        })
    return pd.DataFrame(data)

def dubbing_impact_analysis_in_regions(video_id, region):
    youtube = get_service()
    result = execute_api_request(
        youtube.videos().list,
        part="id,snippet",
        id=video_id,
        regionCode=region
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Video ID": row["id"],
            "Dubbing Impact": row["snippet"]["title"],
            "Region": region
        })
    return pd.DataFrame(data)

def localization_opportunities(video_id):
    youtube = get_service()
    result = execute_api_request(
        youtube.videos().list,
        part="id,snippet",
        id=video_id
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Video ID": row["id"],
            "Localization Opportunities": row["snippet"]["description"]
        })
    return pd.DataFrame(data)

def translation_suggestions(video_id):
    youtube = get_service()
    result = execute_api_request(
        youtube.videos().list,
        part="id,snippet",
        id=video_id
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Video ID": row["id"],
            "Translation Suggestions": row["snippet"]["tags"]
        })
    return pd.DataFrame(data)

def multi_language_performance_analysis(video_id):
    youtube = get_service()
    result = execute_api_request(
        youtube.videos().list,
        part="id,snippet",
        id=video_id
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Video ID": row["id"],
            "Multi-Language Performance": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

def localization_return_on_investment(video_id):
    youtube = get_service()
    result = execute_api_request(
        youtube.videos().list,
        part="id,snippet",
        id=video_id
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Video ID": row["id"],
            "Localization ROI": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

# CROSS-BORDER ANALYSIS FUNCRION
def cross_border_trend_comparison(video_id, region1, region2, start_date, end_date):
    youtube = get_service()
    
    # Fetch data for both region)
    request1 = youtube.reports().query(
        ids=f"channel=={channel_id}",
        startDate=start_date,
        endDate=end_date,
        metrics="views,likes,comments",
        dimensions="country",
        filters=f"video=={video_id};country=={region1}"
    )
    response1 = request1.execute()
    data_region1 = response1.get("rows", [[region1, 0, 0, 0]])[0]
    
    request2 = youtube.reports().query(
        ids=f"channel=={channel_id}",
        startDate=start_date,
        endDate=end_date,
        metrics="views,likes,comments",
        dimensions="country",
        filters=f"video=={video_id};country=={region2}"
    )
    response2 = request2.execute()
    data_region2 = response2.get("rows", [[region2, 0, 0, 0]])[0]
    
    # Prepare data for a DataFrame
    data = [
        {
            "Video ID": video_id,
            "Region": data_region1[0],
            "View Count": data_region1[1],
            "Like Count": data_region1[2],
            "Comment Count": data_region1[3],
            "Engagement Rate": (
                (int(data_region1[2]) + int(data_region1[3])) / int(data_region1[1])
                if int(data_region1[1]) > 0 else 0
            )
        },
        {
            "Video ID": video_id,
            "Region": data_region2[0],
            "View Count": data_region2[1],
            "Like Count": data_region2[2],
            "Comment Count": data_region2[3],
            "Engagement Rate": (
                (int(data_region2[2]) + int(data_region2[3])) / int(data_region2[1])
                if int(data_region2[1]) > 0 else 0
            )
        }
    ]

    # Create a DataFrame from the data
    df = pd.DataFrame(data)
    return df

    # # Extract values from lists
    # regions = [item[0] for item in df['Region']]
    # view_counts1 = [int(item[0]) for item in df['View Count']]
    # view_counts2 = [int(item[1]) for item in df['View Count']]
    # like_counts1 = [int(item[0]) for item in df['Like Count']]
    # like_counts2 = [int(item[1]) for item in df['Like Count']]
    # comment_counts1 = [int(item[0]) for item in df['Comment Count']]
    # comment_counts2 = [int(item[1]) for item in df['Comment Count']]
    # engagement_rates1 = [float(item[0]) for item in df['Engagement Rate']]
    # engagement_rates2 = [float(item[1]) for item in df['Engagement Rate']]

    # # Extract values
    # regions = df['Region'].tolist()
    # view_counts = df['View Count'].tolist()
    # like_counts = df['Like Count'].tolist()
    # comment_counts = df['Comment Count'].tolist()
    
    # # Check 'Engagement Rate' column type
    # print(df['Engagement Rate'].dtype)
    
    # # Convert 'Engagement Rate' to float if necessary
    # engagement_rates = df['Engagement Rate'].apply(lambda x: float(x) if isinstance(x, (int, float)) else 0).tolist()

    # # Plotting
    # fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # axs[0, 0].bar(regions, view_counts)
    # axs[0, 0].set_title('View Count Comparison')

    # axs[0, 1].bar(regions, like_counts)
    # axs[0, 1].set_title('Like Count Comparison')

    # axs[1, 0].bar(regions, comment_counts)
    # axs[1, 0].set_title('Comment Count Comparison')

    # axs[1, 1].bar(regions, engagement_rates)
    # axs[1, 1].set_title('Engagement Rate Comparison')

    # plt.tight_layout()
    # st.pyplot(fig)
    
def multi_country_content_performance(video_id, countries):
    youtube = get_service()
    data = []
    for country in countries:
        result = execute_api_request(
            youtube.videos().list,
            part="id,snippet",
            id=video_id,
            regionCode=country
        )
        rows = result.get("items", [])
        for row in rows:
            data.append({
                "Video ID": row["id"],
                "Country": country,
                "Content Performance": row["snippet"]["title"]
            })
    return pd.DataFrame(data)

def international_reach_analysis(video_id):
    youtube = get_service()
    result = execute_api_request(
        youtube.videos().list,
        part="id,snippet",
        id=video_id
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Video ID": row["id"],
            "International Reach": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

def global_vs_local_metrics(video_id):
    youtube = get_service()
    result = execute_api_request(
        youtube.videos().list,
        part="id,snippet",
        id=video_id
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Video ID": row["id"],
            "Global Metrics": row["snippet"]["title"],
            "Local Metrics": row["snippet"]["description"]
        })
    return pd.DataFrame(data)

def regional_content_adaptation(video_id, regions):
    youtube = get_service()
    data = []
    for region in regions:
        result = execute_api_request(
            youtube.videos().list,
            part="id,snippet",
            id=video_id,
            regionCode=region
        )
        rows = result.get("items", [])
        for row in rows:
            data.append({
                "Video ID": row["id"],
                "Region": region,
                "Content Adaptation": row["snippet"]["title"]
            })
    return pd.DataFrame(data)

def market_engagement_comparison(video_id, markets):
    youtube = get_service()
    data = []
    for market in markets:
        result = execute_api_request(
            youtube.videos().list,
            part="id,snippet",
            id=video_id,
            regionCode=market
        )
        rows = result.get("items", [])
        for row in rows:
            data.append({
                "Video ID": row["id"],
                "Market": market,
                "Engagement Comparison": row["snippet"]["title"]
            })
    return pd.DataFrame(data)

def global_expansion_potential(video_id):
    youtube = get_service()
    result = execute_api_request(
        youtube.videos().list,
        part="id,snippet",
        id=video_id
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Video ID": row["id"],
            "Global Expansion Potential": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

def multi_region_performance(video_id, regions):
    youtube = get_service()
    data = []
    for region in regions:
        result = execute_api_request(
            youtube.videos().list,
            part="id,snippet",
            id=video_id,
            regionCode=region
        )
        rows = result.get("items", [])
        for row in rows:
            data.append({
                "Video ID": row["id"],
                "Region": region,
                "Performance": row["snippet"]["title"]
            })
    return pd.DataFrame(data)

# Regional Content Strategy functions
def country_specific_content_ideas(country):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"top videos in {country}",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Country": country,
            "Content Idea": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

def regional_title_optimization(region, language):
    youtube = get_service()
    result = execute_api_request(
        youtube.videos().list,
        part="id,snippet",
        regionCode=region,
        hl=language
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Region": region,
            "Language": language,
            "Optimized Title": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

def country_thumbnail_preferences(country):
    youtube = get_service()
    result = execute_api_request(
        youtube.videos().list,
        part="id,snippet",
        regionCode=country
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Country": country,
            "Thumbnail Preference": row["snippet"]["thumbnails"]["default"]["url"]
        })
    return pd.DataFrame(data)

def format_analysis_for_region(region):
    youtube = get_service()
    result = execute_api_request(
        youtube.videos().list,
        part="id,snippet",
        regionCode=region
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Region": region,
            "Format": row["snippet"]["categoryId"]
        })
    return pd.DataFrame(data)

def local_keyword_research(country, language):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"top keywords in {country} {language}",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Country": country,
            "Language": language,
            "Keyword": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

def content_style_guidance(region):
    youtube = get_service()
    result = execute_api_request(
        youtube.videos().list,
        part="id,snippet",
        regionCode=region
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Region": region,
            "Content Style": row["snippet"]["description"]
        })
    return pd.DataFrame(data)

def regional_description_templates(region):
    youtube = get_service()
    result = execute_api_request(
        youtube.videos().list,
        part="id,snippet",
        regionCode=region
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Region": region,
            "Description Template": row["snippet"]["description"]
        })
    return pd.DataFrame(data)

def country_tag_suggestions(country):
    youtube = get_service()
    result = execute_api_request(
        youtube.videos().list,
        part="id,snippet",
        regionCode=country
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Country": country,
            "Tag Suggestion": row["snippet"]["tags"]
        })
        
# Local Competition Analysis functions
def country_top_creators(country):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"top creators in {country}",
        type="channel"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Country": country,
            "Creator": row["snippet"]["title"],
            "Subscribers": row["snippet"]["subscriberCount"]
        })
    return pd.DataFrame(data)

def regional_competitor_analysis(region):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"top channels in {region}",
        type="channel"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Region": region,
            "Competitor": row["snippet"]["title"],
            "Subscribers": row["snippet"]["subscriberCount"]
        })
    return pd.DataFrame(data)

def cross_country_creator_comparison(country1, country2):
    youtube = get_service()
    result1 = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"top creators in {country1}",
        type="channel"
    )
    result2 = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"top creators in {country2}",
        type="channel"
    )
    rows1 = result1.get("items", [])
    rows2 = result2.get("items", [])
    data = []
    for row1, row2 in zip(rows1, rows2):
        data.append({
            "Country": [country1, country2],
            "Creator": [row1["snippet"]["title"], row2["snippet"]["title"]],
            "Subscribers": [row1["snippet"]["subscriberCount"], row2["snippet"]["subscriberCount"]]
        })
    return pd.DataFrame(data)

def market_leader_identification(market):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"top channels in {market}",
        type="channel"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Market": market,
            "Leader": row["snippet"]["title"],
            "Subscribers": row["snippet"]["subscriberCount"]
        })
    return pd.DataFrame(data)

def niche_competitor_finder(niche):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"top channels in {niche}",
        type="channel"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Niche": niche,
            "Competitor": row["snippet"]["title"],
            "Subscribers": row["snippet"]["subscriberCount"]
        })
    return pd.DataFrame(data)

def local_channel_strategy_insights(country):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"top channels in {country}",
        type="channel"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Country": country,
            "Channel": row["snippet"]["title"],
            "Strategy": row["snippet"]["description"]
        })
    return pd.DataFrame(data)

def regional_performance_benchmarks(region):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"top channels in {region}",
        type="channel"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Region": region,
            "Channel": row["snippet"]["title"],
            "Views": row["snippet"]["viewCount"],
            "Subscribers": row["snippet"]["subscriberCount"]
        })
    return pd.DataFrame(data)

def market_share_analysis(market):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"top channels in {market}",
        type="channel"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Market": market,
            "Channel": row["snippet"]["title"],
            "Share": row["snippet"]["viewCount"] / sum([r["snippet"]["viewCount"] for r in rows])
        })
    return pd.DataFrame(data)
        
# Time Zone-Based Analysis functions
def best_upload_times_by_region(region):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"best upload times in {region}",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Region": region,
            "Upload Time": row["snippet"]["publishedAt"]
        })
    return pd.DataFrame(data)

def peak_viewing_hours_by_country(country):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"peak viewing hours in {country}",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Country": country,
            "Viewing Hour": row["snippet"]["publishedAt"]
        })
    return pd.DataFrame(data)

def engagement_patterns_by_timezone(timezone):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"engagement patterns in {timezone}",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Timezone": timezone,
            "Engagement": row["snippet"]["likeCount"]
        })
    return pd.DataFrame(data)

def performance_comparison_across_time_zones(timezone1, timezone2):
    youtube = get_service()
    result1 = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"performance in {timezone1}",
        type="video"
    )
    result2 = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"performance in {timezone2}",
        type="video"
    )
    rows1 = result1.get("items", [])
    rows2 = result2.get("items", [])
    data = []
    for row1, row2 in zip(rows1, rows2):
        data.append({
            "Timezone": [timezone1, timezone2],
            "Performance": [row1["snippet"]["viewCount"], row2["snippet"]["viewCount"]]
        })
    return pd.DataFrame(data)

def optimal_posting_schedule_by_region(region):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"optimal posting schedule in {region}",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Region": region,
            "Posting Schedule": row["snippet"]["publishedAt"]
        })
    return pd.DataFrame(data)

def audience_activity_times_by_country(country):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"audience activity times in {country}",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Country": country,
            "Activity Time": row["snippet"]["publishedAt"]
        })
    return pd.DataFrame(data)

def live_stream_timing_analysis_by_region(region):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"live stream timing analysis in {region}",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Region": region,
            "Timing Analysis": row["snippet"]["publishedAt"]
        })
    return pd.DataFrame(data)

def regional_prime_times(region):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"prime times in {region}",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Region": region,
            "Prime Time": row["snippet"]["publishedAt"]
        })
    return pd.DataFrame(data)

# Geographic Performance Metrics functions
def country_watch_time_distribution(country):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        dimensions="country",
        metrics="watchTimeMinutes",
        filters=f"country=={country}"
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Country": row["dimensions"][0],
            "Watch Time": row["metrics"][0]
        })
    return pd.DataFrame(data)

def regional_subscriber_demographics(region):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        dimensions="region",
        metrics="subscribersGained",
        filters=f"region=={region}"
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Region": row["dimensions"][0],
            "Subscribers": row["metrics"][0]
        })
    return pd.DataFrame(data)

def view_duration_pattern_analysis(country):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        dimensions="country",
        metrics="averageViewDuration",
        filters=f"country=={country}"
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Country": row["dimensions"][0],
            "Average View Duration": row["metrics"][0]
        })
    return pd.DataFrame(data)

def cross_regional_engagement_comparison(region1, region2):
    youtube = get_service()
    result1 = execute_api_request(
        youtube.reports().query,
        dimensions="region",
        metrics="likes,comments,shares",
        filters=f"region=={region1}"
    )
    result2 = execute_api_request(
        youtube.reports().query,
        dimensions="region",
        metrics="likes,comments,shares",
        filters=f"region=={region2}"
    )
    rows1 = result1.get("rows", [])
    rows2 = result2.get("rows", [])
    data = []
    for row1, row2 in zip(rows1, rows2):
        data.append({
            "Region": [row1["dimensions"][0], row2["dimensions"][0]],
            "Engagement": [row1["metrics"][0], row2["metrics"][0]]
        })
    return pd.DataFrame(data)

# def country_level_super_chat_revenue(country):
#     youtube = get_service()
#     result = execute_api_request(
#         youtube.reports().query,
#         dimensions="country",
#         metrics="superChatRevenue",
#         filters=f"country=={country}"
#     )
#     rows = result.get("rows", [])
#     data = []
#     for row in rows:
#         data.append({
#             "Country": row["dimensions"][0],
#             "Super Chat Revenue": row["metrics"][0]
#         })
#     return pd.DataFrame(data)

# def regional_membership_growth(region):
#     youtube = get_service()
#     result = execute_api_request(
#         youtube.reports().query,
#         dimensions="region",
#         metrics="membersGained",
#         filters=f"region=={region}"
#     )
#     rows = result.get("rows", [])
#     data = []
#     for row in rows:
#         data.append({
#             "Region": row["dimensions"][0],
#             "Membership Growth": row["metrics"][0]
#         })
#     return pd.DataFrame(data)

# def super_thanks_distribution_insights(country):
#     youtube = get_service()
#     result = execute_api_request(
#         youtube.reporting().query,
#         dimensions="country",
#         metrics="superThanksRevenue",
#         filters=f"country=={country}"
#     )
#     rows = result.get("rows", [])
#     data = []
#     for row in rows:
#         data.append({
#             "Country": row["dimensions"][0],
#             "Super Thanks Revenue": row["metrics"][0]
#         })
#     return pd.DataFrame(data)

# def country_specific_merchandise_sales(country):
#     youtube = get_service()
#     result = execute_api_request(
#         youtube.reporting().query,
#         dimensions="country",
#         metrics="merchandiseSales",
#         filters=f"country=={country}"
#     )
#     rows = result.get("rows", [])
#     data = []
#     for row in rows:
#         data.append({
#             "Country": row["dimensions"][0],
#             "Merchandise Sales": row["metrics"][0]
#         })
#     return pd.DataFrame(data)

# Cultural Trend Analysis functions
def city_regional_trend_tracker(city):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"trending in {city}",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "City": city,
            "Trend": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

def country_seasonal_trends(country):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"{country} seasonal trends",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Country": country,
            "Seasonal Trend": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

def cultural_event_impact_analysis(event):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"{event} impact",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Event": event,
            "Impact": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

def holiday_content_spotlight(holiday):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"{holiday} content",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Holiday": holiday,
            "Content": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

def local_celebrity_trend_monitor(celebrity):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"{celebrity} trending",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Celebrity": celebrity,
            "Trend": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

def regional_meme_tracker(region):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"{region} memes",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Region": region,
            "Meme": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

def local_news_trend_impact(news):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"{news} impact",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "News": news,
            "Impact": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

def festival_content_performance(festival):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"{festival} content",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Festival": festival,
            "Content Performance": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

# Market Research Commands functions
def country_market_sizing(country):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"market size in {country}",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Country": country,
            "Market Size": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

def competition_level_analysis(niche):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"{niche} competition",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Niche": niche,
            "Competition Level": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

def regional_niche_opportunities(region):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"{region} niche opportunities",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Region": region,
            "Niche Opportunity": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

def market_saturation_comparison(market1, market2):
    youtube = get_service()
    result1 = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"{market1} market saturation",
        type="video"
    )
    result2 = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"{market2} market saturation",
        type="video"
    )
    rows1 = result1.get("items", [])
    rows2 = result2.get("items", [])
    data = []
    for row1, row2 in zip(rows1, rows2):
        data.append({
            "Market": [market1, market2],
            "Saturation Level": [row1["snippet"]["title"], row2["snippet"]["title"]]
        })
    return pd.DataFrame(data)

def audience_preference_insights(audience):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"{audience} preferences",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Audience": audience,
            "Preference": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

def content_gap_identification(niche):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"{niche} content gap",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Niche": niche,
            "Content Gap": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

def ad_rate_benchmarking(niche):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"{niche} ad rates",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Niche": niche,
            "Ad Rate": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

def monetization_potential_assessment(niche):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"{niche} monetization potential",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Niche": niche,
            "Monetization Potential": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

# Language-Based Search functions
def language_trending_videos(language):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"trending in {language}",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Language": language,
            "Trending Video": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

def popular_creator_spotlight(language):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"popular creators in {language}",
        type="channel"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Language": language,
            "Popular Creator": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

def topic_trend_analysis(language, topic):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"{topic} in {language}",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Language": language,
            "Topic": topic,
            "Trend": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

def hashtag_intelligence(language, hashtag):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"{hashtag} in {language}",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Language": language,
            "Hashtag": hashtag,
            "Intelligence": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

def cross_linguistic_engagement_comparison(language1, language2):
    youtube = get_service()
    result1 = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"engagement in {language1}",
        type="video"
    )
    result2 = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"engagement in {language2}",
        type="video"
    )
    rows1 = result1.get("items", [])
    rows2 = result2.get("items", [])
    data = []
    for row1, row2 in zip(rows1, rows2):
        data.append({
            "Language": [language1, language2],
            "Engagement": [row1["snippet"]["title"], row2["snippet"]["title"]]
        })
    return pd.DataFrame(data)

def emerging_channel_tracker(language):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"emerging channels in {language}",
        type="channel"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Language": language,
            "Emerging Channel": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

def language_specific_keyword_suggestions(language):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"keyword suggestions in {language}",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Language": language,
            "Keyword Suggestion": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

def viral_short_form_videos(language):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"viral short-form videos in {language}",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Language": language,
            "Viral Short-Form Video": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

# Regional Analysis functions
def country_specific_trending_videos(country):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"trending in {country}",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Country": country,
            "Trending Video": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

def top_videos_by_country(country):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"top videos in {country}",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Country": country,
            "Top Video": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

def keyword_trend_analysis(country, keyword):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"{keyword} in {country}",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Country": country,
            "Keyword": keyword,
            "Trend": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

def cross_country_trend_comparison(country1, country2):
    youtube = get_service()
    result1 = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"trending in {country1}",
        type="video"
    )
    result2 = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"trending in {country2}",
        type="video"
    )
    rows1 = result1.get("items", [])
    rows2 = result2.get("items", [])
    data = []
    for row1, row2 in zip(rows1, rows2):
        data.append({
            "Country": [country1, country2],
            "Trend": [row1["snippet"]["title"], row2["snippet"]["title"]]
        })
    return pd.DataFrame(data)

def emerging_creators_by_country(country):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"emerging creators in {country}",
        type="channel"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Country": country,
            "Emerging Creator": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

def viral_content_tracker(country):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"viral content in {country}",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Country": country,
            "Viral Content": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

def country_level_hashtag_trends(country, hashtag):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"{hashtag} in {country}",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Country": country,
            "Hashtag": hashtag,
            "Trend": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

def popular_music_trends_by_country(country):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"popular music in {country}",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Country": country,
            "Popular Music": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

# Regional Performance Analysis functions
def regional_video_performance(country, video_id):
    youtube = get_service()
    result = execute_api_request(
        youtube.videos().list,
        part="id,snippet,statistics",
        id=video_id,
        regionCode=country
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Country": country,
            "Video Title": row["snippet"]["title"],
            "View Count": row["statistics"]["viewCount"]
        })
    return pd.DataFrame(data)

def country_specific_viewer_demographics(country, video_id):
    youtube = get_service()
    result = execute_api_request(
        youtube.analytics().query,
        dimensions="ageGroup,gender",
        metrics="viewerPercentage",
        filters=f"country=={country};video=={video_id}"
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Country": country,
            "Age Group": row["dimensions"][0],
            "Gender": row["dimensions"][1],
            "Viewer Percentage": row["metrics"][0]
        })
    return pd.DataFrame(data)

def cross_country_view_comparison(country1, country2, video_id):
    youtube = get_service()
    result1 = execute_api_request(
        youtube.videos().list,
        part="id,snippet,statistics",
        id=video_id,
        regionCode=country1
    )
    result2 = execute_api_request(
        youtube.videos().list,
        part="id,snippet,statistics",
        id=video_id,
        regionCode=country2
    )
    rows1 = result1.get("items", [])
    rows2 = result2.get("items", [])
    data = []
    for row1, row2 in zip(rows1, rows2):
        data.append({
            "Country": [country1, country2],
            "View Count": [row1["statistics"]["viewCount"], row2["statistics"]["viewCount"]]
        })
    return pd.DataFrame(data)

def country_level_engagement_rates(country, video_id):
    youtube = get_service()
    result = execute_api_request(
        youtube.videos().list,
        part="id,snippet,statistics",
        id=video_id,
        regionCode=country
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Country": country,
            "Video Title": row["snippet"]["title"],
            "Engagement Rate": row["statistics"]["engagementRate"]
        })
    return pd.DataFrame(data)

def top_performing_regions(video_id):
    youtube = get_service()
    result = execute_api_request(
        youtube.videos().list,
        part="id,snippet,statistics",
        id=video_id
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Region": row["snippet"]["region"],
            "View Count": row["statistics"]["viewCount"]
        })
    return pd.DataFrame(data)

def audience_retention_by_country(country, video_id, start_date, end_date):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        dimensions="elapsedVideoTimeRatio",
        metrics="audienceWatchRatio",
        startDate=start_date,
        endDate=end_date,
        filters=f"country=={country};video=={video_id}"
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Country": country,
            "Audience Retention": row["dimensions"][0],
            "Retention Percentage": row["metrics"][0]
        })
    return pd.DataFrame(data)

def country_originating_traffic(country, video_id, start_date, end_date):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        dimensions="insightTrafficSource",
        metrics="views",
        startDate=start_date,
        endDate=end_date,
        filters=f"country=={country};video=={video_id}"
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Country": country,
            "Traffic Source": row["dimensions"][0],
            "Views": row["metrics"][0]
        })
    return pd.DataFrame(data)

def regional_click_through_rate_comparison(country1, country2, video_id, start_date, end_date):
    youtube = get_service()
    
    # API request for country1
    result1 = execute_api_request(
        youtube.reports().query,
        ids=f"channel=={channel_id}",
        startDate=start_date,
        endDate=end_date,
        dimensions="video",
        metrics="views,estimatedRevenue",
        filters=f"country=={country1};video=={video_id}"
    )
    
    # API request for country2
    result2 = execute_api_request(
        youtube.reports().query,
        ids=f"channel=={channel_id}",
        startDate=start_date,
        endDate=end_date,
        dimensions="video",
        metrics="views,estimatedRevenue",
        filters=f"country=={country2};video=={video_id}"
    )
    
    rows1 = result1.get("rows", [])
    rows2 = result2.get("rows", [])
    
    data = []
    for row1, row2 in zip(rows1, rows2):
        views1 = row1["metrics"][0]["values"][0]
        views2 = row2["metrics"][0]["values"][0]
        
        # Assuming estimatedRevenue as a proxy for impressions
        impressions1 = row1["metrics"][1]["values"][0]
        impressions2 = row2["metrics"][1]["values"][0]
        
        ctr1 = views1 / impressions1
        ctr2 = views2 / impressions2
        
        data.append({
            "Country": [country1, country2],
            "Click-Through Rate": [ctr1, ctr2]
        })
    
    return pd.DataFrame(data)

# NATURAL LANGUAGE QUERIES FUNCTIONS
def boost_video_views(query):
    prompt = f"Provide actionable tips to increase video views for '{query}' content, including optimization strategies and engagement techniques."
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def enhance_click_through_rate(query):
    prompt = f"Suggest proven strategies to enhance click-through rate (CTR) for '{query}' videos, including title optimization, thumbnail design, and metadata improvement."
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def find_profitable_topics(query):
    prompt = f"Identify profitable topics related to '{query}' for YouTube content creators, considering audience demand, competition, and monetization potential."
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def optimal_upload_time(query):
    prompt = f"Determine the optimal upload time for '{query}' videos to maximize engagement, considering audience location, time zones, and platform algorithms."
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def grow_subscribers(query):
    prompt = f"Provide effective strategies to grow subscribers for '{query}' YouTube channels, including content optimization, engagement techniques, and audience retention methods."
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def niche_success_strategies(query):
    prompt = f"Outline niche success strategies for '{query}' YouTube content creators, considering target audience, content differentiation, and marketing tactics."
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def extract_thumbnail(video_id):
    # Extract thumbnail URL using YouTube API
    thumbnail_url = youtube.videos().list(part="snippet", id=video_id).execute()["items"][0]["snippet"]["thumbnails"]["default"]["url"]
    return thumbnail_url

def analyze_thumbnail(thumbnail_url):
    # Download and process thumbnail
    response = requests.get(thumbnail_url)
    img = Image.open(BytesIO(response.content))
    
    # Analyze thumbnail
    dominant_colors = extract_dominant_colors(img)
    composition = analyze_composition(img)
    text_overlay = detect_text_overlay(img)
    
    return dominant_colors, composition, text_overlay

def extract_dominant_colors(img):
    # Simplified color extraction using PIL
    img = img.convert('RGB')
    img = img.resize((150, 150))  # Resize to reduce computation
    img = np.array(img)
    dominant_colors = np.mean(img, axis=(0, 1)).astype(int)
    return dominant_colors

def analyze_composition(img):
    # Simplified composition analysis using OpenCV
    img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    composition = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    return composition

def detect_text_overlay(img):
    # Simplified text overlay detection using OpenCV
    img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text_overlay = cv2.countNonZero(thresh)
    return text_overlay

def thumbnail_improvement(video_id):
    thumbnail_url = extract_thumbnail(video_id)
    dominant_colors, composition, text_overlay = analyze_thumbnail(thumbnail_url)
    video_title = youtube.videos().list(part="snippet", id=video_id).execute()["items"][0]["snippet"]["title"]
    
    # Craft OpenAI prompt
    prompt = f"Analyze the thumbnail for YouTube video '{video_title}' (ID: {video_id}) and suggest improvements. Consider the following analysis:\n\nDominant colors: {dominant_colors}\nComposition: {composition}\nText overlay: {text_overlay}\n\nGoals: Increase clicks, engagement."
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def get_video_metadata(youtube, video_id):
    """Fetch video metadata using YouTube API"""
    video_response = youtube.videos().list(part="snippet", id=video_id).execute()
    video_data = video_response["items"][0]["snippet"]
    return {
        "title": video_data["title"],
        "channel_id": video_data["channelId"],
        "category_id": video_data["categoryId"]
    }

def diagnose_view_drop(video_id, date_range):
    try:
        # Fetch video metrics using YouTube API
        result = execute_api_request(
            youtube.analytics().query,
            dimensions="day",
            metrics="views",
            filters=f"video=={video_id};time={date_range}"
        )
        
        # Fetch video metadata
        video_metadata = get_video_metadata(youtube, video_id)
        video_title = video_metadata["title"]
        channel_name = youtube.channels().list(part="snippet", id=video_metadata["channel_id"]).execute()["items"][0]["snippet"]["title"]
        video_category = video_metadata["category_id"]
        
        # Extract relevant data from YouTube API response
        views_data = result.get("rows", [])
        views = [row["metrics"][0] for row in views_data]
        
        # Craft OpenAI prompt
        prompt = f"Analyze the view drop for YouTube video '{video_title}' (ID: {video_id}) by {channel_name} in the {video_category} category between {date_range}. Consider the following data:\n\nViews: {views}\nProvide insights on potential causes, including content quality, audience engagement, and platform changes."
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error: {e}")
        return None

def search_optimization_tips(query):
    prompt = f"Provide search optimization tips for '{query}' YouTube videos, including keyword research, title optimization, and metadata improvement."
    response = OpenAI.Chat.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def effective_hashtags(query):
    prompt = f"Suggest effective hashtags for '{query}' YouTube videos, considering relevance, popularity, and audience targeting."
    response = OpenAI.Chat.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# ADVANCED EARNINGS METRICS
def earnings_per_engagement_calculator(channel_id, video_id, start_date, end_date):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="estimatedRevenue,views,watchTimeMinutes,averageViewDuration,comments,likes,dislikes,shares",
        startDate=start_date,
        endDate=end_date,
        filters=f"video=={video_id}"
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        metrics = row["metrics"][0]
        earnings = metrics["estimatedRevenue"]
        engagement = sum([
            metrics["views"],
            metrics["watchTimeMinutes"],
            metrics["averageViewDuration"],
            metrics["comments"],
            metrics["likes"],
            metrics["dislikes"],
            metrics["shares"]
        ])
        earnings_per_engagement = float(earnings) / float(engagement) if engagement != 0 else 0
        data.append({
            "Video ID": video_id,
            "Date": row["dimensions"][0],
            "Earnings": earnings,
            "Engagement": engagement,
            "Earnings per Engagement": earnings_per_engagement
        })
    return pd.DataFrame(data)

def revenue_efficiency_ratio(channel_id, video_id, start_date, end_date):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="estimatedRevenue,views",
        startDate=start_date,
        endDate=end_date,
        filters=f"video=={video_id}"
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Video ID": video_id,
            "Date": row["dimensions"][0],
            "Earning": row["metrics"][0],
            "Views": row["metrics"][1],
            "Revenue Efficiency Ratio": float(row["metrics"][0]) / float(row["metrics"][1])
        })
    return pd.DataFrame(data)

def earnings_quality_metrics(channel_id, video_id, start_date, end_date):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="estimatedRevenue,monetizedPlaybacks",
        startDate=start_date,
        endDate=end_date,
        filters=f"video=={video_id}"
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Video ID": video_id,
            "Date": row["dimensions"][0],
            "Earnings": row["metrics"][0],
            "Monetizable Playbacks": row["metrics"][1],
            "Earnings Quality": float(row["metrics"][0]) / float(row["metrics"][1])
        })
    return pd.DataFrame(data)

def revenue_sustainability_score(channel_id, video_id, start_date, end_date):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="estimatedRevenue,views",
        startDate=start_date,
        endDate=end_date,
        filters=f"video=={video_id}"
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Video ID": video_id,
            "Date": row["dimensions"][0],
            "Earnings": row["metrics"][0],
            "Views": row["metrics"][1],
            "Revenue Sustainability Score": float(row["metrics"][0]) / float(row["metrics"][1])
        })
    return pd.DataFrame(data)

def monetization_health_index(channel_id, video_id, start_date, end_date):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="estimatedRevenue,monetizedPlaybacks",
        startDate=start_date,
        endDate=end_date,
        filters=f"video=={video_id}"
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Video ID": video_id,
            "Date": row["dimensions"][0],
            "Earnings": row["metrics"][0],
            "Monetizable Playbacks": row["metrics"][1],
            "Monetization Health Index": float(row["metrics"][0]) / float(row["metrics"][1])
        })
    return pd.DataFrame(data)

def earnings_diversification_ratio(channel_id, video_id, start_date, end_date):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="estimatedRevenue,views",
        startDate=start_date,
        endDate=end_date,
        filters=f"video=={video_id}"
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Video ID": video_id,
            "Date": row["dimensions"][0],
            "Earnings": row["metrics"][0],
            "Views": row["metrics"][1],
            "Earnings Diversification Ratio": float(row["metrics"][0]) / float(row["metrics"][1])
        })
    return pd.DataFrame(data)

def revenue_predictability_analysis(channel_id, video_id, start_date, end_date):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="estimatedRevenue,views",
        startDate=start_date,
        endDate=end_date,
        filters=f"video=={video_id}"
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Video ID": video_id,
            "Date": row["dimensions"][0],
            "Earnings": row["metrics"][0],
            "Views": row["metrics"][1],
            "Revenue Predictability": float(row["metrics"][0]) / float(row["metrics"][1])
        })
    return pd.DataFrame(data)

def earnings_optimization_score(channel_id, video_id, start_date, end_date):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="estimatedRevenue,monetizedPlaybacks",
        startDate=start_date,
        endDate=end_date,
        filters=f"video=={video_id}"
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Video ID": video_id,
            "Date": row["dimensions"][0],
            "Earnings": row["metrics"][0],
            "Monetizable Playbacks": row["metrics"][1],
            "Earnings Optimization Score": float(row["metrics"][0]) / float(row["metrics"][1])
        })
    return pd.DataFrame(data)

def revenue_resilience_metrics(channel_id, video_id, start_date, end_date):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="estimatedRevenue,views",
        startDate=start_date,
        endDate=end_date,
        filters=f"video=={video_id}"
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Video ID": video_id,
            "Date": row["dimensions"][0],
            "Earnings": row["metrics"][0],
            "Views": row["metrics"][1],
            "Revenue Resilience": float(row["metrics"][0]) / float(row["metrics"][1])
        })
    return pd.DataFrame(data)

def earnings_consistency_rating(channel_id, video_id, start_date, end_date):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"video=={video_id}"
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Video ID": video_id,
            "Date": row["dimensions"][0],
            "Earnings": row["metrics"][0],
            "Earnings Consistency Rating": float(row["metrics"][0])
        })
    return pd.DataFrame(data)

# Historical Earnings Data
def historical_earnings_trend(channel_id, video_id, start_date, end_date):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"video=={video_id}"
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "Earnings": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df["Earnings"])
    plt.title("Historical Earnings Trend")
    plt.xlabel("Date")
    plt.ylabel("Earnings")
    st.pyplot(plt)
    return df

def lifetime_revenue_analysis(channel_id, video_id, start_date, end_date):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        ids=f"channel=={channel_id}",
        dimensions="",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"video=={video_id}"
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Lifetime Revenue": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    return df

def earnings_growth_rate(channel_id, video_id, start_date, end_date):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"video=={video_id}"
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "Earnings": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    df["Earnings Growth Rate"] = df["Earnings"].pct_change()
    return df

def revenue_milestones(channel_id, video_id, start_date, end_date):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"video=={video_id}"
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "Earnings": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    milestones = []
    for index, row in df.iterrows():
        if row["Earnings"] >= 1000:
            milestones.append({
                "Date": row["Date"],
                "Milestone": "1000"
            })
        elif row["Earnings"] >= 5000:
            milestones.append({
                "Date": row["Date"],
                "Milestone": "5000"
            })
    df_milestones = pd.DataFrame(milestones)
    return df_milestones

def monthly_earnings_history(channel_id, video_id, start_date, end_date):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        ids=f"channel=={channel_id}",
        dimensions="month",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"video=={video_id}"
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Month": row["dimensions"][0],
            "Earnings": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    return df

def historical_cpm_trends(channel_id, video_id, start_date, end_date):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="cpm",
        startDate=start_date,
        endDate=end_date,
        filters=f"video=={video_id}"
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "CPM": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df["CPM"])
    plt.title("Historical CPM Trends")
    plt.xlabel("Date")
    plt.ylabel("CPM")
    st.pyplot(plt)
    return df

def revenue_pattern_analysis(channel_id, video_id, start_date, end_date):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"video=={video_id}"
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "Earnings": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    df["Moving Average"] = df["Earnings"].rolling(window=7).mean()
    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df["Earnings"])
    plt.plot(df["Date"], df["Moving Average"])
    plt.title("Revenue Pattern Analysis")
    plt.xlabel("Date")
    plt.ylabel("Earnings")
    st.pyplot(plt)
    return df

def earnings_trajectory(channel_id, video_id, start_date, end_date):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"video=={video_id}"
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "Earnings": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    df["Cumulative Earnings"] = df["Earnings"].cumsum()
    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df["Cumulative Earnings"])
    plt.title("Earnings Trajectory")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Earnings")
    st.pyplot(plt)
    return df

def revenue_stability_analysis(channel_id, video_id, start_date, end_date):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"video=={video_id}"
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "Earnings": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    df["Standard Deviation"] = df["Earnings"].rolling(window=30).std()
    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df["Earnings"])
    plt.plot(df["Date"], df["Standard Deviation"])
    plt.title("Revenue Stability Analysis")
    plt.xlabel("Date")
    plt.ylabel("Earnings")
    st.pyplot(plt)
    return df

def earnings_volatility(channel_id, video_id, start_date, end_date):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"video=={video_id}"
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "Earnings": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    df["Volatility"] = df["Earnings"].pct_change().rolling(window=30).std()
    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df["Volatility"])
    plt.title("Earnings Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    st.pyplot(plt)
    return df

# Competitive Earnings Analysis functions
def channel_earnings_comparison(channel_ids, start_date, end_date):
    youtube = get_service()
    data = []
    for channel_id in channel_ids:
        result = execute_api_request(
            youtube.reports().query,
            dimensions="",
            metrics="estimatedRevenue",
            startDate=start_date,
            endDate=end_date,
            filters=f"channel=={channel_id}"
        )
        rows = result.get("rows", [])
        for row in rows:
            data.append({
                "Channel ID": channel_id,
                "Earnings": row["metrics"][0]
            })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.bar(df["Channel ID"], df["Earnings"])
    plt.title("Channel Earnings Comparison")
    plt.xlabel("Channel ID")
    plt.ylabel("Earnings")
    st.pyplot(plt)
    return df

def category_ranking(category, region):
    youtube = get_service()
    data = []
    result = execute_api_request(
        youtube.videos().list,
        part="id,snippet,statistics",
        chart="mostPopular",
        videoCategoryId=category,
        regionCode=region
    )
    rows = result.get("items", [])
    for row in rows:
        data.append({
            "Video Title": row["snippet"]["title"],
            "Channel ID": row["snippet"]["channelId"],
            "Views": row["statistics"]["viewCount"]
        })
    df = pd.DataFrame(data)
    return df

def competitor_revenue_assessment(channel_ids, start_date, end_date):
    youtube = get_service()
    data = []
    for channel_id in channel_ids:
        result = execute_api_request(
            youtube.reports().query,
            dimensions="",
            metrics="estimatedRevenue",
            startDate=start_date,
            endDate=end_date,
            filters=f"channel=={channel_id}"
        )
        rows = result.get("rows", [])
        for row in rows:
            data.append({
                "Channel ID": channel_id,
                "Earnings": row["metrics"][0]
            })
    df = pd.DataFrame(data)
    return df

def earnings_gap_identification(channel_ids, start_date, end_date):
    youtube = get_service()
    data = []
    for channel_id in channel_ids:
        result = execute_api_request(
            youtube.reports().query,
            dimensions="day",
            metrics="estimatedRevenue",
            startDate=start_date,
            endDate=end_date,
            filters=f"channel=={channel_id}"
        )
        data.append(result)

        rows = result.get("rows", [])
        for row in rows:
            data.append({
                "Channel ID": channel_id,
                "Earnings": row["metrics"][0]
            })
    df = pd.DataFrame(data)
    df["Earnings Gap"] = df["Earnings"].max() - df["Earnings"]
    return df

def revenue_competitive_landscape(channel_ids, start_date, end_date):
    youtube = get_service()
    data = []
    for channel_id in channel_ids:
        result = execute_api_request(
            youtube.reports().query,
            dimensions="day",
            metrics="estimatedRevenue",
            startDate=start_date,
            endDate=end_date,
            filters=f"channel=={channel_id}"
        )
        rows = result.get("rows", [])
        for row in rows:
            data.append({
                "Channel ID": channel_id,
                "Date": row["dimensions"][0],
                "Earnings": row["metrics"][0]
            })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.scatter(df["Date"], df["Earnings"])
    plt.title("Revenue Competitive Landscape")
    plt.xlabel("Date")
    plt.ylabel("Earnings")
    st.pyplot(plt)
    return df

def market_share_analysis(channel_ids, start_date, end_date):
    youtube = get_service()
    data = []
    for channel_id in channel_ids:
        result = execute_api_request(
            youtube.reports().query,
            dimensions="day",
            metrics="estimatedRevenue",
            startDate=start_date,
            endDate=end_date,
            filters=f"channel=={channel_id}"
        )
        rows = result.get("rows", [])
        for row in rows:
            data.append({
                "Channel ID": channel_id,
                "Date": row["dimensions"][0],
                "Earnings": row["metrics"][0]
            })
    df = pd.DataFrame(data)
    # Calculate total earnings across all channels
    total_earnings = df["Earnings"].sum()
    
    # Calculate market share for each channel
    df["Market Share"] = df.groupby("Channel ID")["Earnings"].transform(lambda x: x.sum() / total_earnings)
    
    return df

def monetization_strategy_evaluation(channel_ids, start_date, end_date):
    youtube = get_service()
    data = []
    for channel_id in channel_ids:
        result = execute_api_request(
            youtube.reports().query,
            dimensions="day",
            metrics="estimatedRevenue,views",
            startDate=start_date,
            endDate=end_date,
            filters=f"channel=={channel_id}"
        )
        rows = result.get("rows", [])
        for row in rows:
            data.append({
                "Channel ID": channel_id,
                "Date": row["dimensions"][0],
                "Earnings": row["metrics"][0]["values"][0],
                "Views": row["metrics"][0]["values"][1]
            })
    df = pd.DataFrame(data)
    
    # Calculate RPM (Revenue per Mille)
    df["RPM"] = df["Earnings"] / df["Views"] * 1000
    
    # Calculate CPM (Cost per Mille)
    # Assuming average CPM for YouTube is around $2
    df["CPM"] = 2
    
    # Calculate Monetization Strategy (Revenue / Views)
    df["Monetization Strategy"] = df["Earnings"] / df["Views"]
    
    return df

def revenue_positioning_insights(channel_ids, start_date, end_date):
    youtube = get_service()
    data = []
    for channel_id in channel_ids:
        result = execute_api_request(
            youtube.reports().query,
            dimensions="day",
            metrics="estimatedRevenue",
            startDate=start_date,
            endDate=end_date,
            filters=f"channel=={channel_id}"
        )
        rows = result.get("rows", [])
        for row in rows:
            data.append({
                "Channel ID": channel_id,
                "Date": row["dimensions"][0],
                "Earnings": row["metrics"][0]
            })
    df = pd.DataFrame(data)
    
    # Calculate total earnings per channel
    df_channel_earnings = df.groupby("Channel ID")["Earnings"].sum().reset_index()
    
    # Rank channels by total earnings
    df_channel_earnings["Revenue Position"] = df_channel_earnings["Earnings"].rank(method="dense", ascending=False)
    
    # Merge rankings with original DataFrame
    df = pd.merge(df, df_channel_earnings[["Channel ID", "Revenue Position"]], on="Channel ID")
    
    return df

def competitor_performance_benchmarking(channel_ids, start_date, end_date):
    youtube = get_service()
    data = []
    for channel_id in channel_ids:
        result = execute_api_request(
            youtube.reports().query,
            dimensions="day",
            metrics="estimatedRevenue,views",
            startDate=start_date,
            endDate=end_date,
            filters=f"channel=={channel_id}"
        )
        rows = result.get("rows", [])
        for row in rows:
            data.append({
                "Channel ID": channel_id,
                "Date": row["dimensions"][0],
                "Earnings": row["metrics"][0]["values"][0],
                "Views": row["metrics"][0]["values"][1]
            })
    df = pd.DataFrame(data)
    
    # Calculate RPM (Revenue per Mille)
    df["RPM"] = df["Earnings"] / df["Views"] * 1000
    
    # Calculate CPM (Cost per Mille)
    # Assuming average CPM for YouTube is around $2
    df["CPM"] = 2
    
    # Calculate Performance Benchmark (Revenue / Views)
    df["Performance Benchmark"] = df["Earnings"] / df["Views"]
    
    # Calculate Engagement Rate
    df["Engagement Rate"] = df["Views"] / df["Views"].sum()
    
    return df

def earnings_benchmarking_report(channel_ids, start_date, end_date):
    youtube = get_service()
    data = []
    for channel_id in channel_ids:
        result = execute_api_request(
            youtube.reports().query,
            dimensions="day",
            metrics="estimatedRevenue",
            startDate=start_date,
            endDate=end_date,
            filters=f"channel=={channel_id}"
        )
        rows = result.get("rows", [])
        for row in rows:
            data.append({
                "Channel ID": channel_id,
                "Date": row["dimensions"][0],
                "Earnings": row["metrics"][0]
            })
    df = pd.DataFrame(data)
    
    # Calculate total earnings per channel
    df_channel_earnings = df.groupby("Channel ID")["Earnings"].sum().reset_index()
    
    # Rank channels by total earnings
    df_channel_earnings["Earnings Benchmark"] = df_channel_earnings["Earnings"].rank(method="dense", ascending=False)
    
    # Merge rankings with original DataFrame
    df = pd.merge(df, df_channel_earnings[["Channel ID", "Earnings Benchmark"]], on="Channel ID")
    
    # Calculate percentage difference from top-earning channel
    top_earnings = df_channel_earnings["Earnings"].max()
    df["Percentage Difference"] = ((top_earnings - df["Earnings"]) / top_earnings) * 100
    
    return df

# Revenue Optimization Queries functions
def revenue_optimization_tips(channel_id, start_date, end_date):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="estimatedRevenue,views,averageViewDuration,averageViewPercentage",
        startDate=start_date,
        endDate=end_date,
        filters=""
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "Earnings": row["metrics"][0]["values"][0],
            "Views": row["metrics"][0]["values"][1],
            "Average View Duration": row["metrics"][0]["values"][2],
            "Average View Percentage": row["metrics"][0]["values"][3]
        })
    
    df = pd.DataFrame(data)
    return df

def ad_placement_recommendations(video_id):
    youtube = get_service()
    result = execute_api_request(
        youtube.videos().list,
        part="id,snippet,statistics",
        id=video_id
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Video Title": row["snippet"]["title"],
            "Ad Views": row["statistics"]["viewCount"]
        })
    df = pd.DataFrame(data)
    return df

def optimal_video_length_analysis(channel_id, start_date, end_date):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        ids=f"channel=={channel_id}",
        dimensions="videoDuration",
        metrics="estimatedRevenue,views,averageViewDuration",
        startDate=start_date,
        endDate=end_date,
        filters=""
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Video Duration": row["dimensions"][0],
            "Earnings": row["metrics"][0]["values"][0],
            "Views": row["metrics"][0]["values"][1],
            "Average View Duration": row["metrics"][0]["values"][2]
        })
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df["Video Duration"], df["Earnings"])
    plt.title("Optimal Video Length Analysis")
    plt.xlabel("Video Length")
    plt.ylabel("Earnings")
    st.pyplot(plt)
    return df

def best_earning_time_slots(channel_id, start_date, end_date):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        ids=f"channel=={channel_id}",
        dimensions="hour",
        metrics="estimatedRevenue,views",
        startDate=start_date,
        endDate=end_date,
        filters=""
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "Hour": row["dimensions"][1],
            "Earnings": row["metrics"][0]["values"][0],
            "Views": row["metrics"][0]["values"][1]
        })
    
    df = pd.DataFrame(data)
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.bar(df["Hour"], df["Earnings"])
    plt.title("Best Earning Time Slots")
    plt.xlabel("Hour")
    plt.ylabel("Earnings")
    st.pyplot(plt)
    return df

def monetization_strategy_suggestions(channel_id, start_date, end_date):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="estimatedRevenue,views,averageViewDuration,averageViewPercentage",
        startDate=start_date,
        endDate=end_date,
        filters=""
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "Earnings": row["metrics"][0]["values"][0],
            "Views": row["metrics"][0]["values"][1],
            "Average View Duration": row["metrics"][0]["values"][2],
            "Average View Percentage": row["metrics"][0]["values"][3]
        })
    
    df = pd.DataFrame(data)
    return df

def revenue_growth_opportunities(channel_id, start_date, end_date):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        ids=f"channel=={channel_id}",
        dimensions="month",
        metrics="estimatedRevenue,views,averageViewDuration,averageViewPercentage",
        startDate=start_date,
        endDate=end_date,
        filters=""
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Month": row["dimensions"][0],
            "Earnings": row["metrics"][0]["values"][0],
            "Views": row["metrics"][0]["values"][1],
            "Average View Duration": row["metrics"][0]["values"][2],
            "Average View Percentage": row["metrics"][0]["values"][3]
        })
    
    df = pd.DataFrame(data)
    return df

def earnings_improvement_areas(channel_id, start_date, end_date):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="estimatedRevenue,views,averageViewDuration,averageViewPercentage",
        startDate=start_date,
        endDate=end_date,
        filters=""
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "Earnings": row["metrics"][0]["values"][0],
            "Views": row["metrics"][0]["values"][1],
            "Average View Duration": row["metrics"][0]["values"][2],
            "Average View Percentage": row["metrics"][0]["values"][3]
        })
    
    df = pd.DataFrame(data)
    return df

def cpm_optimization_tips(video_id):
    youtube = get_service()
    result = execute_api_request(
        youtube.videos().list,
        part="id,snippet,statistics",
        id=video_id
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Video Title": row["snippet"]["title"],
            "CPM": row["statistics"]["cpm"]
        })
    df = pd.DataFrame(data)
    return df

def revenue_leakage_detection(channel_id, start_date, end_date):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        ids=f"channel=={channel_id}",
        dimensions="day,video",
        metrics="estimatedRevenue,views,averageViewDuration,averageViewPercentage",
        startDate=start_date,
        endDate=end_date,
        filters=""
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "Video": row["dimensions"][1],
            "Earnings": row["metrics"][0]["values"][0],
            "Views": row["metrics"][0]["values"][1],
            "Average View Duration": row["metrics"][0]["values"][2],
            "Average View Percentage": row["metrics"][0]["values"][3]
        })
    
    df = pd.DataFrame(data)
    return df

def untapped_earning_potential_insights(channel_id, start_date, end_date):
    youtube = get_service()
    result = execute_api_request(
        youtube.reports().query,
        ids=f"channel=={channel_id}",
        dimensions="day,video,deviceType",
        metrics="estimatedRevenue,views,averageViewDuration,averageViewPercentage",
        startDate=start_date,
        endDate=end_date,
        filters=""
    )
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "Video": row["dimensions"][1],
            "Device Type": row["dimensions"][2],
            "Earnings": row["metrics"][0]["values"][0],
            "Views": row["metrics"][0]["values"][1],
            "Average View Duration": row["metrics"][0]["values"][2],
            "Average View Percentage": row["metrics"][0]["values"][3]
        })
    
    df = pd.DataFrame(data)
    
    return df

# Audience-Based Revenue Analysis functions
def earnings_by_demographic_segment(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="ageGroup,gender,deviceType",
        metrics="estimatedRevenue,views",
        startDate=start_date,
        endDate=end_date,
        filters=""
    )
    
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Age Group": row["dimensions"][0],
            "Gender": row["dimensions"][1],
            "Device Type": row["dimensions"][2],
            "Earnings": row["metrics"][0]["values"][0],
            "Views": row["metrics"][0]["values"][1]
        })
    
    df = pd.DataFrame(data)
    return df

def age_group_revenue_benchmark(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="ageGroup",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Age Group": row["dimensions"][0],
            "Earnings": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.bar(df["Age Group"], df["Earnings"])
    plt.title("Age Group Revenue Benchmark")
    plt.xlabel("Age Group")
    plt.ylabel("Earnings")
    st.pyplot(plt)
    return df

def viewer_type_performance_metrics(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="audienceType",
        metrics="estimatedRevenue,views",
        startDate=start_date,
        endDate=end_date,
        filters=""
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Viewer Type": row["dimensions"][0],
            "Earnings": row["metrics"][0],
            "Views": row["metrics"][1]
        })
    df = pd.DataFrame(data)
    return df

def watch_time_revenue_optimizer(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        metrics="estimatedRevenue,watchTime",
        dimensions="day",
        startDate=start_date,
        endDate=end_date,
        filters=""
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Watch Time": row["dimensions"][0],
            "Earnings": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.scatter(df["Watch Time"], df["Earnings"])
    plt.title("Watch Time Revenue Optimizer")
    plt.xlabel("Watch Time")
    plt.ylabel("Earnings")
    st.pyplot(plt)
    return df

def location_based_earnings_insights(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="country",
        metrics="estimatedRevenue,views",
        startDate=start_date,
        endDate=end_date,
        filters=""
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Country": row["dimensions"][0],
            "Earnings": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    return df

def device_type_revenue_comparison(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="deviceType",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=""
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Device Type": row["dimensions"][0],
            "Earnings": row["metrics"][0]
        })
        
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.bar(df["Device Type"], df["Earnings"])
    plt.title("Device Type Revenue Comparison")
    plt.xlabel("Device Type")
    plt.ylabel("Earnings")
    st.pyplot(plt)
    return df

def traffic_source_earnings_analyzer(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="trafficSourceDetail",
        metrics="estimatedRevenue,views",
        startDate=start_date,
        endDate=end_date,
        filters=""
    ).execute()
    
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Traffic Source": row["dimensions"][0],
            "Earnings": row["metrics"][0]["values"][0],
            "Views": row["metrics"][0]["values"][1]
        })
    
    df = pd.DataFrame(data)
    
    # Calculate earnings per view
    df["Earnings per View"] = df["Earnings"] / df["Views"]
    
    # Sort traffic sources by earnings in descending order
    df = df.sort_values("Earnings", ascending=False).reset_index(drop=True)
    
    return df

def viewer_engagement_revenue_driver(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="audienceType",
        metrics="estimatedRevenue,averageViewDuration,views",
        startDate=start_date,
        endDate=end_date,
        filters=""
    ).execute()
    
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Audience Type": row["dimensions"][0],
            "Earnings": row["metrics"][0]["values"][0],
            "Average View Duration": row["metrics"][0]["values"][1],
            "Views": row["metrics"][0]["values"][2]
        })
    
    df = pd.DataFrame(data)
    
    # Calculate revenue per view
    df["Revenue per View"] = df["Earnings"] / df["Views"]
    
    # Calculate engagement rate
    df["Engagement Rate"] = df["Average View Duration"] / 3600
    
    # Sort audience types by earnings in descending order
    df = df.sort_values("Earnings", ascending=False).reset_index(drop=True)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df["Engagement Rate"], df["Earnings"])
    plt.title("Viewer Engagement Revenue Driver")
    plt.xlabel("Engagement Rate")
    plt.ylabel("Earnings")
    st.pyplot(plt)
    return df

def subscriber_non_subscriber_revenue_gap(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="subscriberStatus",
        metrics="estimatedRevenue,views",
        startDate=start_date,
        endDate=end_date,
        filters=""
    ).execute()
    
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Subscriber Status": row["dimensions"][0],
            "Earnings": row["metrics"][0]["values"][0],
            "Views": row["metrics"][0]["values"][1]
        })
    
    df = pd.DataFrame(data)
    
    # Calculate revenue per view
    df["Revenue per View"] = df["Earnings"] / df["Views"]
    
    # Sort subscriber status by earnings in descending order
    df = df.sort_values("Earnings", ascending=False).reset_index(drop=True)
    return df

# def member_only_content_revenue_tracker(channel_id):
#     youtube = get_service()
#     result = youtube.reports().query(
#         dimensions="membershipStatus",
#         metrics="estimatedRevenue",
#         filters=f"channel=={channel_id}"
#     ).execute()
#     rows = result.get("rows", [])
#     data = []
#     for row in rows:
#         data.append({
#             "Membership Status": row["dimensions"][0],
#             "Earnings": row["metrics"][0]
#         })
#     df = pd.DataFrame(data)
#     return df

# Content-Specific Earnings functions
def highest_earning_video_identifier(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="video",
        metrics="estimatedRevenue,views,averageViewDuration",
        startDate=start_date,
        endDate=end_date,
        filters=""
    ).execute()
    
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Video Title": row["dimensions"][0],
            "Earnings": row["metrics"][0]["values"][0],
            "Views": row["metrics"][0]["values"][1],
            "Average View Duration": row["metrics"][0]["values"][2]
        })
    
    df = pd.DataFrame(data)
    
    # Calculate revenue per view
    df["Revenue per View"] = df["Earnings"] / df["Views"]
    
    # Sort videos by earnings in descending order
    df = df.sort_values("Earnings", ascending=False).reset_index(drop=True)
    return df

def video_type_revenue_analyzer(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="contentType",
        metrics="estimatedRevenue,views,averageViewDuration",
        startDate=start_date,
        endDate=end_date,
        filters=""
    ).execute()
    
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Content Type": row["dimensions"][0],
            "Earnings": row["metrics"][0]["values"][0],
            "Views": row["metrics"][0]["values"][1],
            "Average View Duration": row["metrics"][0]["values"][2]
        })
    
    df = pd.DataFrame(data)
    
    # Calculate revenue per view
    df["Revenue per View"] = df["Earnings"] / df["Views"]
    
    # Sort content types by earnings in descending order
    df = df.sort_values("Earnings", ascending=False).reset_index(drop=True)
    plt.figure(figsize=(10, 6))
    plt.bar(df["Content Type"], df["Earnings"])
    plt.title("Content Type Revenue Analyzer")
    plt.xlabel("Content Type")
    plt.ylabel("Earnings")
    st.pyplot(plt)
    return df

def content_category_performance_metrics(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="contentCategory",
        metrics="estimatedRevenue,views,averageViewDuration",
        startDate=start_date,
        endDate=end_date,
        filters=""
    ).execute()
    
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Content Category": row["dimensions"][0],
            "Earnings": row["metrics"][0]["values"][0],
            "Views": row["metrics"][0]["values"][1],
            "Average View Duration": row["metrics"][0]["values"][2]
        })
    
    df = pd.DataFrame(data)
    
    # Calculate revenue per view
    df["Revenue per View"] = df["Earnings"] / df["Views"]
    
    # Calculate engagement rate
    df["Engagement Rate"] = df["Average View Duration"] / 3600
    
    # Sort content categories by earnings in descending order
    df = df.sort_values("Earnings", ascending=False).reset_index(drop=True)
    return df

def shorts_long_form_earnings_comparator(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="video",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        video_id = row["dimensions"][0]
        video_duration = get_video_duration(video_id)  # Implement get_video_duration function
        data.append({
            "Video ID": video_id,
            "Video Duration": video_duration,
            "Earnings": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.scatter(df["Video Duration"], df["Earnings"])
    plt.title("Shorts/Long-Form Earnings Comparator")
    plt.xlabel("Video Duration")
    plt.ylabel("Earnings")
    st.pyplot(plt)
    return df

def extract_series_name(video_title):
    """
    Extracts the series name from a video title.

    Args:
        video_title (str): The title of the video.

    Returns:
        str: The extracted series name or None if not found.
    """
    
    # Define common series name patterns
    patterns = [
        r"\[(.*?)\]",  # [Series Name]
        r"\((.*?)\)",  # (Series Name)
        r"\"(.*?)\"",  # "Series Name"
        r"(?:Season \d+: )(.*)",  # Season X: Series Name
        r"(?:Part \d+: )(.*)",  # Part X: Series Name
        r"(.*) - (?:Episode|Ep) \d+",  # Series Name - Episode X
    ]
    
    # Iterate through patterns
    for pattern in patterns:
        match = re.search(pattern, video_title)
        if match:
            # Return the first capturing group (series name)
            return match.group(1).strip()
    
    # If no pattern matches, return None
    return None

def video_series_revenue_calculator(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="video",
        metrics="estimatedRevenue",
        startDate = start_date,
        endDate = end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        video_id = row["dimensions"][0]
        video_title = get_video_title(video_id)  # Implement get_video_title function
        data.append({
            "Video Title": video_title,
            "Earnings": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    # Group by video series (assuming series name is part of video title)
    df['Video Series'] = df['Video Title'].apply(lambda x: extract_series_name(x))
    df_grouped = df.groupby('Video Series')['Earnings'].sum().reset_index()
    return df_grouped

def video_length_revenue_optimizer(channel_id, start_date, end_date):
    youtube = get_service()
    
    # Get video durations using YouTube Data API
    video_durations = []
    request = youtube.videos().list(
        part="contentDetails",
        channelId=channel_id
    )
    response = request.execute()
    for item in response["items"]:
        video_id = item["id"]
        duration = item["contentDetails"]["duration"]
        video_durations.append((video_id, duration))

    # Get earnings by video using YouTube Analytics API
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="video",
        metrics="estimatedRevenue",
        startDate = start_date,
        endDate = end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    revenue_data = []
    for row in rows:
        video_id = row["dimensions"][0]
        earnings = row["metrics"][0]
        revenue_data.append((video_id, earnings))

    # Match video IDs with durations and revenue
    data = []
    for vid, duration in video_durations:
        earnings = next((earnings for vid_, earnings in revenue_data if vid_ == vid), None)
        data.append({
            "Video Length": duration,
            "Earnings": earnings
        })

    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.scatter(df["Video Length"], df["Earnings"])
    plt.title("Video Length Revenue Optimizer")
    plt.xlabel("Video Length")
    plt.ylabel("Earnings")
    st.pyplot(plt)
    return df

def top_revenue_generating_topic_finder(channel_id, start_date, end_date):
    youtube = get_service()
    # Get video metadata using YouTube Data API
    video_metadata = []
    request = youtube.videos().list(
        part="snippet",
        channelId=channel_id
    )
    response = request.execute()
    for item in response["items"]:
        video_id = item["id"]
        title = item["snippet"]["title"]
        description = item["snippet"]["description"]
        tags = item["snippet"]["tags"]
        video_metadata.append((video_id, title, description, tags))

    # Analyze revenue using YouTube Analytics API
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="video",
        metrics="estimatedRevenue",
        startDate = start_date,
        endDate = end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    revenue_data = []
    for row in rows:
        video_id = row["dimensions"][0]
        earnings = row["metrics"][0]
        revenue_data.append((video_id, earnings))

    # Match video IDs with metadata and revenue
    data = []
    for vid, title, desc, tags in video_metadata:
        earnings = next((earnings for vid_, earnings in revenue_data if vid_ == vid), None)
        text = f"{title} {desc} {tags}"
        data.append({
            "Text": text,
            "Earnings": earnings
        })

    # Topic modeling using Latent Dirichlet Allocation (LDA)
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    tfidf = tfidf_vectorizer.fit_transform([row["Text"] for row in data])
    lda = LatentDirichletAllocation(n_components=5)
    topics = lda.fit_transform(tfidf)

    # Get top revenue-generating topics
    topic_earnings = []
    for topic_id in range(topics.shape[1]):
        earnings = 0
        for i, row in enumerate(data):
            if topics[i, topic_id] > 0.5:
                earnings += row["Earnings"]
        topic_earnings.append(earnings)

    # Return top revenue-generating topic
    top_topic_id = topic_earnings.index(max(topic_earnings))
    top_topic = [row["Text"] for i, row in enumerate(data) if topics[i, top_topic_id] > 0.5]
    return top_topic


def video_format_revenue_benchmark(channel_id, start_date, end_date):
    youtube = get_service()
    # Get video formats using YouTube Data API
    video_formats = []
    request = youtube.videos().list(
        part="contentDetails",
        channelId=channel_id
    )
    response = request.execute()
    for item in response["items"]:
        video_id = item["id"]
        video_format = item["contentDetails"]["definition"]  # Get video resolution (e.g., hd, sd)
        video_formats.append((video_id, video_format))

    # Get earnings by video using YouTube Analytics API
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="video",
        metrics="estimatedRevenue",
        startDate = start_date,
        endDate = end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        video_id = row["dimensions"][0]
        earnings = row["metrics"][0]
        # Match video ID with video format
        video_format = next((format for vid, format in video_formats if vid == video_id), None)
        data.append({
            "Video Format": video_format,
            "Earnings": earnings
        })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.bar(df["Video Format"], df["Earnings"])
    plt.title("Video Format Revenue Benchmark")
    plt.xlabel("Video Format")
    plt.ylabel("Earnings")
    plt.show()
    return df

def live_stream_revenue_tracker(channel_id, start_date, end_date):
    youtube = get_service()
    
    # Get live video IDs using YouTube Data API
    live_video_ids = []
    request = youtube.search().list(
        part="id,snippet",
        channelId=channel_id,
        type="video",
        eventType="live"
    )
    response = request.execute()
    for item in response["items"]:
        live_video_ids.append(item["id"]["videoId"])

    # Get earnings by video using YouTube Analytics API
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="video",
        metrics="estimatedRevenue",
        startDate = start_date,
        endDate = end_date,
        filters=f"channel=={channel_id};video=={','.join(live_video_ids)}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        video_id = row["dimensions"][0]
        earnings = row["metrics"][0]
        data.append({
            "Video ID": video_id,
            "Earnings": earnings
        })
    df = pd.DataFrame(data)
    return df

def playlist_revenue_share(channel_id, start_date, end_date):
    youtube = get_service()
    
    # Get playlist IDs using YouTube Data API
    playlist_ids = []
    request = youtube.playlists().list(
        part="id,snippet",
        channelId=channel_id
    )
    response = request.execute()
    for item in response["items"]:
        playlist_ids.append(item["id"])

    # Get video IDs by playlist using YouTube Data API
    playlist_video_ids = {}
    for playlist_id in playlist_ids:
        request = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id
        )
        response = request.execute()
        video_ids = [item["contentDetails"]["videoId"] for item in response["items"]]
        playlist_video_ids[playlist_id] = video_ids

    # Get earnings by video using YouTube Analytics API
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="video",
        metrics="estimatedRevenue",
        startDate = start_date,
        endDate = end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        video_id = row["dimensions"][0]
        earnings = row["metrics"][0]
        for playlist_id, video_ids in playlist_video_ids.items():
            if video_id in video_ids:
                data.append({
                    "Playlist": playlist_id,
                    "Earnings": earnings
                })

    df = pd.DataFrame(data)
    return df

# Revenue Performance Analysis functions
def channel_earnings_comparator(channel_ids, start_date, end_date):
    youtube = get_service()
    data = []
    for channel_id in channel_ids:
        result = youtube.reports().query(
            ids=f"channel=={channel_id}",
            dimensions="",
            metrics="estimatedRevenue",
            startDate=start_date,
            endDate=end_date,
            filters=f"channel=={channel_id}"
        ).execute()
        rows = result.get("rows", [])
        for row in rows:
            data.append({
                "Channel ID": channel_id,
                "Earnings": row["metrics"][0]
            })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.bar(df["Channel ID"], df["Earnings"])
    plt.title("Channel Earnings Comparator")
    plt.xlabel("Channel ID")
    plt.ylabel("Earnings")
    st.pyplot(plt)
    return df

def niche_revenue_benchmarks(niche, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        dimensions="contentCategory",
        metrics="estimatedRevenue,views,averageViewDuration",
        startDate=start_date,
        endDate=end_date,
        filters=f"contentCategory=={niche}"
    ).execute()
    
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Content Category": row["dimensions"][0],
            "Earnings": row["metrics"][0]["values"][0],
            "Views": row["metrics"][0]["values"][1],
            "Average View Duration": row["metrics"][0]["values"][2]
        })
    
    df = pd.DataFrame(data)
    
    # Calculate revenue per view
    df["Revenue per View"] = df["Earnings"] / df["Views"]
    
    # Calculate engagement rate
    df["Engagement Rate"] = df["Average View Duration"] / 3600
    
    # Sort content categories by earnings in descending order
    df = df.sort_values("Earnings", ascending=False).reset_index(drop=True)
    return df

def performance_metrics_dashboard(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="estimatedRevenue,views,likes,comments,shares,averageViewDuration",
        startDate=start_date,
        endDate=end_date,
        filters=""
    ).execute()
    
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "Earnings": row["metrics"][0]["values"][0],
            "Views": row["metrics"][0]["values"][1],
            "Likes": row["metrics"][0]["values"][2],
            "Comments": row["metrics"][0]["values"][3],
            "Shares": row["metrics"][0]["values"][4],
            "Average View Duration": row["metrics"][0]["values"][5]
        })
    
    df = pd.DataFrame(data)
    
    # Calculate engagement rate
    df["Engagement Rate"] = (df["Likes"] + df["Comments"] + df["Shares"]) / df["Views"]
    
    # Calculate revenue per view
    df["Revenue per View"] = df["Earnings"] / df["Views"]
    
    # Calculate average view duration in minutes
    df["Average View Duration (minutes)"] = df["Average View Duration"] / 60
    return df

def earnings_efficiency_optimizer(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="video",
        metrics="estimatedRevenue,views,averageViewDuration",
        startDate=start_date,
        endDate=end_date,
        filters=""
    ).execute()
    
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Video Title": row["dimensions"][0],
            "Earnings": row["metrics"][0]["values"][0],
            "Views": row["metrics"][0]["values"][1],
            "Average View Duration": row["metrics"][0]["values"][2]
        })

    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.scatter(df["Views"], df["Earnings"])
    plt.title("Earnings Efficiency Optimizer")
    plt.xlabel("Views")
    plt.ylabel("Earnings")
    st.pyplot(plt)
    return df

def revenue_enhancement_opportunities(channel_id, start_date, end_date):
    youtube = get_service()
    # Query revenue data
    revenue_result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=""
    ).execute()
    
    revenue_rows = revenue_result.get("rows", [])
    revenue_data = []
    for row in revenue_rows:
        revenue_data.append({
            "Date": row["dimensions"][0],
            "Earnings": row["metrics"][0]["values"][0]
        })
    
    # Query engagement data
    engagement_result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="",
        metrics="views,likes,comments,shares",
        startDate=start_date,
        endDate=end_date,
        filters=""
    ).execute()
    
    engagement_rows = engagement_result.get("rows", [])
    engagement_data = []
    for row in engagement_rows:
        engagement_data.append({
            "Date": row["dimensions"][0],
            "Views": row["metrics"][0]["values"][0],
            "Likes": row["metrics"][0]["values"][1],
            "Comments": row["metrics"][0]["values"][2],
            "Shares": row["metrics"][0]["values"][3]
        })
    
    # Merge revenue and engagement data
    df_revenue = pd.DataFrame(revenue_data)
    df_engagement = pd.DataFrame(engagement_data)
    df_merged = pd.merge(df_revenue, df_engagement, on="Date")
    
    # Identify revenue enhancement opportunities
    df_merged["Opportunity"] = np.where(df_merged["Earnings"] < df_merged["Views"] * 0.01, "Content Improvement", "Marketing Strategies")
    
    return df_merged

def earnings_quality_rating(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="estimatedRevenue,views,averageViewDuration",
        startDate=start_date,
        endDate=end_date,
        filters=""
    ).execute()
    
    rows = result.get("rows", [])
    data = []
    for row in rows:
        earnings = row["metrics"][0]["values"][0]
        views = row["metrics"][0]["values"][1]
        avg_duration = row["metrics"][0]["values"][2]
        
        # Calculate earnings quality metrics
        earnings_per_view = earnings / views
        engagement_rate = avg_duration / 3600
        earnings_quality = earnings_per_view * engagement_rate
        
        data.append({
            "Date": row["dimensions"][0],
            "Earnings": earnings,
            "Views": views,
            "Average View Duration": avg_duration,
            "Earnings per View": earnings_per_view,
            "Engagement Rate": engagement_rate,
            "Earnings Quality Rating": earnings_quality
        })
    
    df = pd.DataFrame(data)
    
    # Normalize earnings quality ratings (0-100 scale)
    df["Earnings Quality Rating"] = (df["Earnings Quality Rating"] - df["Earnings Quality Rating"].min()) / (df["Earnings Quality Rating"].max() - df["Earnings Quality Rating"].min()) * 100
    
    return df

def revenue_stream_efficiency_analyzer(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="",
        metrics="estimatedAdRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=""
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Revenue Stream": ["Advertisements", "Sponsorships", "Merchandise"],
            "Efficiency": [row["metrics"][0] * 0.5, row["metrics"][0] * 0.3, row["metrics"][0] * 0.2]
        })
    df = pd.DataFrame(data)
    return df

def industry_average_earnings_benchmark(channel_id, start_date, end_date, niche):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"niche=={niche}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Niche": niche,
            "Average Earnings": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    return df

def content_hour_revenue_insights(channel_id, start_date, end_date):
    youtube = get_service()
    
    # Get video durations using YouTube Data API
    video_durations = []
    request = youtube.videos().list(
        part="contentDetails",
        channelId=channel_id
    )
    response = request.execute()
    for item in response["items"]:
        video_id = item["id"]
        duration = item["contentDetails"]["duration"]
        video_durations.append((video_id, duration))

    # Get earnings by video using YouTube Analytics API
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="video",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    revenue_data = []
    for row in rows:
        video_id = row["dimensions"][0]
        earnings = row["metrics"][0]
        revenue_data.append((video_id, earnings))

    # Match video IDs with durations and revenue
    data = []
    for vid, duration in video_durations:
        earnings = next((earnings for vid_, earnings in revenue_data if vid_ == vid), None)
        data.append({
            "Video ID": vid,
            "Duration": duration,
            "Earnings": earnings
        })

    df = pd.DataFrame(data)
    return df

def subscriber_to_earnings_ratio(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="estimatedRevenue,subscribersGained",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        date = row["dimensions"][0]
        earnings = next((metric["values"][0] for metric in row["metrics"] if metric["name"] == "earnings"), None)
        subscribers = next((metric["values"][0] for metric in row["metrics"] if metric["name"] == "subscribersGained"), None)
        data.append({
            "Date": date,
            "Subscribers": subscribers,
            "Earnings": earnings
        })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.scatter(df["Subscribers"], df["Earnings"])
    plt.title("Subscriber-to-Earnings Ratio")
    plt.xlabel("Subscribers")
    plt.ylabel("Earnings")
    st.pyplot(plt)
    return df

# Monetization Metrics functions
def cpm_trend_tracker(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="cpm",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "CPM": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df["CPM"])
    plt.title("CPM Trend Tracker")
    plt.xlabel("Date")
    plt.ylabel("CPM")
    plt.show()
    return df

def channel_rpm_analyzer(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="grossRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        date = row["dimensions"][0]
        rpm = row["metrics"][0]
        data.append({
            "Date": date,
            "RPM": rpm
        })
    df = pd.DataFrame(data)
    return df

def monetized_playback_metrics(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="",
        metrics="monetizedPlaybacks",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Monetized Playbacks": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    return df

def ad_impression_valuation(channel_id, start_date, end_start):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="",
        metrics="adImpressions",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Ad Impressions": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    return df

def view_revenue_calculator(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="",
        metrics="views,estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Views": row["metrics"][0],
            "Earnings": row["metrics"][1]
        })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.scatter(df["Views"], df["Earnings"])
    plt.title("View Revenue Calculator")
    plt.xlabel("Views")
    plt.ylabel("Earnings")
    st.pyplot(plt)
    return df

def ecpm_estimator(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="",
        metrics="ecpm",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "ECPM": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    return df

def monetization_rate_benchmark(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="",
        metrics="monetizationRate",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Monetization Rate": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    return df

def ad_density_optimizer(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="",
        metrics="adDensity",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Ad Density": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.bar(df.index, df["Ad Density"])
    plt.title("Ad Density Optimizer")
    plt.xlabel("Index")
    plt.ylabel("Ad Density")
    st.pyplot(plt)
    return df

def skip_rate_earnings_impact(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="",
        metrics="skipRate,earnings",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Skip Rate": row["metrics"][0],
            "Earnings": row["metrics"][1]
        })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.scatter(df["Skip Rate"], df["Earnings"])
    plt.title("Skip Rate Earnings Impact")
    plt.xlabel("Skip Rate")
    plt.ylabel("Earnings")
    st.pyplot(plt)
    return df

def subscriber_revenue_insights(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="",
        metrics="subscribers,estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Subscribers": row["metrics"][0],
            "Earnings": row["metrics"][1]
        })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.scatter(df["Subscribers"], df["Earnings"])
    plt.title("Subscriber Revenue Insights")
    plt.xlabel("Subscribers")
    plt.ylabel("Earnings")
    st.pyplot(plt)
    return df

# Geographic Revenue Analysis functions
def geographic_earnings_insights(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="country",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Country": row["dimensions"][0],
            "Earnings": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.bar(df["Country"], df["Earnings"])
    plt.title("Geographic Earnings Insights")
    plt.xlabel("Country")
    plt.ylabel("Earnings")
    st.pyplot(plt)
    return df

def regional_revenue_rankings(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        idds=f"channel=={channel_id}",
        dimensions="region",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Region": row["dimensions"][0],
            "Earnings": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    df = df.sort_values("Earnings", ascending=False)
    return df

def country_level_cpm_analysis(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="country",
        metrics="cpm",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Country": row["dimensions"][0],
            "CPM": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.bar(df["Country"], df["CPM"])
    plt.title("Country-Level CPM Analysis")
    plt.xlabel("Country")
    plt.ylabel("CPM")
    st.pyplot(plt)
    return df

def revenue_share_by_region(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="region",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Region": row["dimensions"][0],
            "Earnings Share": row["metrics"][0] / sum([r["metrics"][0] for r in rows])
        })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.pie(df["Earnings Share"], labels=df["Region"], autopct='%1.1f%%')
    plt.title("Revenue Share by Region")
    st.pyplot(plt)
    return df

def market_specific_earnings_potential(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="country",
        metrics="subscribers,estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Market": row["dimensions"][0],
            "Earnings Potential": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    return df

def cross_regional_earnings_comparison(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="region",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Region": row["dimensions"][0],
            "Earnings": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.bar(df["Region"], df["Earnings"])
    plt.title("Cross-Regional Earnings Comparison")
    plt.xlabel("Region")
    plt.ylabel("Earnings")
    st.pyplot(plt)
    return df

def top_paying_markets(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="market",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Market": row["dimensions"][0],
            "Earnings": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    df = df.sort_values("Earnings", ascending=False)
    return df

def regional_revenue_composition(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="region",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Region": row["dimensions"][0],
            "Earnings Share": row["metrics"][0] / sum([r["metrics"][0] for r in rows])
        })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.pie(df["Earnings Share"], labels=df["Region"], autopct='%1.1f%%')
    plt.title("Regional Revenue Composition")
    st.pyplot(plt)
    return df

def international_earnings_trends(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="country",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Country": row["dimensions"][0],
            "Earnings": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.bar(df["Country"], df["Earnings"])
    plt.title("International Earnings Trends")
    plt.xlabel("Country")
    plt.ylabel("Earnings")
    st.pyplot(plt)
    return df

def global_local_earnings_balance(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Global Earnings": row["metrics"][0] * 0.7,
            "Local Earnings": row["metrics"][0] * 0.3
        })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.bar(df.index, df["Global Earnings"], label="Global Earnings")
    plt.bar(df.index, df["Local Earnings"], label="Local Earnings")
    plt.title("Global/Local Earnings Balance")
    plt.xlabel("Index")
    plt.ylabel("Earnings")
    plt.legend()
    st.pyplot(plt)
    return df

# Report Generation functions
def generate_earnings_report(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "Earnings": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df["Earnings"])
    plt.title("Earnings Report")
    plt.xlabel("Date")
    plt.ylabel("Earnings")
    st.pyplot(plt)
    return df

def export_revenue_analysis(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Revenue": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    return df

def create_earnings_forecast_report(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "Earnings": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df["Earnings"])
    plt.title("Earnings Forecast Report")
    plt.xlabel("Date")
    plt.ylabel("Earnings")
    st.pyplot(plt)
    return df

def generate_revenue_breakdown_pdf(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
        "Revenue": row["metrics"][0]
    })
    df = pd.DataFrame(data)

    # Create a PDF object
    pdf = FPDF()
    pdf.add_page()
    
    # Set the title font and size
    pdf.set_font("Arial", size=15)
    pdf.cell(200, 10, txt="Revenue Breakdown Report", ln=True, align='C')
    pdf.ln(10)  # Add a line break

    # Set the font for the content
    pdf.set_font("Arial", size=10)

    # Iterate through the DataFrame and add revenue data to the PDF
    for index, row in df.iterrows():
        pdf.cell(200, 10, txt=f"Revenue: ${row['Revenue']:.2f}", ln=True, align='L')

    # Save the PDF to a file
    pdf.output("revenue_breakdown.pdf")

    return df


def export_earnings_comparison(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "Earnings": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    df.to_csv("earnings_comparison.csv", index=False)
    return df


def create_monetization_strategy_report(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="",
        metrics="estimatedRevenue,monetizedPlaybacks,adImpressions",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        date = row["dimensions"][0]
        revenue = row["metrics"][0]
        monetized_playbacks = row["metrics"][1]
        ad_impressions = row["metrics"][2]
        data.append({
            "Date": date,
            "Revenue": revenue,
            "Monetized Playbacks": monetized_playbacks,
            "Ad Impressions": ad_impressions
        })
    df = pd.DataFrame(data)
    return df


def generate_revenue_optimization_report(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Revenue": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    return df


def export_earnings_analytics(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "Earnings": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    df.to_excel("earnings_analytics.xlsx", index=False)
    return df


def create_revenue_stream_analysis(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Revenue Stream": row["dimensions"][0],
            "Revenue": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    return df


def generate_earnings_benchmark_report(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Earnings Benchmark": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    return df

# Historical Data functions
def video_performance_history(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="views,estimatedRevenue",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "Views": row["metrics"][0],
            "Earnings": row["metrics"][1]
        })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df["Views"], label="Views")
    plt.plot(df["Date"], df["Earnings"], label="Earnings")
    plt.title("Video Performance History")
    plt.xlabel("Date")
    plt.ylabel("Views/Earnings")
    plt.legend()
    st.pyplot(plt)
    return df

# def keyword_trend_analysis(channel_id):
#     youtube = get_service()
#     result = youtube.reports().query(
#         dimensions="keyword",
#         metrics="views",
#         filters=f"channel=={channel_id}"
#     ).execute()
#     rows = result.get("rows", [])
#     data = []
#     for row in rows:
#         data.append({
#             "Keyword": row["dimensions"][0],
#             "Views": row["metrics"][0]
#         })
#     df = pd.DataFrame(data)
#     return df

def year_over_year_growth_comparison(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="year",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Year": row["dimensions"][0],
            "Earnings": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.bar(df["Year"], df["Earnings"])
    plt.title("Year-over-Year Growth Comparison")
    plt.xlabel("Year")
    plt.ylabel("Earnings")
    st.pyplot(plt)
    return df


def seasonal_trend_identification(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="month",
        metrics="views",
        startDate=start_date,
        endDate=end_date
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Month": row["dimensions"][0],
            "Views": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.plot(df["Month"], df["Views"])
    plt.title("Seasonal Trend Identification")
    plt.xlabel("Month")
    plt.ylabel("Views")
    st.pyplot(plt)
    return df


def engagement_history_analysis(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="likes,comments,shares",
        startDate=start_date,
        endDate=end_date
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "Likes": row["metrics"][0],
            "Comments": row["metrics"][1],
            "Shares": row["metrics"][2]
        })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df["Likes"], label="Likes")
    plt.plot(df["Date"], df["Comments"], label="Comments")
    plt.plot(df["Date"], df["Shares"], label="Shares")
    plt.title("Engagement History Analysis")
    plt.xlabel("Date")
    plt.ylabel("Engagement")
    plt.legend()
    st.pyplot(plt)
    return df


def viewership_trend_tracking(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="views",
        startDate=start_date,
        endDate=end_date
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "Views": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df["Views"])
    plt.title("Viewership Trend Tracking")
    plt.xlabel("Date")
    plt.ylabel("Views")
    st.pyplot(plt)
    return df


def subscriber_growth_history(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="subscribersGained",
        startDate=start_date,
        endDate=end_date
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "Subscribers Gained": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df["Subscribers Gained"])
    plt.title("Subscriber Growth History")
    plt.xlabel("Date")
    plt.ylabel("Subscribers Gained")
    st.pyplot(plt)
    return df


def monthly_performance_comparison(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="month",
        metrics="estimatedRevenue",
        startDate=start_date,
        endDate=end_date
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Month": row["dimensions"][0],
            "Earnings": row["metrics"][0]
        })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.bar(df["Month"], df["Earnings"])
    plt.title("Monthly Performance Comparison")
    plt.xlabel("Month")
    plt.ylabel("Earnings")
    st.pyplot(plt)
    return df

# Competition Analysis functions
def niche_competitor_analysis(channel_ids):
    youtube = get_service()
    data = []
    for channel_id in channel_ids:
        result = youtube.search().list(
            part="id,snippet",
            type="channel",
            maxResults=50,
            relatedToChannelId=channel_id
        ).execute()
        channels = result.get("items", [])
        for channel in channels:
            data.append({
                "Channel Name": channel["snippet"]["title"],
                "Subscriber Count": channel["snippet"]["subscriberCount"],
                "Competitor Of": channel_id
            })
    df = pd.DataFrame(data)
    return df

def channel_comparison(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="day",
        metrics="views,estimatedRevenue",
        startDate=start_date,
        endDaate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Date": row["dimensions"][0],
            "Views": row["metrics"][0],
            "Earnings": row["metrics"][1]
        })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.bar(df["Date"], df["Views"], label="Views")
    plt.plot(df["Date"], df["Earnings"], label="Earnings")
    plt.title("Channel Insights")
    plt.xlabel("Date")
    plt.ylabel("Views/Earnings")
    plt.legend()
    st.pyplot(plt)
    return df

def competitor_video_strategy(channel_id):
    youtube = get_service()
    result = youtube.search().list(
        part="id,snippet",
        type="video",
        maxResults=50,
        relatedToVideoId=channel_id
    ).execute()
    videos = result.get("items", [])
    data = []
    for video in videos:
        data.append({
            "Video Title": video["snippet"]["title"],
            "Video Views": video["snippet"]["viewCount"]
        })
    df = pd.DataFrame(data)
    return df


def tag_analysis(channel_id):
    youtube = get_service()
    result = youtube.videos().list(
        part="id,snippet",
        id=channel_id
    ).execute()
    video = result.get("items", [])[0]
    tags = video["snippet"]["tags"]
    data = []
    for tag in tags:
        data.append({
            "Tag": tag
        })
    df = pd.DataFrame(data)
    return df


def title_comparison(channel_id):
    youtube = get_service()
    result = youtube.search().list(
        part="id,snippet",
        type="video",
        maxResults=50,
        relatedToVideoId=channel_id
    ).execute()
    videos = result.get("items", [])
    data = []
    for video in videos:
        data.append({
            "Video Title": video["snippet"]["title"],
            "Video Views": video["snippet"]["viewCount"]
        })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.bar(df["Video Title"], df["Video Views"])
    plt.title("Title Comparison")
    plt.xlabel("Video Title")
    plt.ylabel("Video Views")
    plt.show()
    return df


def thumbnail_analysis(channel_id):
    youtube = get_service()
    result = youtube.videos().list(
        part="id,snippet",
        id=channel_id
    ).execute()
    video = result.get("items", [])[0]
    thumbnail = video["snippet"]["thumbnails"]["default"]["url"]
    data = []
    data.append({
        "Thumbnail URL": thumbnail
    })
    df = pd.DataFrame(data)
    return df


def upload_pattern_insights(channel_id):
    youtube = get_service()
    result = youtube.search().list(
        part="id,snippet",
        type="video",
        maxResults=50,
        channelId=channel_id
    ).execute()
    videos = result.get("items", [])
    data = []
    for video in videos:
        data.append({
            "Video Title": video["snippet"]["title"],
            "Upload Date": video["snippet"]["publishedAt"]
        })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.plot(df["Upload Date"], range(len(df["Upload Date"])), marker='o')
    plt.title("Upload Pattern Insights")
    plt.xlabel("Upload Date")
    plt.ylabel("Video Count")
    plt.show()
    return df


def engagement_metric_analysis(channel_id, start_date, end_date):
    youtube = get_service()
    result = youtube.reports().query(
        ids=f"channel=={channel_id}",
        dimensions="video",
        metrics="likes,comments,shares",
        startDate=start_date,
        endDate=end_date,
        filters=f"channel=={channel_id}"
    ).execute()
    rows = result.get("rows", [])
    data = []
    for row in rows:
        data.append({
            "Video Title": row["dimensions"][0],
            "Likes": row["metrics"][0],
            "Comments": row["metrics"][1],
            "Shares": row["metrics"][2]
        })
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.plot(df["Video Title"], df["Likes"], label="Likes")
    plt.plot(df["Video Title"], df["Comments"], label="Comments")
    plt.plot(df["Video Title"], df["Shares"], label="Shares")
    plt.title("Engagement Metric Analysis")
    plt.xlabel("Video Title")
    plt.ylabel("Engagement")
    plt.legend()
    st.pyplot(plt)
    return df

# Title of the app
st.title("YouTube Viral Chatbot")

# Function to extract video ID from the URL
def get_video_id(url):
    # Simple extraction logic; adjust as necessary for different URL formats
    if "watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return None

# Sidebar for options
st.sidebar.header("Select an Option")

# Dropdown menu options
options = [
    "Basic Video Analysis", "Channel Analysis", "Content Strategy", "Trending Analysis", 
    "Keyword Research", "Performance Insights", "Basic Earnings Estimates", 
    "Revenue Stream Analysis", "Earnings by Time Period", "Regional Compliance Checks", 
    "Localization", "Cross-Border Analysis", "Regional Content Strategy", 
    "Local Competition Analysis", "Time Zone-Based Analysis", "Geographic Performance Metrics", 
    "Cultural Trend Analysis", "Market Research Commands", "Language-Based Search", 
    "Regional Analysis", "Regional Performance Analysis", "Natural Language Queries", 
    "Advanced Earnings Metrics", "Historical Earnings Data", "Competitive Earnings Analysis", 
    "Revenue Optimization Queries", "Audience-Based Revenue Analysis", 
    "Content-Specific Earnings", "Revenue Performance Analysis", "Monetization Metrics", 
    "Geographic Revenue Analysis", "Report Generation", "Historical Data", 
    "Competition Analysis"
]

selected_option = st.sidebar.selectbox("Choose an analysis type:", options)

# Placeholder for the content based on the selected option
if selected_option:
    st.write(f"You selected: **{selected_option}**")
    # You can expand here with additional functionality for each option

# Basic Video Analysis functionality
if selected_option == "Basic Video Analysis":
    st.write("### Basic Video Analysis")

    # Input for YouTube API Key
    youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")

    # Set up YouTube API client only if YouTube API key is provided
    if youtube_api_key:
        youtube = build("youtube", "v3", developerKey=youtube_api_key)
    else:
        youtube = None  # Set to None if key is not provided

    # Input for YouTube video URL
    video_url = st.text_input("Enter the YouTube video URL:")

    # Check if the video URL is provided
    if video_url:
        video_id = get_video_id(video_url)  # Extract the video ID from the URL

        # Show options for basic video analysis
        analysis_option = st.selectbox(
            "Choose an analysis option:",
            [
                "Analyze video",
                "Get video metadata",
                "Show video tags",
                "Extract video keywords",
                "Get video performance metrics",
                "Summarize video statistics",
                "Show video engagement metrics",
                "Compare metrics between videos",
            ]
        )

        # Conditional logic based on user selection
        if st.button("Analyze Video"):
            if video_id:
                if analysis_option == "Compare metrics between videos":
                    # Ask for second video URL for comparison
                    second_video_url = st.text_input("Enter the second YouTube Video URL for comparison:")
                    if second_video_url:
                        second_video_id = get_video_id(second_video_url)  # Extract ID from the second URL
                        if second_video_id:
                            with st.spinner("Comparing metrics..."):
                                comparison_result = compare_metrics_between_videos(video_id, second_video_id)
                                st.success("Metrics comparison completed successfully!")
                                st.subheader("Comparison Results:")
                                st.write(comparison_result)
                        else:
                            st.error("Invalid second video URL.")
                else:
                    with st.spinner("Analyzing video..."):
                        # Call the appropriate analysis function based on selection
                        if analysis_option == "Analyze video":
                            result = get_video_metadata(video_id)
                            st.success("Video analysis completed successfully!")
                        elif analysis_option == "Get video metadata":
                            result = get_video_metadata(video_id)
                            st.success("Video metadata retrieved successfully!")
                        elif analysis_option == "Show video tags":
                            result = show_video_tags(video_id)
                            st.success("Video tags retrieved successfully!")
                        elif analysis_option == "Extract video keywords":
                            result = extract_video_keywords(video_id)
                            st.success("Video keywords extracted successfully!")
                        elif analysis_option == "Get video performance metrics":
                            result = get_video_performance_metrics(video_id)
                            st.success("Video performance metrics obtained successfully!")
                        elif analysis_option == "Summarize video statistics":
                            result = summarize_video_statistics(video_id)
                            st.success("Video statistics summarized successfully!")
                        elif analysis_option == "Show video engagement metrics":
                            result = show_video_engagement_metrics(video_id)
                            st.success("Video engagement metrics displayed successfully!")

                    # Display the result
                    if result:
                        st.subheader("Analysis Results:")
                        st.write(result)
            else:
                st.error("Invalid YouTube video URL.")


elif selected_option == "Channel Analysis":
    st.write("### Channel Analysis")
    
    # Input for YouTube API Key
    youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")

    # Set up YouTube API client only if YouTube API key is provided
    if youtube_api_key:
        youtube = build("youtube", "v3", developerKey=youtube_api_key)
    else:
        youtube = None  # Set to None if key is not provided
        
    channel_id = st.text_input("Enter Channel ID:")

    if channel_id:
        st.write("### Choose Channel Analysis Option")
        channel_analysis_option = st.selectbox("Select an analysis option:", [
            "Analyze channel",
            "Show top performing videos",
            "Get channel growth metrics",
            "Compare channels",
            "Show channel engagement trends",
            "Analyze upload schedule",
            "Get subscriber growth rate"
        ])

        # Button to execute analysis
        if st.button("Analyze Channel"):
            if channel_id:
                with st.spinner("Analyzing channel..."):
                    try:
                        if channel_analysis_option == "Compare channels":
                            channel_id_2 = st.text_input("Enter second Channel ID for comparison:")
                            if channel_id_2:
                                result = compare_channels(channel_id, channel_id_2)
                                st.success("Channels compared successfully!")
                            else:
                                st.error("Please enter the second channel ID for comparison.")
                        else:
                            # Execute the corresponding function based on user selection
                            if channel_analysis_option == "Analyze channel":
                                result = analyze_channel(channel_id)
                                st.success("Channel analysis completed!")
                            elif channel_analysis_option == "Show top performing videos":
                                result = show_top_performing_videos(channel_id)
                                st.success("Top performing videos retrieved successfully!")
                            elif channel_analysis_option == "Get channel growth metrics":
                                result = get_channel_growth_metrics(channel_id)
                                st.success("Channel growth metrics obtained successfully!")
                            elif channel_analysis_option == "Show channel engagement trends":
                                result = show_channel_engagement_trends(channel_id)
                                st.success("Engagement trends analyzed successfully!")
                            elif channel_analysis_option == "Analyze upload schedule":
                                result = analyze_upload_schedule(channel_id)
                                st.success("Upload schedule analyzed successfully!")
                            elif channel_analysis_option == "Get subscriber growth rate":
                                result = get_subscriber_growth_rate(channel_id)
                                st.success("Subscriber growth rate retrieved successfully!")
                        
                        # Display results
                        st.subheader("Channel Analysis Results")
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            st.json(result)  # Display results in JSON format for clarity

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            else:
                st.error("Please enter a valid Channel ID.")


elif selected_option == "Content Strategy":
    st.write("### Content Strategy")
    
    # Input for OpenAI API Key
    api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
    
    client = OpenAI(api_key=api_key)

    # Check if API key is loaded
    if api_key is None:
        st.error("OpenAI API key is missing. Please set it in the .env file.")
    else:
        pytrends = TrendReq(hl='en-US', tz=360)
        
        # User option selection
        option = st.selectbox(
            "Select a content strategy option:",
            [
                "Suggest video ideas",
                "Generate title ideas",
                "Optimize description",
                "Get thumbnail suggestions",
                "Recommend upload time",
                "Get content calendar suggestions",
                "Analyze best posting times",
            ]
        )

        # Input field for related topic
        related_topic = st.text_input("Enter a topic or keyword")

        # Button to process the request

        if st.button("Submit"):
            with st.spinner("Processing..."):
                try:
                    if option == "Suggest video ideas":
                        if not related_topic:
                            st.error("Please enter a topic.")
                        else:
                            results = suggest_video_ideas(related_topic)
                            st.subheader("Video Ideas:")
                            st.write(results)
                            st.success("Video ideas generated successfully!")

                    elif option == "Generate title ideas":
                        if not related_topic:
                            st.error("Please enter a related topic.")
                        else:
                            results = generate_title_ideas(related_topic)
                            st.subheader("Title Ideas:")
                            st.write(results)
                            st.success("Title ideas generated successfully!")

                    elif option == "Optimize description":
                        if not related_topic:
                            st.error("Please enter a keyword.")
                        else:
                            results = optimize_description(related_topic)
                            st.subheader("Optimized Description:")
                            st.write(results)
                            st.success("Description optimized successfully!")

                    elif option == "Get thumbnail suggestions":
                        if not related_topic:
                            st.error("Please enter a keyword.")
                        else:
                            results = get_thumbnail_suggestions(related_topic)
                            st.subheader("Thumbnail Suggestions:")
                            st.write(results)
                            st.success("Thumbnail suggestions generated successfully!")

                    elif option == "Recommend upload time":
                        results = recommend_upload_time(related_topic)
                        st.subheader("Recommended Upload Time:")
                        st.write(results)
                        st.success("Upload time recommended successfully!")

                    elif option == "Get content calendar suggestions":
                        results = get_content_calendar_suggestions(related_topic)
                        st.subheader("Content Calendar Suggestions:")
                        st.write(results)
                        st.success("Content calendar suggestions generated successfully!")

                    elif option == "Analyze best posting times":
                        results = analyze_best_posting_times(related_topic)
                        st.subheader("Best Posting Times:")
                        st.write(results)
                        st.success("Best posting times analyzed successfully!")

                except Exception as e:
                    st.error(f"An error occurred: {e}")

                
elif selected_option == "Trending Analysis":
    st.write("### Trending Analysis")
    
    youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")

    # Set up YouTube API client only if YouTube API key is provided
    if youtube_api_key:
        youtube = build("youtube", "v3", developerKey=youtube_api_key)
    else:
        youtube = None  # Set to None if key is not provided
    
    # User option selection for trending analysis
    trending_option = st.selectbox(
        "Select a trending analysis option:",
        [
            "What's trending now",
            "Show trending topics",
            "Get trending hashtags",
            "Show viral videos",
            "Analyze trends",  
            "Compare trend data", 
            "Show rising trends", 
            "Get weekly trend report"
        ]
    )
    
    # Input for region code
    geo = st.text_input("Enter a region code (e.g., 'US', 'IN'):", "US")

    # Input for keyword in case of trend analysis
    keyword = st.text_input("Enter a keyword for trend analysis:")

    # Additional input for comparing trend data only if that option is selected
    if trending_option == "Compare trend data":
        keyword2 = st.text_input("Enter a second keyword for comparison:")
    else:
        keyword2 = None  # Reset keyword2 if not comparing

    # Button to process trending analysis request
    if st.button("Submit Trending Analysis"):
        with st.spinner("Processing..."):
            try:
                if trending_option == "Compare trend data" and not keyword2:
                    st.error("Please enter a second keyword for comparison.")
                elif not keyword and trending_option in ["Analyze trends", "Get trending hashtags", "Show rising trends", "Get weekly trend report"]:
                    st.error("Please enter a keyword for trend analysis.")
                else:
                    # Call the appropriate function based on the selected option
                    if trending_option == "What's trending now":
                        trending_data = get_trending_topics(geo)
                        st.success("Processing complete!")
                        st.write(trending_data)
                    elif trending_option == "Show trending topics":
                        trending_data = get_trending_topics(geo)
                        st.success("Processing complete!")
                        st.write(trending_data)
                    elif trending_option == "Get trending hashtags":
                        hashtags = get_trending_hashtags(geo)
                        st.success("Processing complete!")
                        st.write(hashtags)
                    elif trending_option == "Show viral videos":
                        viral_videos = get_viral_videos(geo)
                        st.success("Processing complete!")
                        st.write(viral_videos)
                    elif trending_option == "Analyze trends":
                        trend_analysis = analyze_trends(geo)
                        st.success("Processing complete!")
                        st.write(trend_analysis)
                    elif trending_option == "Compare trend data" and keyword2:  # Ensure keyword2 is provided
                        comparison_data = compare_trend_data(keyword, geo)
                        st.success("Processing complete!")
                        st.write(comparison_data)
                    elif trending_option == "Show rising trends":
                        rising_trends = show_rising_trends(geo)
                        st.success("Processing complete!")
                        st.write(rising_trends)
                    elif trending_option == "Get weekly trend report":
                        weekly_report = get_weekly_trend_report(geo, keyword=keyword)
                        st.success("Processing complete!")
                        st.write(weekly_report)
            except Exception as e:
                st.error(f"An error occurred: {e}")


elif selected_option == "Keyword Research":
    st.write("### Keyword Research")
    
    youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")

    # Set up YouTube API client only if YouTube API key is provided
    if youtube_api_key:
        youtube = build("youtube", "v3", developerKey=youtube_api_key)
    else:
        youtube = None  # Set to None if key is not provided
        
    keyword_option = st.selectbox(
        "Select a trending analysis option:",
        [
            "Research keywords",
            "Get search volume", 
            "Show related keywords", 
            "Analyze keyword competition", 
            "Get best tags", 
            "Show keyword trends", 
            "Compare keywords", 
            "Generate tag suggestions"
        ]
    )
    
    # Input for keyword in case of trend analysis
    keyword = st.text_input("Enter a keyword for research: ")
    location = st.text_input("Please enter the location you want to search in:")
    
    # Additional input for comparing trend data only if that option is selected
    keyword2 = None  # Initialize keyword2
    if keyword_option == "Compare keywords":
        keyword2 = st.text_input("Enter a second keyword for comparison:")
        
    if keyword_option == "Research keywords":
        api_key = st.text_input("Enter your SERPAPI KEY")
        
        
    # Button to process trending analysis request
    if st.button("Submit keyword for research"):
        with st.spinner("Processing..."):
            try:
                if keyword_option == "Compare keywords" and not keyword2:
                    st.error("Please enter a second keyword for comparison.")
                elif not keyword and keyword_option in ["Research keywords", "Get search volume", 
                                                        "Show related keywords", "Analyze keyword competition"]:
                    st.error("Please enter a keyword for trend analysis.")
                else:
                    # Call the appropriate function based on the selected option
                    if keyword_option == "Research keywords":
                        result = research_keywords(keyword, location, api_key)
                        st.success("Processing complete!")
                        st.write(result)  # Display the result
                    elif keyword_option == "Get search volume":
                        result = get_search_volume(keyword, location)
                        st.success("Processing complete!")
                        st.write(result)  # Display the result
                    elif keyword_option == "Show related keywords":
                        result = show_related_keywords(keyword, location)
                        st.success("Processing complete!")
                        st.write(result)  # Display the result
                    elif keyword_option == "Analyze keyword competition":
                        result = analyze_keyword_competition(keyword, location)
                        st.success("Processing complete!")
                        st.write(result)  # Display the result
                    elif keyword_option == "Get best tags":
                        result = get_best_tags(keyword, location)
                        st.success("Processing complete!")
                        st.write(result)  # Display the result
                    elif keyword_option == "Show keyword trends":
                        result = show_keyword_trends(keyword, location)
                        st.success("Processing complete!")
                        st.write(result)  # Display the result
                    elif keyword_option == "Compare keywords" and keyword2:
                        result = compare_keywords(keyword, keyword2, location)
                        st.success("Processing complete!")
                        st.write(result)  # Display the result
                    elif keyword_option == "Generate tag suggestions":
                        with st.sidebar:
                            st.header("OpenAI API Key")
                            st.info("To generate tag suggestions, please input your OpenAI API key.")
                            openai_api_key = st.text_input("OpenAI API Key", type="password")
                            if st.button("Generate Tags"):
                                result = generate_tags(keyword, location, openai_api_key)
                                st.success("Processing complete!")
                                st.write(result)  # Display the result
            except Exception as e:
                st.error(f"An error occurred: {e}")


# Main Streamlit code
elif selected_option == "Performance Insights":
    st.write("### Performance Insights")
    
    # Upload the OAuth JSON file
    uploaded_file = st.sidebar.file_uploader("Upload your OAuth JSON file", type=["json"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = "temp_client_secret.json"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
    
    SCOPES = ["https://www.googleapis.com/auth/yt-analytics.readonly"]
    # https://developers.google.com/identity/protocols/oauth2/scopes

    # https://github.com/googleapis/google-api-python-client/blob/main/docs/dyn/index.md
    API_SERVICE_NAME = "youtubeAnalytics"
    API_VERSION = "v2"
    CLIENT_SECRETS_FILE = temp_file_path

    def get_service():
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_local_server()
        return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

    def execute_api_request(client_library_function, **kwargs):
        response = client_library_function(
            **kwargs
        ).execute()
        print(response)
        return response

    # Allow user to select a performance insight option
    performance_option = st.selectbox(
        "Select the performance insights option:",
        [
            "Show traffic sources",
            "Get audience retention data",
            "Analyze click-through rate",
            "Show viewer demographics",
            "Get watch time analytics",
            "Show engagement patterns",
            "Compare performance metrics",
            "Get viewer behavior insights"
        ]
    )

    # Input fields for channel ID, start date, and end date
    channel_id = st.text_input("Enter Channel ID")
    start_date = st.date_input("Enter Start Date (YYYY-MM-DD)")
    end_date = st.date_input("Enter End Date (YYYY-MM-DD)")

    # Input fields for video ID (only shown for audience retention data)
    if performance_option == "Get audience retention data":
        video_id = st.text_input("Enter Video ID")
        
    if performance_option == "Compare performance metrics":
        channel_id = st.text_input("Enter Channel ID")
        start_date = st.date_input("Enter Start Date (YYYY-MM-DD)")
        end_date = st.date_input("Enter End Date (YYYY-MM-DD)")
        metrics1 = st.text_input("Enter the first metrics")
        metrics2 = st.text_input("Enter the second metrics")

    # Generate the report based on the selected option
    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            try:
                if performance_option == "Show traffic sources":
                    result = show_traffic_sources(channel_id, start_date, end_date)
                    st.success("Report generated!")
                    st.write(result)
                elif performance_option == "Get audience retention data":
                    result = get_audience_retention_data(video_id)
                    st.success("Report generated!")
                    st.write(result)
                elif performance_option == "Analyze click-through rate":
                    result = analyze_click_through_rate(channel_id, start_date, end_date)
                    st.success("Report generated!")
                    st.write(result)
                elif performance_option == "Show viewer demographics":
                    result = show_viewer_demographics(channel_id, start_date, end_date)
                    st.success("Report generated!")
                    st.write(result)
                elif performance_option == "Get watch time analytics":
                    result = get_watch_time_analytics(channel_id, start_date, end_date)
                    st.success("Report generated!")
                    st.write(result)
                elif performance_option == "Show engagement patterns":
                    result = show_engagement_patterns(channel_id, start_date, end_date)
                    st.success("Report generated!")
                    st.write(result)
                elif performance_option == "Compare performance metrics":
                    result = compare_performance_metrics(channel_id, start_date, end_date, metrics1, metrics2)
                    st.success("Report generated!")
                    st.write(result)
                elif performance_option == "Get viewer behavior insights":
                    result = get_viewer_behavior_insights(channel_id, start_date, end_date)
                    st.success("Report generated!")
                    st.write(result)
            except Exception as e:
                st.error(f"An error occurred: {e}")

            
elif selected_option == "Basic Earnings Estimates":
    st.write("### Basic Earnings Estimates")
    
    # Upload the OAuth JSON file
    uploaded_file = st.sidebar.file_uploader("Upload your OAuth JSON file", type=["json"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = "temp_client_secret.json"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
    
    SCOPES = ["https://www.googleapis.com/auth/youtube",
              "https://www.googleapis.com/auth/yt-analytics-monetary.readonly",
              "https://www.googleapis.com/auth/yt-analytics.readonly"]
    # https://developers.google.com/identity/protocols/oauth2/scopes

    # https://github.com/googleapis/google-api-python-client/blob/main/docs/dyn/index.md
    API_SERVICE_NAME = "youtubeAnalytics"
    API_VERSION = "v2"
    CLIENT_SECRETS_FILE = temp_file_path

    def get_service():
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_local_server()
        return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

    def execute_api_request(client_library_function, **kwargs):
        response = client_library_function(
            **kwargs
        ).execute()
        print(response)
        return response
    
    basic_earnings_estimates_option = st.selectbox(
        "Basic Earnings Estimates:", 
        [
         "Channel Earnings Estimator", 
         "Revenue Analysis Report", 
         "URL Earnings Potential", 
         "Monthly Channel Earnings Forecast", 
         "Detailed Earnings Breakdown", 
         "URL Revenue Potential Analysis", 
         "Channel Earnings Projections", 
         "CPM Estimator"   
        ]
    )
    
    # Input fields for channel ID, start date, and end date
    if basic_earnings_estimates_option == "Channel Earnings Estimator":
        channel_id = st.text_input("Enter Channel ID")
        start_date = st.date_input("Enter Start Date (YYYY-MM-DD)")
        end_date = st.date_input("Enter End Date (YYYY-MM-DD)")
        
    if basic_earnings_estimates_option == "Revenue Analysis Report":
        channel_id = st.text_input("Enter Channel ID")
        start_date = st.date_input("Enter Start Date (YYYY-MM-DD)")
        end_date = st.date_input("Enter End Date (YYYY-MM-DD)")
        
    if basic_earnings_estimates_option == "Monthly Channel Earnings Forecast":
        channel_id = st.text_input("Enter Channel ID")
        start_date = st.date_input("Enter Start Date (YYYY-MM-DD)")
        end_date = st.date_input("Enter End Date (YYYY-MM-DD)")
        
    if basic_earnings_estimates_option == "Detailed Earnings Breakdown":
        channel_id = st.text_input("Enter Channel ID")
        start_date = st.date_input("Enter Start Date (YYYY-MM-DD)")
        end_date = st.date_input("Enter End Date (YYYY-MM-DD)")
        
    if basic_earnings_estimates_option == "Channel Earnings Projections":
        channel_id = st.text_input("Enter Channel ID")
        start_date = st.date_input("Enter Start Date (YYYY-MM-DD)")
        end_date = st.date_input("Enter End Date (YYYY-MM-DD)")
        
    if basic_earnings_estimates_option == "CPM Estimator":
        channel_id = st.text_input("Enter Channel ID")
        start_date = st.date_input("Enter Start Date (YYYY-MM-DD)")
        end_date = st.date_input("Enter End Date (YYYY-MM-DD)")
    
    if basic_earnings_estimates_option == "URL Earnings Potential":
        video_id = st.text_input("Enter Video ID")
        start_date = st.date_input("Enter Start Date (YYYY-MM-DD)")
        end_date = st.date_input("Enter End Date (YYYY-MM-DD)")
        
    if basic_earnings_estimates_option == "URL Revenue Potential Analysis":
        video_id = st.text_input("Enter Video ID")
        start_date = st.date_input("Enter Start Date (YYYY-MM-DD)")
        end_date = st.date_input("Enter End Date (YYYY-MM-DD)")
    
    
    if st.button("Estimate Basic Earnings"):
        with st.spinner("Processing..."):
            try:
                if basic_earnings_estimates_option == "Channel Earnings Estimator":
                    result = channel_earnings_estimator(channel_id, start_date, end_date)
                    st.success("Processing complete!")
                    st.write(result)
                elif basic_earnings_estimates_option == "Revenue Analysis Report":
                    result = revenue_analysis_report(channel_id, start_date, end_date)
                    st.success("Processing complete!")
                    st.write(result)
                elif basic_earnings_estimates_option == "URL Earnings Potential":
                    result = url_earnings_potential(video_id, start_date, end_date)
                    st.success("Processing complete!")
                    st.write(result)
                elif basic_earnings_estimates_option == "Monthly Channel Earnings Forecast":
                    result = monthly_channel_earnings_forecast(channel_id, start_date, end_date)
                    st.success("Processing complete!")
                    st.write(result)
                elif basic_earnings_estimates_option == "Detailed Earnings Breakdown":
                    result = detailed_earnings_breakdown(channel_id, start_date, end_date)
                    st.success("Processing complete!")
                    st.write(result)
                elif basic_earnings_estimates_option == "URL Revenue Potential Analysis":
                    result = url_revenue_potential_analysis(video_id, start_date, end_date)
                    st.success("Processing complete!")
                    st.write(result)
                elif basic_earnings_estimates_option == "Channel Earnings Projections":
                    result = channel_earnings_projections(channel_id, start_date, end_date)
                    st.success("Processing complete!")
                    st.write(result)
                elif basic_earnings_estimates_option == "CPM Estimator":
                    result = cpm_estimator(channel_id, start_date, end_date)
                    st.success("Processing complete!")
                    st.write(result)
            except Exception as e:
                st.error(f"An error occurred: {e}")


elif selected_option == "Revenue Stream Analysis":
    st.write("### Revenue Stream Analysis")
    
    # Upload the OAuth JSON file
    uploaded_file = st.sidebar.file_uploader("Upload your OAuth JSON file", type=["json"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = "temp_client_secret.json"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
    
    SCOPES = ["https://www.googleapis.com/auth/youtube",
              "https://www.googleapis.com/auth/yt-analytics.readonly",
              "https://www.googleapis.com/auth/yt-analytics-monetary.readonly"]
    # https://developers.google.com/identity/protocols/oauth2/scopes

    # https://github.com/googleapis/google-api-python-client/blob/main/docs/dyn/index.md
    API_SERVICE_NAME = "youtubeAnalytics"
    API_VERSION = "v2"
    CLIENT_SECRETS_FILE = temp_file_path

    def get_service():
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_local_server()
        return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

    def execute_api_request(client_library_function, **kwargs):
        response = client_library_function(
            **kwargs
        ).execute()
        print(response)
        return response
    
    Revenue_Stream_Analysis_option = st.selectbox(
        "Revenue Stream Analysis:",
        [
            "Channel Ad Revenue Report", 
            "Super Chat Earnings Tracker", 
            "YouTube Premium Revenue Share Calculator"
        ]
    )
    
    channel_id = st.text_input("Enter Channel ID")
    start_date = st.date_input("Enter Start Date (YYYY-MM-DD)")
    end_date = st.date_input("Enter End Date (YYYY-MM-DD)")
    
    if st.button("Generate Revenue Stream Analysis"):
        with st.spinner("Processing..."):
            try:
                if Revenue_Stream_Analysis_option == "Channel Ad Revenue Report":
                    result = channel_ad_revenue_report(channel_id, start_date, end_date)
                    st.success("Processing complete!")
                    st.write(result)
                elif Revenue_Stream_Analysis_option == "YouTube Premium Revenue Share Calculator":
                    result = youtube_premium_revenue_share_calculator(channel_id, start_date, end_date)
                    st.success("Processing complete!")
                    st.write(result)
            except Exception as e:
                st.error(f"An error occurred: {e}")

            
elif selected_option == "Earnings by Time Period":
    st.write("### Earnings by Time Period")
    
    # Upload the OAuth JSON file
    uploaded_file = st.sidebar.file_uploader("Upload your OAuth JSON file", type=["json"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = "temp_client_secret.json"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
    
    SCOPES = ["https://www.googleapis.com/auth/youtube",
              "https://www.googleapis.com/auth/yt-analytics.readonly",
              "https://www.googleapis.com/auth/yt-analytics-monetary.readonly"]
    # https://developers.google.com/identity/protocols/oauth2/scopes

    # https://github.com/googleapis/google-api-python-client/blob/main/docs/dyn/index.md
    API_SERVICE_NAME = "youtubeAnalytics"
    API_VERSION = "v2"
    CLIENT_SECRETS_FILE = temp_file_path

    def get_service():
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_local_server()
        return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

    def execute_api_request(client_library_function, **kwargs):
        response = client_library_function(
            **kwargs
        ).execute()
        print(response)
        return response
    
    Earnings_by_Time_Period_option = st.selectbox(
        "Earnings by Time Period:",
        [
            "30-Day Revenue Snapshot", 
            "Annualized Revenue Estimate", 
            "Quarterly Earnings Analysis", 
            "Monthly Growth Tracker", 
            "Weekly Earnings Summary", 
            "Earnings Comparison Tool", 
            "Year-over-Year Growth Rate", 
            "6-Month Revenue Outlook"
        ]
    )
    
    channel_id = st.text_input("channel_id")
    col1, col2 = st.columns(2)
    start_date = col1.data_input("Start Date")
    end_date = col2.data_input("Enter Date")
    
    if Earnings_by_Time_Period_option == "Quarterly Earnings Analysis":
        channel_id = st.text_input("channel_id")
        quarter = st.selectbox("Select the quarter of the year", ["Q1 (Jan-Mar)", "Q2 (Apr-Jun)", "Q3 (Jul-Sep)", "Q4 (Oct-Dec)"])
        # Then, you can convert the selected quarter to a number (1, 2, 3, or 4) for use in your `quarterly_earnings_analysis` function:

        if quarter == "Q1 (Jan-Mar)":
            quarter_number = 1
        elif quarter == "Q2 (Apr-Jun)":
            quarter_number = 2
        elif quarter == "Q3 (Jul-Sep)":
            quarter_number = 3
        else:
            quarter_number = 4
    
    if Earnings_by_Time_Period_option == "Earnings Comparison Tool":
        channel_id = st.text_input("channel_id")
        comparison_type = st.selectbox("Select comparison type", ["Specific date", "Last year", "Same quarter last year"])
    
        if comparison_type == "Specific date":
            comparison_date = st.date_input("Select comparison date")
        elif comparison_type == "Last year":
            comparison_date = datetime.today() - timedelta(days=365)
        else:
            # Calculate same quarter last year
            quarter = (datetime.today().month - 1) // 3 + 1
            year = datetime.today().year - 1
            comparison_date = datetime(year, (quarter - 1) * 3 + 1, 1)
    
    if st.button("Generate Earnings by Time Period"):
        with st.spinner("Processing..."):
            try:
                if Earnings_by_Time_Period_option == "30-Day Revenue Snapshot":
                    result = thirty_day_revenue_snapshot(channel_id, start_date, end_date)
                    st.success("Processing complete!")
                    st.write(result)
                elif Earnings_by_Time_Period_option == "Annualized Revenue Estimate":
                    result = annualized_revenue_estimate(channel_id, start_date, end_date)
                    st.success("Processing complete!")
                    st.write(result)
                elif Earnings_by_Time_Period_option == "Quarterly Earnings Analysis":
                    result = quarterly_earnings_analysis(channel_id, quarter)
                    st.success("Processing complete!")
                    st.write(result)
                elif Earnings_by_Time_Period_option == "Monthly Growth Tracker":
                    result = monthly_growth_tracker(channel_id, start_date, end_date)
                    st.success("Processing complete!")
                    st.write(result)
                elif Earnings_by_Time_Period_option == "Weekly Earnings Summary":
                    result = weekly_earnings_summary(channel_id, start_date, end_date)
                    st.success("Processing complete!")
                    st.write(result)
                elif Earnings_by_Time_Period_option == "Earnings Comparison Tool":
                    result = earnings_comparison_tool(channel_id, comparison_date)
                    st.success("Processing complete!")
                    st.write(result)
                elif Earnings_by_Time_Period_option == "Year-over-Year Growth Rate":
                    result = year_over_year_growth_rate(channel_id, start_date, end_date)
                    st.success("Processing complete!")
                    st.write(result)
                elif Earnings_by_Time_Period_option == "6-Month Revenue Outlook":
                    result = six_month_revenue_outlook(channel_id, start_date, end_date)
                    st.success("Processing complete!")
                    st.write(result)
            except Exception as e:
                st.error(f"An error occurred: {e}")

            
elif selected_option == "Regional Compliance Checks":
    st.write("### Regional Compliance Checks")
    
    # Upload the OAuth JSON file
    uploaded_file = st.sidebar.file_uploader("Upload your OAuth JSON file", type=["json"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = "temp_client_secret.json"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
    
    SCOPES = ["https://www.googleapis.com/auth/youtube",
              "https://www.googleapis.com/auth/yt-analytics.readonly"]
    # https://developers.google.com/identity/protocols/oauth2/scopes

    # https://github.com/googleapis/google-api-python-client/blob/main/docs/dyn/index.md
    API_SERVICE_NAME = "youtube"
    API_VERSION = "v3"
    CLIENT_SECRETS_FILE = temp_file_path

    def get_service():
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_local_server()
        return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

    def execute_api_request(client_library_function, **kwargs):
        response = client_library_function(
            **kwargs
        ).execute()
        print(response)
        return response
    
    regional_compliance_checks_option = st.selectbox(
        "Regional Compliance Checks:",
        [
            "Check content restrictions by country", 
            "Age rating requirements by region", 
            "Monetization policies by country", 
            "Content guidelines analysis by region", 
            "Restricted keywords by country", 
            "Copyright rules by region", 
            "Advertising restrictions check", 
            "Regional policy compliance"
        ]
    )
    
    if regional_compliance_checks_option == "Check content restrictions by country":
        channel_id = st.text_input("Enter Channel ID")
        country = st.text_input("Enter region code e.g US, GB")
        
    if regional_compliance_checks_option == "Age rating requirements by region":
        channel_id = st.text_input("Enter Channel ID")
        region = st.selectbox("Select region", [
            "North America",
            "South America",
            "Europe",
            "Asia",
            "Africa",
            "Australia/Oceania"
        ])
    
    if regional_compliance_checks_option == "Monetization policies by country":
        channel_id = st.text_input("Enter Channel ID")
        country = st.text_input("Enter region code e.g US, GB")
        
    if regional_compliance_checks_option == "Content guidelines analysis by region":
        channel_id = st.text_input("Enter Channel ID")
        region = st.selectbox("Select region", [
            "North America",
            "South America",
            "Europe",
            "Asia",
            "Africa",
            "Australia/Oceania"
        ])
        
    if regional_compliance_checks_option == "Restricted keywords by country":
        channel_id = st.text_input("Enter Channel ID")
        country = st.text_input("Enter region code e.g US, GB")
        
    if regional_compliance_checks_option == "Copyright rules by region":
        channel_id = st.text_input("Enter Channel ID")
        region = st.selectbox("Select region", [
            "North America",
            "South America",
            "Europe",
            "Asia",
            "Africa",
            "Australia/Oceania"
        ])
        
    if regional_compliance_checks_option == "Advertising restrictions check":
        channel_id = st.text_input("Enter Channel ID")
        
    if regional_compliance_checks_option == "Regional policy compliance":
        channel_id = st.text_input("Enter Channel ID")
        region = st.selectbox("Select region", [
            "North America",
            "South America",
            "Europe",
            "Asia",
            "Africa",
            "Australia/Oceania"
        ])
    
    if st.button("Check Regional Compliance"):
        with st.spinner("Processing..."):
            try:
                if regional_compliance_checks_option == "Content Restrictions by Country":
                    result = content_restrictions_by_country(channel_id, country)
                    st.success("Processing complete!")
                    st.write(result)
                elif regional_compliance_checks_option == "Age Rating Requirements by Region":
                    result = age_rating_requirements_by_region(channel_id, region)
                    st.success("Processing complete!")
                    st.write(result)
                elif regional_compliance_checks_option == "Monetization Policies by Country":
                    result = monetization_policies_by_country(channel_id, country)
                    st.success("Processing complete!")
                    st.write(result)
                elif regional_compliance_checks_option == "Content Guidelines Analysis by Region":
                    result = content_guidelines_analysis_by_region(channel_id, region)
                    st.success("Processing complete!")
                    st.write(result)
                elif regional_compliance_checks_option == "Restricted Keywords by Country":
                    result = restricted_keywords_by_country(channel_id, country)
                    st.success("Processing complete!")
                    st.write(result)
                elif regional_compliance_checks_option == "Copyright Rules by Region":
                    result = copyright_rules_by_region(channel_id, region)
                    st.success("Processing complete!")
                    st.write(result)
                elif regional_compliance_checks_option == "Advertising Restrictions Check":
                    result = advertising_restrictions_check(channel_id)
                    st.success("Processing complete!")
                    st.write(result)
                elif regional_compliance_checks_option == "Regional Policy Compliance":
                    result = regional_policy_compliance(channel_id, region)
                    st.success("Processing complete!")
                    st.write(result)
            except Exception as e:
                st.error(f"An error occurred: {e}")
            
elif selected_option == "Localization":
    st.write("### Localization")
    
    # Upload the OAuth JSON file
    uploaded_file = st.sidebar.file_uploader("Upload your OAuth JSON file", type=["json"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = "temp_client_secret.json"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
    
    SCOPES = ["https://www.googleapis.com/auth/youtube",
              "https://www.googleapis.com/auth/yt-analytics.readonly"]
    # https://developers.google.com/identity/protocols/oauth2/scopes

    # https://github.com/googleapis/google-api-python-client/blob/main/docs/dyn/index.md
    API_SERVICE_NAME = "youtube"
    API_VERSION = "v3"
    CLIENT_SECRETS_FILE = temp_file_path

    def get_service():
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_local_server()
        return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

    def execute_api_request(client_library_function, **kwargs):
        response = client_library_function(
            **kwargs
        ).execute()
        print(response)
        return response
    
    localization_option = st.selectbox(
        "Localization",
        [
            "Check translation quality by language", 
            "Caption performance in countries", 
            "Subtitle recommendations", 
            "Dubbing impact analysis in regions", 
            "Localization opportunities", 
            "Translation suggestions", 
            "Multi-language performance analysis", 
            "Localization return on investment"
        ]
    )
    
    if localization_option == "Translation Quality by Language":
        video_id = st.text_input("Enter Video ID")
        language = st.text_input("Enter the language")
        
    if localization_option == "Caption Performance in Countries":
        video_id = st.text_input("Enter Video ID")
        country = st.text_input("Enter the Country e.g US, GB")
        
    if localization_option == "Subtitle Recommendations":
        video_id = st.text_input("Enter Video ID")
        
    if localization_option == "Dubbing Impact Analysis in Regions":
        video_id = st.text_input("Enter Video ID")
        region = st.selectbox("Select region", [
            "North America",
            "South America",
            "Europe",
            "Asia",
            "Africa",
            "Australia/Oceania"
        ])
        
    if localization_option == "Localization Opportunities":
        video_id = st.text_input("Enter Video ID")
        
    if localization_option == "Translation Suggestions":
        video_id = st.text_input("Enter Video ID")
        
    if localization_option == "Multi-Language Performance Analysis":
        video_id = st.text_input("Enter Video ID")
        
    if localization_option == "Localization Return on Investment":
        video_id = st.text_input("Enter Video ID")
    
    if st.button("Localization"):
        with st.spinner("Generating localization insights..."):
            try:
                if localization_option == "Translation Quality by Language":
                    result = translation_quality_by_language(video_id, language)
                    st.write(result)
                elif localization_option == "Caption Performance in Countries":
                    result = caption_performance_in_countries(video_id, country)
                    st.write(result)
                elif localization_option == "Subtitle Recommendations":
                    result = subtitle_recommendations(video_id)
                    st.write(result)
                elif localization_option == "Dubbing Impact Analysis in Regions":
                    result = dubbing_impact_analysis_in_regions(video_id, region)
                    st.write(result)
                elif localization_option == "Localization Opportunities":
                    result = localization_opportunities(video_id)
                    st.write(result)
                elif localization_option == "Translation Suggestions":
                    result = translation_suggestions(video_id)
                    st.write(result)
                elif localization_option == "Multi-Language Performance Analysis":
                    result = multi_language_performance_analysis(video_id)
                    st.write(result)
                elif localization_option == "Localization Return on Investment":
                    result = localization_return_on_investment(video_id)
                    st.write(result)
            except Exception as e:
                st.error(f"Localization insights generation failed: {str(e)}")

            
elif selected_option == "Cross-Border Analysis":
    st.write("### Cross-Border Analysis")
    
    # Upload the OAuth JSON file
    uploaded_file = st.sidebar.file_uploader("Upload your OAuth JSON file", type=["json"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = "temp_client_secret.json"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
    
    SCOPES = ["https://www.googleapis.com/auth/youtube",
              "https://www.googleapis.com/auth/yt-analytics.readonly"]
    # https://developers.google.com/identity/protocols/oauth2/scopes

    # https://github.com/googleapis/google-api-python-client/blob/main/docs/dyn/index.md
    API_SERVICE_NAME = "youtubeAnalytics"
    API_VERSION = "v2"
    CLIENT_SECRETS_FILE = temp_file_path

    def get_service():
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_local_server()
        return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

    def execute_api_request(client_library_function, **kwargs):
        response = client_library_function(
            **kwargs
        ).execute()
        print(response)
        return response
    
    cross_border_analysis_option = st.selectbox(
        "Cross-Border Analysis:",
        [
            "Cross-Border Trend Comparison", 
            "Multi-Country Content Performance", 
            "International Reach Analysis", 
            "Global vs Local Metrics", 
            "Regional Content Adaptation", 
            "Market Engagement Comparison", 
            "Global Expansion Potential", 
            "Multi-Region Performance"
        ]
    )
    
    if cross_border_analysis_option == "Cross-Border Trend Comparison":
        # Get start and end dates from user
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date")
        end_date = col2.date_input("End Date")
        channel_id = st.text_input("Enter Channel ID")
        video_id = st.text_input("Enter Video ID")
        region1 = st.text_input("Enter the first region (e.g US, IN, GB)")
        region2 = st.text_input("Enter the second region (e.g US, IN, GB)")
    
    if cross_border_analysis_option == "Multi-Country Content Performance":
        # Input for YouTube API Key
        youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")

        # Set up YouTube API client only if YouTube API key is provided
        if youtube_api_key:
            youtube = build("youtube", "v3", developerKey=youtube_api_key)
        else:
            youtube = None  # Set to None if key is not provided

        video_id = st.text_input("Enter Video ID")
        countries = st.text_input("Enter the countries")
        
    if cross_border_analysis_option == "International Reach Analysis":
        youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")

        # Set up YouTube API client only if YouTube API key is provided
        if youtube_api_key:
            youtube = build("youtube", "v3", developerKey=youtube_api_key)
        else:
            youtube = None  # Set to None if key is not provided
        video_id = st.text_input("Enter Video ID")
        
    if cross_border_analysis_option == "Global vs Local Metrics":
        youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")

        # Set up YouTube API client only if YouTube API key is provided
        if youtube_api_key:
            youtube = build("youtube", "v3", developerKey=youtube_api_key)
        else:
            youtube = None  # Set to None if key is not provided
        video_id = st.text_input("Enter Video ID")
        
    if cross_border_analysis_option == "Regional Content Adaptation":
        youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")

        # Set up YouTube API client only if YouTube API key is provided
        if youtube_api_key:
            youtube = build("youtube", "v3", developerKey=youtube_api_key)
        else:
            youtube = None  # Set to None if key is not provided
        video_id = st.text_input("Enter Video ID")
        regions = st.selectbox("Select region", [
            "North America",
            "South America",
            "Europe",
            "Asia",
            "Africa",
            "Australia/Oceania"
        ])
        
    if cross_border_analysis_option == "Market Engagement Comparison":
        youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")

        # Set up YouTube API client only if YouTube API key is provided
        if youtube_api_key:
            youtube = build("youtube", "v3", developerKey=youtube_api_key)
        else:
            youtube = None  # Set to None if key is not provided
        markets = st.selectbox("Select markets (e.g., countries, regions)", 
        [
            "USA",
            "Canada",
            "UK",
            "Australia",
            "North America",
            "Europe",
            "Asia-Pacific",
            "Other (specify)"
        ])

        if markets == "Other (specify)":
            markets = st.text_input("Enter custom market(s) (comma-separated)")
            
    if cross_border_analysis_option == "Global Expansion Potential":
        youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")

        # Set up YouTube API client only if YouTube API key is provided
        if youtube_api_key:
            youtube = build("youtube", "v3", developerKey=youtube_api_key)
        else:
            youtube = None  # Set to None if key is not provided
        video_id = st.text_input("Enter Video ID")
        
    if cross_border_analysis_option == "Multi-Region Performance":
        youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")

        # Set up YouTube API client only if YouTube API key is provided
        if youtube_api_key:
            youtube = build("youtube", "v3", developerKey=youtube_api_key)
        else:
            youtube = None  # Set to None if key is not provided
            
        video_id = st.text_input("Enter Video ID")
        regions = st.selectbox("Select region", [
            "North America",
            "South America",
            "Europe",
            "Asia",
            "Africa",
            "Australia/Oceania"
        ])
    
    if st.button("Generate Cross-Border Analysis"):
        if cross_border_analysis_option == "Cross-Border Trend Comparison":
            result = cross_border_trend_comparison(video_id, region1, region2, start_date, end_date)
            st.write(result)
        elif cross_border_analysis_option == "Multi-Country Content Performance":
            result = multi_country_content_performance(video_id, countries)
            st.write(result)
        elif cross_border_analysis_option == "International Reach Analysis":
            result = international_reach_analysis(video_id)
            st.write(result)
        elif cross_border_analysis_option == "Global vs Local Metrics":
            result = global_vs_local_metrics(video_id)
            st.write(result)
        elif cross_border_analysis_option == "Regional Content Adaptation":
            result = regional_content_adaptation(video_id, regions)
            st.write(result)
        elif cross_border_analysis_option == "Market Engagement Comparison":
            result = market_engagement_comparison(video_id, markets)
            st.write(result)
        elif cross_border_analysis_option == "Global Expansion Potential":
            result = global_expansion_potential(video_id)
            st.write(result)
        elif cross_border_analysis_option == "Multi-Region Performance":
            result = multi_region_performance(video_id, regions)
            st.write(result)
        else:
            result = "Invalid Opiton"
            st.write(result)
            
elif selected_option == "Regional Content Strategy":
    st.write("### Regional Content Strategy")
    
    # Upload the OAuth JSON file
    uploaded_file = st.sidebar.file_uploader("Upload your OAuth JSON file", type=["json"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = "temp_client_secret.json"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
    
    SCOPES = ["https://www.googleapis.com/auth/yt-analytics.readonly"]
    # https://developers.google.com/identity/protocols/oauth2/scopes

    # https://github.com/googleapis/google-api-python-client/blob/main/docs/dyn/index.md
    API_SERVICE_NAME = "youtube"
    API_VERSION = "v3"
    CLIENT_SECRETS_FILE = temp_file_path

    def get_service():
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_local_server()
        return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

    def execute_api_request(client_library_function, **kwargs):
        response = client_library_function(
            **kwargs
        ).execute()
        print(response)
        return response
    
    regional_content_strategy_option = st.selectbox(
        "Regional Content Strategy:",
        [
            "Country-Specific Content Ideas",
            "Regional Title Optimization", 
            "Country Thumbnail Preferences", 
            "Format Analysis for Region", 
            "Local Keyword Research", 
            "Content Style Guidance", 
            "Regional Description Templates", 
            "Country Tag Suggestions"
        ]
    )
    
    if regional_content_strategy_option == "Country-Specific Content Ideas":
        country = st.text_input("Enter the Country")
        
    if regional_content_strategy_option == "Regional Title Optimization":
        language = st.text_input("Enter the language")
        region = st.selectbox("Select region", [
            "North America",
            "South America",
            "Europe",
            "Asia",
            "Africa",
            "Australia/Oceania"
        ])
        
    if regional_content_strategy_option == "Country Thumbnail Preferences":
        country = st.text_input("Enter the Country")
        
    if regional_content_strategy_option == "Format Analysis for Region":
        region = st.selectbox("Select region", [
            "North America",
            "South America",
            "Europe",
            "Asia",
            "Africa",
            "Australia/Oceania"
        ])
        
    if regional_content_strategy_option == "Local Keyword Research":
        country = st.text_input("Enter the Country")
        language = st.text_input("Enter the language")
        
    if regional_content_strategy_option == "Content Style Guidance":
        region = st.selectbox("Select region", [
            "North America",
            "South America",
            "Europe",
            "Asia",
            "Africa",
            "Australia/Oceania"
        ])
        
    if regional_content_strategy_option == "Regional Description Templates":
        region = st.selectbox("Select region", [
            "North America",
            "South America",
            "Europe",
            "Asia",
            "Africa",
            "Australia/Oceania"
        ])
        
    if regional_content_strategy_option == "Country Tag Suggestions":
        language = st.text_input("Enter the language")
    
    if st.button("Generate Regional Content Strategy"):
        with st.spinner("Generating regional content strategy..."):
            try:
                if regional_content_strategy_option == "Country-Specific Content Ideas":
                    result = country_specific_content_ideas(country)
                    st.write(result)
                elif regional_content_strategy_option == "Regional Title Optimization":
                    result = regional_title_optimization(region, language)
                    st.write(result)
                elif regional_content_strategy_option == "Country Thumbnail Preferences":
                    result = country_thumbnail_preferences(country)
                    st.write(result)
                elif regional_content_strategy_option == "Format Analysis for Region":
                    result = format_analysis_for_region(region)
                    st.write(result)
                elif regional_content_strategy_option == "Local Keyword Research":
                    result = local_keyword_research(country, language)
                    st.write(result)
                elif regional_content_strategy_option == "Content Style Guidance":
                    result = content_style_guidance(region)
                    st.write(result)
                elif regional_content_strategy_option == "Regional Description Templates":
                    result = regional_description_templates(region)
                    st.write(result)
                elif regional_content_strategy_option == "Country Tag Suggestions":
                    result = country_tag_suggestions(country)
                    st.write(result)
            except Exception as e:
                st.error(f"Content strategy generation failed: {str(e)}")


elif selected_option == "Local Competition Analysis":
    st.write("### Local Competition Analysis")
    
    # Upload the OAuth JSON file
    uploaded_file = st.sidebar.file_uploader("Upload your OAuth JSON file", type=["json"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = "temp_client_secret.json"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
    
    SCOPES = ["https://www.googleapis.com/auth/yt-analytics.readonly",
              "https://www.googleapis.com/auth/youtube"]
    # https://developers.google.com/identity/protocols/oauth2/scopes

    # https://github.com/googleapis/google-api-python-client/blob/main/docs/dyn/index.md
    API_SERVICE_NAME = "youtube"
    API_VERSION = "v3"
    CLIENT_SECRETS_FILE = temp_file_path

    def get_service():
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_local_server()
        return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

    def execute_api_request(client_library_function, **kwargs):
        response = client_library_function(
            **kwargs
        ).execute()
        print(response)
        return response
    
    local_competition_analysis_option = st.selectbox(
        "Local Competition Analysis:",
        [
            "Country Top Creators", 
            "Regional Competitor Analysis", 
            "Cross-Country Creator Comparison", 
            "Market Leader Identification", 
            "Niche Competitor Finder", 
            "Local Channel Strategy Insights", 
            "Regional Performance Benchmarks", 
            "Market Share Analysis"
        ]
    )
    
    if local_competition_analysis_option == "Country Top Creators":
        country = st.text_input("Enter the Country")
        
    if local_competition_analysis_option == "Regional Competitor Analysis":
        region = st.selectbox("Select region", [
            "North America",
            "South America",
            "Europe",
            "Asia",
            "Africa",
            "Australia/Oceania"
        ])
        
    if local_competition_analysis_option == "Cross-Country Creator Comparison":
        country1 = st.text_input("Enter the first country")
        country2 = st.text_input("Enter the second country")
        
    if local_competition_analysis_option == "Market Leader Identification":
        market = st.selectbox("Select markets (e.g., countries, regions)", 
        [
            "USA",
            "Canada",
            "UK",
            "Australia",
            "North America",
            "Europe",
            "Asia-Pacific",
            "Other (specify)"
        ])

        if market == "Other (specify)":
            market = st.text_input("Enter custom market(s) (comma-separated)")
            
    if local_competition_analysis_option == "Niche Competitor Finder":
        niches = [
            "health and wellness",
            "personal finance",
            "gaming",
            "technology",
            "marketing",
            "travel",
            "food",
            "fitness"
        ]

        st.title("Niche Competitor Finder")
        niche = st.selectbox("Select a niche", niches)
        
    if local_competition_analysis_option == "Local Channel Strategy Insights":
        country = st.text_input("Enter the Country")
        
    if local_competition_analysis_option == "Regional Performance Benchmarks":
        region = st.selectbox("Select region", [
            "North America",
            "South America",
            "Europe",
            "Asia",
            "Africa",
            "Australia/Oceania"
        ])
        
    if local_competition_analysis_option == "Market Share Analysis":
        market = st.selectbox("Select markets (e.g., countries, regions)", 
        [
            "USA",
            "Canada",
            "UK",
            "Australia",
            "North America",
            "Europe",
            "Asia-Pacific",
            "Other (specify)"
        ])

        if market == "Other (specify)":
            market = st.text_input("Enter custom market(s) (comma-separated)")
    
    if st.button("Perform Local Competition Analysis"):
        if local_competition_analysis_option == "Country Top Creators":
            result = country_top_creators(country)
            st.write(result)
        elif local_competition_analysis_option == "Regional Competitor Analysis":
            result = regional_competitor_analysis(region)
            st.write(result)
        elif local_competition_analysis_option == "Cross-Country Creator Comparison":
            result = cross_country_creator_comparison(country1, country2)
            st.write(result)
        elif local_competition_analysis_option == "Market Leader Identification":
            result = market_leader_identification(market)
            st.write(result)
        elif local_competition_analysis_option == "Niche Competitor Finder":
            result = niche_competitor_finder(niche)
            st.write(result)
        elif local_competition_analysis_option == "Local Channel Strategy Insights":
            result = local_channel_strategy_insights(country)
            st.write(result)
        elif local_competition_analysis_option == "Regional Performance Benchmarks":
            result = regional_performance_benchmarks(region)
            st.write(result)
        elif local_competition_analysis_option == "Market Share Analysis":
            result = market_share_analysis(market)
            st.write(result)
        else:
            result = "Invalid option"
            st.write(result)
            
elif selected_option == "Time Zone-Based Analysis":
    st.write("### Time Zone-Based Analysis")
    
    # Upload the OAuth JSON file
    uploaded_file = st.sidebar.file_uploader("Upload your OAuth JSON file", type=["json"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = "temp_client_secret.json"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
    
    SCOPES = ["https://www.googleapis.com/auth/yt-analytics.readonly",
              "https://www.googleapis.com/auth/youtube"]
    # https://developers.google.com/identity/protocols/oauth2/scopes

    # https://github.com/googleapis/google-api-python-client/blob/main/docs/dyn/index.md
    API_SERVICE_NAME = "youtube"
    API_VERSION = "v3"
    CLIENT_SECRETS_FILE = temp_file_path

    def get_service():
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_local_server()
        return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

    def execute_api_request(client_library_function, **kwargs):
        response = client_library_function(
            **kwargs
        ).execute()
        print(response)
        return response
    
    time_zone_based_analysis_option = st.selectbox(
        "Time Zone-Based Analysis:",
        [
            "Best upload times by region", 
            "Peak viewing hours by country", 
            "Engagement patterns by timezone", 
            "Performance comparison across time zones", 
            "Optimal posting schedule by region", 
            "Audience activity times by country", 
            "Live stream timing analysis by region", 
            "Regional prime times"
        ]
    )
    
    timezones = pytz.common_timezones
    
    if time_zone_based_analysis_option == "Best Upload Times by Region":
        region = st.selectbox("Select region", [
            "North America",
            "South America",
            "Europe",
            "Asia",
            "Africa",
            "Australia/Oceania"
        ])
        
    if time_zone_based_analysis_option == "Peak Viewing Hours by Country":
        country = st.text_input("Enter the Country")
        
    if time_zone_based_analysis_option == "Engagement Patterns by Timezone":
        timezone = st.selectbox("Select timezone", timezones)
        
    if time_zone_based_analysis_option == "Performance Comparison Across Time Zones":
        timezone1 = st.selectbox("Select timezone", timezones)
        timezone2 = st.selectbox("Select timezone", timezones)
        
    if time_zone_based_analysis_option == "Optimal Posting Schedule by Region":
        region = st.selectbox("Select region", [
            "North America",
            "South America",
            "Europe",
            "Asia",
            "Africa",
            "Australia/Oceania"
        ])
        
    if time_zone_based_analysis_option == "Audience Activity Times by Country":
        country = st.text_input("Enter the Country")
        
    if time_zone_based_analysis_option == "Live Stream Timing Analysis by Region":
        region = st.selectbox("Select region", [
            "North America",
            "South America",
            "Europe",
            "Asia",
            "Africa",
            "Australia/Oceania"
        ])
        
    if time_zone_based_analysis_option == "Regional Prime Times":
        region = st.selectbox("Select region", [
            "North America",
            "South America",
            "Europe",
            "Asia",
            "Africa",
            "Australia/Oceania"
        ])
    
    if st.button("Perform Time Zone-Based Analysis"):
        with st.spinner("Performing time zone-based analysis..."):
            try:
                if time_zone_based_analysis_option == "Best Upload Times by Region":
                    result = best_upload_times_by_region(region)
                    st.write(result)
                elif time_zone_based_analysis_option == "Peak Viewing Hours by Country":
                    result = peak_viewing_hours_by_country(country)
                    st.write(result)
                elif time_zone_based_analysis_option == "Engagement Patterns by Timezone":
                    result = engagement_patterns_by_timezone(timezone)
                    st.write(result)
                elif time_zone_based_analysis_option == "Performance Comparison Across Time Zones":
                    result = performance_comparison_across_time_zones(timezone1, timezone2)
                    st.write(result)
                elif time_zone_based_analysis_option == "Optimal Posting Schedule by Region":
                    result = optimal_posting_schedule_by_region(region)
                    st.write(result)
                elif time_zone_based_analysis_option == "Audience Activity Times by Country":
                    result = audience_activity_times_by_country(country)
                    st.write(result)
                elif time_zone_based_analysis_option == "Live Stream Timing Analysis by Region":
                    result = live_stream_timing_analysis_by_region(region)
                    st.write(result)
                elif time_zone_based_analysis_option == "Regional Prime Times":
                    result = regional_prime_times(region)
                    st.write(result)
            except Exception as e:
                st.error(f"Time zone-based analysis failed: {str(e)}")

            
elif selected_option == "Geographic Performance Metrics":
    st.write("### Geographic Performance Metrics")
    
    # Upload the OAuth JSON file
    uploaded_file = st.sidebar.file_uploader("Upload your OAuth JSON file", type=["json"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = "temp_client_secret.json"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
    
    SCOPES = ["https://www.googleapis.com/auth/yt-analytics.readonly",
              "https://www.googleapis.com/auth/youtube"]
    # https://developers.google.com/identity/protocols/oauth2/scopes

    # https://github.com/googleapis/google-api-python-client/blob/main/docs/dyn/index.md
    API_SERVICE_NAME = "youtube"
    API_VERSION = "v3"
    CLIENT_SECRETS_FILE = temp_file_path

    def get_service():
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_local_server()
        return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

    def execute_api_request(client_library_function, **kwargs):
        response = client_library_function(
            **kwargs
        ).execute()
        print(response)
        return response
    
    geographic_performance_metrics_option = st.selectbox(
        "Geographic Performance Metrics:",
        [
            "Country Watch Time Distribution", 
            "Regional Subscriber Demographics", 
            "View Duration Pattern Analysis", 
            "Cross-Regional Engagement Comparison"
        ]
    )
    
    if geographic_performance_metrics_option == "Country Watch Time Distribution":
        country = st.text_input("Enter the Country")
        
    if geographic_performance_metrics_option == "Regional Subscriber Demographics":
        region = st.selectbox("Select region", [
            "North America",
            "South America",
            "Europe",
            "Asia",
            "Africa",
            "Australia/Oceania"
        ])
        
    if geographic_performance_metrics_option == "View Duration Pattern Analysis":
        country = st.text_input("Enter the Country")
        
    if geographic_performance_metrics_option == "Cross-Regional Engagement Comparison":
        region1 = st.selectbox("Select region", [
            "North America",
            "South America",
            "Europe",
            "Asia",
            "Africa",
            "Australia/Oceania"
        ])
        region2 = st.selectbox("Select region", [
            "North America",
            "South America",
            "Europe",
            "Asia",
            "Africa",
            "Australia/Oceania"
        ])
    
    if geographic_performance_metrics_option == "Country-Level Super Chat Revenue":
        country = st.text_input("Enter the Country")
        
    if geographic_performance_metrics_option == "Regional Membership Growth":
        region = st.selectbox("Select region", [
            "North America",
            "South America",
            "Europe",
            "Asia",
            "Africa",
            "Australia/Oceania"
        ])
        
    if geographic_performance_metrics_option == "Super Thanks Distribution Insights":
        country = st.text_input("Enter the Country")
        
    if geographic_performance_metrics_option == "Country-Specific Merchandise Sales":
        country = st.text_input("Enter the Country")
        
    if st.button("Geographic Performance Metrics"):
        with st.spinner("Analyzing geographic performance metrics..."):
            try:
                if geographic_performance_metrics_option == "Country Watch Time Distribution":
                    result = country_watch_time_distribution(country)
                    st.write(result)
                elif geographic_performance_metrics_option == "Regional Subscriber Demographics":
                    result = regional_subscriber_demographics(region)
                    st.write(result)
                elif geographic_performance_metrics_option == "View Duration Pattern Analysis":
                    result = view_duration_pattern_analysis(country)
                    st.write(result)
                elif geographic_performance_metrics_option == "Cross-Regional Engagement Comparison":
                    result = cross_regional_engagement_comparison(region1, region2)
                    st.write(result)
            except Exception as e:
                st.error(f"Geographic performance metrics analysis failed: {str(e)}")
            
elif selected_option == "Cultural Trend Analysis":
    st.write("### Cultural Trend Analysis")
    
    # Upload the OAuth JSON file
    uploaded_file = st.sidebar.file_uploader("Upload your OAuth JSON file", type=["json"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = "temp_client_secret.json"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
    
    SCOPES = ["https://www.googleapis.com/auth/yt-analytics.readonly",
              "https://www.googleapis.com/auth/youtube"]
    # https://developers.google.com/identity/protocols/oauth2/scopes

    # https://github.com/googleapis/google-api-python-client/blob/main/docs/dyn/index.md
    API_SERVICE_NAME = "youtube"
    API_VERSION = "v3"
    CLIENT_SECRETS_FILE = temp_file_path

    def get_service():
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_local_server()
        return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

    def execute_api_request(client_library_function, **kwargs):
        response = client_library_function(
            **kwargs
        ).execute()
        print(response)
        return response
    
    cultural_trend_analysis_option = st.selectbox(
        "Cultural Trend Analysis:",
        [
             "City/Regional Trend Tracker", 
             "Country Seasonal Trends", 
             "Cultural Event Impact Analysis", 
             "Holiday Content Spotlight", 
             "Local Celebrity Trend Monitor", 
             "Regional Meme Tracker", 
             "Local News Trend Impact", 
             "Festival Content Performance"
        ]
    )
    
    if cultural_trend_analysis_option == "City/Regional Trend Tracker":
        city = st.text_input("Enter the City")
        
    if cultural_trend_analysis_option == "Country Seasonal Trends":
        country = st.text_input("Enter the Country")
    if cultural_trend_analysis_option == "Cultural Event Impact Analysis":
        event = st.text_input("Enter event")
        
    if cultural_trend_analysis_option == "Holiday Content Spotlight":
        holiday = st.text_input("Enter holiday")
        
    if cultural_trend_analysis_option == "Local Celebrity Trend Monitor":
        celebrity = st.text_input("Enter celebrity")
        
    if cultural_trend_analysis_option == "Regional Meme Tracker":
        region = st.selectbox("Select region", [
            "North America",
            "South America",
            "Europe",
            "Asia",
            "Africa",
            "Australia/Oceania"
        ])
        
    if cultural_trend_analysis_option == "Local News Trend Impact":
        news_api_key = st.text_input("YOUR_NEWS_API_KEY")
        news_api_url = f"https://newsapi.org/v2/top-headlines?apiKey={news_api_key}"

        response = requests.get(news_api_url)
        news_data = response.json()

        news_topics = [article["title"] for article in news_data["articles"]]
        news = news_topics
       
    if cultural_trend_analysis_option == "Festival Content Performance":    
        festival_api_key = st.text_input("YOUR_FESTIVAL_API_KEY")
        festival_api_url = f"https://api.predicthq.com/v1/events/?apiKey={festival_api_key}"

        response = requests.get(festival_api_url)
        festival_data = response.json()

        festivals = [event["name"] for event in festival_data["events"]]
        festival = festivals
    
    if st.button("Cultural Trend Analysis"):
        with st.spinner("Performing cultural trend analysis..."):
            try:
                if cultural_trend_analysis_option == "City/Regional Trend Tracker":
                    result = city_regional_trend_tracker(city)
                    st.write(result)
                elif cultural_trend_analysis_option == "Country Seasonal Trends":
                    result = country_seasonal_trends(country)
                    st.write(result)
                elif cultural_trend_analysis_option == "Cultural Event Impact Analysis":
                    result = cultural_event_impact_analysis(event)
                    st.write(result)
                elif cultural_trend_analysis_option == "Holiday Content Spotlight":
                    result = holiday_content_spotlight(holiday)
                    st.write(result)
                elif cultural_trend_analysis_option == "Local Celebrity Trend Monitor":
                    result = local_celebrity_trend_monitor(celebrity)
                    st.write(result)
                elif cultural_trend_analysis_option == "Regional Meme Tracker":
                    result = regional_meme_tracker(region)
                    st.write(result)
                elif cultural_trend_analysis_option == "Local News Trend Impact":
                    result = local_news_trend_impact(news)
                    st.write(result)
                elif cultural_trend_analysis_option == "Festival Content Performance":
                    result = festival_content_performance(festival)
                    st.write(result)
            except Exception as e:
                st.error(f"Cultural trend analysis failed: {str(e)}")
            
elif selected_option == "Market Research Commands":
    st.write("### Market Research Commands")
    
    # Upload the OAuth JSON file
    uploaded_file = st.sidebar.file_uploader("Upload your OAuth JSON file", type=["json"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = "temp_client_secret.json"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
    
    SCOPES = ["https://www.googleapis.com/auth/yt-analytics.readonly",
              "https://www.googleapis.com/auth/youtube"]
    # https://developers.google.com/identity/protocols/oauth2/scopes

    # https://github.com/googleapis/google-api-python-client/blob/main/docs/dyn/index.md
    API_SERVICE_NAME = "youtube"
    API_VERSION = "v3"
    CLIENT_SECRETS_FILE = temp_file_path

    def get_service():
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_local_server()
        return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

    def execute_api_request(client_library_function, **kwargs):
        response = client_library_function(
            **kwargs
        ).execute()
        print(response)
        return response
    
    market_research_command_option = st.selectbox(
        "Market Research Commands:",
        [
            "Country Market Sizing", 
            "Competition Level Analysis", 
            "Regional Niche Opportunities", 
            "Market Saturation Comparison", 
            "Audience Preference Insights", 
            "Content Gap Identification", 
            "Ad Rate Benchmarking", 
            "Monetization Potential Assessment"
        ]
    )
    
    if market_research_command_option == "Country Market Sizing":
        country = st.text_input("Enter the Country")
        
    if market_research_command_option == "Competition Level Analysis":
        niches = [
            "health and wellness",
            "personal finance",
            "gaming",
            "technology",
            "marketing",
            "travel",
            "food",
            "fitness"
        ]

        st.title("Niche Competitor Finder")
        niche = st.selectbox("Select a niche", niches)
        
    if market_research_command_option == "Regional Niche Opportunities":
        region = st.selectbox("Select region", [
            "North America",
            "South America",
            "Europe",
            "Asia",
            "Africa",
            "Australia/Oceania"
        ])
        
    if market_research_command_option == "Market Saturation Comparison":
        market1 = st.selectbox("Select markets (e.g., countries, regions)", 
        [
            "USA",
            "Canada",
            "UK",
            "Australia",
            "North America",
            "Europe",
            "Asia-Pacific",
            "Other (specify)"
        ])

        if market1 == "Other (specify)":
            market1 = st.text_input("Enter custom market(s) (comma-separated)")
        
        market2 = st.selectbox("Select markets (e.g., countries, regions)", 
        [
            "USA",
            "Canada",
            "UK",
            "Australia",
            "North America",
            "Europe",
            "Asia-Pacific",
            "Other (specify)"
        ])

        if market2 == "Other (specify)":
            market2 = st.text_input("Enter custom market(s) (comma-separated)")
        
    if market_research_command_option == "Audience Preference Insights":
        audience = st.text_input("Enter Preferred Audience")
        
    if market_research_command_option == "Content Gap Identification":
        niches = [
            "health and wellness",
            "personal finance",
            "gaming",
            "technology",
            "marketing",
            "travel",
            "food",
            "fitness"
        ]

        st.title("Niche Competitor Finder")
        niche = st.selectbox("Select a niche", niches)
        
    if market_research_command_option == "Ad Rate Benchmarking":
        niches = [
            "health and wellness",
            "personal finance",
            "gaming",
            "technology",
            "marketing",
            "travel",
            "food",
            "fitness"
        ]

        st.title("Niche Competitor Finder")
        niche = st.selectbox("Select a niche", niches)
        
    if market_research_command_option == "Monetization Potential Assessment":
        niches = [
            "health and wellness",
            "personal finance",
            "gaming",
            "technology",
            "marketing",
            "travel",
            "food",
            "fitness"
        ]

        st.title("Niche Competitor Finder")
        niche = st.selectbox("Select a niche", niches)
    
    if st.button("Perform Market Research"):
        with st.spinner("Performing market research..."):
            try:
                if market_research_command_option == "Country Market Sizing":
                    result = country_market_sizing(country)
                    st.write(result)
                elif market_research_command_option == "Competition Level Analysis":
                    result = competition_level_analysis(niche)
                    st.write(result)
                elif market_research_command_option == "Regional Niche Opportunities":
                    result = regional_niche_opportunities(region)
                    st.write(result)
                elif market_research_command_option == "Market Saturation Comparison":
                    result = market_saturation_comparison(market1, market2)
                    st.write(result)
                elif market_research_command_option == "Audience Preference Insights":
                    result = audience_preference_insights(audience)
                    st.write(result)
                elif market_research_command_option == "Content Gap Identification":
                    result = content_gap_identification(niche)
                    st.write(result)
                elif market_research_command_option == "Ad Rate Benchmarking":
                    result = ad_rate_benchmarking(niche)
                    st.write(result)
                elif market_research_command_option == "Monetization Potential Assessment":
                    result = monetization_potential_assessment(niche)
                    st.write(result)
            except Exception as e:
                st.error(f"Market research failed: {str(e)}")
            
elif selected_option == "Language-Based Search":
    st.write("### Language-Based Search")
    
    # Upload the OAuth JSON file
    uploaded_file = st.sidebar.file_uploader("Upload your OAuth JSON file", type=["json"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = "temp_client_secret.json"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
    
    SCOPES = ["https://www.googleapis.com/auth/yt-analytics.readonly",
              "https://www.googleapis.com/auth/youtube"]
    # https://developers.google.com/identity/protocols/oauth2/scopes

    # https://github.com/googleapis/google-api-python-client/blob/main/docs/dyn/index.md
    API_SERVICE_NAME = "youtube"
    API_VERSION = "v3"
    CLIENT_SECRETS_FILE = temp_file_path

    def get_service():
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_local_server()
        return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

    def execute_api_request(client_library_function, **kwargs):
        response = client_library_function(
            **kwargs
        ).execute()
        print(response)
        return response
    
    language_Based_search_option = st.selectbox(
        "Language-Based Search:",
        [
            "Language Trending Videos", 
            "Popular Creator Spotlight", 
            "Topic Trend Analysis", 
            "Hashtag Intelligence", 
            "Cross-Linguistic Engagement Comparison", 
            "Emerging Channel Tracker", 
            "Language-Specific Keyword Suggestions", 
            "Viral Short-Form"
        ]
    )
    
    if language_Based_search_option == "Language Trending Videos":
        language = st.text_input("Enter the language")
        
    if language_Based_search_option == "Popular Creator Spotlight":
        language = st.text_input("Enter the language")
        
    if language_Based_search_option == "Topic Trend Analysis":
        language = st.text_input("Enter the language")
        topic = st.text_input("What topic are you interested in?")
        
    if language_Based_search_option == "Hashtag Intelligence":
        language = st.text_input("Enter the language")
        hashtag = st.text_input("Enter the hashtag")
        
    if language_Based_search_option == "Cross-Linguistic Engagement Comparison":
        language1 = st.text_input("Enter the language")
        language2 = st.text_input("Enter the language")
        
    if language_Based_search_option == "Emerging Channel Tracker":
        language = st.text_input("Enter the language")
        
    if language_Based_search_option == "Language-Specific Keyword Suggestions":
        language = st.text_input("Enter the language")
        
    if language_Based_search_option == "Viral Short-Form Videos":
        language = st.text_input("Enter the language")
    
    if st.button("Perform Language-Based Search"):
        with st.spinner("Performing language-based search..."):
            try:
                if language_Based_search_option == "Language Trending Videos":
                    result = language_trending_videos(language)
                    st.success("Search complete!")
                    st.write(result)
                elif language_Based_search_option == "Popular Creator Spotlight":
                    result = popular_creator_spotlight(language)
                    st.success("Search complete!")
                    st.write(result)
                elif language_Based_search_option == "Topic Trend Analysis":
                    result = topic_trend_analysis(language, topic)
                    st.success("Search complete!")
                    st.write(result)
                elif language_Based_search_option == "Hashtag Intelligence":
                    result = hashtag_intelligence(language, hashtag)
                    st.success("Search complete!")
                    st.write(result)
                elif language_Based_search_option == "Cross-Linguistic Engagement Comparison":
                    result = cross_linguistic_engagement_comparison(language1, language2)
                    st.success("Search complete!")
                    st.write(result)
                elif language_Based_search_option == "Emerging Channel Tracker":
                    result = emerging_channel_tracker(language)
                    st.success("Search complete!")
                    st.write(result)
                elif language_Based_search_option == "Language-Specific Keyword Suggestions":
                    result = language_specific_keyword_suggestions(language)
                    st.success("Search complete!")
                    st.write(result)
                elif language_Based_search_option == "Viral Short-Form Videos":
                    result = viral_short_form_videos(language)
                    st.success("Search complete!")
                    st.write(result)
            except Exception as e:
                st.error(f"Search failed: {str(e)}")
            
elif selected_option == "Regional Analysis":
    st.write("### Regional Analysis")
    
    # Upload the OAuth JSON file
    uploaded_file = st.sidebar.file_uploader("Upload your OAuth JSON file", type=["json"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = "temp_client_secret.json"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
    
    SCOPES = ["https://www.googleapis.com/auth/yt-analytics.readonly",
              "https://www.googleapis.com/auth/youtube"]
    # https://developers.google.com/identity/protocols/oauth2/scopes

    # https://github.com/googleapis/google-api-python-client/blob/main/docs/dyn/index.md
    API_SERVICE_NAME = "youtube"
    API_VERSION = "v3"
    CLIENT_SECRETS_FILE = temp_file_path

    def get_service():
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_local_server()
        return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

    def execute_api_request(client_library_function, **kwargs):
        response = client_library_function(
            **kwargs
        ).execute()
        print(response)
        return response
    
    regional_analysis_option = st.selectbox(
        "Regional Analysis:",
        [
            "Country-Specific Trending Videos", 
            "Top Videos by Country", 
            "Keyword Trend Analysis", 
            "Cross-Country Trend Comparison", 
            "Emerging Creators by Country", 
            "Viral Content Tracker", 
            "Country-Level Hashtag Trends", 
            "Popular Music Trends by Country"
        ]
    )
    
    if regional_analysis_option == "Country-Specific Trending Videos":
        country = st.text_input("Enter the Country")
        
    if regional_analysis_option == "Top Videos by Country":
        country = st.text_input("Enter the Country")
    
    if regional_analysis_option == "Keyword Trend Analysis":
        country = st.text_input("Enter the Country")
        keyword = st.text_input("Enter the keyword")
        
    if regional_analysis_option == "Cross-Country Trend Comparison":
        country1 = st.text_input("Enter the Country")
        country2 = st.text_input("Enter the Country")
        
    if regional_analysis_option == "Emerging Creators by Country":
        country = st.text_input("Enter the Country")
        
    if regional_analysis_option == "Viral Content Tracker":
        country = st.text_input("Enter the Country")
        
    if regional_analysis_option == "Country-Level Hashtag Trends":
        country = st.text_input("Enter the Country")
        hashtag = st.text_input("Enter the hashtag")
        
    if regional_analysis_option == "Popular Music Trends by Country":
        country = st.text_input("Enter the Country")
    
    if st.button("Perform Regional Analysis"):
        with st.spinner("Performing regional analysis..."):
            try:
                if regional_analysis_option == "Country-Specific Trending Videos":
                    result = country_specific_trending_videos(country)
                    st.success("Analysis complete!")
                    st.write(result)
                elif regional_analysis_option == "Top Videos by Country":
                    result = top_videos_by_country(country)
                    st.success("Analysis complete!")
                    st.write(result)
                elif regional_analysis_option == "Keyword Trend Analysis":
                    result = keyword_trend_analysis(country, keyword)
                    st.success("Analysis complete!")
                    st.write(result)
                elif regional_analysis_option == "Cross-Country Trend Comparison":
                    result = cross_country_trend_comparison(country1, country2)
                    st.success("Analysis complete!")
                    st.write(result)
                elif regional_analysis_option == "Emerging Creators by Country":
                    result = emerging_creators_by_country(country)
                    st.success("Analysis complete!")
                    st.write(result)
                elif regional_analysis_option == "Viral Content Tracker":
                    result = viral_content_tracker(country)
                    st.success("Analysis complete!")
                    st.write(result)
                elif regional_analysis_option == "Country-Level Hashtag Trends":
                    result = country_level_hashtag_trends(country, hashtag)
                    st.success("Analysis complete!")
                    st.write(result)
                elif regional_analysis_option == "Popular Music Trends by Country":
                    result = popular_music_trends_by_country(country)
                    st.success("Analysis complete!")
                    st.write(result)
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
            
elif selected_option == "Regional Performance Analysis":
    st.write("### Regional Performance Analysis")
    
    # Upload the OAuth JSON file
    uploaded_file = st.sidebar.file_uploader("Upload your OAuth JSON file", type=["json"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = "temp_client_secret.json"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
    
    SCOPES = ["https://www.googleapis.com/auth/yt-analytics.readonly",
              "https://www.googleapis.com/auth/youtube",
              "https://www.googleapis.com/auth/yt-analytics-monetary.readonly"]
    # https://developers.google.com/identity/protocols/oauth2/scopes

    # https://github.com/googleapis/google-api-python-client/blob/main/docs/dyn/index.md
    API_SERVICE_NAME = "youtubeAnalytics"
    API_VERSION = "v2"
    CLIENT_SECRETS_FILE = temp_file_path

    def get_service():
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_local_server()
        return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

    def execute_api_request(client_library_function, **kwargs):
        response = client_library_function(
            **kwargs
        ).execute()
        print(response)
        return response
    
    regional_performance_analysis_option = st.selectbox(
        "Regional Performance Analysis:",
        [
            "Regional Video Performance", 
            "Country-Specific Viewer Demographics", 
            "Cross-Country View Comparison", 
            "Country-Level Engagement Rates", 
            "Top-Performing Regions", 
            "Audience Retention by Country", 
            "Country-Originating Traffic", 
            "Regional Click-Through Rate Comparison"
        ]
    )
    
    if regional_performance_analysis_option == "Regional Video Performance":
        # Input for YouTube API Key
        youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")

        # Set up YouTube API client only if YouTube API key is provided
        if youtube_api_key:
            youtube = build("youtube", "v3", developerKey=youtube_api_key)
        else:
            youtube = None  # Set to None if key is not provided
        country = st.text_input("Enter the Country e.g. US, GB, IN")
        video_id = st.text_input("Enter the Video ID")
        
    if regional_performance_analysis_option == "Country-Specific Viewer Demographics":
        country = st.text_input("Enter the Country e.g. US, GB, IN")
        video_id = st.text_input("Enter the Video ID")
        # Input for YouTube API Key
        youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")

        # Set up YouTube API client only if YouTube API key is provided
        if youtube_api_key:
            youtube = build("youtube", "v3", developerKey=youtube_api_key)
        else:
            youtube = None  # Set to None if key is not provided
        
    if regional_performance_analysis_option == "Cross-Country View Comparison":
        country1 = st.text_input("Enter the Country e.g. US, GB, IN")
        country2 = st.text_input("Enter the Country e.g. US, GB, IN")
        video_id = st.text_input("Enter the Video ID")
        # Input for YouTube API Key
        youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")

        # Set up YouTube API client only if YouTube API key is provided
        if youtube_api_key:
            youtube = build("youtube", "v3", developerKey=youtube_api_key)
        else:
            youtube = None  # Set to None if key is not provided
        
    if regional_performance_analysis_option == "Country-Level Engagement Rates":
        country = st.text_input("Enter the Country e.g. US, GB, IN")
        video_id = st.text_input("Enter the Video ID")
        
    if regional_performance_analysis_option == "Top-Performing Regions":
        video_id = st.text_input("Enter the Video ID")
        # Input for YouTube API Key
        youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")

        # Set up YouTube API client only if YouTube API key is provided
        if youtube_api_key:
            youtube = build("youtube", "v3", developerKey=youtube_api_key)
        else:
            youtube = None  # Set to None if key is not provided
        
    if regional_performance_analysis_option == "Audience Retention by Country":
        country = st.text_input("Enter the Country e.g. US, GB, IN")
        video_id = st.text_input("Enter the Video ID")
        
    if regional_performance_analysis_option == "Country-Originating Traffic":
        country = st.text_input("Enter the Country e.g. US, GB, IN")
        video_id = st.text_input("Enter the Video ID")
        
    if regional_performance_analysis_option == "Regional Click-Through Rate Comparison":
        country1 = st.text_input("Enter the Country e.g. US, GB, IN")
        country2 = st.text_input("Enter the Country e.g. US, GB, IN")
        video_id = st.text_input("Enter the Video ID")
        
    if regional_performance_analysis_option == "Audience Retention by Country":
        country = st.text_input("Enter country")
        video_id = st.text_input("Enter Video ID")
        start_date = st.date_input("Start Date")
        end_date = st.date_input("End Date")
        
    if regional_performance_analysis_option == "Country-Originating Traffic":
        country = st.text_input("Enter country")
        video_id = st.text_input("Enter Video ID")
        start_date = st.date_input("Start Date")
        end_date = st.date_input("End Date")
        
    if regional_performance_analysis_option == "Regional Click-Through Rate Comparison":
        country1 = st.text_input("Enter country")
        country2 = st.text_input("Enter country")
        video_id = st.text_input("Enter Video ID")
        start_date = st.date_input("Start Date")
        end_date = st.date_input("End Date")
    
    if st.button("Regional Performance Analysis"):
        with st.spinner("Analyzing regional performance..."):
            try:
                if regional_performance_analysis_option == "Regional Video Performance":
                    result = regional_video_performance(country, video_id)
                    st.success("Analysis complete!")
                    st.write(result)
                elif regional_performance_analysis_option == "Country-Specific Viewer Demographics":
                    result = country_specific_viewer_demographics(country, video_id)
                    st.success("Analysis complete!")
                    st.write(result)
                elif regional_performance_analysis_option == "Cross-Country View Comparison":
                    result = cross_country_view_comparison(country1, country2, video_id)
                    st.success("Analysis complete!")
                    st.write(result)
                elif regional_performance_analysis_option == "Country-Level Engagement Rates":
                    result = country_level_engagement_rates(country, video_id)
                    st.success("Analysis complete!")
                    st.write(result)
                elif regional_performance_analysis_option == "Top-Performing Regions":
                    result = top_performing_regions(video_id)
                    st.success("Analysis complete!")
                    st.write(result)
                elif regional_performance_analysis_option == "Audience Retention by Country":
                    result = audience_retention_by_country(country, video_id, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif regional_performance_analysis_option == "Country-Originating Traffic":
                    result = country_originating_traffic(country, video_id, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif regional_performance_analysis_option == "Regional Click-Through Rate Comparison":
                    result = regional_click_through_rate_comparison(country1, country2, video_id, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
            except Exception  as e:
                    st.error(f"Generation failed: {str(e)}")
            
elif selected_option == "Natural Language Queries":
    st.write("### Natural Language Queries")
    
    api_key = st.sidebar.text_input("Enter your OpenAI Api Key", type="password")
    
    client = OpenAI(api_key=api_key)
    
    # Input for YouTube API Key
    youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")

    # Set up YouTube API client only if YouTube API key is provided
    if youtube_api_key:
        youtube = build("youtube", "v3", developerKey=youtube_api_key)
    else:
        youtube = None  # Set to None if key is not provided

    # Input for YouTube video URL
    video_url = st.text_input("Enter the YouTube video URL:")
    
    natural_language_queries_option = st.selectbox(
        "Natural Language Queries:",
        [
            "Boost Video Views", 
            "Enhance Click-Through Rate", 
            "Find Profitable Topics", 
            "Optimal Upload Time", 
            "Grow Subscribers", 
            "Niche Success Strategies", 
            "Thumbnail Improvement", 
            "Diagnose View Drop", 
            "Search Optimization Tips", 
            "Effective Hashtags"
        ]
    )
    
    if natural_language_queries_option == "Boost Video Views":
        query = st.text_input("What is your question?")
    
    if natural_language_queries_option == "Enhance Click-Through Rate":
        query = st.text_input("What is your question?")
        
    if natural_language_queries_option == "Find Profitable Topics":
        query = st.text_input("What is your question?")
        
    if natural_language_queries_option == "Optimal Upload Time":
        query = st.text_input("Ask about the best time to upload video")
        
    if natural_language_queries_option == "Grow Subscribers":
        query = st.text_input("Ask about how to grow subscribers")
        
    if natural_language_queries_option == "Niche Success Strategies":
        query = st.text_input("Ask about niche success strategies")
        
    if natural_language_queries_option == "Thumbnail Improvement":
        video_id = st.text_input("Enter Video ID")
        
    if natural_language_queries_option == "Diagnose View Drop":
        # Upload the OAuth JSON file
        uploaded_file = st.sidebar.file_uploader("Upload your OAuth JSON file", type=["json"])
        
        if uploaded_file is not None:
            # Save the uploaded file temporarily
            temp_file_path = "temp_client_secret.json"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
        
        SCOPES = ["https://www.googleapis.com/auth/yt-analytics.readonly"]
        # https://developers.google.com/identity/protocols/oauth2/scopes

        # https://github.com/googleapis/google-api-python-client/blob/main/docs/dyn/index.md
        API_SERVICE_NAME = "youtubeAnalytics"
        API_VERSION = "v2"
        CLIENT_SECRETS_FILE = temp_file_path

        def get_service():
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
            credentials = flow.run_local_server()
            return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

        def execute_api_request(client_library_function, **kwargs):
            response = client_library_function(
                **kwargs
            ).execute()
            print(response)
            return response
        
        video_id = st.text_input("Enter Video ID")
        date_range = st.text_input("format YYYY-MM-DD/YYYY-MM-DD")
        
    if natural_language_queries_option == "Search Optimization Tips":
        query = st.text_input("Ask about optimization tips")
        
    if natural_language_queries_option == "Effective Hashtags":
        query = st.text_input("Ask about effective hashtags")
        
    if st.button("Natural Language Queries"):
        with st.spinner("Generating data..."):
            try:
                if natural_language_queries_option == "Boost Video Views":
                    result = boost_video_views(query)
                    st.success("Generation complete!")
                    st.write(result)
                elif natural_language_queries_option == "Enhance Click-Through Rate":
                    result = enhance_click_through_rate(query)
                    st.success("Generation complete!")
                    st.write(result)
                elif natural_language_queries_option == "Find Profitable Topics":
                    result = find_profitable_topics(query)
                    st.success("Generation complete!")
                    st.write(result)
                elif natural_language_queries_option == "Optimal Upload Time":
                    result = optimal_upload_time(query)
                    st.success("Generation complete!")
                    st.write(result)
                elif natural_language_queries_option == "Grow Subscribers":
                    result = grow_subscribers(query)
                    st.success("Generation complete!")
                    st.write(result)
                elif natural_language_queries_option == "Niche Success Strategies":
                    result = niche_success_strategies(query)
                    st.success("Generation complete!")
                    st.write(result)
                elif natural_language_queries_option == "Thumbnail Improvement":
                    result = thumbnail_improvement(video_id)
                    st.success("Generation complete!")
                    st.write(result)
                elif natural_language_queries_option == "Diagnose View Drop":
                    result = diagnose_view_drop(video_id, date_range)
                    st.success("Generation complete!")
                    st.write(result)
                elif natural_language_queries_option == "Search Optimization Tips":
                    result = search_optimization_tips(query)
                    st.success("Generation complete!")
                    st.write(result)
                elif natural_language_queries_option == "Effective Hashtags":
                    result = effective_hashtags(query)
                    st.success("Generation complete!")
                    st.write(result)
            except Exception  as e:
                    st.error(f"Generation failed: {str(e)}")
                
elif selected_option == "Advanced Earnings Metrics":
    st.write("### Advanced Earnings Metrics")
    
    # Upload the OAuth JSON file
    uploaded_file = st.sidebar.file_uploader("Upload your OAuth JSON file", type=["json"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = "temp_client_secret.json"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
    
    SCOPES = ["https://www.googleapis.com/auth/yt-analytics.readonly",
              "https://www.googleapis.com/auth/yt-analytics-monetary.readonly"]
    # https://developers.google.com/identity/protocols/oauth2/scopes

    # https://github.com/googleapis/google-api-python-client/blob/main/docs/dyn/index.md
    API_SERVICE_NAME = "youtubeAnalytics"
    API_VERSION = "v2"
    CLIENT_SECRETS_FILE = temp_file_path

    def get_service():
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_local_server()
        return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

    def execute_api_request(client_library_function, **kwargs):
        response = client_library_function(
            **kwargs
        ).execute()
        print(response)
        return response
    
    advanced_earnings_metrics_option = st.selectbox(
        "Advanced Earnings Metrics:",
        [
            "Earnings per Engagement Calculator", 
            "Revenue Efficiency Ratio", 
            "Earnings Quality Metrics", 
            "Revenue Sustainability Score", 
            "Monetization Health Index", 
            "Earnings Diversification Ratio", 
            "Revenue Predictability Analysis", 
            "Earnings Optimization Score", 
            "Revenue Resilience Metrics", 
            "Earnings Consistency Rating"
        ]
    )
    
    channel_id = st.text_input("Enter Channel ID")
    video_id = st.text_input("Enter Video ID")
    start_date = st.text_input("Start Date")
    end_date = st.text_input("End Date")
    
    if st.button("View Advanced Earnings Metrics"):
        with st.spinner("Generating data..."):
            try:
                if advanced_earnings_metrics_option == "Earnings per Engagement Calculator":
                    result = earnings_per_engagement_calculator(channel_id, video_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif advanced_earnings_metrics_option == "Revenue Efficiency Ratio":
                    result = revenue_efficiency_ratio(channel_id, video_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif advanced_earnings_metrics_option == "Earnings Quality Metrics":
                    result = earnings_quality_metrics(channel_id, video_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif advanced_earnings_metrics_option == "Revenue Sustainability Score":
                    result = revenue_sustainability_score(channel_id, video_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif advanced_earnings_metrics_option == "Monetization Health Index":
                    result = monetization_health_index(channel_id, video_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif advanced_earnings_metrics_option == "Earnings Diversification Ratio":
                    result = earnings_diversification_ratio(channel_id, video_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif advanced_earnings_metrics_option == "Revenue Predictability Analysis":
                    result = revenue_predictability_analysis(channel_id, video_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif advanced_earnings_metrics_option == "Earnings Optimization Score":
                    result = earnings_optimization_score(channel_id, video_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif advanced_earnings_metrics_option == "Revenue Resilience Metrics":
                    result = revenue_resilience_metrics(channel_id, video_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif advanced_earnings_metrics_option == "Earnings Consistency Rating":
                    result = earnings_consistency_rating(channel_id, video_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
            except Exception  as e:
                st.error(f"Generation failed: {str(e)}")
                
            
elif selected_option == "Historical Earnings Data":
    st.write("### Historical Earnings Data")
    
    # Upload the OAuth JSON file
    uploaded_file = st.sidebar.file_uploader("Upload your OAuth JSON file", type=["json"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = "temp_client_secret.json"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
    
    SCOPES = ["https://www.googleapis.com/auth/yt-analytics.readonly",
              "https://www.googleapis.com/auth/yt-analytics-monetary.readonly",
              "https://www.googleapis.com/auth/youtube"]
    # https://developers.google.com/identity/protocols/oauth2/scopes

    # https://github.com/googleapis/google-api-python-client/blob/main/docs/dyn/index.md
    API_SERVICE_NAME = "youtubeAnalytics"
    API_VERSION = "v2"
    CLIENT_SECRETS_FILE = temp_file_path

    def get_service():
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_local_server()
        return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

    def execute_api_request(client_library_function, **kwargs):
        response = client_library_function(
            **kwargs
        ).execute()
        print(response)
        return response
    
    historical_earnings_data_option = st.selectbox(
        "Historical Earnings Data:",
        [
            "Historical Earnings Trend", 
            "Lifetime Revenue Analysis", 
            "Earnings Growth Rate", 
            "Revenue Milestones", 
            "Monthly Earnings History", 
            "Historical CPM Trends", 
            "Revenue Pattern Analysis", 
            "Earnings Trajectory", 
            "Revenue Stability Analysis", 
            "Earnings Volatility"
        ]
    )
    
    channel_id = st.text_input("Enter Channel ID")
    video_id = st.text_input("Enter Video ID")
    # Create two columns
    col1, col2 = st.columns(2)

    # Create date input fields in the columns
    start_date = col1.date_input("Start Date")
    end_date = col2.date_input("End Date")
    
    if st.button("View Historical Earnings Data"):
        with st.spinner("Generating data..."):
            try:
                if historical_earnings_data_option == "Historical Earnings Trend":
                    result = historical_earnings_trend(channel_id, video_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif historical_earnings_data_option == "Lifetime Revenue Analysis":
                    result = lifetime_revenue_analysis(channel_id, video_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif historical_earnings_data_option == "Earnings Growth Rate":
                    result = earnings_growth_rate(channel_id, video_id, start_date, end_date)
                    st.write(result)
                elif historical_earnings_data_option == "Revenue Milestones":
                    result = revenue_milestones(channel_id, video_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif historical_earnings_data_option == "Monthly Earnings History":
                    result = monthly_earnings_history(channel_id, video_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif historical_earnings_data_option == "Historical CPM Trends":
                    result = historical_cpm_trends(channel_id, video_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif historical_earnings_data_option == "Revenue Pattern Analysis":
                    result = revenue_pattern_analysis(channel_id, video_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif historical_earnings_data_option == "Earnings Trajectory":
                    result = earnings_trajectory(channel_id, video_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif historical_earnings_data_option == "Revenue Stability Analysis":
                    result = revenue_stability_analysis(channel_id, video_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif historical_earnings_data_option == "Earnings Volatility":
                    result = earnings_volatility(channel_id, video_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
            except Exception  as e:
                st.error(f"Generation failed: {str(e)}")
            
elif selected_option == "Competitive Earnings Analysis":
    st.write("### Competitive Earnings Analysis")
    
    # Upload the OAuth JSON file
    uploaded_file = st.sidebar.file_uploader("Upload your OAuth JSON file", type=["json"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = "temp_client_secret.json"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
    
    SCOPES = ["https://www.googleapis.com/auth/yt-analytics.readonly",
              "https://www.googleapis.com/auth/yt-analytics-monetary.readonly"]
    # https://developers.google.com/identity/protocols/oauth2/scopes

    # https://github.com/googleapis/google-api-python-client/blob/main/docs/dyn/index.md
    API_SERVICE_NAME = "youtubeAnalytics"
    API_VERSION = "v2"
    CLIENT_SECRETS_FILE = temp_file_path

    def get_service():
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_local_server()
        return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

    def execute_api_request(client_library_function, **kwargs):
        response = client_library_function(
            **kwargs
        ).execute()
        print(response)
        return response
    
    competitive_earnings_analysis_option = st.selectbox(
        "Competitive Earnings Analysis:",
        [
            "Channel Earnings Comparison", 
            "Category Ranking", 
            "Competitor Revenue Assessment", 
            "Earnings Gap Identification", 
            "Revenue Competitive Landscape", 
            "Market Share Analysis", 
            "Monetization Strategy Evaluation", 
            "Revenue Positioning Insights", 
            "Competitor Performance Benchmarking", 
            "Earnings Benchmarking Report"
        ]
    )
    
    if competitive_earnings_analysis_option == "Channel Earnings Comparison":
        channel_ids_input = st.text_input("Channel IDs")
        # Create two columns
        col1, col2 = st.columns(2)

        # Create date input fields in the columns
        start_date = col1.date_input("Start Date")
        end_date = col2.date_input("End Date")

        # Convert input string to list of channel IDs
        if channel_ids_input:
            channel_ids = [id.strip() for id in channel_ids_input.split(",")]
        else:
            channel_ids = []
        
    if competitive_earnings_analysis_option == "Category Ranking":
        
        # Input for YouTube API Key
        youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")

        # Set up YouTube API client only if YouTube API key is provided
        if youtube_api_key:
            youtube = build("youtube", "v3", developerKey=youtube_api_key)
        else:
            youtube = None  # Set to None if key is not provided
        
        with st.form("category_ranking_form"):
            category = st.text_input("Enter Category (e.g., 'music', 'gaming', etc.)")
            region = st.selectbox("Select Region", ["US", "UK", "CA", "IN", "Other"])
            if region == "Other":
                region = st.text_input("Enter Custom Region")
                
    if competitive_earnings_analysis_option == "Competitor Revenue Assessment":
        channel_ids_input = st.text_input("Channel IDs")
        # Create two columns
        col1, col2 = st.columns(2)

        # Create date input fields in the columns
        start_date = col1.date_input("Start Date")
        end_date = col2.date_input("End Date")

        # Convert input string to list of channel IDs
        if channel_ids_input:
            channel_ids = [id.strip() for id in channel_ids_input.split(",")]
        else:
            channel_ids = []
            
    if competitive_earnings_analysis_option == "Earnings Gap Identification":
        channel_ids_input = st.text_input("Channel IDs")
        # Create two columns
        col1, col2 = st.columns(2)

        # Create date input fields in the columns
        start_date = col1.date_input("Start Date")
        end_date = col2.date_input("End Date")


        # Convert input string to list of channel IDs
        if channel_ids_input:
            channel_ids = [id.strip() for id in channel_ids_input.split(",")]
        else:
            channel_ids = []
            
    if competitive_earnings_analysis_option == "Revenue Competitive Landscape":
        channel_ids_input = st.text_input("Channel IDs")
        # Create two columns
        col1, col2 = st.columns(2)

        # Create date input fields in the columns
        start_date = col1.date_input("Start Date")
        end_date = col2.date_input("End Date")

        # Convert input string to list of channel IDs
        if channel_ids_input:
            channel_ids = [id.strip() for id in channel_ids_input.split(",")]
        else:
            channel_ids = []
            
    if competitive_earnings_analysis_option == "Market Share Analysis":
        channel_ids_input = st.text_input("Channel IDs")
        # Create two columns
        col1, col2 = st.columns(2)

        # Create date input fields in the columns
        start_date = col1.date_input("Start Date")
        end_date = col2.date_input("End Date")


        # Convert input string to list of channel IDs
        if channel_ids_input:
            channel_ids = [id.strip() for id in channel_ids_input.split(",")]
        else:
            channel_ids = []
            
    if competitive_earnings_analysis_option == "Monetization Strategy Evaluation":
        channel_ids_input = st.text_input("Channel IDs")
        # Create two columns
        col1, col2 = st.columns(2)

        # Create date input fields in the columns
        start_date = col1.date_input("Start Date")
        end_date = col2.date_input("End Date")

        # Convert input string to list of channel IDs
        if channel_ids_input:
            channel_ids = [id.strip() for id in channel_ids_input.split(",")]
        else:
            channel_ids = []
            
    if competitive_earnings_analysis_option == "Revenue Positioning Insights":
        channel_ids_input = st.text_input("Channel IDs")
        # Create two columns
        col1, col2 = st.columns(2)

        # Create date input fields in the columns
        start_date = col1.date_input("Start Date")
        end_date = col2.date_input("End Date")

        # Convert input string to list of channel IDs
        if channel_ids_input:
            channel_ids = [id.strip() for id in channel_ids_input.split(",")]
        else:
            channel_ids = []
            
    if competitive_earnings_analysis_option == "Competitor Performance Benchmarking":
        channel_ids_input = st.text_input("Channel IDs")
        # Create two columns
        col1, col2 = st.columns(2)

        # Create date input fields in the columns
        start_date = col1.date_input("Start Date")
        end_date = col2.date_input("End Date")

        # Convert input string to list of channel IDs
        if channel_ids_input:
            channel_ids = [id.strip() for id in channel_ids_input.split(",")]
        else:
            channel_ids = []
            
    if competitive_earnings_analysis_option == "Earnings Benchmarking Report":
        channel_ids_input = st.text_input("Channel IDs")
        # Create two columns
        col1, col2 = st.columns(2)

        # Create date input fields in the columns
        start_date = col1.date_input("Start Date")
        end_date = col2.date_input("End Date")

        # Convert input string to list of channel IDs
        if channel_ids_input:
            channel_ids = [id.strip() for id in channel_ids_input.split(",")]
        else:
            channel_ids = []
            
    if st.button("Perform Competitive Earnings Analysis"):
        with st.spinner("Performing analysis..."):
            try:
                if competitive_earnings_analysis_option == "Channel Earnings Comparison":
                    result = channel_earnings_comparison(channel_ids, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif competitive_earnings_analysis_option == "Category Ranking":
                    result = category_ranking(category, region)
                    st.success("Analysis complete!")
                    st.write(result)
                elif competitive_earnings_analysis_option == "Competitor Revenue Assessment":
                    st.success("Analysis complete!")
                    result = competitor_revenue_assessment(channel_ids, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif competitive_earnings_analysis_option == "Earnings Gap Identification":
                    result = earnings_gap_identification(channel_ids, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif competitive_earnings_analysis_option == "Revenue Competitive Landscape":
                    result = revenue_competitive_landscape(channel_ids, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif competitive_earnings_analysis_option == "Market Share Analysis":
                    result = market_share_analysis(channel_ids, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif competitive_earnings_analysis_option == "Monetization Strategy Evaluation":
                    result = monetization_strategy_evaluation(channel_ids, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif competitive_earnings_analysis_option == "Revenue Positioning Insights":
                    result = revenue_positioning_insights(channel_ids, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif competitive_earnings_analysis_option == "Competitor Performance Benchmarking":
                    result = competitor_performance_benchmarking(channel_ids, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif competitive_earnings_analysis_option == "Earnings Benchmarking Report":
                    result = earnings_benchmarking_report(channel_ids, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                
elif selected_option == "Revenue Optimization Queries":
    st.write("### Revenue Optimization Queries")
    
    # Upload the OAuth JSON file
    uploaded_file = st.sidebar.file_uploader("Upload your OAuth JSON file", type=["json"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = "temp_client_secret.json"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
    
    SCOPES = ["https://www.googleapis.com/auth/yt-analytics.readonly",
              "https://www.googleapis.com/auth/yt-analytics-monetary.readonly"]
    # https://developers.google.com/identity/protocols/oauth2/scopes

    # https://github.com/googleapis/google-api-python-client/blob/main/docs/dyn/index.md
    API_SERVICE_NAME = "youtubeAnalytics"
    API_VERSION = "v2"
    CLIENT_SECRETS_FILE = temp_file_path

    def get_service():
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_local_server()
        return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

    def execute_api_request(client_library_function, **kwargs):
        response = client_library_function(
            **kwargs
        ).execute()
        print(response)
        return response
    
    revenue_optimization_queries_option = st.selectbox(
        "Revenue Optimization Queries:",
        [
            "Revenue Optimization Tips", 
            "Ad Placement Recommendations", 
            "Optimal Video Length Analysis", 
            "Best Earning Time Slots", 
            "Monetization Strategy Suggestions", 
            "Revenue Growth Opportunities", 
            "Earnings Improvement Areas", 
            "CPM Optimization Tips", 
            "Revenue Leakage Detection", 
            "Untapped Earning Potential Insights"
        ]
    )
    
    if revenue_optimization_queries_option == "CPM Optimization Tips":
        # Input for YouTube API Key
        youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")

        # Set up YouTube API client only if YouTube API key is provided
        if youtube_api_key:
            youtube = build("youtube", "v3", developerKey=youtube_api_key)
        else:
            youtube = None  # Set to None if key is not provided
            
        video_id = st.text_input("Enter Video ID")
        
    if revenue_optimization_queries_option == "Revenue Optimization Tips":
        channel_id = st.text_input("Enter Channel ID")
        col1, col2 = st.columns(2)
        # Create date input fields in the columns
        start_date = col1.date_input("Start Date")
        end_date = col2.date_input("End Date")
        
    if revenue_optimization_queries_option == "Ad Placement Recommendations":
        # Input for YouTube API Key
        youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")

        # Set up YouTube API client only if YouTube API key is provided
        if youtube_api_key:
            youtube = build("youtube", "v3", developerKey=youtube_api_key)
        else:
            youtube = None  # Set to None if key is not provided
            
        video_id = st.text_input("Enter Video ID")
        
    if revenue_optimization_queries_option == "Optimal Video Length Analysis":
        channel_id = st.text_input("Enter Channel ID")
        col1, col2 = st.columns(2)
        # Create date input fields in the columns
        start_date = col1.date_input("Start Date")
        end_date = col2.date_input("End Date")
        
    if revenue_optimization_queries_option == "Best Earning Time Slots":
        channel_id = st.text_input("Enter Channel ID")
        col1, col2 = st.columns(2)
        # Create date input fields in the columns
        start_date = col1.date_input("Start Date")
        end_date = col2.date_input("End Date")
        
    if revenue_optimization_queries_option == "Monetization Strategy Suggestions":
        channel_id = st.text_input("Enter Channel ID")
        col1, col2 = st.columns(2)
        # Create date input fields in the columns
        start_date = col1.date_input("Start Date")
        end_date = col2.date_input("End Date")
        
    if revenue_optimization_queries_option == "Earnings Improvement Areas":
        channel_id = st.text_input("Enter Channel ID")
        col1, col2 = st.columns(2)
        # Create date input fields in the columns
        start_date = col1.date_input("Start Date")
        end_date = col2.date_input("End Date")
        
    if revenue_optimization_queries_option == "Revenue Leakage Detection":
        channel_id = st.text_input("Enter Channel ID")
        col1, col2 = st.columns(2)
        # Create date input fields in the columns
        start_date = col1.date_input("Start Date")
        end_date = col2.date_input("End Date")
        
    if revenue_optimization_queries_option == "Untapped Earning Potential Insights":
        channel_id = st.text_input("Enter Channel ID")
        col1, col2 = st.columns(2)
        # Create date input fields in the columns
        start_date = col1.date_input("Start Date")
        end_date = col2.date_input("End Date")

    if st.button("Revenue Optimization Queries"):
        with st.spinner("Generating result..."):
            try:
                if revenue_optimization_queries_option == "Revenue Optimization Tips":
                    result = revenue_optimization_tips(channel_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif revenue_optimization_queries_option == "Ad Placement Recommendations":
                    result = ad_placement_recommendations(video_id)
                    st.success("Generation complete!")
                    st.write(result)
                elif revenue_optimization_queries_option == "Optimal Video Length Analysis":
                    result = optimal_video_length_analysis(channel_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif revenue_optimization_queries_option == "Best Earning Time Slots":
                    result = best_earning_time_slots(channel_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif revenue_optimization_queries_option == "Monetization Strategy Suggestions":
                    result = monetization_strategy_suggestions(channel_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif revenue_optimization_queries_option == "Revenue Growth Opportunities":
                    result = revenue_growth_opportunities(channel_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif revenue_optimization_queries_option == "Earnings Improvement Areas":
                    result = earnings_improvement_areas(channel_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif revenue_optimization_queries_option == "CPM Optimization Tips":
                    result = cpm_optimization_tips(video_id)
                    st.success("Generation complete!")
                    st.write(result)
                elif revenue_optimization_queries_option == "Revenue Leakage Detection":
                    result = revenue_leakage_detection(channel_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif revenue_optimization_queries_option == "Untapped Earning Potential Insights":
                    result = untapped_earning_potential_insights(channel_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
            except Exception as e:
                st.error(f"Generation failed: {str(e)}") 
                
elif selected_option == "Audience-Based Revenue Analysis":
    st.write("### Audience-Based Revenue Analysis")
    
    # Upload the OAuth JSON file
    uploaded_file = st.sidebar.file_uploader("Upload your OAuth JSON file", type=["json"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = "temp_client_secret.json"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
    
    SCOPES = ["https://www.googleapis.com/auth/yt-analytics.readonly",
              "https://www.googleapis.com/auth/yt-analytics-monetary.readonly"]
    # https://developers.google.com/identity/protocols/oauth2/scopes

    # https://github.com/googleapis/google-api-python-client/blob/main/docs/dyn/index.md
    API_SERVICE_NAME = "youtubeAnalytics"
    API_VERSION = "v2"
    CLIENT_SECRETS_FILE = temp_file_path

    def get_service():
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_local_server()
        return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

    def execute_api_request(client_library_function, **kwargs):
        response = client_library_function(
            **kwargs
        ).execute()
        print(response)
        return response
    
    audience_based_revenue_analysis_option = st.selectbox(
        "Audience-Based Revenue Analysis:",
        [
            "Earnings by Demographic Segment", 
            "Age Group Revenue Benchmark", 
            "Viewer Type Performance Metrics", 
            "Watch Time Revenue Optimizer", 
            "Location-Based Earnings Insights", 
            "Device Type Revenue Comparison", 
            "Traffic Source Earnings Analyzer", 
            "Viewer Engagement Revenue Driver", 
            "Subscriber/Non-Subscriber Revenue Gap", 
            "Member-Only Content Revenue Tracker"
        ]
    )
    
    channel_id = st.text_input("Enter Channel ID")
    
    if st.button("Perform Audience-Based Revenue Analysis"):
        with st.spinner("Performing analysis..."):
            try:
                if audience_based_revenue_analysis_option == "Earnings by Demographic Segment":
                    result = earnings_by_demographic_segment(channel_id)
                    st.success("Analysis complete!")
                    st.write(result)
                elif audience_based_revenue_analysis_option == "Age Group Revenue Benchmark":
                    result = age_group_revenue_benchmark(channel_id)
                    st.success("Analysis complete!")
                    st.write(result)
                elif audience_based_revenue_analysis_option == "Viewer Type Performance Metrics":
                    result = viewer_type_performance_metrics(channel_id)
                    st.success("Analysis complete!")
                    st.write(result)
                elif audience_based_revenue_analysis_option == "Watch Time Revenue Optimizer":
                    result = watch_time_revenue_optimizer(channel_id)
                    st.success("Analysis complete!")
                    st.write(result)
                elif audience_based_revenue_analysis_option == "Location-Based Earnings Insights":
                    result = location_based_earnings_insights(channel_id)
                    st.success("Analysis complete!")
                    st.write(result)
                elif audience_based_revenue_analysis_option == "Device Type Revenue Comparison":
                    result = device_type_revenue_comparison(channel_id)
                    st.success("Analysis complete!")
                    st.write(result)
                elif audience_based_revenue_analysis_option == "Traffic Source Earnings Analyzer":
                    result = traffic_source_earnings_analyzer(channel_id)
                    st.success("Analysis complete!")
                    st.write(result)
                elif audience_based_revenue_analysis_option == "Viewer Engagement Revenue Driver":
                    result = viewer_engagement_revenue_driver(channel_id)
                    st.success("Analysis complete!")
                    st.write(result)
                elif audience_based_revenue_analysis_option == "Subscriber/Non-Subscriber Revenue Gap":
                    result = subscriber_non_subscriber_revenue_gap(channel_id)
                    st.success("Analysis complete!")
                    st.write(result)
            except Exception as e:
                st.error(f"Generation failed: {str(e)}") 
                
elif selected_option == "Content-Specific Earnings":
    st.write("### Content-Specific Earnings")
    
    # Upload the OAuth JSON file
    uploaded_file = st.sidebar.file_uploader("Upload your OAuth JSON file", type=["json"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = "temp_client_secret.json"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
    
    SCOPES = ["https://www.googleapis.com/auth/yt-analytics.readonly",
              "https://www.googleapis.com/auth/yt-analytics-monetary.readonly"]
    # https://developers.google.com/identity/protocols/oauth2/scopes

    # https://github.com/googleapis/google-api-python-client/blob/main/docs/dyn/index.md
    API_SERVICE_NAME = "youtubeAnalytics"
    API_VERSION = "v2"
    CLIENT_SECRETS_FILE = temp_file_path

    def get_service():
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_local_server()
        return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

    def execute_api_request(client_library_function, **kwargs):
        response = client_library_function(
            **kwargs
        ).execute()
        print(response)
        return response
    
    content_specific_earnings_option = st.selectbox(
        "Content-Specific Earnings:",
        [
            "Highest-Earning Video Identifier", 
            "Video Type Revenue Analyzer", 
            "Content Category Performance Metrics", 
            "Shorts/Long-Form Earnings Comparator", 
            "Video Series Revenue Calculator", 
            "Video Length Revenue Optimizer", 
            "Top Revenue-Generating Topic Finder", 
            "Video Format Revenue Benchmark", 
            "Live Stream Revenue Tracker", 
            "Playlist Revenue Share"
        ]
    )
    
    if content_specific_earnings_option == "Shorts/Long-Form Earnings Comparator":
        def get_video_duration(video_id, youtube_api_key):
            youtube = build("youtube", "v3", developerKey=youtube_api_key)
            request = youtube.videos().list(
                part="contentDetails",
                id=video_id
            )
            response = request.execute()
            duration = response["items"][0]["contentDetails"]["duration"]
            return duration

        youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")
        video_id = st.text_input("Enter Video ID")

        if youtube_api_key and video_id:
            duration = get_video_duration(video_id, youtube_api_key)
            st.write(f"Video Duration: {duration}")
        else:
            st.error("Please enter both YouTube API Key and Video ID")
    
    if content_specific_earnings_option == "Video Series Revenue Calculator":        
        def get_video_title(video_id, youtube_api_key):
            # Implement YouTube API video title retrieval
            youtube = build("youtube", "v3", developerKey=youtube_api_key)
            request = youtube.videos().list(
                part="snippet",
                id=video_id
            )
            response = request.execute()
            title = response["items"][0]["snippet"]["title"]
            return title
    
        youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")
        video_id = st.text_input("Enter Video ID")

        if youtube_api_key and video_id:
            duration = get_video_duration(video_id, youtube_api_key)
            st.write(f"Video Duration: {duration}")
        else:
            st.error("Please enter both YouTube API Key and Video ID")
            
    if content_specific_earnings_option == "Video Length Revenue Optimizer":
        youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")
        video_id = st.text_input("Enter Video ID")

        if youtube_api_key and video_id:
            duration = get_video_duration(video_id, youtube_api_key)
            st.write(f"Video Duration: {duration}")
        else:
            st.error("Please enter both YouTube API Key and Video ID")
            
    if content_specific_earnings_option == "Top Revenue-Generating Topic Finder":
        youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")
        video_id = st.text_input("Enter Video ID")

        if youtube_api_key and video_id:
            duration = get_video_duration(video_id, youtube_api_key)
            st.write(f"Video Duration: {duration}")
        else:
            st.error("Please enter both YouTube API Key and Video ID")
            
    if content_specific_earnings_option == "Video Format Revenue Benchmark":
        youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")
        video_id = st.text_input("Enter Video ID")

        if youtube_api_key and video_id:
            duration = get_video_duration(video_id, youtube_api_key)
            st.write(f"Video Duration: {duration}")
        else:
            st.error("Please enter both YouTube API Key and Video ID")
            
    if content_specific_earnings_option == "Live Stream Revenue Tracker":
        youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")
        video_id = st.text_input("Enter Video ID")

        if youtube_api_key and video_id:
            duration = get_video_duration(video_id, youtube_api_key)
            st.write(f"Video Duration: {duration}")
        else:
            st.error("Please enter both YouTube API Key and Video ID")
            
    if content_specific_earnings_option == "Playlist Revenue Share":
        youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")
        video_id = st.text_input("Enter Video ID")

        if youtube_api_key and video_id:
            duration = get_video_duration(video_id, youtube_api_key)
            st.write(f"Video Duration: {duration}")
        else:
            st.error("Please enter both YouTube API Key and Video ID")
        
    channel_id = st.text_input("Enter Channel ID")
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start Date")
    end_date = col2.date_input("End Date")
    
    if st.button("View Earnings"):
        with st.spinner("Processing..."):
            try:
                if content_specific_earnings_option == "Highest-Earning Video Identifier":
                    result = highest_earning_video_identifier(channel_id, start_date, end_date)
                    st.success("Processing complete!")
                    st.write(result)
                elif content_specific_earnings_option == "Video Type Revenue Analyzer":
                    result = video_type_revenue_analyzer(channel_id, start_date, end_date)
                    st.success("Processing complete!")
                    st.write(result)
                elif content_specific_earnings_option == "Content Category Performance Metrics":
                    result = content_category_performance_metrics(channel_id, start_date, end_date)
                    st.success("Processing complete!")
                    st.write(result)
                elif content_specific_earnings_option == "Shorts/Long-Form Earnings Comparator":
                    result = shorts_long_form_earnings_comparator(channel_id, start_date, end_date)
                    st.success("Processing complete!")
                    st.write(result)
                elif content_specific_earnings_option == "Video Series Revenue Calculator":
                    result = video_series_revenue_calculator(channel_id, start_date, end_date)
                    st.success("Processing complete!")
                    st.write(result)
                elif content_specific_earnings_option == "Video Length Revenue Optimizer":
                    result = video_length_revenue_optimizer(channel_id, start_date, end_date)
                    st.success("Processing complete!")
                    st.write(result)
                elif content_specific_earnings_option == "Top Revenue-Generating Topic Finder":
                    result = top_revenue_generating_topic_finder(channel_id, start_date, end_date)
                    st.success("Processing complete!")
                    st.write(result)
                elif content_specific_earnings_option == "Video Format Revenue Benchmark":
                    result = video_format_revenue_benchmark(channel_id, start_date, end_date)
                    st.success("Processing complete!")
                    st.write(result)
                elif content_specific_earnings_option == "Live Stream Revenue Tracker":
                    result = live_stream_revenue_tracker(channel_id, start_date, end_date)
                    st.success("Processing complete!")
                    st.write(result)
                elif content_specific_earnings_option == "Playlist Revenue Share":
                    result = playlist_revenue_share(channel_id, start_date, end_date)
                    st.success("Processing complete!")
                    st.write(result)
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
                
elif selected_option == "Revenue Performance Analysis":
    st.write("### Revenue Performance Analysis")
    
    # Upload the OAuth JSON file
    uploaded_file = st.sidebar.file_uploader("Upload your OAuth JSON file", type=["json"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = "temp_client_secret.json"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
    
    SCOPES = ["https://www.googleapis.com/auth/yt-analytics.readonly",
              "https://www.googleapis.com/auth/yt-analytics-monetary.readonly"]
    # https://developers.google.com/identity/protocols/oauth2/scopes

    # https://github.com/googleapis/google-api-python-client/blob/main/docs/dyn/index.md
    API_SERVICE_NAME = "youtubeAnalytics"
    API_VERSION = "v2"
    CLIENT_SECRETS_FILE = temp_file_path

    def get_service():
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_local_server()
        return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

    def execute_api_request(client_library_function, **kwargs):
        response = client_library_function(
            **kwargs
        ).execute()
        print(response)
        return response
    
    revenue_performance_analysis_option = st.selectbox(
        "Revenue Performance Analysis:",
        [
            "Channel Earnings Comparator", 
            "Niche Revenue Benchmarks", 
            "Performance Metrics Dashboard", 
            "Earnings Efficiency Optimizer", 
            "Revenue Enhancement Opportunities", 
            "Earnings Quality Rating", 
            "Revenue Stream Efficiency Analyzer", 
            "Industry Average Earnings Benchmark", 
            "Content Hour Revenue Insights", 
            "Subscriber-to-Earnings Ratio"
        ]
    )
    
    if revenue_performance_analysis_option == "Channel Earnings Comparator":
        channel_ids_input = st.text_input("Channel IDs")

        # Convert input string to list of channel IDs
        if channel_ids_input:
            channel_ids = [id.strip() for id in channel_ids_input.split(",")]
        else:
            channel_ids = []
            
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date")
        end_date = col2.date_input("End Date")
        
    if revenue_performance_analysis_option == "Niche Revenue Benchmarks":
        niches = [
            "health and wellness",
            "personal finance",
            "gaming",
            "technology",
            "marketing",
            "travel",
            "food",
            "fitness"
        ]

        st.title("Niche Competitor Finder")
        niche = st.selectbox("Select a niche", niches)
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date")
        end_date = col2.date_input("End Date")
        
    if revenue_performance_analysis_option == "Performance Metrics Dashboard":
        channel_id = st.text_input("Enter Channel ID")
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date")
        end_date = col2.date_input("End Date")
        
    if revenue_performance_analysis_option == "Earnings Efficiency Optimizer":
        channel_id = st.text_input("Enter Channel ID")
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date")
        end_date = col2.date_input("End Date")
        
    if revenue_performance_analysis_option == "Revenue Enhancement Opportunities":
        channel_id = st.text_input("Enter Channel ID")
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date")
        end_date = col2.date_input("End Date")
        
    if revenue_performance_analysis_option == "Earnings Quality Rating":
        channel_id = st.text_input("Enter Channel ID")
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date")
        end_date = col2.date_input("End Date")
        
    if revenue_performance_analysis_option == "Revenue Stream Efficiency Analyzer":
        channel_id = st.text_input("Enter Channel ID")
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date")
        end_date = col2.date_input("End Date")
        
    if revenue_performance_analysis_option == "Industry Average Earnings Benchmark":
        niches = [
            "health and wellness",
            "personal finance",
            "gaming",
            "technology",
            "marketing",
            "travel",
            "food",
            "fitness"
        ]

        st.title("Niche Competitor Finder")
        niche = st.selectbox("Select a niche", niches)
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date")
        end_date = col2.date_input("End Date")
        
    if revenue_performance_analysis_option == "Content Hour Revenue Insights":
        youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")
        video_id = st.text_input("Enter Video ID")

        if youtube_api_key and video_id:
            duration = get_video_duration(video_id, youtube_api_key)
            st.write(f"Video Duration: {duration}")
        else:
            st.error("Please enter both YouTube API Key and Video ID")
        channel_id = st.text_input("Enter Channel ID")
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date")
        end_date = col2.date_input("End Date")
        
    if revenue_performance_analysis_option == "Subscriber-to-Earnings Ratio":
        channel_id = st.text_input("Enter Channel ID")
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date")
        end_date = col2.date_input("End Date")
        
    if st.button("Analyze Revenue Performance"):
        with st.spinner("Analyzing revenue performance..."):
            try:
                if revenue_performance_analysis_option == "Channel Earnings Comparator":
                    result = channel_earnings_comparator(channel_ids, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif revenue_performance_analysis_option == "Niche Revenue Benchmarks":
                    result = niche_revenue_benchmarks(niche, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif revenue_performance_analysis_option == "Performance Metrics Dashboard":
                    result = performance_metrics_dashboard(channel_id, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif revenue_performance_analysis_option == "Earnings Efficiency Optimizer":
                    result = earnings_efficiency_optimizer(channel_id, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif revenue_performance_analysis_option == "Revenue Enhancement Opportunities":
                    result = revenue_enhancement_opportunities(channel_id, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif revenue_performance_analysis_option == "Earnings Quality Rating":
                    result = earnings_quality_rating(channel_id, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif revenue_performance_analysis_option == "Revenue Stream Efficiency Analyzer":
                    result = revenue_stream_efficiency_analyzer(channel_id, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif revenue_performance_analysis_option == "Industry Average Earnings Benchmark":
                    result = industry_average_earnings_benchmark(niche, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif revenue_performance_analysis_option == "Content Hour Revenue Insights":
                    result = content_hour_revenue_insights(channel_id, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif revenue_performance_analysis_option == "Subscriber-to-Earnings Ratio":
                    result = subscriber_to_earnings_ratio(channel_id, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
                
elif selected_option == "Monetization Metrics":
    st.write("### Monetization Metrics")
    
    # Upload the OAuth JSON file
    uploaded_file = st.sidebar.file_uploader("Upload your OAuth JSON file", type=["json"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = "temp_client_secret.json"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
    
    SCOPES = ["https://www.googleapis.com/auth/yt-analytics.readonly",
              "https://www.googleapis.com/auth/yt-analytics-monetary.readonly"]
    # https://developers.google.com/identity/protocols/oauth2/scopes

    # https://github.com/googleapis/google-api-python-client/blob/main/docs/dyn/index.md
    API_SERVICE_NAME = "youtubeAnalytics"
    API_VERSION = "v2"
    CLIENT_SECRETS_FILE = temp_file_path

    def get_service():
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_local_server()
        return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

    def execute_api_request(client_library_function, **kwargs):
        response = client_library_function(
            **kwargs
        ).execute()
        print(response)
        return response
    
    monetization_metrics = st.selectbox(
        "Monetization Metrics:",
        [
            "CPM Trend Tracker", 
            "Channel RPM Analyzer", 
            "Monetized Playback Metrics", 
            "Ad Impression Valuation", 
            "View Revenue Calculator", 
            "ECPM Estimator", 
            "Monetization Rate Benchmark", 
            "Ad Density Optimizer", 
            "Skip Rate Earnings Impact", 
            "Subscriber Revenue Insights"
        ]
    )
    
    channel_id = st.text_input("Enter Channel ID")
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start Date")
    end_date = col2.date_input("End Date")
    
    if st.button("View Monetization Metrics"):
        with st.spinner("Generating..."):
            try:
                if monetization_metrics == "CPM Trend Tracker":
                    result = cpm_trend_tracker(channel_id)
                    st.success("Generation complete!")
                    st.write(result)
                elif monetization_metrics == "Channel RPM Analyzer":
                    result = channel_rpm_analyzer(channel_id)
                    st.success("Generation complete!")
                    st.write(result)
                elif monetization_metrics == "Monetized Playback Metrics":
                    result = monetized_playback_metrics(channel_id)
                    st.success("Generation complete!")
                    st.write(result)
                elif monetization_metrics == "Ad Impression Valuation":
                    result = ad_impression_valuation(channel_id)
                    st.success("Generation complete!")
                    st.write(result)
                elif monetization_metrics == "View Revenue Calculator":
                    result = view_revenue_calculator(channel_id)
                    st.success("Generation complete!")
                    st.write(result)
                elif monetization_metrics == "ECPM Estimator":
                    result = ecpm_estimator(channel_id)
                    st.success("Generation complete!")
                    st.write(result)
                elif monetization_metrics == "Monetization Rate Benchmark":
                    result = monetization_rate_benchmark(channel_id)
                    st.success("Generation complete!")
                    st.write(result)
                elif monetization_metrics == "Ad Density Optimizer":
                    result = ad_density_optimizer(channel_id)
                    st.success("Generation complete!")
                    st.write(result)
                elif monetization_metrics == "Skip Rate Earnings Impact":
                    result = skip_rate_earnings_impact(channel_id)
                    st.success("Generation complete!")
                    st.write(result)
                elif monetization_metrics == "Subscriber Revenue Insights":
                    result = subscriber_revenue_insights(channel_id)
                    st.success("Generation complete!")
                    st.write(result)
            except Exception as e:
                st.error(f"Generation failed: {str(e)}")
                
elif selected_option == "Geographic Revenue Analysis":
    st.write("### Geographic Revenue Analysis")
    
    # Upload the OAuth JSON file
    uploaded_file = st.sidebar.file_uploader("Upload your OAuth JSON file", type=["json"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = "temp_client_secret.json"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
    
    SCOPES = ["https://www.googleapis.com/auth/yt-analytics.readonly",
              "https://www.googleapis.com/auth/yt-analytics-monetary.readonly"]
    # https://developers.google.com/identity/protocols/oauth2/scopes

    # https://github.com/googleapis/google-api-python-client/blob/main/docs/dyn/index.md
    API_SERVICE_NAME = "youtubeAnalytics"
    API_VERSION = "v2"
    CLIENT_SECRETS_FILE = temp_file_path

    def get_service():
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_local_server()
        return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

    def execute_api_request(client_library_function, **kwargs):
        response = client_library_function(
            **kwargs
        ).execute()
        print(response)
        return response
    
    geographic_revenue_analysis = st.selectbox(
        "Geographic Revenue Analysis:",
        [
            "Geographic Earnings Insights", 
            "Regional Revenue Rankings", 
            "Country-Level CPM Analysis", 
            "Revenue Share by Region", 
            "Market-Specific Earnings Potential", 
            "Cross-Regional Earnings Comparison", 
            "Top-Paying Markets", 
            "Regional Revenue Composition", 
            "International Earnings Trends", 
            "Global/Local Earnings Balance"
        ]
    )
    
    channel_id = st.text_input("Enter Channel ID")
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start Date")
    end_date = col2.date_input("End Date")
    
    if st.button("Analyze Geographic Revenue"):
        with st.spinner("Analyzing..."):
            try:
                if geographic_revenue_analysis == "Geographic Earnings Insights":
                    result = geographic_earnings_insights(channel_id, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif geographic_revenue_analysis == "Regional Revenue Rankings":
                    result = regional_revenue_rankings(channel_id, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif geographic_revenue_analysis == "Country-Level CPM Analysis":
                    result = country_level_cpm_analysis(channel_id, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif geographic_revenue_analysis == "Revenue Share by Region":
                    result = revenue_share_by_region(channel_id, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif geographic_revenue_analysis == "Market-Specific Earnings Potential":
                    result = market_specific_earnings_potential(channel_id, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif geographic_revenue_analysis == "Cross-Regional Earnings Comparison":
                    result = cross_regional_earnings_comparison(channel_id, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif geographic_revenue_analysis == "Top-Paying Markets":
                    result = top_paying_markets(channel_id, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif geographic_revenue_analysis == "Regional Revenue Composition":
                    result = regional_revenue_composition(channel_id, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif geographic_revenue_analysis == "International Earnings Trends":
                    result = international_earnings_trends(channel_id, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif geographic_revenue_analysis == "Global/Local Earnings Balance":
                    result = global_local_earnings_balance(channel_id, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                
elif selected_option == "Report Generation":
    st.write("### Report Generation")
    
    # Upload the OAuth JSON file
    uploaded_file = st.sidebar.file_uploader("Upload your OAuth JSON file", type=["json"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = "temp_client_secret.json"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
    
    SCOPES = ["https://www.googleapis.com/auth/yt-analytics.readonly",
              "https://www.googleapis.com/auth/yt-analytics-monetary.readonly"]
    # https://developers.google.com/identity/protocols/oauth2/scopes

    # https://github.com/googleapis/google-api-python-client/blob/main/docs/dyn/index.md
    API_SERVICE_NAME = "youtubeAnalytics"
    API_VERSION = "v2"
    CLIENT_SECRETS_FILE = temp_file_path

    def get_service():
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_local_server()
        return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

    def execute_api_request(client_library_function, **kwargs):
        response = client_library_function(
            **kwargs
        ).execute()
        print(response)
        return response
    
    report_generation_option = st.selectbox(
        "Report Generation:",
        [
            "Generate earnings report",
            "Export revenue analysis",
            "Create earnings forecast report",
            "Generate revenue breakdown PDF",
            "Export earnings comparison",
            "Create monetization strategy report",
            "Generate revenue optimization report",
            "Export earnings analytics",
            "Create revenue stream analysis",
            "Generate earnings benchmark report"
        ]
    )
    
    channel_id = st.text_input("Enter Channel ID")
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start Date")
    end_date = col2.date_input("End Date")
    
    
    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            try:
                if report_generation_option == "Generate Earnings Report":
                    result = generate_earnings_report(channel_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif report_generation_option == "Export Revenue Analysis":
                    result = export_revenue_analysis(channel_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif report_generation_option == "Create Earnings Forecast Report":
                    result = create_earnings_forecast_report(channel_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif report_generation_option == "Generate Revenue Breakdown PDF":
                    result = generate_revenue_breakdown_pdf(channel_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif report_generation_option == "Export Earnings Comparison":
                    result = export_earnings_comparison(channel_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif report_generation_option == "Create Monetization Strategy Report":
                    result = create_monetization_strategy_report(channel_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif report_generation_option == "Generate Revenue Optimization Report":
                    result = generate_revenue_optimization_report(channel_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif report_generation_option == "Export Earnings Analytics":
                    result = export_earnings_analytics(channel_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif report_generation_option == "Create Revenue Stream Analysis":
                    result = create_revenue_stream_analysis(channel_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif report_generation_option == "Generate Earnings Benchmark Report":
                    result = generate_earnings_benchmark_report(channel_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
            except Exception as e:
                st.error(f"Generation failed: {str(e)}")
                
elif selected_option == "Historical Data":
    st.write("### Historical Data")
    
    # Upload the OAuth JSON file
    uploaded_file = st.sidebar.file_uploader("Upload your OAuth JSON file", type=["json"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = "temp_client_secret.json"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
    
    SCOPES = ["https://www.googleapis.com/auth/yt-analytics.readonly",
              "https://www.googleapis.com/auth/yt-analytics-monetary.readonly"]
    # https://developers.google.com/identity/protocols/oauth2/scopes

    # https://github.com/googleapis/google-api-python-client/blob/main/docs/dyn/index.md
    API_SERVICE_NAME = "youtubeAnalytics"
    API_VERSION = "v2"
    CLIENT_SECRETS_FILE = temp_file_path

    def get_service():
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_local_server()
        return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

    def execute_api_request(client_library_function, **kwargs):
        response = client_library_function(
            **kwargs
        ).execute()
        print(response)
        return response
    
    historical_data_option = st.selectbox(
        "Historical Data:",
        [
            "video Performance History", 
            "Keyword Trend Analysis", 
            "Year-over-Year Growth Comparison", 
            "Seasonal Trend Identification", 
            "Engagement History Analysis", 
            "Viewership Trend Tracking", 
            "Subscriber Growth History", 
            "Monthly Performance Comparison"
        ]
    )
    
    channel_id = st.text_input("Enter Channel ID")
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start Date")
    end_date = col2.date_input("End Date")
    
    if st.button("View Historical Data"):
        with st.spinner("Generating historical data..."):
            try:
                if historical_data_option == "Video Performance History":
                    result = video_performance_history(channel_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif historical_data_option == "Keyword Trend Analysis":
                    result = keyword_trend_analysis(channel_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif historical_data_option == "Year-over-Year Growth Comparison":
                    result = year_over_year_growth_comparison(channel_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif historical_data_option == "Seasonal Trend Identification":
                    result = seasonal_trend_identification(channel_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif historical_data_option == "Engagement History Analysis":
                    result = engagement_history_analysis(channel_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif historical_data_option == "Viewership Trend Tracking":
                    result = viewership_trend_tracking(channel_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif historical_data_option == "Subscriber Growth History":
                    result = subscriber_growth_history(channel_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
                elif historical_data_option == "Monthly Performance Comparison":
                    result = monthly_performance_comparison(channel_id, start_date, end_date)
                    st.success("Generation complete!")
                    st.write(result)
            except Exception as e:
                st.error(f"Generation failed: {str(e)}")
                
elif selected_option == "Competition Analysis":
    st.write("### Competition Analysis")
    
    # Upload the OAuth JSON file
    uploaded_file = st.sidebar.file_uploader("Upload your OAuth JSON file", type=["json"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = "temp_client_secret.json"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
    
    SCOPES = ["https://www.googleapis.com/auth/yt-analytics.readonly",
              "https://www.googleapis.com/auth/yt-analytics-monetary.readonly"]
    # https://developers.google.com/identity/protocols/oauth2/scopes

    # https://github.com/googleapis/google-api-python-client/blob/main/docs/dyn/index.md
    API_SERVICE_NAME = "youtubeAnalytics"
    API_VERSION = "v2"
    CLIENT_SECRETS_FILE = temp_file_path

    def get_service():
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_local_server()
        return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

    def execute_api_request(client_library_function, **kwargs):
        response = client_library_function(
            **kwargs
        ).execute()
        print(response)
        return response
    
    competition_analysis_option = st.selectbox(
        "Competition Analysis:",
        [
            "Niche Competitor Analysis", 
            "Channel Comparison", 
            "Competitor Video Strategy", 
            "Tag Analysis", 
            "Title Comparison", 
            "Thumbnail Analysis", 
            "Upload Pattern Insights", 
            "Engagement Metric Analysis"
        ]
    )
    
    if competition_analysis_option == "Niche Competitor Analysis":
        youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")

        # Set up YouTube API client only if YouTube API key is provided
        if youtube_api_key:
            youtube = build("youtube", "v3", developerKey=youtube_api_key)
        else:
            youtube = None  # Set to None if key is not provided

        
        channel_ids_input = st.text_input("Enter Channel IDs (comma-separated)", "")

        # Convert input string to list of channel IDs
        if channel_ids_input:
            channel_ids = [id.strip() for id in channel_ids_input.split(",")]
        else:
            channel_ids = []
            
    if competition_analysis_option == "Channel Comparison":
        channel_id = st.text_input("Enter Channel ID")
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date")
        end_date = col2.date_inpur("End Date")
        
    if competition_analysis_option == "Competitor Video Strategy":
        youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")

        # Set up YouTube API client only if YouTube API key is provided
        if youtube_api_key:
            youtube = build("youtube", "v3", developerKey=youtube_api_key)
        else:
            youtube = None  # Set to None if key is not provided

        channel_id = st.text_input("Enter Channel ID")
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date")
        end_date = col2.date_inpur("End Date")
        
    if competition_analysis_option == "Tag Analysis":
        youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")

        # Set up YouTube API client only if YouTube API key is provided
        if youtube_api_key:
            youtube = build("youtube", "v3", developerKey=youtube_api_key)
        else:
            youtube = None  # Set to None if key is not provided

        channel_id = st.text_input("Enter Channel ID")
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date")
        end_date = col2.date_inpur("End Date")
        
    if competition_analysis_option == "Title Comparison":
        youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")

        # Set up YouTube API client only if YouTube API key is provided
        if youtube_api_key:
            youtube = build("youtube", "v3", developerKey=youtube_api_key)
        else:
            youtube = None  # Set to None if key is not provided

        channel_id = st.text_input("Enter Channel ID")
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date")
        end_date = col2.date_inpur("End Date")
        
    if competition_analysis_option == "Thumbnail Analysis":
        youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")

        # Set up YouTube API client only if YouTube API key is provided
        if youtube_api_key:
            youtube = build("youtube", "v3", developerKey=youtube_api_key)
        else:
            youtube = None  # Set to None if key is not provided

        channel_id = st.text_input("Enter Channel ID")
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date")
        end_date = col2.date_inpur("End Date")
        
    if competition_analysis_option == "Upload Pattern Insights":
        youtube_api_key = st.sidebar.text_input("Enter your YouTube API Key:", type="password")

        # Set up YouTube API client only if YouTube API key is provided
        if youtube_api_key:
            youtube = build("youtube", "v3", developerKey=youtube_api_key)
        else:
            youtube = None  # Set to None if key is not provided

        channel_id = st.text_input("Enter Channel ID")
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date")
        end_date = col2.date_inpur("End Date")
        
    if competition_analysis_option == "Engagement Metric Analysis":
        channel_id = st.text_input("Enter Channel ID")
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date")
        end_date = col2.date_inpur("End Date")
    
    if st.button("Analyze Competitors"):
        with st.spinner("Analyzing..."):
            try:
                if competition_analysis_option == "Niche Competitor Analysis":
                    result = niche_competitor_analysis(channel_ids)
                    st.success("Analysis complete!")
                    st.write(result)
                elif competition_analysis_option == "Channel Comparison":
                    result = channel_comparison(channel_id, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif competition_analysis_option == "Competitor Video Strategy":
                    result = competitor_video_strategy(channel_id, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif competition_analysis_option == "Tag Analysis":
                    result = tag_analysis(channel_id, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif competition_analysis_option == "Title Comparison":
                    result = title_comparison(channel_id, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif competition_analysis_option == "Thumbnail Analysis":
                    result = thumbnail_analysis(channel_id, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result) 
                elif competition_analysis_option == "Upload Pattern Insights":
                    result = upload_pattern_insights(channel_id, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
                elif competition_analysis_option == "Engagement Metric Analysis":
                    result = engagement_metric_analysis(channel_id, start_date, end_date)
                    st.success("Analysis complete!")
                    st.write(result)
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")