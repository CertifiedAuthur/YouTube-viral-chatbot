import streamlit as st
import requests
import pytz
import time
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from collections import Counter
from serpapi import GoogleSearch
import pytrends
from googleapiclient.discovery import build
from openai import OpenAI
from streamlit_js import st_js, st_js_blocking
from google_auth_oauthlib.flow import Flow
import matplotlib.pyplot as plt
from newsapi import NewsApiClient
import plotly.express as px
from pytrends.request import TrendReq
from pytrends.exceptions import ResponseError
from datetime import datetime
from dateutil import parser
import pandas as pd
import numpy as np
import json
import base64
import re


submitted = False  # Define it globally or within a function

def on_submit():
    global submitted
    submitted = True  # Set to True when form is submitted

if submitted:  # Ensure 'submitted' is accessible
    print("Form was submitted")




# Helper function to convert large numbers to thousands, millions, etc.
def format_number(number):
    if number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.1f}B"
    elif number >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"
    elif number >= 1_000:
        return f"{number / 1_000:.1f}K"
    return str(number)

def get_channel_id(Channel_url):
    # Fetch the page content
    response = requests.get(Channel_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find channel ID in the page metadata
    channel_link = soup.find("link", {"rel": "canonical"})
    if channel_link:
        canonical_url = channel_link["href"]
        channel_id = canonical_url.split("/")[-1]
        return channel_id
    else:
        return None

def extract_channel_id(video_url):
    video_id = extract_video_id(video_url)
    youtube = get_service()
    response = execute_api_request(
        youtube.videos().list,
        part="snippet",
        id=video_id
    )
    
    if 'items' in response:
        video_snippet = response['items'][0]['snippet']
        channel_id = video_snippet['channelId']
        return channel_id
    else:
        return None
    
def extract_video_id(video_url):
    # Split the URL by slashes and take the last part
  try:  
    video_id = video_url.split('/')[-1]
    
    # If the video ID contains a query string (e.g., ?si=...)
    if '?' in video_id:
        video_id = video_id.split('?')[0]
    
    # return video_id
        print(f"Extracted Video ID: {video_id}")
  except Exception as e:
      print(f"Error: {str(e)}")

# BASIC VIDEO ANALYSIS
def get_video_metrics(video_url):
    try:
        video_id = extract_video_id(video_url)
        youtube = get_service()
        response = execute_api_request(
            youtube.videos().list,
            part="statistics",
            id=video_id
        )
        return response
    except Exception as e:
        print(f"Error fetching video metrics: {e}")


    if 'items' in response and response['items']:
        stats = response['items'][0]['statistics']

        # Create DataFrame
        data = {
        "Metric": ["View Count", "Like Count", "Comment Count"],
        "Value": [int(stats.get('viewCount', 0)), int(stats.get('likeCount', 0)), int(stats.get('commentCount', 0))]
        }
        df = pd.DataFrame(data)

        # Plot
        st.write("### Video Metrics")
        st.bar_chart(df.set_index("Metric")["Value"])

        return df
    else:
        st.write(f"No data found for video {video_id}")
        return None
    
def get_video_comments(video_url, max_results=10):
    video_id = extract_video_id(video_url)
    youtube = get_service()
    response = execute_api_request(
        youtube.commentThreads().list,
        part="snippet",
        videoId=video_id,
        maxResults=max_results
    )
    
    if 'items' in response:
        comments = [{
            "Comment": item['snippet']['topLevelComment']['snippet']['textDisplay'],
            "Author": item['snippet']['topLevelComment']['snippet']['authorDisplayName'],
            "Published At": item['snippet']['topLevelComment']['snippet']['publishedAt']
        } for item in response['items']]
        
        df = pd.DataFrame(comments)
        return df
    else:
        st.write(f"No comments found for video {video_id}")
        return None


def get_video_details(video_url):
    video_id = extract_video_id(video_url)
    youtube = get_service()
    response = execute_api_request(
        youtube.videos().list,
        part="snippet,statistics,contentDetails",
        id=video_id
    )
    
    if 'items' in response:
        details = response['items'][0]
        snippet = details['snippet']
        statistics = details['statistics']
        content_details = details['contentDetails']
        
        # Parse publishedAt date
        published_at = datetime.strptime(snippet.get('publishedAt'), '%Y-%m-%dT%H:%M:%SZ')
        published_at_formatted = published_at.strftime('%B %d, %Y at %I:%M %p')
        
        # Format duration
        duration = content_details.get('duration')
        duration = duration.replace("PT", "")
        hours = re.search(r"(\d+)H", duration)
        minutes = re.search(r"(\d+)M", duration)
        seconds = re.search(r"(\d+)S", duration)
        
        formatted_duration = ""
        if hours:
            formatted_duration += f"{hours.group(1)} hours "
        if minutes:
            formatted_duration += f"{minutes.group(1)} minutes "
        if seconds:
            formatted_duration += f"{seconds.group(1)} seconds"
        
        data = {
            "Metric": [
                "Title",
                "Description",
                "Published At",
                "Duration",
                "View Count",
                "Like Count",
                "Dislike Count",
                "Comment Count",
                "Category"
            ],
            "Value": [
                snippet.get('title'),
                snippet.get('description'),
                published_at_formatted,
                formatted_duration,
                statistics.get('viewCount'),
                statistics.get('likeCount'),
                statistics.get('dislikeCount'),
                statistics.get('commentCount'),
                snippet.get('categoryId')
            ]
        }
        
        df = pd.DataFrame(data)
        return df
    else:
        st.write(f"No data found for video {video_id}")
    
def show_video_engagement_metrics(video_url):
    video_id = extract_video_id(video_url)
    
    # Use YouTube API to retrieve video statistics
    youtube = get_service()
    request = youtube.videos().list(
        part="statistics",
        id=video_id
    )
    response = request.execute()
    
    # Extract statistics
    statistics = response['items'][0]['statistics']
    
    # Calculate engagement rate
    if 'viewCount' not in statistics or 'likeCount' not in statistics:
        return pd.DataFrame({"error": ["Insufficient data for engagement calculation"]})
    
    engagement_rate = (
        (int(statistics["likeCount"]) + int(statistics.get("commentCount", 0)))
        / int(statistics["viewCount"]) * 100
    )
    
    # Create DataFrame
    data = {
        "Video URL": [video_url],
        "Engagement Rate": [f"{engagement_rate:.2f}%"],
        "View Count": [statistics["viewCount"]],
        "Like Count": [statistics["likeCount"]],
        "Comment Count": [statistics.get("commentCount", 0)]
    }
    
    return pd.DataFrame(data)

def compare_metrics_between_videos(video_url_1, video_url_2):
    video_1_id = extract_video_id(video_url_1)
    video_2_id = extract_video_id(video_url_2)
    
    video_1_stats = get_video_details(video_url_1)
    video_2_stats = get_video_details(video_url_2)
    
    # Access values from DataFrames
    view_count_1 = video_1_stats.loc[video_1_stats['Metric'] == 'View Count', 'Value'].values[0]
    view_count_2 = video_2_stats.loc[video_2_stats['Metric'] == 'View Count', 'Value'].values[0]
    
    like_count_1 = video_1_stats.loc[video_1_stats['Metric'] == 'Like Count', 'Value'].values[0]
    like_count_2 = video_2_stats.loc[video_2_stats['Metric'] == 'Like Count', 'Value'].values[0]
    
    comment_count_1 = video_1_stats.loc[video_1_stats['Metric'] == 'Comment Count', 'Value'].values[0]
    comment_count_2 = video_2_stats.loc[video_2_stats['Metric'] == 'Comment Count', 'Value'].values[0]
    
    # Create DataFrame for plotting
    comparison_df = pd.DataFrame({
        "Metric": ["Views", "Likes", "Comments"],
        video_url_1: [int(view_count_1), int(like_count_1), int(comment_count_1)],
        video_url_2: [int(view_count_2), int(like_count_2), int(comment_count_2)],
    })
    
    # Plot side-by-side bars
    fig = px.bar(comparison_df, x="Metric", y=[video_url_1, video_url_2], barmode="group", title="Video Comparison")
    
    # Update labels
    fig.update_layout(
        xaxis_title="Metric",
        yaxis_title="Count",
        legend_title="Video"
    )
    
    st.plotly_chart(fig)
def extract_video_keywords(video_url):
    video_id = extract_video_id(video_url)
    # Extract video keywords based on tags and description
    youtube = get_service()
    
    # Get video metadata
    metadata_response = execute_api_request(
        youtube.videos().list,
        part="snippet",
        id=video_id
    )
    
    if 'items' in metadata_response:
        metadata = metadata_response['items'][0]['snippet']
        
        # Get video tags
        tags_response = execute_api_request(
            youtube.videos().list,
            part="snippet",
            id=video_id
        )
        
        tags = tags_response['items'][0]['snippet'].get('tags', [])
        
        # Combine tags and description keywords
        keywords = tags + metadata.get('description', '').split()
        
        # Remove duplicates and return
        return {"keywords": list(set(keywords))}
    else:
        return {"error": "Video metadata not found"}

# CHANNEL ANALYSIS
def get_channel_analytics(channel_url):
    # Get channel ID
    channel_id = get_channel_id(channel_url)
    if not channel_id:
        return None
    
    youtube = get_service()
    
    # Get channel info
    request = youtube.channels().list(
        part="snippet,statistics,brandingSettings",
        id=channel_id
    )
    response = request.execute()
    
    if not response["items"]:
        st.write(f"No data found for channel {channel_id}")
        return None
    
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

    # Format published_at date
    published_at = parser.isoparse(channel_info['published_at'])
    channel_info['published_at'] = published_at.strftime('%B %d, %Y at %I:%M %p')
    
    # Create DataFrame for analytics
    stats = channel["statistics"]
    data = {
        "Metric": ["Subscriber Count", "Total Views", "Total Videos"],
        "Value": [int(stats.get('subscriberCount', 0)), int(stats.get('viewCount', 0)), int(stats.get('videoCount', 0))]
    }
    df = pd.DataFrame(data)
    
    # Display channel info
    st.write(f"### {channel_info['title']} Channel Analytics")
    st.write(f"**Channel Description:** {channel_info['description']}")
    st.write(f"**Country:** {channel_info['country']}")
    st.write(f"**Published At:** {channel_info['published_at']}")
    
    # Plot chart
    st.write("### Channel Metrics")
    st.bar_chart(df.set_index("Metric")["Value"])

def show_top_performing_videos(channel_url, max_results=20):
    channel_id = get_channel_id(channel_url)
    if not channel_id:
        st.error("Invalid channel URL.")
        return None

    youtube = get_service()

    def video_metrics(video_id):
        try:
            request = youtube.videos().list(
                part="statistics",
                id=video_id
            )
            response = request.execute()
            metrics = response["items"][0]["statistics"]
            return {
                "view_count": int(metrics.get("viewCount", 0)),
                "like_count": int(metrics.get("likeCount", 0)),
                "comment_count": int(metrics.get("commentCount", 0))
            }
        except Exception as e:
            st.error(f"Error fetching video metrics: {e}")
            return {
                "view_count": 0,
                "like_count": 0,
                "comment_count": 0
            }

    request = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        maxResults=max_results
    )
    response = request.execute()

    video_items = [item for item in response["items"] if item["id"]["kind"] == "youtube#video"]

    top_videos = [
        {
            "title": item["snippet"]["title"],
            "video_id": item["id"]["videoId"],
            "description": item["snippet"]["description"],
            "published_at": item["snippet"]["publishedAt"],
            **video_metrics(item["id"]["videoId"])
        }
        for item in video_items
    ]

    for video in top_videos:
        published_at = datetime.strptime(video['published_at'], '%Y-%m-%dT%H:%M:%SZ')
        video['published_at'] = published_at.strftime('%B %d, %Y at %I:%M %p')

    df = pd.DataFrame(top_videos).sort_values(by='view_count', ascending=False)

    st.write(f"### Top Performing Videos on {channel_url}")
    st.bar_chart(df[["view_count", "like_count", "comment_count"]])
    st.write(df)

def compare_channels(channel_url_1, channel_url_2):
    # Retrieve channel analytics
    channel_1_info = get_channel_analytics(channel_url_1)
    channel_2_info = get_channel_analytics(channel_url_2)
    
    # Check for errors in retrieving channel data
    if channel_1_info is None or channel_2_info is None:
        return {"error": "Failed to retrieve channel data"}
    
    if "error" in channel_1_info or "error" in channel_2_info:
        return {"error": "One or both channels not found"}

    # Organize data for comparison
    metrics = ["View Count", "Subscriber Count", "Video Count"]
    
    # Check for missing values
    channel_1_data = [
        channel_1_info.get("view_count", 0),
        channel_1_info.get("subscriber_count", 0),
        channel_1_info.get("video_count", 0)
    ]
    channel_2_data = [
        channel_2_info.get("view_count", 0),
        channel_2_info.get("subscriber_count", 0),
        channel_2_info.get("video_count", 0)
    ]
    
    # Create DataFrames
    df_1 = pd.DataFrame({
        "Metric": metrics,
        "Value": channel_1_data
    })
    df_2 = pd.DataFrame({
        "Metric": metrics,
        "Value": channel_2_data
    })
    
    # Define positions for the bars
    x = np.arange(len(metrics))
    width = 0.35  # Width of the bars

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, channel_1_data, width, label=f'Channel A ({channel_url_1})', color='red')
    ax.bar(x + width/2, channel_2_data, width, label=f'Channel B ({channel_url_2})', color='blue')

    # Add labels and title
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Count")
    ax.set_title("Comparison of Channel Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Display the plot
    plt.tight_layout()
    st.pyplot(plt)
    # Display DataFrames
    st.subheader(f"Channel A ({channel_url_1}) Analytics")
    st.dataframe(df_1)
    
    st.subheader(f"Channel B ({channel_url_2}) Analytics")
    st.dataframe(df_2)

def show_channel_engagement_trends(channel_url):
    # Extract channel ID from URL
    channel_id = get_channel_id(channel_url)
    if not channel_id:
        return pd.DataFrame({"Error": ["Invalid channel URL"]})

    # Get uploaded videos
    youtube = get_service()
    request = youtube.search().list(
        part="id,snippet",
        channelId=channel_id,
        order="viewCount",
        maxResults=10
    )
    response = request.execute()

    # Define function to get video performance metrics
    def get_video_performance_metrics(video_id):
        youtube = get_service()
        request = youtube.videos().list(
            part="statistics",
            id=video_id
        )
        response = request.execute()

        if not response["items"]:
            return {"error": "Video not found"}

        video_stats = response["items"][0]["statistics"]
        metrics = {
            "view_count": video_stats.get("viewCount"),
            "like_count": video_stats.get("likeCount"),
            "dislike_count": video_stats.get("dislikeCount"),
            "favorite_count": video_stats.get("favoriteCount"),
            "comment_count": video_stats.get("commentCount")
        }

        return metrics


    # Get video metrics for top 10 videos
    video_metrics = []
    for item in response["items"]:
        if item["id"]["kind"] == "youtube#video":
            video_id = item["id"]["videoId"]
            metrics = get_video_performance_metrics(video_id)
            video_metrics.append({
                "Video Title": item["snippet"]["title"],
                "Video ID": video_id,
                "View Count": int(metrics["view_count"] or 0),
                "Like Count": int(metrics["like_count"] or 0),
                "Dislike Count": int(metrics["dislike_count"] or 0),
                "Favorite Count": int(metrics["favorite_count"] or 0),
                "Comment Count": int(metrics["comment_count"] or 0)
            })

    # Create DataFrame
    df = pd.DataFrame(video_metrics)

    return df

def analyze_upload_schedule(channel_url):
    # Extract channel ID from URL
    channel_id = get_channel_id(channel_url)
    if not channel_id:
        return {"error": "Invalid channel URL"}

    youtube = get_service()
    request = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        order="date",
        maxResults=50
    )
    response = request.execute()

    # Extract upload dates
    upload_dates = [item["snippet"]["publishedAt"] for item in response["items"] if item["id"]["kind"] == "youtube#video"]
    upload_dates = pd.to_datetime(upload_dates)
    
    # Extract day names
    import calendar
    upload_days = upload_dates.map(lambda x: calendar.day_name[x.weekday()])
    
    # Calculate day distribution
    day_distribution = upload_days.value_counts().to_frame("Count")
    day_distribution.index.name = "Day"
    
    # Plot line chart
    st.write("### Upload Schedule")
    day_distribution["Count"] = day_distribution["Count"].astype(int)
    chart_data = day_distribution.reset_index()
    chart_data = chart_data.sort_values(by='Day', key=lambda x: pd.Categorical(x, categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']))
    st.line_chart(chart_data.set_index("Day"))

    # Return day distribution as DataFrame
    return day_distribution

def get_subscriber_growth_rate(channel_url):
    # Extract channel ID from URL
    channel_id = get_channel_id(channel_url)
    if not channel_id:
        return {"error": "Invalid channel URL"}

    # Fetch channel details
    youtube = get_service()
    channel_response = youtube.channels().list(
        part="snippet,statistics",
        id=channel_id
    ).execute()
    
    if not channel_response['items']:
        return {"error": "Channel not found."}

    # Extract relevant data
    channel_info = channel_response['items'][0]
    published_at = pd.to_datetime(channel_info['snippet']['publishedAt'])  # Channel creation date
    subscriber_count_start = 0  # Start with 0 subscribers

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

    # Create DataFrame
    data = {
        "Metric": ["Subscriber Count Start", "Subscriber Count Now", "Days Active", "Growth Rate (%)"],
        "Value": [subscriber_count_start, subscriber_count_now, days_active, growth_rate]
    }
    df = pd.DataFrame(data)

    # Display results
    st.write(f"### {channel_info['snippet']['title']} Subscriber Growth Rate")
    st.bar_chart(df.set_index("Metric")["Value"])
    
    return df

# YOUTUBE SEARCH 
def search_youtube(query, max_results=5):
    youtube = get_service()
    response = execute_api_request(
        youtube.search().list,
        part="snippet",
        q=query,
        type="video,channel,playlist",
        maxResults=max_results
    )
    
    if 'items' in response:
        data = [{
            "Title": item['snippet']['title'],
            "Type": item['id']['kind'].split('#')[-1],  # Extract video, channel, or playlist
            "Description": item['snippet']['description']
        } for item in response['items']]
        
        df = pd.DataFrame(data)
        return df
    else:
        st.write(f"No results found for query '{query}'")
        return None
    
# CONTENT STRATEGY
# Function to suggest video ideas
def suggest_video_categories(related_topic):
    prompt = f"Suggest YouTube video categories related to {related_topic}:"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    # Extract categories from response
    categories = response.choices[0].message.content
    return categories

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
    title_ideas = response.choices[0].message.content
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
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    recommended_time = response.choices[0].message.content
    return recommended_time

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
        content_calendar = response.choices[0].message.content
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
pytrends = TrendReq()

def get_trending_topics(pn, max_results=20):
    trending_searches_df = pytrends.trending_searches(pn=pn)
    trending_topics_df = trending_searches_df.head(max_results).reset_index()
    trending_topics_df.columns = ['Rank', 'Trending Topic']
    return trending_topics_df

def get_trending_hashtags(region_code="US", max_results=20):
    trending_videos_df = get_viral_videos(region_code, max_results)
    
    # Check if DataFrame is not empty
    if trending_videos_df.empty:
        st.error("No trending videos found")
        return None
    
    # Extract hashtags from 'Tags' column
    hashtags = trending_videos_df['Tags'].str.split(',').explode()
    hashtags = hashtags.str.strip()  # Remove leading/trailing whitespace
    
    # Remove empty strings and duplicates
    unique_hashtags = hashtags[hashtags.ne('')].unique()
    
    # Create DataFrame with trending hashtags
    trending_hashtags_df = pd.DataFrame(unique_hashtags[:max_results], columns=['Trending Hashtag'])
    
    return trending_hashtags_df

def get_viral_videos(region_code: str = "US", max_results: int = 20) -> pd.DataFrame:
    youtube = get_service()
    request = youtube.videos().list(
        part="snippet,statistics",
        chart="mostPopular",
        regionCode=region_code,
        maxResults=max_results
    )
    response = request.execute()
    
    trending_videos = [
        {
            "Title": item["snippet"]["title"],
            "Video ID": item["id"],
            "Description": item["snippet"]["description"],
            "Tags": ", ".join(item.get("snippet", {}).get("tags", [])),
            "View Count": item["statistics"].get("viewCount", 0),
            "Like Count": item["statistics"].get("likeCount", 0),
        }
        for item in response["items"]
    ]
    
    trending_videos_df = pd.DataFrame(trending_videos)
    
    return trending_videos_df

def extract_categories_info(categories_dict):
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
    
    # Convert to DataFrame and sort in descending order
    df = pd.DataFrame(categories)
    df = df.sort_values(by=['Category'], ascending=False).reset_index(drop=True)
    
    return df

def show_rising_trends(keyword, pn):
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


def get_search_volume(timeframe, kw_list, location):
    try:
        # Split kw_list into individual keywords
        keywords = [kw.strip() for kw in kw_list]
        
        pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo=location, gprop="")
        search_volume = pytrends.interest_over_time()
        
        if search_volume.empty:
            return pd.DataFrame()  # Return empty DataFrame
        
        # Convert to DataFrame and reset index
        df = search_volume[keywords].reset_index()
        
        # Convert 'date' column to datetime format
        df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

def show_related_keywords(kw_list, location):
    try:
        # Split kw_list into individual keywords
        keyword = kw_list[0].strip()
        
        # Build payload for Google Trends
        pytrends.build_payload([keyword], cat=0, timeframe="now 7-d", geo=location, gprop="")
        
        # Get rising related queries
        related_keywords = pytrends.suggestions(keyword=keyword)
        
        # Return related queries as DataFrame
        df = pd.DataFrame(related_keywords)
        
        return df
    
    except ResponseError as e:
        if e.response.status_code == 429:  # If rate limit error
            st.write("Rate limit exceeded, retrying in 60 seconds...")
            time.sleep(60)  # Wait 60 seconds before retrying
            return show_related_keywords(kw_list, location)  # Retry the request
        
        return pd.DataFrame({"error": [str(e)]})


def analyze_keyword_competition(kw_list, location):
    try:
        # Split kw_list into individual keywords
        keyword = kw_list[0].strip()
        
        # Build payload for Google Trends
        pytrends.build_payload([keyword], cat=0, timeframe="now 7-d", geo=location, gprop="")
        
        # Get interest by region
        interest_by_region = pytrends.interest_by_region(
            resolution='COUNTRY', 
            inc_low_vol=True, 
            inc_geo_code=False
        )
        
        # Check if interest_by_region is empty
        if interest_by_region.empty:
            return pd.DataFrame()  # Return empty DataFrame
        
        # Convert to DataFrame 
        df = interest_by_region
        
        return df

    except ResponseError as e:
        if e.response.status_code == 429:  # If rate limit error
            st.write("Rate limit exceeded, retrying in 60 seconds...")
            time.sleep(60)  # Wait 60 seconds before retrying
            return analyze_keyword_competition(kw_list, location)  # Retry the request
        
        return pd.DataFrame({"error": [str(e)]})

def show_keyword_trends(timeframe, kw_list, location):
    try:
        # Extract keyword and build Google Trends payload
        keyword = kw_list[0].strip()
        pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo=location, gprop="")
        
        # Retrieve interest over time data
        trends = pytrends.interest_over_time()
        
        # Handle empty trends data
        if trends.empty:
            return pd.DataFrame({"error": [f"No trends found for '{keyword}'"]})
        
        # Prepare trends data for plotting
        trends.reset_index(inplace=True)
        trends['date'] = pd.to_datetime(trends['date'])
        
        # Plot trends
        plt.figure(figsize=(10, 6))
        plt.plot(trends['date'], trends[keyword])
        plt.title(f"{keyword} Trends over Time")
        plt.xlabel("Date")
        plt.ylabel("Interest")
        plt.grid(True)
        plt.tight_layout()
        
        # Display plot using Streamlit
        st.pyplot(plt)
        
        return trends
    
    except ResponseError as e:
        if e.response.status_code == 429:  # If rate limit error
            st.write("Rate limit exceeded, retrying in 60 seconds...")
            time.sleep(60)  # Wait 60 seconds before retrying
            return show_keyword_trends(timeframe, kw_list, location)  # Retry the request
        
        return pd.DataFrame({"error": [str(e)]})

def compare_keywords(keyword1, keyword2, location, timeframe="now 7-d"):
    try:
        # Build payload for Google Trends comparison
        pytrends.build_payload([keyword1, keyword2], cat=0, timeframe=timeframe, geo=location, gprop="")
        
        # Retrieve interest over time data
        comparison_data = pytrends.interest_over_time()
        
        # Check if comparison data is empty
        if comparison_data.empty:
            return pd.DataFrame({"error": [f"No comparison data found for '{keyword1}' and '{keyword2}'"]})
        
        # Prepare comparison data for plotting
        comparison_data.reset_index(inplace=True)
        
        # Plot comparison data
        plt.figure(figsize=(12, 6))
        plt.plot(comparison_data['date'], comparison_data[keyword1], marker='o', label=keyword1)
        plt.plot(comparison_data['date'], comparison_data[keyword2], marker='s', label=keyword2)
        plt.title(f"Interest Over Time: {keyword1} vs {keyword2}")
        plt.xlabel("Date")
        plt.ylabel("Interest")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Display plot using Streamlit
        st.pyplot(plt)
        
        return comparison_data
    
    except ResponseError as e:
        if e.response.status_code == 429:  # If rate limit error
            st.write("Rate limit exceeded, retrying in 60 seconds...")
            time.sleep(60)  # Wait 60 seconds before retrying
            return compare_keywords(keyword1, keyword2, location, timeframe)  # Retry the request
        
        return pd.DataFrame({"error": [str(e)]})
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame({"error": [str(e)]})

def generate_tags(keyword, location):
    # Define prompt for tag generation
    prompt = f"Generate 10 relevant tags for '{keyword} that is mostly used in {location}'"

    # Use OpenAI API to generate tags
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    # Extract generated tags
    generated_tags = response.choices[0].message.content.splitlines()

    # Remove duplicates and add hashtag
    generated_tags = list(set([f"#{tag.replace(' ', '')}" for tag in generated_tags]))

    # Create DataFrame with generated tags
    df = pd.DataFrame(generated_tags, columns=["Tag"])

    # Return DataFrame
    return df

# Function to get view count from a video and estimate earnings
def estimate_earnings(video_url):
    video_id = extract_video_id(video_url)
    youtube = get_service()
    response = execute_api_request(
        youtube.videos().list,
        part="statistics",
        id=video_id
    )
    
    if 'items' in response:
        stats = response['items'][0]['statistics']
        view_count = int(stats.get('viewCount', 0))
        
        # Get channel statistics
        channel_response = execute_api_request(
            youtube.channels().list,
            part="statistics",
            id=extract_channel_id(video_url)
        )
        
        if 'items' in channel_response:
            channel_stats = channel_response['items'][0]['statistics']
            subscriber_count = int(channel_stats.get('subscriberCount', 0))
            
            # Only calculate earnings for channels with more than 1000 subscribers
            if subscriber_count >= 1000:
                # Define minimum and maximum CPI (cost per 1000 views)
                cpi_min = 0.2  # $0.2 per 1000 views (e.g., Africa)
                cpi_max = 4.0  # $4 per 1000 views (e.g., USA)
                
                # Calculate minimum and maximum earnings
                min_earnings = (view_count * cpi_min) / 1000
                max_earnings = (view_count * cpi_max) / 1000
                
                # Plot earnings as a range
                st.write("### Earnings Estimation")
                st.write(f"Estimated Earnings Range: &#36;{min_earnings:.2f} - &#36;{max_earnings:.2f}")
                
                # Display earnings range as a bar chart
                fig = plt.figure(figsize=(8, 6))
                plt.bar(['Min Earnings', 'Max Earnings'], [min_earnings, max_earnings])
                plt.xlabel('Earnings Range')
                plt.ylabel('Earnings ($)')
                plt.title('Earnings Estimation Range')
                st.pyplot(fig)
                
                return min_earnings, max_earnings
            else:
                st.write("Channel does not meet monetization requirements (<1000 subscribers).")
                return None
        else:
            st.write(f"No channel data found for video {video_id}")
            return None
    else:
        st.write(f"No data found for video {video_id}")
        return None
    

# Function to get video tags and rankings
def get_video_tags(video_url):
    video_id = extract_video_id(video_url)
    youtube = get_service()
    response = execute_api_request(
        youtube.videos().list,
        part="snippet,statistics",
        id=video_id
    )
    
    if 'items' in response:
        item = response['items'][0]
        tags = item['snippet'].get('tags', [])
        view_count = item['statistics'].get('viewCount', 0)
        
        # Create DataFrame
        data = {
            "Tag": tags,
            "Ranking": [i + 1 for i in range(len(tags))],
            "View Count": [format_number(int(view_count))] * len(tags)
        }
        df = pd.DataFrame(data)
        
        st.write("### Video Tags and Rankings")
        return df 
    else:
        st.write(f"No data found for video {video_id}")
        return None
    
# Function to get trending keywords based on region
def get_trending_keywords(country):
    pytrends = TrendReq()
    
    # Get trending searches for the specified region
    trending_searches_df = pytrends.trending_searches(pn=country)
    
    # Simulate search volumes (you can implement your logic here)
    trending_data = {
        "Keyword": trending_searches_df[0].tolist(),
        "Search Volume": [np.random.randint(100_000, 2_000_000) for _ in range(len(trending_searches_df))],
        "Region": [country] * len(trending_searches_df)
    }
    
    df = pd.DataFrame(trending_data)
    df["Search Volume"] = df["Search Volume"].apply(format_number)
    return df

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

def country_title_optimization(country, language):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"optimized titles in {country} {language}",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Country": country,
            "Language": language,
            "Optimized Title": row["snippet"]["title"]
        })
    return pd.DataFrame(data)

def country_thumbnail_preferences(country):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"thumbnail preferences in {country}",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Country": country,
            "Thumbnail Preference": row["snippet"]["thumbnails"]["default"]["url"]
        })
    return pd.DataFrame(data)

def format_analysis_for_country(country):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"video formats in {country}",
        type="video"
    )
    rows = result.get("items", [])
    data = []

    for row in rows:
        video_id = row["id"]["videoId"]
        video_result = execute_api_request(
            youtube.videos().list,
            part="id,snippet",
            id=video_id
        )
        video_rows = video_result.get("items", [])

        if video_rows:
            video_row = video_rows[0]
            category_id = video_row["snippet"]["categoryId"]
            data.append({
                "Country": country,
                "Video ID": video_id,
                "Format": category_id
            })

    return pd.DataFrame(data)

def local_keyword_research(country, language):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"keywords in {country} {language}",
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

def content_style_guidance(country):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"content styles in {country}",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        video_id = row["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        data.append({
            "Country": country,
            "Video Title": row["snippet"]["title"],
            "Content Style": row["snippet"]["description"],
            "Video URL": video_url
        })
    return pd.DataFrame(data)

def country_description_templates(country):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"description templates in {country}",
        type="video"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        data.append({
            "Country": country,
            "Description Template": row["snippet"]["description"]
        })
    return pd.DataFrame(data)

        
# Local Competition Analysis functions
def country_top_creators(country):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"most subscribed creators in {country}",
        type="channel"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        channel_id = row["id"]["channelId"]
        channel_result = execute_api_request(
            youtube.channels().list,
            part="statistics",
            id=channel_id
        )
        channel_rows = channel_result.get("items", [])
        
        if channel_rows:
            subscriber_count = channel_rows[0]["statistics"]["subscriberCount"]
            data.append({
                "Country": country,
                "Creator": row["snippet"]["title"],
                "Subscribers": subscriber_count
            })
    return pd.DataFrame(data)

def country_competitor_analysis(country):
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
        channel_id = row["id"]["channelId"]
        channel_result = execute_api_request(
            youtube.channels().list,
            part="statistics",
            id=channel_id
        )
        channel_rows = channel_result.get("items", [])
        
        if channel_rows:
            subscriber_count = channel_rows[0]["statistics"].get("subscriberCount", "Not Available")
            data.append({
                "Country": country,
                "Competitor": row["snippet"]["title"],
                "Subscribers": subscriber_count
            })
    return pd.DataFrame(data)


def cross_country_creator_comparison(country1, country2):
    youtube = get_service()
    result1 = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"most subscribed creators in {country1}",
        type="channel"
    )
    result2 = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"most subscribed creators in {country2}",
        type="channel"
    )
    rows1 = result1.get("items", [])
    rows2 = result2.get("items", [])
    data = []
    for row1, row2 in zip(rows1, rows2):
        channel_id1 = row1["id"]["channelId"]
        channel_id2 = row2["id"]["channelId"]
        channel_result1 = execute_api_request(
            youtube.channels().list,
            part="statistics",
            id=channel_id1
        )
        channel_result2 = execute_api_request(
            youtube.channels().list,
            part="statistics",
            id=channel_id2
        )
        channel_rows1 = channel_result1.get("items", [])
        channel_rows2 = channel_result2.get("items", [])
        
        if channel_rows1 and channel_rows2:
            subscriber_count1 = channel_rows1[0]["statistics"].get("subscriberCount", "Not Available")
            subscriber_count2 = channel_rows2[0]["statistics"].get("subscriberCount", "Not Available")
            data.append({
                "Country": [country1, country2],
                "Creator": [row1["snippet"]["title"], row2["snippet"]["title"]],
                "Subscribers": [subscriber_count1, subscriber_count2]
            })
    return pd.DataFrame(data)


def industry_leader_identification(industry):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"top channels in {industry}",
        type="channel"
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        channel_id = row["id"]["channelId"]
        channel_result = execute_api_request(
            youtube.channels().list,
            part="statistics",
            id=channel_id
        )
        channel_rows = channel_result.get("items", [])
        
        if channel_rows:
            subscriber_count = channel_rows[0]["statistics"].get("subscriberCount", "Not Available")
            channel_url = f"https://www.youtube.com/channel/{channel_id}"
            data.append({
                "Industry": industry,
                "Leader": row["snippet"]["title"],
                "Subscribers": subscriber_count,
                "Channel URL": channel_url
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
        channel_id = row["id"]["channelId"]
        channel_result = execute_api_request(
            youtube.channels().list,
            part="snippet,statistics",
            id=channel_id
        )
        channel_rows = channel_result.get("items", [])
        
        if channel_rows:
            subscriber_count = channel_rows[0]["statistics"].get("subscriberCount", "Not Available")
            data.append({
                "Country": country,
                "Channel": row["snippet"]["title"],
                "Strategy": row["snippet"]["description"],
                "Subscribers": subscriber_count
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
        channel_id = row["id"]["channelId"]
        channel_result = execute_api_request(
            youtube.channels().list,
            part="statistics",
            id=channel_id
        )
        channel_rows = channel_result.get("items", [])
        
        if channel_rows:
            subscriber_count = channel_rows[0]["statistics"].get("subscriberCount", "Not Available")
            view_count = channel_rows[0]["statistics"].get("viewCount", "Not Available")
            channel_url = f"https://www.youtube.com/channel/{channel_id}"
            data.append({
                "Region": region,
                "Channel": row["snippet"]["title"],
                "Channel URL": channel_url,
                "Views": view_count,
                "Subscribers": subscriber_count
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
    total_views = 0
    
    for row in rows:
        channel_id = row["id"]["channelId"]
        channel_result = execute_api_request(
            youtube.channels().list,
            part="statistics",
            id=channel_id
        )
        channel_rows = channel_result.get("items", [])
        
        if channel_rows:
            view_count = int(channel_rows[0]["statistics"].get("viewCount", 0))
            total_views += view_count
    
    for row in rows:
        channel_id = row["id"]["channelId"]
        channel_result = execute_api_request(
            youtube.channels().list,
            part="statistics",
            id=channel_id
        )
        channel_rows = channel_result.get("items", [])
        
        if channel_rows:
            view_count = int(channel_rows[0]["statistics"].get("viewCount", 0))
            subscriber_count = channel_rows[0]["statistics"].get("subscriberCount", "Not Available")
            market_share = view_count / total_views
            channel_url = f"https://www.youtube.com/channel/{channel_id}"
            data.append({
                "Market": market,
                "Channel": row["snippet"]["title"],
                "Channel URL": channel_url,
                "Views": view_count,
                "Subscribers": subscriber_count,
                "Market Share": market_share
            })
    return pd.DataFrame(data)
        
# Time Zone-Based Analysis functions
from datetime import datetime

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
        published_at = row["snippet"]["publishedAt"]
        dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
        formatted_published_at = dt.strftime("%Y-%m-%d %H:%M:%S")
        video_id = row["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        data.append({
            "Region": region,
            "Upload Time": formatted_published_at,
            "Video Title": row["snippet"]["title"],
            "Video URL": video_url
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
        published_at = row["snippet"]["publishedAt"]
        dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
        formatted_published_at = dt.strftime("%Y-%m-%d %H:%M:%S")
        video_id = row["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        data.append({
            "Country": country,
            "Viewing Hour": formatted_published_at,
            "Video Title": row["snippet"]["title"],
            "Video URL": video_url
        })
    return pd.DataFrame(data)

def engagement_patterns_by_timezone(timezone):
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"popular videos in {timezone}",
        type="video",
        maxResults=100
    )
    rows = result.get("items", [])
    data = []
    for row in rows:
        video_id = row["id"]["videoId"]
        channel_id = row["snippet"]["channelId"]
        channel_url = f"https://www.youtube.com/channel/{channel_id}"
        video_result = execute_api_request(
            youtube.videos().list,
            part="statistics",
            id=video_id
        )
        video_rows = video_result.get("items", [])
        if video_rows:
            video_stats = video_rows[0]["statistics"]
            data.append({
                "Timezone": timezone,
                "Video Title": row["snippet"]["title"],
                "Channel": row["snippet"]["channelTitle"],
                "Channel URL": channel_url,
                "Views": video_stats.get("viewCount", 0),
                "Likes": video_stats.get("likeCount", 0),
                "Comments": video_stats.get("commentCount", 0)
            })
    return pd.DataFrame(data)

def performance_comparison_across_time_zones(timezone1, timezone2):
    youtube = get_service()
    result1 = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"popular videos in {timezone1}",
        type="video",
        maxResults=100
    )
    result2 = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        q=f"popular videos in {timezone2}",
        type="video",
        maxResults=100
    )
    rows1 = result1.get("items", [])
    rows2 = result2.get("items", [])
    data = []
    for row1, row2 in zip(rows1, rows2):
        video_id1 = row1["id"]["videoId"]
        video_id2 = row2["id"]["videoId"]
        video_result1 = execute_api_request(
            youtube.videos().list,
            part="statistics",
            id=video_id1
        )
        video_result2 = execute_api_request(
            youtube.videos().list,
            part="statistics",
            id=video_id2
        )
        video_rows1 = video_result1.get("items", [])
        video_rows2 = video_result2.get("items", [])
        if video_rows1 and video_rows2:
            video_stats1 = video_rows1[0]["statistics"]
            video_stats2 = video_rows2[0]["statistics"]
            view_count1 = video_stats1.get("viewCount", "N/A")
            view_count2 = video_stats2.get("viewCount", "N/A")
            like_count1 = video_stats1.get("likeCount", "N/A")
            like_count2 = video_stats2.get("likeCount", "N/A")
            comment_count1 = video_stats1.get("commentCount", "N/A")
            comment_count2 = video_stats2.get("commentCount", "N/A")
            data.append({
                "Timezone": [timezone1, timezone2],
                "Video Title": [row1["snippet"]["title"], row2["snippet"]["title"]],
                "Views": [view_count1, view_count2],
                "Likes": [like_count1, like_count2],
                "Comments": [comment_count1, comment_count2]
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
        published_at = row["snippet"]["publishedAt"]
        dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
        formatted_published_at = dt.strftime("%B %d, %Y %I:%M %p")
        video_id = row["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        data.append({
            "Region": region,
            "Video Title": row["snippet"]["title"],
            "Posting Schedule": formatted_published_at,
            "Video URL": video_url
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
        published_at = row["snippet"]["publishedAt"]
        dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
        formatted_published_at = dt.strftime("%B %d, %Y %I:%M %p")
        video_id = row["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        data.append({
            "Country": country,
            "Video Title": row["snippet"]["title"],
            "Activity Time": formatted_published_at,
            "Video URL": video_url
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
        published_at = row["snippet"]["publishedAt"]
        dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
        formatted_published_at = dt.strftime("%B %d, %Y %I:%M %p")
        video_id = row["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        data.append({
            "Region": region,
            "Video Title": row["snippet"]["title"],
            "Timing Analysis": formatted_published_at,
            "Video URL": video_url
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
        published_at = row["snippet"]["publishedAt"]
        dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
        formatted_published_at = dt.strftime("%B %d, %Y %I:%M %p")
        video_id = row["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        data.append({
            "Region": region,
            "Video Title": row["snippet"]["title"],
            "Prime Time": formatted_published_at,
            "Video URL": video_url
        })
    return pd.DataFrame(data)

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
        channel_id = row["snippet"]["channelId"]
        channel_url = f"https://www.youtube.com/channel/{channel_id}"
        published_at = row["snippet"]["publishedAt"]
        dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
        formatted_published_at = dt.strftime("%B %d, %Y %I:%M %p")
        data.append({
            "City": city,
            "Trend": row["snippet"]["title"],
            "Channel": row["snippet"]["channelTitle"],
            "Channel URL": channel_url,
            "Published At": formatted_published_at
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
        channel_id = row["snippet"]["channelId"]
        channel_url = f"https://www.youtube.com/channel/{channel_id}"
        published_at = row["snippet"]["publishedAt"]
        dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
        formatted_published_at = dt.strftime("%B %d, %Y %I:%M %p")
        data.append({
            "Country": country,
            "Seasonal Trend": row["snippet"]["title"],
            "Channel": row["snippet"]["channelTitle"],
            "Channel URL": channel_url,
            "Published At": formatted_published_at
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
        video_id = row["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        channel_id = row["snippet"]["channelId"]
        channel_url = f"https://www.youtube.com/channel/{channel_id}"
        published_at = row["snippet"]["publishedAt"]
        dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
        formatted_published_at = dt.strftime("%B %d, %Y %I:%M %p")
        data.append({
            "Event": event,
            "Impact": row["snippet"]["title"],
            "Channel": row["snippet"]["channelTitle"],
            "Channel URL": channel_url,
            "Video Title": row["snippet"]["title"],
            "Video URL": video_url,
            "Published At": formatted_published_at
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
        video_id = row["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        channel_id = row["snippet"]["channelId"]
        channel_url = f"https://www.youtube.com/channel/{channel_id}"
        published_at = row["snippet"]["publishedAt"]
        dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
        formatted_published_at = dt.strftime("%B %d, %Y %I:%M %p")
        data.append({
            "Holiday": holiday,
            "Content": row["snippet"]["title"],
            "Channel": row["snippet"]["channelTitle"],
            "Channel URL": channel_url,
            "Video URL": video_url,
            "Published At": formatted_published_at
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
        video_id = row["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        data.append({
            "Celebrity": celebrity,
            "Trend": row["snippet"]["title"],
            "Video URL": video_url
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
        video_id = row["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        data.append({
            "Region": region,
            "Meme": row["snippet"]["title"],
            "Video URL": video_url
        })
    return pd.DataFrame(data)

def local_news_trend_impact(api_key, location, news_type):
    # Initialize News API client
    newsapi = NewsApiClient(api_key=api_key)

    try:
        # Get news from News API
        if news_type == "Top Headlines":
            news_data = newsapi.get_top_headlines(q=location, language='en', country='us')
        
        # Check if news data is available
        if news_data and "articles" in news_data:
            news_titles = [article["title"] for article in news_data["articles"]]

            # Initialize YouTube API client
            youtube = get_service()

            # Analyze news impact on YouTube
            data = []
            for news in news_titles:
                result = youtube.search().list(part="id,snippet", q=f"{news} impact", type="video").execute()
                # Check if API result is valid
                if result and "items" in result:
                    rows = result.get("items", [])
                    for row in rows:
                        video_id = row["id"]["videoId"]
                        video_url = f"https://www.youtube.com/watch?v={video_id}"
                        data.append({
                            "News": news,
                            "Impact": row["snippet"]["title"],
                            "Video URL": video_url
                        })

            # Return news impact analysis as DataFrame
            return pd.DataFrame(data)
        else:
            print("No news data available")
    except Exception as e:
        print(f"Error: {str(e)}")


def festival_content_performance(festival, api_key):
    # YouTube API integration
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
        video_id = row["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        data.append({
            "Festival": festival,
            "Content Title": row["snippet"]["title"],
            "Video URL": video_url
        })

    # PredictHQ API integration
    url = f"https://api.predicthq.com/v1/events/"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }
    params = {
        "q": festival
    }
    response = requests.get(url, headers=headers, params=params)
    festival_data = response.json()

    # Extract festival names from PredictHQ response
    festivals = [event.get("title", "") for event in festival_data["results"]]

    festival_data_df = pd.DataFrame(data)
    
    # Display PredictHQ event data
    st.write("\n### **PredictHQ Event Data**")
    st.write("#### Upcoming {} Events".format(festival))
    for i, event in enumerate(festivals, start=1):
        st.write(f"{i}. **{event.title()}**")

    st.write("### **{} Cultural Trend Analysis**".format(festival))
    st.write("\n#### **YouTube Video Data**")
    return festival_data_df #, festivals

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
        video_id = row["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        data.append({
            "Country": country,
            "Market Size": row["snippet"]["title"],
            "Video URL": video_url
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
        video_id = row["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        data.append({
            "Niche": niche,
            "Competition Level": row["snippet"]["title"],
            "Video URL": video_url
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
        video_id = row["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        data.append({
            "Region": region,
            "Niche Opportunity": row["snippet"]["title"],
            "Video URL": video_url
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
        video_id1 = row1["id"]["videoId"]
        video_url1 = f"https://www.youtube.com/watch?v={video_id1}"
        video_id2 = row2["id"]["videoId"]
        video_url2 = f"https://www.youtube.com/watch?v={video_id2}"
        data.append({
            "Market": [market1, market2],
            "Saturation Level": [row1["snippet"]["title"], row2["snippet"]["title"]],
            "Video URL": [video_url1, video_url2]
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
        video_id = row["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        data.append({
            "Audience": audience,
            "Preference": row["snippet"]["title"],
            "Video URL": video_url
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
        video_id = row["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        data.append({
            "Niche": niche,
            "Content Gap": row["snippet"]["title"],
            "Video URL": video_url
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
        video_id = row["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        data.append({
            "Niche": niche,
            "Ad Rate": row["snippet"]["title"],
            "Video URL": video_url
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
        video_id = row["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        data.append({
            "Niche": niche,
            "Monetization Potential": row["snippet"]["title"],
            "Video URL": video_url
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
        video_id = row["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        data.append({
            "Language": language,
            "Trending Video": row["snippet"]["title"],
            "Video URL": video_url
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
        channel_id = row["id"]["channelId"]
        channel_url = f"https://www.youtube.com/channel/{channel_id}"
        data.append({
            "Language": language,
            "Popular Creator": row["snippet"]["title"],
            "Channel URL": channel_url
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
        video_id = row["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        data.append({
            "Language": language,
            "Topic": topic,
            "Trend": row["snippet"]["title"],
            "Video URL": video_url
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
        video_id = row["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        data.append({
            "Language": language,
            "Hashtag": hashtag,
            "Intelligence": row["snippet"]["title"],
            "Video URL": video_url
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
        video_id1 = row1["id"]["videoId"]
        video_url1 = f"https://www.youtube.com/watch?v={video_id1}"
        video_id2 = row2["id"]["videoId"]
        video_url2 = f"https://www.youtube.com/watch?v={video_id2}"
        data.append({
            "Language": [language1, language2],
            "Engagement": [row1["snippet"]["title"], row2["snippet"]["title"]],
            "Video URL": [video_url1, video_url2]
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
        channel_id = row["id"]["channelId"]
        channel_url = f"https://www.youtube.com/channel/{channel_id}"
        data.append({
            "Language": language,
            "Emerging Channel": row["snippet"]["title"],
            "Channel URL": channel_url
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
        video_id = row["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        data.append({
            "Language": language,
            "Keyword Suggestion": row["snippet"]["title"],
            "Video URL": video_url
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
        video_id = row["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        data.append({
            "Language": language,
            "Viral Short-Form Video": row["snippet"]["title"],
            "Video URL": video_url
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
        video_id = row["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        data.append({
            "Country": country,
            "Trending Video": row["snippet"]["title"],
            "Video URL": video_url
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
        video_id = row["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        data.append({
            "Country": country,
            "Top Video": row["snippet"]["title"],
            "Video URL": video_url
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
        video_id = row["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        data.append({
            "Country": country,
            "Keyword": keyword,
            "Trend": row["snippet"]["title"],
            "Video URL": video_url
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
        video_id1 = row1["id"]["videoId"]
        video_url1 = f"https://www.youtube.com/watch?v={video_id1}"
        video_id2 = row2["id"]["videoId"]
        video_url2 = f"https://www.youtube.com/watch?v={video_id2}"
        data.append({
            "Country": [country1, country2],
            "Trend": [row1["snippet"]["title"], row2["snippet"]["title"]],
            "Video URL": [video_url1, video_url2]
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
        video_id = row["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        data.append({
            "Country": country,
            "Viral Content": row["snippet"]["title"],
            "Video URL": video_url
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
        video_id = row["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        data.append({
            "Country": country,
            "Hashtag": hashtag,
            "Trend": row["snippet"]["title"],
            "Video URL": video_url
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
        video_id = row["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        data.append({
            "Country": country,
            "Popular Music": row["snippet"]["title"],
            "Video URL": video_url
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
    # Authenticate with YouTube API
    youtube = get_service()
    
    # Check if video exists
    video_response = youtube.videos().list(
        part="id,snippet",
        id=video_id
    ).execute()
    
    if video_response["items"]:
        # Extract thumbnail URL
        thumbnail_url = video_response["items"][0]["snippet"]["thumbnails"]["default"]["url"]
        return thumbnail_url
    else:
        print(f"Video {video_id} not found")
        return None

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
    img_array = np.array(img)
    
    # Flatten the image array and count colors
    pixels = img_array.reshape(-1, img_array.shape[-1])
    color_counts = Counter(map(tuple, pixels))
    most_common_colors = color_counts.most_common(3)  # Get the 3 most common colors
    return [color for color, count in most_common_colors]

def analyze_composition(img):
    # Calculate the aspect ratio and dimensions
    width, height = img.size
    aspect_ratio = width / height
    return {
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio
    }

def detect_text_overlay(img):
    # Convert to grayscale and analyze pixel values for text overlay
    img_gray = img.convert('L')  # Convert to grayscale
    img_array = np.array(img_gray)
    
    # Count the number of white pixels (potential text overlay)
    white_pixels = np.sum(img_array > 200)  # Assuming text overlay is bright
    total_pixels = img_array.size
    text_overlay_ratio = white_pixels / total_pixels
    
    return text_overlay_ratio

def thumbnail_improvement(video_url):
    video_id = extract_video_id(video_url)
    youtube = get_service()
    
    # Extract thumbnail URL
    thumbnail_url = extract_thumbnail(video_id)
    
    # Analyze thumbnail
    dominant_colors, composition, text_overlay = analyze_thumbnail(thumbnail_url)
    
    # Fetch video title
    video_title_response = execute_api_request(
        youtube.videos().list,
        part="snippet",
        id=video_id
    )
    video_title = video_title_response["items"][0]["snippet"]["title"]
    
    # Craft OpenAI prompt
    prompt = f"Analyze the thumbnail for YouTube video '{video_title}' (ID: {video_id}) and suggest improvements. Consider the following analysis:\n\nDominant colors: {dominant_colors}\nComposition: {composition}\nText overlay ratio: {text_overlay}\n\nGoals: Increase clicks, engagement."
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    st.success("Thumbnail improvement suggestions generated!")
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

def search_optimization_tips(query):
    prompt = f"Provide search optimization tips for '{query}' YouTube videos, including keyword research, title optimization, and metadata improvement."
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def effective_hashtags(query):
    prompt = f"Suggest effective hashtags for '{query}' YouTube videos, considering relevance, popularity, and audience targeting."
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


# Competition Analysis functions
def competitor_video_strategy(channel_url):
    channel_id = get_channel_id(channel_url)
    youtube = get_service()
    
    # Define search query to find competitor videos
    search_queries = [
        f"videos like channel {channel_id}",
        f"similar to {channel_id}",
        f"{channel_id} competitors"
    ]
    
    for search_query in search_queries:
        result = execute_api_request(
            youtube.search().list,
            part="id,snippet",
            type="video",
            maxResults=50,
            q=search_query
        )
        videos = result.get("items", [])
        if videos:
            data = []
            for video in videos:
                video_id = video["id"]["videoId"]
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                video_stats = youtube.videos().list(
                    part="statistics",
                    id=video_id
                ).execute()
                view_count = video_stats["items"][0]["statistics"]["viewCount"]
                data.append({
                    "Video Title": video["snippet"]["title"],
                    "Video URL": video_url,
                    "Video Views": view_count
                })
            df = pd.DataFrame(data)
            return df
    return pd.DataFrame()

def title_comparison(channel_url):
    channel_id = get_channel_id(channel_url)
    youtube = get_service()
    
    # Validate channel_id
    if channel_id is None:
        st.error("Invalid channel URL")
        return pd.DataFrame()
    
    # Retrieve videos from the channel
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        type="video",
        maxResults=50,
        channelId=channel_id
    )
    
    videos = result.get("items", [])
    data = []
    for video in videos:
        video_id = video["id"]["videoId"]
        video_stats = youtube.videos().list(
            part="statistics",
            id=video_id
        ).execute()
        view_count = int(video_stats["items"][0]["statistics"]["viewCount"])
        data.append({
            "Video Title": video["snippet"]["title"],
            "Video Views": view_count
        })
    
    df = pd.DataFrame(data)
    
    # Plot bar chart using Plotly
    fig = px.bar(df, x="Video Title", y="Video Views", title="Title Comparison")
    fig.update_layout(xaxis_tickangle=-90)  # Rotate x-axis labels for readability
    fig.update_yaxes(nticks=5)  # Set y-axis ticks to 5
    
    st.plotly_chart(fig, use_container_width=True)
    
    return df


def upload_pattern_insights(channel_url):
    channel_id = get_channel_id(channel_url)
    youtube = get_service()
    result = execute_api_request(
        youtube.search().list,
        part="id,snippet",
        type="video",
        maxResults=50,
        channelId=channel_id
    )
    videos = result.get("items", [])
    data = []
    for video in videos:
        published_at = video["snippet"]["publishedAt"]
        # Convert to datetime and format
        published_at = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
        published_at = published_at.strftime("%B %d, %Y")
        data.append({
            "Video Title": video["snippet"]["title"],
            "Upload Date": published_at
        })
    df = pd.DataFrame(data)
    # Sort by upload date for plotting
    df = df.sort_values("Upload Date")
    plt.figure(figsize=(10, 6))
    plt.plot(df["Upload Date"], range(len(df["Upload Date"])), marker='o')
    plt.title("Upload Pattern Insights")
    plt.xlabel("Upload Date")
    plt.ylabel("Video Count")
    plt.xticks(rotation=45)  # Rotate x-axis labels for readability
    st.pyplot(plt)
    return df


image_path = "YoutubeViralChatbot.png"
with open(image_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
st.image(f"data:image/png;base64,{encoded_string}", use_column_width=True)


# st.title("YouTube Viral ChatBot")

# api_key_input = st.sidebar.text_input("ENTER YOUR YOUTUBE API KEY", type="password")
# st.session_state["api_key"] = api_key_input

# Initialize YouTube API service
def get_service():
    try:
        service = build('youtube', "v3", developerKey=st.session_state["api_key"])
        return service
    except Exception as e:
        st.error(f"Error building YouTube service: {e}")
        return None

def execute_api_request(client_library_function, **kwargs):
    service = get_service()
    if service is None:
        return None
    try:
        response = client_library_function(**kwargs).execute()
        return response
    except Exception as e:
        st.error(f"API request failed: {e}")
        return None



# options = [
#     "Basic Video Analysis", "Channel Analysis", "YouTube Search","Earnings Estimation", "Trending Keywords",
#     "Content Strategy", "Trending Analysis", "Keyword Research", "Regional Content Strategy", 
#     "Local Competition Analysis", "Time Zone-Based Analysis", 
#     "Cultural Trend Analysis", "Market Research Commands", "Language-Based Search", 
#     "Regional Analysis", "Natural Language Queries","Competition Analysis"
# ]

# selected_option = st.sidebar.selectbox("Choose an analysis type", options)

# Placeholder for the content based on the selected option
# if selected_option:
#     st.write(f"You selected: **{selected_option}**")
    
# if selected_option == "Public Channel Analytics":
#     Channel_url = st.text_input("Enter enter Channel Url", "")
    
# if selected_option == "Video Metrics":
#     video_url = st.text_input("Enter Video Url", "")
    
# if selected_option == "Video Engagement Rate":
#     video_url = st.text_input("Enter Video Url")
    
# if selected_option == "YouTube Search":
#     search_query = st.text_input("Enter Search Query", "")
    
# if selected_option == "Channel Information":
#     channel_url = st.text_input("Enter Channel Url for Info", "")
    
# if selected_option == "Playlist Details":
#     channel_url = st.text_input("Enter Channel Url", "")
    
# if selected_option == "Video Comments":
#     video_url = st.text_input("Enter Video Url for Comments", "")
    
# if selected_option == "Video Details":
#     video_url = st.text_input("Enter Video Url for Details", "")
    
# if selected_option == "Earnings Estimation":
#     video_url = st.text_input("Enter Video Url for Earnings Estimation", "")
#     # cpm = st.number_input("Enter CPM (Cost Per 1000 Impressions)", min_value=0.01, value=2.0, step=0.5)
    
# if selected_option == "Trending Keywords":
#     country = st.text_input("Enter Region Code (e.g., 'united_states', 'india')")
    
# if selected_option  == "Video Tags and Rankings":
#     video_url = st.text_input("Enter Video Url for Tags", "")

                
# Video Analysis
# if selected_option == "Basic Video Analysis":
#     st.write("### Basic Video Analysis")
#     # Show options for basic video analysis
#     analysis_option = st.selectbox(
#         "Choose an analysis option:",
#         [
#             "Video Metrics", 
#             "Video Engagement Rate",
#             "Video Comments",
#             "Video Details",
#             "Video Tags and Rankings",
#             "Extract video keywords",
#             "Compare metrics between videos",
#         ]
#     )

#     if analysis_option == "Compare metrics between videos":
#         first_video_url = st.text_input("Enter first YouTube Video URL for comparison")
#         second_video_url = st.text_input("Enter the second YouTube Video URL for comparison:")
        
#         if st.button("Analyze Video"):
#             if second_video_url:
#                 with st.spinner("Comparing metrics..."):
#                     comparison_result = compare_metrics_between_videos(first_video_url, second_video_url)
#                     st.success("Metrics comparison completed successfully!")
#             else:
#                 st.error("Invalid second video URL.")
#     else:
#         video_url = st.text_input("Enter YouTube Video URL")
        
#         if st.button("Analyze Video"):
#             if video_url:
#                 with st.spinner("Analyzing video..."):
#                     # Call the appropriate analysis function based on selection
#                     if analysis_option == "Video Metrics":
#                         metrics_df = get_video_metrics(video_url)
#                         st.success("Retrieval Completed!")
#                         st.write(metrics_df)

#                     elif analysis_option == "Video Engagement Rate":
#                         metrics_df = show_video_engagement_metrics(video_url)
#                         st.success("Retrieval Completed!")
#                         st.write(metrics_df)
                                    
#                     elif analysis_option == "Video Comments":
#                         comments_df = get_video_comments(video_url)
#                         st.success("Retrieval Completed!")
#                         st.write(comments_df)
                                        
#                     elif analysis_option == "Video Details":
#                         details_df = get_video_details(video_url)
#                         st.success("Retrieval Completed!")
#                         st.write(details_df)
                                        
#                     elif analysis_option == "Video Tags and Rankings":
#                         tags_df = get_video_tags(video_url)
#                         st.success("Retrieval Completed!")
#                         st.write(tags_df)
                       
#                     elif analysis_option == "Extract video keywords":
#                         kkeywords = extract_video_keywords(video_url)
#                         st.success("Keywords retrieved!")
#                         st.write(kkeywords)
#             else:
#                 st.error("Invalid video URL.")
                                    
# elif selected_option == "Channel Analysis":
#     st.write("### Choose Channel Analysis Option")
#     channel_analysis_option = st.selectbox("Select an analysis option:", [
#         "Analyze channel",
#         "Show top performing videos",
#         "Compare channels",
#         "Show channel engagement trends",
#         "Analyze upload schedule",
#         "Get subscriber growth rate"
#     ])
    
#     channel_url = None
#     channel_url_1 = None
#     channel_url_2 = None
    
#     if channel_analysis_option == "Compare channels":
#         channel_url_1 = st.text_input("Enter first Channel URL for comparison:")
#         channel_url_2 = st.text_input("Enter second Channel URL for comparison:")
#     else:
#         channel_url = st.text_input("Enter Channel URL")
    
#     # Button to execute analysis
#     if st.button("Analyze Channel"):
#         with st.spinner("Analyzing channel..."):
#             try:
#                 if channel_analysis_option == "Compare channels":
#                     if channel_url_1 and channel_url_2:
#                         result = compare_channels(channel_url_1, channel_url_2)
#                         st.success("Channels compared successfully!")
#                     else:
#                         st.error("Please enter both channel URLs.")
#                 elif channel_url:  # Execute analysis for other options
#                     # Execute the corresponding function based on user selection
#                     if channel_analysis_option == "Analyze channel":
#                         st.success("Channel analysis completed!")
#                         analytics_df = get_channel_analytics(channel_url)
#                     elif channel_analysis_option == "Show top performing videos":
#                         result = show_top_performing_videos(channel_url)
#                         st.success("Top performing videos retrieved successfully!")
#                     elif channel_analysis_option == "Show channel engagement trends":
#                         result = show_channel_engagement_trends(channel_url)
#                         st.success("Engagement trends analyzed successfully!")
#                         st.write(result)
#                     elif channel_analysis_option == "Analyze upload schedule":
#                         result = analyze_upload_schedule(channel_url)
#                         st.success("Upload schedule analyzed successfully!")
#                         st.write(result)
#                     elif channel_analysis_option == "Get subscriber growth rate":
#                         result = get_subscriber_growth_rate(channel_url)
#                         st.success("Subscriber growth rate retrieved successfully!")
#                         st.write(result)
#                 else:
#                     st.error("Please enter a valid Channel URL.")
#             except Exception as e:
#                 st.error(f"An error occurred: {e}")


# elif selected_option == "YouTube Search":
#     if st.button("Search"):
#         with st.spinner("Searching..."):
#             if search_query:
#                 results_df = search_youtube(search_query)
#                 st.success("Searched Successfully!")
#                 st.write(results_df)
                

# elif selected_option == "Content Strategy":
#     st.write("### Content Strategy")
    
#     # Input for OpenAI API Key
#     api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
#     client = OpenAI(api_key=api_key)
#     pytrends = TrendReq(hl='en-US', tz=360)
    
#     # User option selection
#     option = st.selectbox(
#         "Select a content strategy option:",
#         [
#             "Suggest video categories",
#             "Generate title ideas",
#             "Optimize description",
#             "Get thumbnail suggestions",
#             "Recommend upload time",
#             "Get content calendar suggestions",
#             "Analyze best posting times",
#         ]
#     )

#     # Input field for related topic
#     related_topic = st.text_input("Enter a topic or keyword")

#     if not related_topic and option not in ["Recommend upload time"]:
#         st.error("Please enter a topic.")

#     # Button to process the request
#     if st.button("Submit"):
#         with st.spinner("Processing..."):
#             try:
#                 if option == "Suggest video categories":
#                     results = suggest_video_categories(related_topic)
#                     st.success("Video ideas generated successfully!")
#                     st.write(results)

#                 elif option == "Generate title ideas":
#                     results = generate_title_ideas(related_topic)
#                     st.subheader("Title Ideas:")
#                     st.success("Title ideas generated successfully!")
#                     st.write(results)

#                 elif option == "Optimize description":
#                     results = optimize_description(related_topic)
#                     st.subheader("Optimized Description:")
#                     st.success("Description optimized successfully!")
#                     st.write(results)

#                 elif option == "Get thumbnail suggestions":
#                     results = get_thumbnail_suggestions(related_topic)
#                     st.subheader("Thumbnail Suggestions:")
#                     st.success("Thumbnail suggestions generated successfully!")
#                     st.write(results)

#                 elif option == "Recommend upload time":
#                     results = recommend_upload_time(related_topic)
#                     st.subheader("Recommended Upload Time:")
#                     st.success("Upload time recommended successfully!")
#                     st.write(results)

#                 elif option == "Get content calendar suggestions":
#                     results = get_content_calendar_suggestions(related_topic)
#                     st.subheader("Content Calendar Suggestions:")
#                     st.success("Content calendar suggestions generated successfully!")
#                     st.write(results)

#                 elif option == "Analyze best posting times":
#                     results = analyze_best_posting_times(related_topic)
#                     st.subheader("Best Posting Times:")
#                     st.success("Best posting times analyzed successfully!")
#                     st.write(results)

#             except Exception as e:
#                 st.error(f"An error occurred: {e}")
                

# elif selected_option == "Trending Analysis":
#     st.write("### Trending Analysis")
    
#     # User option selection for trending analysis
#     trending_option = st.selectbox(
#         "Select a trending analysis option:",
#         [
#             "What's trending now",
#             "Get trending hashtags",
#             "Show viral videos", 
#             "Show rising trends", 
#             "Get weekly trend report"
#         ]
#     )
    
#     if trending_option == "Get weekly trend report":
#         keyword = st.text_input("Enter a keyword for trend analysis:")
#     else:
#         keyword = None
        
#     if trending_option == "What's trending now":
#         geo = st.text_input("Enter a region code (e.g., united_states, india):")
#     else:
#           geo = st.text_input("Enter a region code (e.g., US, IN):")

#     # Button to process trending analysis request
#     if st.button("Submit Trending Analysis"):
#         with st.spinner("Processing..."):
#             try:
#                 if trending_option == "What's trending now":
#                     trending_data = get_trending_topics(geo)
#                     st.success("Processing complete!")
#                     st.write(trending_data)
#                 elif trending_option == "Get trending hashtags":
#                     hashtags = get_trending_hashtags(geo)
#                     st.success("Processing complete!")
#                     st.write(hashtags)
#                 elif trending_option == "Show viral videos":
#                     viral_videos = get_viral_videos(geo)
#                     st.success("Processing complete!")
#                     st.write(viral_videos)
#                 elif trending_option == "Show rising trends":
#                     rising_trends = show_rising_trends(keyword, geo)
#                     st.success("Processing complete!")
#                     st.write(rising_trends)
#                 elif trending_option == "Get weekly trend report":
#                     weekly_report = get_weekly_trend_report(geo, keyword=keyword)
#                     st.success("Processing complete!")
#                     st.write(weekly_report)
#             except Exception as e:
#                 st.error(f"An error occurred: {e}")
                

# # Keyword Research
# elif selected_option == "Keyword Research":
#     st.write("### Keyword Research")

#     # Keyword research options
#     keyword_options = [
#         "Research keywords",
#         "Get search volume", 
#         "Show related keywords", 
#         "Analyze keyword competition",
#         "Show keyword trends", 
#         "Compare keywords", 
#         "Generate tag suggestions"
#     ]

#     keyword_option = st.selectbox("Select a keyword research option:", keyword_options)

#     # Input forms
#     with st.form("keyword_research_form"):
#         if keyword_option in ["Show related keywords", "Analyze keyword competition"]:
#             keyword = st.text_input("Enter a keyword:")
#             location = st.text_input("Enter location (e.g. US, IN):")

#         elif keyword_option == "Compare keywords":
#             keyword = st.text_input("Enter first keyword:")
#             keyword2 = st.text_input("Enter second keyword:")
#             location = st.text_input("Enter location (e.g. US, IN):")
            
#         elif keyword_option == "Show related keywords":
#             keyword = st.text_input("Enter keyword(s) (comma-separated):")
#             location = st.text_input("Enter location (e.g., US, IN)")

#         elif keyword_option == "Get search volume":
#             keyword = st.text_input("Enter keyword(s) (comma-separated):")
#             timeframe = st.selectbox("Select timeframe:", ["now 7-d", "today 5-y", "past 12-m"])
#             location = st.text_input("Enter location (e.g., US, IN)")
            
#         elif keyword_option == "Show keyword trends":
#             keyword = st.text_input("Enter keyword(s) (comma-separated):")
#             timeframe = st.selectbox("Select timeframe:", ["now 7-d", "today 5-y", "past 12-m"])
#             location = st.text_input("Enter location (e.g., US, IN)")
            
#         elif keyword_option == "Research keywords":
#             location = st.text_input("Enter location (e.g., United State)")
#             keyword = st.text_input("Enter keyword:")
#             api_key = st.text_input("Enter SERPAPI KEY", type="password")
        
#         elif keyword_option == "Generate tag suggestions":
#             openai_api_key = st.text_input("OpenAI API Key", type="password")
#             location = st.text_input("Enter location (e.g., United State)")
#             keyword = st.text_input("Enter keyword:")

#         submitted = st.form_submit_button("Submit")

#     # Process keyword research request
#     if submitted:
#         with st.spinner("Processing..."):
#             try:
#                 if keyword_option == "Compare keywords" and not keyword2:
#                     st.error("Please enter a second keyword for comparison.")
#                 elif not keyword and keyword_option in ["Research keywords", "Show related keywords", "Analyze keyword competition"]:
#                     st.error("Please enter a keyword for trend analysis.")
#                 else:
#                     # Call the appropriate function based on the selected option
#                     if keyword_option == "Research keywords":
#                         result = research_keywords(keyword, location, api_key)
#                     elif keyword_option == "Get search volume":
#                         result = get_search_volume(timeframe, [kw.strip() for kw in keyword.split(",")], location)
#                     elif keyword_option == "Show related keywords":
#                         result = show_related_keywords(keyword, location)
#                     elif keyword_option == "Analyze keyword competition":
#                         result = analyze_keyword_competition(keyword, location)
#                     elif keyword_option == "Show keyword trends":
#                         result = show_keyword_trends(timeframe, [kw.strip() for kw in keyword.split(",")], location)
#                     elif keyword_option == "Compare keywords":
#                         result = compare_keywords(keyword, keyword2, location)
#                     elif keyword_option == "Generate tag suggestions":
#                         client = OpenAI(api_key=openai_api_key)
#                         result = generate_tags(keyword, location)

#                     st.success("Processing complete!")
#                     st.write(result)
#             except Exception as e:
#                 st.error(f"An error occurred: {e}")
            
# elif selected_option == "Regional Content Strategy":
#     st.write("### Regional Content Strategy")
    
#     regional_content_strategy_option = st.selectbox(
#         "Regional Content Strategy:",
#         [
#             "Country-Specific Top Videos",
#             "Country Title Optimization", 
#             "Country Thumbnail Preferences", 
#             "Format Analysis for Country", 
#             "Local Keyword Research", 
#             "Content Style Guidance", 
#             "Country Description Templates", 
#         ]
#     )
    
#     if regional_content_strategy_option == "Country-Specific Top Videos":
#         country = st.text_input("Enter the Country (e.g GH, US, FRA)")
        
#     if regional_content_strategy_option == "Country Title Optimization":
#         language = st.text_input("Enter the language (e.g en, fra, es)")
#         country = st.text_input("Enter the Country (e.g GH, US)")
        
#     if regional_content_strategy_option == "Country Thumbnail Preferences":
#         country = st.text_input("Enter the Country (e.g GH, US, FRA)")
        
#     if regional_content_strategy_option == "Format Analysis for Country":
#         country = st.text_input("Enter the Country (e.g GH, US, FRA)")
        
#     if regional_content_strategy_option == "Local Keyword Research":
#         country = st.text_input("Enter the Country (e.g GH, US, FRA)")
#         language = st.text_input("Enter the language (e.g en, fra, es)")
        
#     if regional_content_strategy_option == "Content Style Guidance":
#         country = st.text_input("Enter the Country (e.g GH, US, FRA)")
        
#     if regional_content_strategy_option == "Country Description Templates":
#         country = st.text_input("Enter the Country (e.g GH, US, FRA)")
        
#     if regional_content_strategy_option == "Country Tag Suggestions":
#         language = st.text_input("Enter the language (e.g en, fra, es)")
#         country = st.text_input("Enter the Country (e.g GH, US, FRA)")
    
#     if st.button("Generate Regional Content Strategy"):
#         with st.spinner("Generating regional content strategy..."):
#             try:
#                 if regional_content_strategy_option == "Country-Specific Top Videos":
#                     result = country_specific_content_ideas(country)
#                     st.success("Content strategy generated successfully!")
#                     st.write(result)
#                 elif regional_content_strategy_option == "Country Title Optimization":
#                     result = country_title_optimization(country, language)
#                     st.success("Title optimization strategy generated successfully!")
#                     st.write(result)
#                 elif regional_content_strategy_option == "Country Thumbnail Preferences":
#                     result = country_thumbnail_preferences(country)
#                     st.success("Thumbnail preferences strategy generated successfully!")
#                     st.write(result)
#                 elif regional_content_strategy_option == "Format Analysis for Country":
#                     result = format_analysis_for_country(country)
#                     st.success("Format analysis strategy generated successfully!")
#                     st.write(result)
#                 elif regional_content_strategy_option == "Local Keyword Research":
#                     result = local_keyword_research(country, language)
#                     st.success("Keyword research strategy generated successfully!")
#                     st.write(result)
#                 elif regional_content_strategy_option == "Content Style Guidance":
#                     result = content_style_guidance(country)
#                     st.success("Content style guidance generated successfully!")
#                     st.write(result)
#                 elif regional_content_strategy_option == "Country Description Templates":
#                     result = country_description_templates(country)
#                     st.success("Description templates generated successfully!")
#                     st.write(result)
#             except Exception as e:
#                 st.error(f"Content strategy generation failed: {str(e)}")
                
# elif selected_option == "Local Competition Analysis":
#     st.write("### Local Competition Analysis")
 
    
#     local_competition_analysis_option = st.selectbox(
#         "Local Competition Analysis:",
#         [
#             "Country Top Creators", 
#             "Country Competitor Analysis", 
#             "Cross-Country Creator Comparison", 
#             "Industry Leader Identification", 
#             "Local Channel Strategy Insights", 
#             "Regional Performance Benchmarks", 
#             "Market Share Analysis"
#         ]
#     )
    
#     if local_competition_analysis_option == "Country Top Creators":
#         country = st.text_input("Enter the Country")
        
#     if local_competition_analysis_option == "Country Competitor Analysis":
#         country = st.text_input("Enter the Country")
        
#     if local_competition_analysis_option == "Cross-Country Creator Comparison":
#         country1 = st.text_input("Enter the first country")
#         country2 = st.text_input("Enter the second country")
        
#     if local_competition_analysis_option == "Industry Leader Identification":
#         industry = st.selectbox("Select industry", 
#         [
#             "Health and Wellness",
#             "Finance",
#             "Gaming",
#             "Technology",
#             "Marketing",
#             "Travel",
#             "Food",
#             "Fitness",
#             "Other (specify)"
#         ])

#         if industry == "Other (specify)":
#             industry = st.text_input("Enter custom industry")
            
#     if local_competition_analysis_option == "Niche Competitor Finder":
#         niches = [
#             "health and wellness",
#             "personal finance",
#             "gaming",
#             "technology",
#             "marketing",
#             "travel",
#             "food",
#             "fitness"
#         ]

#         st.title("Niche Competitor Finder")
#         niche = st.selectbox("Select a niche", niches)
        
#     if local_competition_analysis_option == "Local Channel Strategy Insights":
#         country = st.text_input("Enter the Country")
        
#     if local_competition_analysis_option == "Regional Performance Benchmarks":
#         region = st.selectbox("Select region", [
#             "North America",
#             "South America",
#             "Europe",
#             "Asia",
#             "Africa",
#             "Australia/Oceania"
#         ])
        
#     if local_competition_analysis_option == "Market Share Analysis":
#         market = st.selectbox("Select markets (e.g., countries, regions)", 
#         [
#             "USA",
#             "Canada",
#             "UK",
#             "Australia",
#             "North America",
#             "Europe",
#             "Asia-Pacific",
#             "Other (specify)"
#         ])

#         if market == "Other (specify)":
#             market = st.text_input("Enter custom market(s) (comma-separated)")
    
#     if st.button("Perform Local Competition Analysis"):
#         with st.spinner("Analysis ongoing..."):
#             try:
#                 if local_competition_analysis_option == "Country Top Creators":
#                     result = country_top_creators(country)
#                     st.success("Top creators analysis completed!")
#                     st.write(result)
#                 elif local_competition_analysis_option == "Country Competitor Analysis":
#                     result = country_competitor_analysis(country)
#                     st.success("Competitor analysis completed!")
#                     st.write(result)
#                 elif local_competition_analysis_option == "Cross-Country Creator Comparison":
#                     result = cross_country_creator_comparison(country1, country2)
#                     st.success("Cross-country comparison completed!")
#                     st.write(result)
#                 elif local_competition_analysis_option == "Industry Leader Identification":
#                     result = industry_leader_identification(industry)
#                     st.success("Industry leader identification completed!")
#                     st.write(result)
#                 elif local_competition_analysis_option == "Local Channel Strategy Insights":
#                     result = local_channel_strategy_insights(country)
#                     st.success("Local channel strategy insights completed!")
#                     st.write(result)
#                 elif local_competition_analysis_option == "Regional Performance Benchmarks":
#                     result = regional_performance_benchmarks(region)
#                     st.success("Regional performance benchmarks completed!")
#                     st.write(result)
#                 elif local_competition_analysis_option == "Market Share Analysis":
#                     result = market_share_analysis(market)
#                     st.success("Market share analysis completed!")
#                     st.write(result)
#                 else:
#                     result = "Invalid option"
#                     st.write(result)
#             except Exception as e:
#                 st.error(f"Local competition analysis failed: {str(e)}")
            
# elif selected_option == "Time Zone-Based Analysis":
#     st.write("### Time Zone-Based Analysis")
    
#     time_zone_based_analysis_option = st.selectbox(
#         "Time Zone-Based Analysis:",
#         [
#             "Best Upload Times by Region", 
#             "Peak Viewing Hours by Country", 
#             "Engagement Patterns by Timezone", 
#             "Performance Comparison Across Time Zones", 
#             "Optimal Posting Schedule by Region", 
#             "Audience Activity Times by Country", 
#             "Live Stream Timing Analysis by Region", 
#             "Regional Prime Times"
#         ]
#     )
    
#     timezones = pytz.common_timezones
    
#     if time_zone_based_analysis_option == "Best Upload Times by Region":
#         region = st.selectbox("Select region", [
#             "North America",
#             "South America",
#             "Europe",
#             "Asia",
#             "Africa",
#             "Australia/Oceania"
#         ])
        
#     if time_zone_based_analysis_option == "Peak Viewing Hours by Country":
#         country = st.text_input("Enter the Country (e.g US, IN)")
        
#     if time_zone_based_analysis_option == "Engagement Patterns by Timezone":
#         timezone = st.selectbox("Select timezone", timezones)
        
#     if time_zone_based_analysis_option == "Performance Comparison Across Time Zones":
#         timezone1 = st.selectbox("Select timezone 1", timezones, key="timezone1")
#         timezone2 = st.selectbox("Select timezone 2", timezones, key="timezone2")
        
#     if time_zone_based_analysis_option == "Optimal Posting Schedule by Region":
#         region = st.selectbox("Select region", [
#             "North America",
#             "South America",
#             "Europe",
#             "Asia",
#             "Africa",
#             "Australia/Oceania"
#         ])
        
#     if time_zone_based_analysis_option == "Audience Activity Times by Country":
#         country = st.text_input("Enter the Country (e.g US, IN)")
        
#     if time_zone_based_analysis_option == "Live Stream Timing Analysis by Region":
#         region = st.selectbox("Select region", [
#             "North America",
#             "South America",
#             "Europe",
#             "Asia",
#             "Africa",
#             "Australia/Oceania"
#         ])
        
#     if time_zone_based_analysis_option == "Regional Prime Times":
#         region = st.selectbox("Select region", [
#             "North America",
#             "South America",
#             "Europe",
#             "Asia",
#             "Africa",
#             "Australia/Oceania"
#         ])
    
#     if st.button("Perform Time Zone-Based Analysis"):
#         with st.spinner("Performing time zone-based analysis..."):
#             try:
#                 if time_zone_based_analysis_option == "Best Upload Times by Region":
#                     result = best_upload_times_by_region(region)
#                     st.write(result)
#                 elif time_zone_based_analysis_option == "Peak Viewing Hours by Country":
#                     result = peak_viewing_hours_by_country(country)
#                     st.write(result)
#                 elif time_zone_based_analysis_option == "Engagement Patterns by Timezone":
#                     result = engagement_patterns_by_timezone(timezone)
#                     st.write(result)
#                 elif time_zone_based_analysis_option == "Performance Comparison Across Time Zones":
#                     result = performance_comparison_across_time_zones(timezone1, timezone2)
#                     st.write(result)
#                 elif time_zone_based_analysis_option == "Optimal Posting Schedule by Region":
#                     result = optimal_posting_schedule_by_region(region)
#                     st.write(result)
#                 elif time_zone_based_analysis_option == "Audience Activity Times by Country":
#                     result = audience_activity_times_by_country(country)
#                     st.write(result)
#                 elif time_zone_based_analysis_option == "Live Stream Timing Analysis by Region":
#                     result = live_stream_timing_analysis_by_region(region)
#                     st.write(result)
#                 elif time_zone_based_analysis_option == "Regional Prime Times":
#                     result = regional_prime_times(region)
#                     st.write(result)
#             except Exception as e:
#                 st.error(f"Time zone-based analysis failed: {str(e)}")
                

# elif selected_option == "Cultural Trend Analysis":
#     st.write("### Cultural Trend Analysis")
    
#     cultural_trend_analysis_option = st.selectbox(
#         "Cultural Trend Analysis:",
#         [
#              "City/Regional Trend Tracker", 
#              "Country Seasonal Trends", 
#              "Cultural Event Impact Analysis", 
#              "Holiday Content Spotlight", 
#              "Local Celebrity Trend Monitor", 
#              "Regional Meme Tracker", 
#              "Local News Trend Impact", 
#              "Festival Content Performance"
#         ]
#     )
    
#     if cultural_trend_analysis_option == "City/Regional Trend Tracker":
#         city = st.text_input("Enter the City")
        
#     if cultural_trend_analysis_option == "Country Seasonal Trends":
#         country = st.text_input("Enter the Country")
#     if cultural_trend_analysis_option == "Cultural Event Impact Analysis":
#         event = st.text_input("Enter event")
        
#     if cultural_trend_analysis_option == "Holiday Content Spotlight":
#         holiday = st.text_input("Enter holiday")
        
#     if cultural_trend_analysis_option == "Local Celebrity Trend Monitor":
#         celebrity = st.text_input("Enter celebrity")
        
#     if cultural_trend_analysis_option == "Regional Meme Tracker":
#         region = st.selectbox("Select region", [
#             "North America",
#             "South America",
#             "Europe",
#             "Asia",
#             "Africa",
#             "Australia/Oceania"
#         ])
        
#     if cultural_trend_analysis_option == "Local News Trend Impact":
#         news_type = st.selectbox("Select News Type", ["Top Headlines"])
#         location = st.text_input("Enter Location")
#         api_key = st.text_input("Enter News API Key", type="password")
       
#     if cultural_trend_analysis_option == "Festival Content Performance":
#         festival_name = st.text_input("Festival Name")
#         festival_api_key = st.text_input("Enter PredictHQ API KEY", type="password")
    
#     if st.button("Cultural Trend Analysis"):
#         with st.spinner("Performing cultural trend analysis..."):
#             try:
#                 if cultural_trend_analysis_option == "City/Regional Trend Tracker":
#                     result = city_regional_trend_tracker(city)
#                     st.write(result)
#                 elif cultural_trend_analysis_option == "Country Seasonal Trends":
#                     result = country_seasonal_trends(country)
#                     st.write(result)
#                 elif cultural_trend_analysis_option == "Cultural Event Impact Analysis":
#                     result = cultural_event_impact_analysis(event)
#                     st.write(result)
#                 elif cultural_trend_analysis_option == "Holiday Content Spotlight":
#                     result = holiday_content_spotlight(holiday)
#                     st.write(result)
#                 elif cultural_trend_analysis_option == "Local Celebrity Trend Monitor":
#                     result = local_celebrity_trend_monitor(celebrity)
#                     st.write(result)
#                 elif cultural_trend_analysis_option == "Regional Meme Tracker":
#                     result = regional_meme_tracker(region)
#                     st.write(result)
#                 elif cultural_trend_analysis_option == "Local News Trend Impact":
#                     result = local_news_trend_impact(api_key, location, news_type)
#                     st.write(result)
#                 elif cultural_trend_analysis_option == "Festival Content Performance":
#                     result = festival_content_performance(festival_name, festival_api_key)
#                     st.write(result)
#             except Exception as e:
#                 st.error(f"Cultural trend analysis failed: {str(e)}")
                
                
# elif selected_option == "Market Research Commands":
#     st.write("### Market Research Commands")
    
#     market_research_command_option = st.selectbox(
#         "Market Research Commands:",
#         [
#             "Country Market Sizing", 
#             "Competition Level Analysis", 
#             "Regional Niche Opportunities", 
#             "Market Saturation Comparison", 
#             "Audience Preference Insights", 
#             "Content Gap Identification", 
#             "Ad Rate Benchmarking", 
#             "Monetization Potential Assessment"
#         ]
#     )
    
#     if market_research_command_option == "Country Market Sizing":
#         country = st.text_input("Enter the Country")
        
#     if market_research_command_option == "Competition Level Analysis":
#         niches = [
#             "health and wellness",
#             "personal finance",
#             "gaming",
#             "technology",
#             "marketing",
#             "travel",
#             "food",
#             "fitness"
#         ]

#         st.title("Niche Competitor Finder")
#         niche = st.selectbox("Select a niche", niches)
        
#     if market_research_command_option == "Regional Niche Opportunities":
#         region = st.selectbox("Select region", [
#             "North America",
#             "South America",
#             "Europe",
#             "Asia",
#             "Africa",
#             "Australia/Oceania"
#         ])
        
#     if market_research_command_option == "Market Saturation Comparison":
#         market1 = st.selectbox("Select markets (e.g., countries, regions)", 
#             [
#                 "USA",
#                 "Canada",
#                 "UK",
#                 "Australia",
#                 "North America",
#                 "Europe",
#                 "Asia-Pacific",
#                 "Other (specify)"
#             ], key="market1")

#         if market1 == "Other (specify)":
#             market1 = st.text_input("Enter custom market(s) (comma-separated)", key="market1_input")

#         market2 = st.selectbox("Select markets (e.g., countries, regions)", 
#             [
#                 "USA",
#                 "Canada",
#                 "UK",
#                 "Australia",
#                 "North America",
#                 "Europe",
#                 "Asia-Pacific",
#                 "Other (specify)"
#             ], key="market2")

#         if market2 == "Other (specify)":
#             market2 = st.text_input("Enter custom market(s) (comma-separated)", key="market2_input")
        
#     if market_research_command_option == "Audience Preference Insights":
#         audience = st.text_input("Enter Preferred Audience")
        
#     if market_research_command_option == "Content Gap Identification":
#         niches = [
#             "health and wellness",
#             "personal finance",
#             "gaming",
#             "technology",
#             "marketing",
#             "travel",
#             "food",
#             "fitness"
#         ]

#         st.title("Niche Competitor Finder")
#         niche = st.selectbox("Select a niche", niches)
        
#     if market_research_command_option == "Ad Rate Benchmarking":
#         niches = [
#             "health and wellness",
#             "personal finance",
#             "gaming",
#             "technology",
#             "marketing",
#             "travel",
#             "food",
#             "fitness"
#         ]

#         st.title("Niche Competitor Finder")
#         niche = st.selectbox("Select a niche", niches)
        
#     if market_research_command_option == "Monetization Potential Assessment":
#         niches = [
#             "health and wellness",
#             "personal finance",
#             "gaming",
#             "technology",
#             "marketing",
#             "travel",
#             "food",
#             "fitness"
#         ]

#         st.title("Niche Competitor Finder")
#         niche = st.selectbox("Select a niche", niches)
    
#     if st.button("Perform Market Research"):
#         with st.spinner("Performing market research..."):
#             try:
#                 if market_research_command_option == "Country Market Sizing":
#                     result = country_market_sizing(country)
#                     st.write(result)
#                 elif market_research_command_option == "Competition Level Analysis":
#                     result = competition_level_analysis(niche)
#                     st.write(result)
#                 elif market_research_command_option == "Regional Niche Opportunities":
#                     result = regional_niche_opportunities(region)
#                     st.write(result)
#                 elif market_research_command_option == "Market Saturation Comparison":
#                     result = market_saturation_comparison(market1, market2)
#                     st.write(result)
#                 elif market_research_command_option == "Audience Preference Insights":
#                     result = audience_preference_insights(audience)
#                     st.write(result)
#                 elif market_research_command_option == "Content Gap Identification":
#                     result = content_gap_identification(niche)
#                     st.write(result)
#                 elif market_research_command_option == "Ad Rate Benchmarking":
#                     result = ad_rate_benchmarking(niche)
#                     st.write(result)
#                 elif market_research_command_option == "Monetization Potential Assessment":
#                     result = monetization_potential_assessment(niche)
#                     st.write(result)
#             except Exception as e:
#                 st.error(f"Market research failed: {str(e)}")
                
                
# elif selected_option == "Language-Based Search":
#     st.write("### Language-Based Search")
    
#     language_Based_search_option = st.selectbox(
#         "Language-Based Search:",
#         [
#             "Language Trending Videos", 
#             "Popular Creator Spotlight", 
#             "Topic Trend Analysis", 
#             "Hashtag Intelligence", 
#             "Cross-Linguistic Engagement Comparison", 
#             "Emerging Channel Tracker", 
#             "Language-Specific Keyword Suggestions", 
#             "Viral Short-Form Videos"
#         ]
#     )
    
#     if language_Based_search_option == "Language Trending Videos":
#         language = st.text_input("Enter the language (e.g en, fra, es)")
        
#     if language_Based_search_option == "Popular Creator Spotlight":
#         language = st.text_input("Enter the language (e.g en, fra, es)")
        
#     if language_Based_search_option == "Topic Trend Analysis":
#         language = st.text_input("Enter the language (e.g en, fra, es)")
#         topic = st.text_input("What topic are you interested in?")
        
#     if language_Based_search_option == "Hashtag Intelligence":
#         language = st.text_input("Enter the language (e.g en, fra, es)")
#         hashtag = st.text_input("Enter the hashtag")
        
#     if language_Based_search_option == "Cross-Linguistic Engagement Comparison":
#         language1 = st.text_input("Enter the language (e.g en, fra, es)", key="language1")
#         language2 = st.text_input("Enter the language (e.g en, fra, es)", key="language2")
        
#     if language_Based_search_option == "Emerging Channel Tracker":
#         language = st.text_input("Enter the language (e.g en, fra, es)")
        
#     if language_Based_search_option == "Language-Specific Keyword Suggestions":
#         language = st.text_input("Enter the language (e.g en, fra, es)")
        
#     if language_Based_search_option == "Viral Short-Form Videos":
#         language = st.text_input("Enter the language (e.g en, fra, es)")
    
#     if st.button("Perform Language-Based Search"):
#         with st.spinner("Performing language-based search..."):
#             try:
#                 if language_Based_search_option == "Language Trending Videos":
#                     result = language_trending_videos(language)
#                     st.success("Search complete!")
#                     st.write(result)
#                 elif language_Based_search_option == "Popular Creator Spotlight":
#                     result = popular_creator_spotlight(language)
#                     st.success("Search complete!")
#                     st.write(result)
#                 elif language_Based_search_option == "Topic Trend Analysis":
#                     result = topic_trend_analysis(language, topic)
#                     st.success("Search complete!")
#                     st.write(result)
#                 elif language_Based_search_option == "Hashtag Intelligence":
#                     result = hashtag_intelligence(language, hashtag)
#                     st.success("Search complete!")
#                     st.write(result)
#                 elif language_Based_search_option == "Cross-Linguistic Engagement Comparison":
#                     result = cross_linguistic_engagement_comparison(language1, language2)
#                     st.success("Search complete!")
#                     st.write(result)
#                 elif language_Based_search_option == "Emerging Channel Tracker":
#                     result = emerging_channel_tracker(language)
#                     st.success("Search complete!")
#                     st.write(result)
#                 elif language_Based_search_option == "Language-Specific Keyword Suggestions":
#                     result = language_specific_keyword_suggestions(language)
#                     st.success("Search complete!")
#                     st.write(result)
#                 elif language_Based_search_option == "Viral Short-Form Videos":
#                     result = viral_short_form_videos(language)
#                     st.success("Search complete!")
#                     st.write(result)
#             except Exception as e:
#                 st.error(f"Search failed: {str(e)}")
            
# elif selected_option == "Regional Analysis":
#     st.write("### Regional Analysis")
    
#     regional_analysis_option = st.selectbox(
#         "Regional Analysis:",
#         [
#             "Country-Specific Trending Videos", 
#             "Top Videos by Country", 
#             "Keyword Trend Analysis", 
#             "Cross-Country Trend Comparison", 
#             "Viral Content Tracker", 
#             "Country-Level Hashtag Trends", 
#             "Popular Music Trends by Country"
#         ]
#     )
    
#     if regional_analysis_option == "Country-Specific Trending Videos":
#         country = st.text_input("Enter the Country")
        
#     if regional_analysis_option == "Top Videos by Country":
#         country = st.text_input("Enter the Country")
    
#     if regional_analysis_option == "Keyword Trend Analysis":
#         country = st.text_input("Enter the Country")
#         keyword = st.text_input("Enter the keyword")
        
#     if regional_analysis_option == "Cross-Country Trend Comparison":
#         country1 = st.text_input("Enter the Country", key="country1")
#         country2 = st.text_input("Enter the Country", key="country2")
        
#     if regional_analysis_option == "Viral Content Tracker":
#         country = st.text_input("Enter the Country")
        
#     if regional_analysis_option == "Country-Level Hashtag Trends":
#         country = st.text_input("Enter the Country")
#         hashtag = st.text_input("Enter the hashtag")
        
#     if regional_analysis_option == "Popular Music Trends by Country":
#         country = st.text_input("Enter the Country")
    
#     if st.button("Perform Regional Analysis"):
#         with st.spinner("Performing regional analysis..."):
#             try:
#                 if regional_analysis_option == "Country-Specific Trending Videos":
#                     result = country_specific_trending_videos(country)
#                     st.success("Analysis complete!")
#                     st.write(result)
#                 elif regional_analysis_option == "Top Videos by Country":
#                     result = top_videos_by_country(country)
#                     st.success("Analysis complete!")
#                     st.write(result)
#                 elif regional_analysis_option == "Keyword Trend Analysis":
#                     result = keyword_trend_analysis(country, keyword)
#                     st.success("Analysis complete!")
#                     st.write(result)
#                 elif regional_analysis_option == "Cross-Country Trend Comparison":
#                     result = cross_country_trend_comparison(country1, country2)
#                     st.success("Analysis complete!")
#                     st.write(result)
#                 elif regional_analysis_option == "Viral Content Tracker":
#                     result = viral_content_tracker(country)
#                     st.success("Analysis complete!")
#                     st.write(result)
#                 elif regional_analysis_option == "Country-Level Hashtag Trends":
#                     result = country_level_hashtag_trends(country, hashtag)
#                     st.success("Analysis complete!")
#                     st.write(result)
#                 elif regional_analysis_option == "Popular Music Trends by Country":
#                     result = popular_music_trends_by_country(country)
#                     st.success("Analysis complete!")
#                     st.write(result)
#             except Exception as e:
#                 st.error(f"Analysis failed: {str(e)}")
            
# elif selected_option == "Natural Language Queries":
#     st.write("### Natural Language Queries")
    
#     api_key = st.sidebar.text_input("Enter your OpenAI Api Key", type="password")
    
#     client = OpenAI(api_key=api_key)
    
#     natural_language_queries_option = st.selectbox(
#         "Natural Language Queries:",
#         [
#             "Boost Video Views", 
#             "Enhance Click-Through Rate", 
#             "Find Profitable Topics", 
#             "Optimal Upload Time", 
#             "Grow Subscribers", 
#             "Niche Success Strategies", 
#             "Thumbnail Improvement", 
#             "Search Optimization Tips", 
#             "Effective Hashtags"
#         ]
#     )
    
#     if natural_language_queries_option == "Boost Video Views":
#         query = st.text_input("What is your question?")
    
#     if natural_language_queries_option == "Enhance Click-Through Rate":
#         query = st.text_input("What is your question?")
        
#     if natural_language_queries_option == "Find Profitable Topics":
#         query = st.text_input("What is your question?")
        
#     if natural_language_queries_option == "Optimal Upload Time":
#         query = st.text_input("Ask about the best time to upload video")
        
#     if natural_language_queries_option == "Grow Subscribers":
#         query = st.text_input("Ask about how to grow subscribers")
        
#     if natural_language_queries_option == "Niche Success Strategies":
#         query = st.text_input("Ask about niche success strategies")
        
#     if natural_language_queries_option == "Thumbnail Improvement":
#         video_url = st.text_input("Enter Video URL")
        
#     if natural_language_queries_option == "Search Optimization Tips":
#         query = st.text_input("Ask about optimization tips")
        
#     if natural_language_queries_option == "Effective Hashtags":
#         query = st.text_input("Ask about effective hashtags")
        
#     if st.button("Natural Language Queries"):
#         with st.spinner("Generating data..."):
#             try:
#                 if natural_language_queries_option == "Boost Video Views":
#                     result = boost_video_views(query)
#                     st.success("Generation complete!")
#                     st.write(result)
#                 elif natural_language_queries_option == "Enhance Click-Through Rate":
#                     result = enhance_click_through_rate(query)
#                     st.success("Generation complete!")
#                     st.write(result)
#                 elif natural_language_queries_option == "Find Profitable Topics":
#                     result = find_profitable_topics(query)
#                     st.success("Generation complete!")
#                     st.write(result)
#                 elif natural_language_queries_option == "Optimal Upload Time":
#                     result = optimal_upload_time(query)
#                     st.success("Generation complete!")
#                     st.write(result)
#                 elif natural_language_queries_option == "Grow Subscribers":
#                     result = grow_subscribers(query)
#                     st.success("Generation complete!")
#                     st.write(result)
#                 elif natural_language_queries_option == "Niche Success Strategies":
#                     result = niche_success_strategies(query)
#                     st.success("Generation complete!")
#                     st.write(result)
#                 elif natural_language_queries_option == "Thumbnail Improvement":
#                     result = thumbnail_improvement(video_url)
#                     st.success("Generation complete!")
#                     st.write(result)
#                 elif natural_language_queries_option == "Search Optimization Tips":
#                     result = search_optimization_tips(query)
#                     st.success("Generation complete!")
#                     st.write(result)
#                 elif natural_language_queries_option == "Effective Hashtags":
#                     result = effective_hashtags(query)
#                     st.success("Generation complete!")
#                     st.write(result)
#             except Exception  as e:
#                     st.error(f"Generation failed: {str(e)}")

# elif selected_option == "Earnings Estimation":
#     if st.button("Estimate Earnings"):
#         with st.spinner("Retrieving earnings..."):
#             earnings_df = estimate_earnings(video_url)
#             st.success("Retrieval Completed!")
            
# elif selected_option == "Trending Keywords":
#     if st.button("Get Trending Keywords"):
#         with st.spinner("Retrieving treding keywords..."):
#             trending_df = get_trending_keywords(country)
#             st.success("Retrieval Completed!")
#             st.write(trending_df)
            
            
# elif selected_option == "Competition Analysis":
#     st.write("### Competition Analysis")
#     competition_analysis_option = st.selectbox(
#         "Competition Analysis:",
#         [
#             "Competitor Video Strategy", 
#             "Title Comparison", 
#             "Upload Pattern Insights"
#         ]
#     )
            
#     if competition_analysis_option == "Competitor Video Strategy":
#         channel_url = st.text_input("Enter Channel URL")
        
#     if competition_analysis_option == "Title Comparison":
#         channel_url = st.text_input("Enter Channel URL")
        
#     if competition_analysis_option == "Upload Pattern Insights":
#         channel_url = st.text_input("Enter Channel URL")
    
#     if st.button("Analyze Competitors"):
#         with st.spinner("Analyzing..."):
#             try:
#                 if competition_analysis_option == "Competitor Video Strategy":
#                     result = competitor_video_strategy(channel_url)
#                     st.success("Analysis complete!")
#                     st.write(result)
#                 elif competition_analysis_option == "Title Comparison":
#                     result = title_comparison(channel_url)
#                     st.success("Analysis complete!")
#                     st.write(result)
#                 elif competition_analysis_option == "Upload Pattern Insights":
#                     result = upload_pattern_insights(channel_url)
#                     st.success("Analysis complete!")
#                     st.write(result)
#             except Exception as e:
#                 st.error(f"Analysis failed: {str(e)}")



# Mine



if "api_key" not in st.session_state:
    st.session_state["api_key"] = None
if "video_url" not in st.session_state:
    st.session_state["video_url"] = None
if "analysis_ready" not in st.session_state:
    st.session_state["analysis_ready"] = False


# Sidebar for Navigation
import streamlit as st

# Check if the API key has been entered
if st.session_state.get("api_key") is None:
    # Hide the sidebar when API key is not entered
    st.title("YouTube Viral ChatBot")
    st.write("""
        Please input your YouTube API key below to get started with analyzing videos and channels.
    """)

    # Input field for the YouTube API Key
    api_key_input = st.text_input("Enter Your YouTube API Key:", type="password", key="api_key_input_home_unique")

    # Store the API key in session state if provided
    if api_key_input:
        # Add a placeholder for key validation (you can replace this with actual validation logic)
        if api_key_input:
           st.session_state["api_key"] = api_key_input
           st.success("API Key has been successfully entered!")
else:
    # Sidebar for Navigation - only shows once API key is entered
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Home", "Basic Video Analysis", "Channel Analysis", "YouTube Search", "Earnings Estimation", 
        "Trending Keywords", "Content Strategy", "Trending Analysis", "Keyword Research", 
        "Regional Content Strategy", "Time Zone-Based Analysis", "Cultural Trend Analysis", 
        "Market Research Commands", "Language-Based Search", "Regional Analysis", "Natural Language Queries", 
        "Competition Analysis"
    ], key="unique_sidebar_radio")
    
    # Display the selected page (for demonstration, assuming the logic of page content is handled elsewhere)
    st.write(f"You selected: {page}")


# Homepage (YouTube API Key Input)
if st.session_state["api_key"] is None:
    # st.title("YouTube Viral ChatBot")
    # st.write("""
    #     Please input your YouTube API key below to get started with analyzing videos and channels.
    # """)

    # Input field for the YouTube API Key
    # api_key_input = st.text_input("Enter Your YouTube API Key:", type="password", key="api_key_input_home")

    # Store the API key in session state if provided
    if api_key_input:
        # Add a placeholder for key validation (you can replace this with actual validation logic)
        if len(api_key_input) == 39:  # Example: YouTube API keys are typically 39 characters long
            st.session_state["api_key"] = api_key_input
            st.success("API Key has been successfully entered!")
        else:
            st.error("Invalid API Key. Please try again.")

# Video URL Input (After API Key is Entered)
elif st.session_state["video_url"] is None:
    st.title("Enter YouTube Video URL")
    st.write("""
        Enter a valid YouTube video URL below to start analyzing its metrics.
    """)

    # Input field for the YouTube Video URL
    video_url_input = st.text_input(
        "Enter the YouTube Video URL:", 
        help="Paste the URL of the video you want to analyze. E.g., https://www.youtube.com/watch?v=abc123", 
        key="video_url_input_home"
    )

    # Analyze Video Button
    if st.button("Analyze Video", key="analyze_video_button"):
        if "youtube.com/watch?v=" in video_url_input:  # Example validation for YouTube URLs
            st.session_state["video_url"] = video_url_input
            st.session_state["analysis_ready"] = True
            st.success(f"Video URL successfully analyzed: {video_url_input}")
        else:
            st.error("Invalid YouTube video URL. Please try again.")
            



# Sidebar Navigation (After Video Analysis)
else:
    st.title("YouTube Analytics")
    st.write(f"You are analyzing the video: {st.session_state['video_url']}")

    if page == "Home":
        st.write("Navigate using the sidebar to explore video analysis options.")
    elif page == "Basic Video Analysis":
        st.title("### Basic Video Analysis")
        st.write("Here is the analysis for the selected video.")
        # video analysis logic here
        # Select the analysis option
        analysis_option = st.selectbox(
            "Choose an analysis option:",
            [
                "Video Metrics",
                "Video Engagement Rate",
                "Video Comments",
                "Video Details",
                "Video Tags and Rankings",
                "Extract video keywords",
                "Compare metrics between videos",
            ],
            key="analysis_option_selector"
        )

        # Option: Compare Metrics Between Videos
        if analysis_option == "Compare metrics between videos":
            first_video_url = st.text_input("Enter the first YouTube Video URL for comparison:", key="first_video_url")
            second_video_url = st.text_input("Enter the second YouTube Video URL for comparison:", key="second_video_url")

            if st.button("Analyze Video", key="compare_button"):
                if first_video_url and second_video_url:
                    with st.spinner("Comparing metrics..."):
                        # Call the comparison function
                        comparison_result = compare_metrics_between_videos(first_video_url, second_video_url)
                        st.success("Metrics comparison completed successfully!")
                        st.write(comparison_result)
                else:
                    st.error("Please provide both video URLs for comparison.")

        # Options: Other Analyses
        else:
            video_url = st.text_input("Enter YouTube Video URL:", key="single_video_url")

            if st.button("Analyze Video", key="analyze_single_video"):
                if video_url:
                    with st.spinner(f"Analyzing {analysis_option} for the video..."):
                        # Call the appropriate function based on the selected option
                        if analysis_option == "Video Metrics":
                            metrics_df = get_video_metrics(video_url)
                            st.success("Metrics retrieved successfully!")
                            st.write(metrics_df)

                        elif analysis_option == "Video Engagement Rate":
                            engagement_df = show_video_engagement_metrics(video_url)
                            st.success("Engagement metrics retrieved successfully!")
                            st.write(engagement_df)

                        elif analysis_option == "Video Comments":
                            comments_df = get_video_comments(video_url)
                            st.success("Comments retrieved successfully!")
                            st.write(comments_df)

                        elif analysis_option == "Video Details":
                            details_df = get_video_details(video_url)
                            st.success("Video details retrieved successfully!")
                            st.write(details_df)

                        elif analysis_option == "Video Tags and Rankings":
                            tags_df = get_video_tags(video_url)
                            st.success("Tags and rankings retrieved successfully!")
                            st.write(tags_df)

                        elif analysis_option == "Extract video keywords":
                            keywords = extract_video_keywords(video_url)
                            st.success("Keywords extracted successfully!")
                            st.write(keywords)
                else:
                    st.error("Please provide a valid YouTube video URL.")

    elif page == "Channel Analysis":
        st.title("### Choose Channel Analysis Option")
        st.write("Analyze the channel associated with this video.")
        # Channel analysis logic here

        # Select analysis option
        channel_analysis_option = st.selectbox("Select an analysis option:", [
            "Analyze channel",
            "Show top performing videos",
            "Compare channels",
            "Show channel engagement trends",
            "Analyze upload schedule",
            "Get subscriber growth rate"
        ])

        # Initialize variables
        channel_url = None
        channel_url_1 = None
        channel_url_2 = None

        # For comparing channels
        if channel_analysis_option == "Compare channels":
            channel_url_1 = st.text_input("Enter first Channel URL for comparison:")
            channel_url_2 = st.text_input("Enter second Channel URL for comparison:")
        else:
            # For other options, just one channel URL input
            channel_url = st.text_input("Enter Channel URL:")

        # Button to execute analysis
        if st.button("Analyze Channel"):
            with st.spinner("Analyzing channel..."):
                try:
                    # Compare two channels
                    if channel_analysis_option == "Compare channels":
                        if channel_url_1 and channel_url_2:
                            result = compare_channels(channel_url_1, channel_url_2)
                            st.success("Channels compared successfully!")
                            st.write(result)  # Display comparison results
                        else:
                            st.error("Please enter both channel URLs.")

                    # Execute analysis for other options
                    elif channel_url:
                        if channel_analysis_option == "Analyze channel":
                            st.success("Channel analysis completed!")
                            analytics_df = get_channel_analytics(channel_url)
                            st.write(analytics_df)  # Show channel analytics

                        elif channel_analysis_option == "Show top performing videos":
                            result = show_top_performing_videos(channel_url)
                            st.success("Top performing videos retrieved successfully!")
                            st.write(result)

                        elif channel_analysis_option == "Show channel engagement trends":
                            result = show_channel_engagement_trends(channel_url)
                            st.success("Engagement trends analyzed successfully!")
                            st.write(result)

                        elif channel_analysis_option == "Analyze upload schedule":
                            result = analyze_upload_schedule(channel_url)
                            st.success("Upload schedule analyzed successfully!")
                            st.write(result)

                        elif channel_analysis_option == "Get subscriber growth rate":
                            result = get_subscriber_growth_rate(channel_url)
                            st.success("Subscriber growth rate retrieved successfully!")
                            st.write(result)
                    else:
                        st.error("Please enter a valid Channel URL.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    elif page == "YouTube Search":
        st.title("YouTube Search")
        st.write("Search for related videos or keywords.")
        # YouTube search logic here

        # Input field for the search query
        search_query = st.text_input("Enter keywords to search for YouTube videos:")

        # Execute search when the user clicks "Search"
        if st.button("Search"):
            with st.spinner("Searching..."):
                if search_query:
                    try:
                        # Call the search function (search_youtube)
                        results_df = search_youtube(search_query)
                        st.success("Searched Successfully!")
                        st.write(results_df)  # Display search results in a table or another format
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                else:
                    st.warning("Please enter a valid search query.")






    elif page == "Earnings Estimation":
         st.title("Earnings Estimation")
         st.write("Estimate earnings based on video views and engagement.")
        # Earnings estimation logic

        # Input field for the video URL
         video_url = st.text_input("Enter YouTube Video URL:", key="video_url_earnings")

        # Estimate earnings when the user clicks "Estimate Earnings"
         if st.button("Estimate Earnings"):
           with st.spinner("Retrieving earnings..."):
            if video_url:
                try:
                    # Call the estimate_earnings function
                    earnings_df = estimate_earnings(video_url)
                    st.success("Earnings estimation completed!")
                    st.write(earnings_df)  # Display the earnings results
                except Exception as e:
                    st.error(f"An error occurred while estimating earnings: {e}")
            else:
                st.warning("Please enter a valid YouTube video URL.")



    elif page == "Trending Keywords":
         st.title("Trending Keywords")
         st.write("Discover trending keywords related to YouTube content.")
         # Trending keywords logic here

          # Input for country (or use a predefined list of countries)
         country = st.selectbox("Select country for trending keywords:", ["US", "UK", "IN", "DE", "FR", "BR"])

           # Button to get trending keywords
    if st.button("Get Trending Keywords"):
           with st.spinner("Retrieving trending keywords..."):
            try:
                # Call the function to get trending keywords
                trending_df = get_trending_keywords(country)
                st.success("Trending keywords retrieved successfully!")
                st.write(trending_df)  # Display trending keywords
            except Exception as e:
                st.error(f"An error occurred: {e}")







    elif page == "Content Strategy":
        st.title("Content Strategy")
        st.write("Create a content strategy based on analytics.")
         # Content strategy logic here

         # Input for OpenAI API Key on the page (not in the sidebar)
        api_key = st.text_input("Enter your OpenAI API Key:", type="password", key="api_key_input_content_strategy")
    
        if not api_key:
           st.warning("Please enter your OpenAI API Key to proceed with content generation.")
        else:
        # If API key is provided, proceed with content strategy tasks
           client = OpenAI(api_key=api_key)
           pytrends = TrendReq(hl='en-US', tz=360)
        
        # User option selection for content strategy
        option = st.selectbox(
            "Select a content strategy option:",
            [
                "Suggest video categories",
                "Generate title ideas",
                "Optimize description",
                "Get thumbnail suggestions",
                "Recommend upload time",
                "Get content calendar suggestions",
                "Analyze best posting times",
            ]
        )

        # Input field for related topic or keyword
        related_topic = st.text_input("Enter a topic or keyword")

        if not related_topic and option not in ["Recommend upload time"]:
            st.error("Please enter a topic.")

        # Button to process the request
        if st.button("Submit"):
            with st.spinner("Processing..."):
                try:
                    # Execute the chosen option
                    if option == "Suggest video categories":
                        results = suggest_video_categories(related_topic)
                        st.success("Video categories generated successfully!")
                        st.write(results)

                    elif option == "Generate title ideas":
                        results = generate_title_ideas(related_topic)
                        st.subheader("Title Ideas:")
                        st.success("Title ideas generated successfully!")
                        st.write(results)

                    elif option == "Optimize description":
                        results = optimize_description(related_topic)
                        st.subheader("Optimized Description:")
                        st.success("Description optimized successfully!")
                        st.write(results)

                    elif option == "Get thumbnail suggestions":
                        results = get_thumbnail_suggestions(related_topic)
                        st.subheader("Thumbnail Suggestions:")
                        st.success("Thumbnail suggestions generated successfully!")
                        st.write(results)

                    elif option == "Recommend upload time":
                        results = recommend_upload_time(related_topic)
                        st.subheader("Recommended Upload Time:")
                        st.success("Upload time recommended successfully!")
                        st.write(results)

                    elif option == "Get content calendar suggestions":
                        results = get_content_calendar_suggestions(related_topic)
                        st.subheader("Content Calendar Suggestions:")
                        st.success("Content calendar suggestions generated successfully!")
                        st.write(results)

                    elif option == "Analyze best posting times":
                        results = analyze_best_posting_times(related_topic)
                        st.subheader("Best Posting Times:")
                        st.success("Best posting times analyzed successfully!")
                        st.write(results)

                except Exception as e:
                    st.error(f"An error occurred: {e}")

                    
                    





    elif page == "Trending Analysis":
        st.title("Trending Analysis")
        st.write("Analyze trending topics and videos on YouTube.")
           #trending analysis logic here
           # User option selection for trending analysis
        trending_option = st.selectbox(
          "Select a trending analysis option:",
          [
            "What's trending now",
            "Get trending hashtags",
            "Show viral videos", 
            "Show rising trends", 
            "Get weekly trend report"
        ]
    )

          # Input fields based on user selection
        if trending_option == "Get weekly trend report":
          keyword = st.text_input("Enter a keyword for trend analysis:")
        else:
          keyword = None
        
        if trending_option == "What's trending now":
           geo = st.text_input("Enter a region code (e.g., united_states, india):")
        else:
          geo = st.text_input("Enter a region code (e.g., US, IN):")

    # Button to process trending analysis request
    if st.button("Submit Trending Analysis"):
        with st.spinner("Processing..."):
            try:
                if trending_option == "What's trending now":
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

                elif trending_option == "Show rising trends":
                    if keyword:
                        rising_trends = show_rising_trends(keyword, geo)
                        st.success("Processing complete!")
                        st.write(rising_trends)
                    else:
                        st.warning("Please enter a keyword for rising trends.")

                elif trending_option == "Get weekly trend report":
                    if keyword:
                        weekly_report = get_weekly_trend_report(geo, keyword=keyword)
                        st.success("Processing complete!")
                        st.write(weekly_report)
                    else:
                        st.warning("Please enter a keyword for the weekly trend report.")
            except Exception as e:
                st.error(f"An error occurred: {e}")






    elif page == "Keyword Research":
        st.title("Keyword Research")
        st.write("Perform keyword research for YouTube videos.")
           # Keyword research options
        keyword_options = [
             "Research keywords",
             "Get search volume", 
             "Show related keywords", 
             "Analyze keyword competition",
             "Show keyword trends", 
             "Compare keywords", 
            "Generate tag suggestions"
    ]

             # User selects a keyword research option
        keyword_option = st.selectbox("Select a keyword research option:", keyword_options)

            # Input forms for user-provided data
        with st.form("keyword_research_form"):
         if keyword_option in ["Show related keywords", "Analyze keyword competition"]:
            keyword = st.text_input("Enter a keyword:")
            location = st.text_input("Enter location (e.g., US, IN):")

         elif keyword_option == "Compare keywords":
            keyword = st.text_input("Enter first keyword:")
            keyword2 = st.text_input("Enter second keyword:")
            location = st.text_input("Enter location (e.g., US, IN):")
            
         elif keyword_option == "Show related keywords":
            keyword = st.text_input("Enter keyword(s) (comma-separated):")
            location = st.text_input("Enter location (e.g., US, IN)")

         elif keyword_option == "Get search volume":
            keyword = st.text_input("Enter keyword(s) (comma-separated):")
            timeframe = st.selectbox("Select timeframe:", ["now 7-d", "today 5-y", "past 12-m"])
            location = st.text_input("Enter location (e.g., US, IN)")
            
         elif keyword_option == "Show keyword trends":
            keyword = st.text_input("Enter keyword(s) (comma-separated):")
            timeframe = st.selectbox("Select timeframe:", ["now 7-d", "today 5-y", "past 12-m"])
            location = st.text_input("Enter location (e.g., US, IN)")
            
         elif keyword_option == "Research keywords":
            location = st.text_input("Enter location (e.g., United States):")
            keyword = st.text_input("Enter keyword:")
            api_key = st.text_input("Enter SERPAPI Key:", type="password")
        
         elif keyword_option == "Generate tag suggestions":
            openai_api_key = st.text_input("Enter OpenAI API Key:", type="password")
            location = st.text_input("Enter location (e.g., United States):")
            keyword = st.text_input("Enter keyword:")

        # Submit button for the form
        submitted = st.form_submit_button("Submit")

    # Process keyword research request after form submission
    if submitted:
        with st.spinner("Processing..."):
            try:
                if keyword_option == "Compare keywords" and not keyword2:
                    st.error("Please enter a second keyword for comparison.")
                elif not keyword and keyword_option in ["Research keywords", "Show related keywords", "Analyze keyword competition"]:
                    st.error("Please enter a keyword for analysis.")
                else:
                    # Execute the appropriate logic based on the selected option
                    if keyword_option == "Research keywords":
                        result = research_keywords(keyword, location, api_key)
                    elif keyword_option == "Get search volume":
                        result = get_search_volume(timeframe, [kw.strip() for kw in keyword.split(",")], location)
                    elif keyword_option == "Show related keywords":
                        result = show_related_keywords(keyword, location)
                    elif keyword_option == "Analyze keyword competition":
                        result = analyze_keyword_competition(keyword, location)
                    elif keyword_option == "Show keyword trends":
                        result = show_keyword_trends(timeframe, [kw.strip() for kw in keyword.split(",")], location)
                    elif keyword_option == "Compare keywords":
                        result = compare_keywords(keyword, keyword2, location)
                    elif keyword_option == "Generate tag suggestions":
                        client = OpenAI(api_key=openai_api_key)
                        result = generate_tags(keyword, location)

                    st.success("Processing complete!")
                    st.write(result)
            except Exception as e:
                st.error(f"An error occurred: {e}")





    elif page == "Regional Content Strategy":
        st.title("Regional Content Strategy")
        st.write("Develop content strategies based on regional trends.")

         # User selects a regional content strategy option
        regional_content_strategy_option = st.selectbox(
            "Regional Content Strategy:",
            [
                "Country-Specific Top Videos",
                "Country Title Optimization", 
                "Country Thumbnail Preferences", 
                "Format Analysis for Country", 
                "Local Keyword Research", 
                "Content Style Guidance", 
                "Country Description Templates",
            ]
        )


    # # Inputs based on selected strategy option
    # country = None
    # language = None

        if regional_content_strategy_option == "Country-Specific Top Videos":
           country = st.text_input("Enter the Country (e.g., GH, US, FRA):")
        if not country:
            st.warning("Please enter a valid country.")
        
        elif regional_content_strategy_option == "Country Title Optimization":
             language = st.text_input("Enter the Language (e.g., en, fra, es):")
             country = st.text_input("Enter the Country (e.g., GH, US):")
        
        elif regional_content_strategy_option == "Country Thumbnail Preferences":
             country = st.text_input("Enter the Country (e.g., GH, US, FRA):")
        
        elif regional_content_strategy_option == "Format Analysis for Country":
             country = st.text_input("Enter the Country (e.g., GH, US, FRA):")
        
        elif regional_content_strategy_option == "Local Keyword Research":
             country = st.text_input("Enter the Country (e.g., GH, US, FRA):")
             language = st.text_input("Enter the Language (e.g., en, fra, es):")
        
        elif regional_content_strategy_option == "Content Style Guidance":
             country = st.text_input("Enter the Country (e.g., GH, US, FRA):")
        
        elif regional_content_strategy_option == "Country Description Templates":
             country = st.text_input("Enter the Country (e.g., GH, US, FRA):")
    
    # Button to generate regional content strategy
    if st.button("Generate Regional Content Strategy"):
        with st.spinner("Generating regional content strategy..."):
            try:
                # Execute the appropriate logic based on the selected option
                if regional_content_strategy_option == "Country-Specific Top Videos":
                    result = country_specific_content_ideas(country)
                    st.success("Content strategy generated successfully!")
                    st.write(result)

                elif regional_content_strategy_option == "Country Title Optimization":
                    result = country_title_optimization(country, language)
                    st.success("Title optimization strategy generated successfully!")
                    st.write(result)

                elif regional_content_strategy_option == "Country Thumbnail Preferences":
                    result = country_thumbnail_preferences(country)
                    st.success("Thumbnail preferences strategy generated successfully!")
                    st.write(result)

                elif regional_content_strategy_option == "Format Analysis for Country":
                    result = format_analysis_for_country(country)
                    st.success("Format analysis strategy generated successfully!")
                    st.write(result)

                elif regional_content_strategy_option == "Local Keyword Research":
                    result = local_keyword_research(country, language)
                    st.success("Keyword research strategy generated successfully!")
                    st.write(result)

                elif regional_content_strategy_option == "Content Style Guidance":
                    result = content_style_guidance(country)
                    st.success("Content style guidance generated successfully!")
                    st.write(result)

                elif regional_content_strategy_option == "Country Description Templates":
                    result = country_description_templates(country)
                    st.success("Description templates generated successfully!")
                    st.write(result)

            except Exception as e:
                import traceback
                st.error(f"Content strategy generation failed: {str(e)}")
                st.text(traceback.format_exc())




    elif page == "Time Zone-Based Analysis":
        st.title("Time Zone-Based Analysis")
        st.write("Analyze video performance based on time zones.")
    
        # User selects a time zone-based analysis option
        time_zone_based_analysis_option = st.selectbox(
           "Select a time zone-based analysis option:",
          [
              "Best Upload Times by Region", 
              "Peak Viewing Hours by Country", 
              "Engagement Patterns by Timezone", 
              "Performance Comparison Across Time Zones", 
              "Optimal Posting Schedule by Region", 
              "Audience Activity Times by Country", 
              "Live Stream Timing Analysis by Region", 
              "Regional Prime Times"
          ]
        )

    # List of common timezones
        timezones = pytz.common_timezones

    # Input fields based on the selected option
    # region = None
    # country = None
    # timezone = None
    # timezone1 = None
    # timezone2 = None

        if time_zone_based_analysis_option in [
        "Best Upload Times by Region", 
        "Optimal Posting Schedule by Region", 
        "Live Stream Timing Analysis by Region", 
        "Regional Prime Times"
    ]:
            region = st.selectbox("Select region:", [
            "North America",
            "South America",
            "Europe",
            "Asia",
            "Africa",
            "Australia/Oceania"
        ])

        elif time_zone_based_analysis_option in [
        "Peak Viewing Hours by Country", 
        "Audience Activity Times by Country"
    ]:
            country = st.text_input("Enter the Country (e.g., US, IN):")

        elif time_zone_based_analysis_option == "Engagement Patterns by Timezone":
            timezone = st.selectbox("Select timezone:", timezones)

        elif time_zone_based_analysis_option == "Performance Comparison Across Time Zones":
            timezone1 = st.selectbox("Select timezone 1:", timezones, key="timezone1")
            timezone2 = st.selectbox("Select timezone 2:", timezones, key="timezone2")

    # Button to perform analysis
    if st.button("Perform Time Zone-Based Analysis"):
        with st.spinner("Performing time zone-based analysis..."):
            try:
                # Execute the appropriate logic based on the selected option
                if time_zone_based_analysis_option == "Best Upload Times by Region":
                    result = best_upload_times_by_region(region)
                    st.success("Analysis complete!")
                    st.write(result)

                elif time_zone_based_analysis_option == "Peak Viewing Hours by Country":
                    result = peak_viewing_hours_by_country(country)
                    st.success("Analysis complete!")
                    st.write(result)

                elif time_zone_based_analysis_option == "Engagement Patterns by Timezone":
                    result = engagement_patterns_by_timezone(timezone)
                    st.success("Analysis complete!")
                    st.write(result)

                elif time_zone_based_analysis_option == "Performance Comparison Across Time Zones":
                    result = performance_comparison_across_time_zones(timezone1, timezone2)
                    st.success("Analysis complete!")
                    st.write(result)

                elif time_zone_based_analysis_option == "Optimal Posting Schedule by Region":
                    result = optimal_posting_schedule_by_region(region)
                    st.success("Analysis complete!")
                    st.write(result)

                elif time_zone_based_analysis_option == "Audience Activity Times by Country":
                    result = audience_activity_times_by_country(country)
                    st.success("Analysis complete!")
                    st.write(result)

                elif time_zone_based_analysis_option == "Live Stream Timing Analysis by Region":
                    result = live_stream_timing_analysis_by_region(region)
                    st.success("Analysis complete!")
                    st.write(result)

                elif time_zone_based_analysis_option == "Regional Prime Times":
                    result = regional_prime_times(region)
                    st.success("Analysis complete!")
                    st.write(result)

            except Exception as e:
                st.error(f"Time zone-based analysis failed: {str(e)}")



    elif page == "Cultural Trend Analysis":
        st.title("Cultural Trend Analysis")
        st.write("Analyze cultural trends impacting content creation.")
         # Cultural trend analysis logic here

         # User selects a cultural trend analysis option
        cultural_trend_analysis_option = st.selectbox(
           "Cultural Trend Analysis::",
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

        # Inputs based on the selected option
        city = None
        country = None
        event = None
        holiday = None
        celebrity = None
        region = None
        news_type = None
        location = None
        api_key = None
        festival_name = None
        festival_api_key = None

        if cultural_trend_analysis_option == "City/Regional Trend Tracker":
           city = st.text_input("Enter the City")
        
        elif cultural_trend_analysis_option == "Country Seasonal Trends":
           country = st.text_input("Enter the Country")
        
        elif cultural_trend_analysis_option == "Cultural Event Impact Analysis":
           event = st.text_input("Enter event")
        
        elif cultural_trend_analysis_option == "Holiday Content Spotlight":
           holiday = st.text_input("Enter holiday")
        
        elif cultural_trend_analysis_option == "Local Celebrity Trend Monitor":
             celebrity = st.text_input("Enter celebrity")
        
        elif cultural_trend_analysis_option == "Regional Meme Tracker":
             region = st.selectbox("Select region", [
            "North America",
            "South America",
            "Europe",
            "Asia",
            "Africa",
            "Australia/Oceania"
            ])
        
        elif cultural_trend_analysis_option == "Local News Trend Impact":
             news_type = st.selectbox("Select News Type", ["Top Headlines"])
             location = st.text_input("Enter Location")
             api_key = st.text_input("Enter News API Key", type="password")
       
        elif cultural_trend_analysis_option == "Festival Content Performance":
            festival_name = st.text_input("Festival Name")
            festival_api_key = st.text_input("Enter PredictHQ API KEY", type="password")

    # Button to perform cultural trend analysis
    if st.button("Cultural Trend Analysis"):
        with st.spinner("Performing cultural trend analysis..."):
            try:
                # Execute the appropriate logic based on the selected option
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
                    result = local_news_trend_impact(api_key, location, news_type)
                    st.write(result)
                    
                elif cultural_trend_analysis_option == "Festival Content Performance":
                    result = festival_content_performance(festival_name, festival_api_key)
                    st.write(result)

            except Exception as e:
                st.error(f"Cultural trend analysis failed: {str(e)}")





    elif page == "Market Research Commands":
         st.title("Market Research Commands")
         st.write("Perform market research using YouTube data.")
       # Market research commands logic here

       # User selects a market research command option
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

         # Input fields based on the selected option
         country = None
         niche = None
         region = None
         market1 = None
         market2 = None
         audience = None

         if market_research_command_option == "Country Market Sizing":
            country = st.text_input("Enter the Country")
        
         elif market_research_command_option == "Competition Level Analysis":
            niches = ["health and wellness", "personal finance", "gaming", "technology", "marketing", "travel", "food", "fitness"]
            st.title("Niche Competitor Finder")
            niche = st.selectbox("Select a niche", niches)
        
         elif market_research_command_option == "Regional Niche Opportunities":
            region = st.selectbox("Select region", ["North America", "South America", "Europe", "Asia", "Africa", "Australia/Oceania"])
        
         elif market_research_command_option == "Market Saturation Comparison":
           market1 = st.selectbox("Select markets (e.g., countries, regions)", ["USA", "Canada", "UK", "Australia", "North America", "Europe", "Asia-Pacific", "Other (specify)"], key="market1")
         if market1 == "Other (specify)":
            market1 = st.text_input("Enter custom market(s) (comma-separated)", key="market1_input")

            market2 = st.selectbox("Select markets (e.g., countries, regions)", ["USA", "Canada", "UK", "Australia", "North America", "Europe", "Asia-Pacific", "Other (specify)"], key="market2")
         if market2 == "Other (specify)":
            market2 = st.text_input("Enter custom market(s) (comma-separated)", key="market2_input")

         elif market_research_command_option == "Audience Preference Insights":
            audience = st.text_input("Enter Preferred Audience")
        
         elif market_research_command_option == "Content Gap Identification":
           niches = ["health and wellness", "personal finance", "gaming", "technology", "marketing", "travel", "food", "fitness"]
           st.title("Niche Competitor Finder")
           niche = st.selectbox("Select a niche", niches)
        
         elif market_research_command_option == "Ad Rate Benchmarking":
           niches = ["health and wellness", "personal finance", "gaming", "technology", "marketing", "travel", "food", "fitness"]
           st.title("Niche Competitor Finder")
           niche = st.selectbox("Select a niche", niches)
        
         elif market_research_command_option == "Monetization Potential Assessment":
           niches = ["health and wellness", "personal finance", "gaming", "technology", "marketing", "travel", "food", "fitness"]
           st.title("Niche Competitor Finder")
           niche = st.selectbox("Select a niche", niches)

    # Button to perform market research
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









    elif page == "Language-Based Search":
        st.title("Language-Based Search")
    st.write("Search YouTube content based on different languages.")

    # language-based search logic here
    language_Based_search_option = st.selectbox(
        "Select a language-based search option:",
        [
            "Language Trending Videos", 
            "Popular Creator Spotlight", 
            "Topic Trend Analysis", 
            "Hashtag Intelligence", 
            "Cross-Linguistic Engagement Comparison", 
            "Emerging Channel Tracker", 
            "Language-Specific Keyword Suggestions", 
            "Viral Short-Form Videos"
        ]
    )

    # Initialize variables for input fields
    language = None
    topic = None
    hashtag = None
    language1 = None
    language2 = None

    # Conditional inputs based on selected option
    if language_Based_search_option in ["Language Trending Videos", "Popular Creator Spotlight", 
                                        "Emerging Channel Tracker", "Language-Specific Keyword Suggestions", 
                                        "Viral Short-Form Videos"]:
        language = st.text_input("Enter the language (e.g., en, fra, es)")

    if language_Based_search_option == "Topic Trend Analysis":
        language = st.text_input("Enter the language (e.g., en, fra, es)")
        topic = st.text_input("What topic are you interested in?")

    elif language_Based_search_option == "Hashtag Intelligence":
        language = st.text_input("Enter the language (e.g., en, fra, es)")
        hashtag = st.text_input("Enter the hashtag")

    elif language_Based_search_option == "Cross-Linguistic Engagement Comparison":
        language1 = st.text_input("Enter the first language (e.g., en, fra, es)", key="language1")
        language2 = st.text_input("Enter the second language (e.g., en, fra, es)", key="language2")

    # Button to perform language-based search
    if st.button("Perform Language-Based Search"):
        if not language and not (language1 and language2):  # Check if necessary inputs are filled
            st.error("Please provide the required inputs.")
        else:
            with st.spinner("Performing language-based search..."):
                try:
                    # Execute the appropriate logic based on the selected option
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







    elif page == "Regional Analysis": 
        st.title("Regional Analysis")
        st.write("Analyze video performance across different regions.")

        # User selects a regional analysis option
        regional_analysis_option = st.selectbox(
         "Select a regional analysis option:",
           [
              "Country-Specific Trending Videos", 
              "Top Videos by Country", 
              "Keyword Trend Analysis", 
              "Cross-Country Trend Comparison", 
              "Viral Content Tracker", 
              "Country-Level Hashtag Trends", 
              "Popular Music Trends by Country"
           ]
    )

           # Initialize input fields based on selected option
        country = None
        keyword = None
        country1 = None
        country2 = None
        hashtag = None

         # Conditional inputs based on selected option
        if regional_analysis_option in ["Country-Specific Trending Videos", "Top Videos by Country", 
                                    "Viral Content Tracker", "Country-Level Hashtag Trends", 
                                    "Popular Music Trends by Country"]:
          country = st.text_input("Enter the Country")

        if regional_analysis_option == "Keyword Trend Analysis":
          country = st.text_input("Enter the Country")
          keyword = st.text_input("Enter the keyword")

        elif regional_analysis_option == "Cross-Country Trend Comparison":
          country1 = st.text_input("Enter the first Country", key="country1")
          country2 = st.text_input("Enter the second Country", key="country2")

        if regional_analysis_option == "Country-Level Hashtag Trends":
          hashtag = st.text_input("Enter the hashtag")

           # Button to perform regional analysis
        if st.button("Perform Regional Analysis"):
           if not country and not (country1 and country2):    # Ensure required fields are filled
            st.error("Please provide the required inputs.")
           else:
              with st.spinner("Performing regional analysis..."):
                try:
                    # Execute the appropriate logic based on the selected option
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







    elif page == "Natural Language Queries":
        st.title("Natural Language Queries")
        st.write("Query YouTube data using natural language processing.")

       # Input for OpenAI API Key
        api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
    
        if not api_key:
          st.warning("Please enter your OpenAI API key to proceed.")
        else:
          client = OpenAI(api_key=api_key)

        # Select a query option
        natural_language_queries_option = st.selectbox(
            "Select a natural language query option:",
            [
                "Boost Video Views", 
                "Enhance Click-Through Rate", 
                "Find Profitable Topics", 
                "Optimal Upload Time", 
                "Grow Subscribers", 
                "Niche Success Strategies", 
                "Thumbnail Improvement", 
                "Search Optimization Tips", 
                "Effective Hashtags"
            ]
        )

        # Input fields based on the selected query option
        query = None
        video_url = None

        # Conditional inputs based on selected option
        if natural_language_queries_option in ["Boost Video Views", "Enhance Click-Through Rate", 
                                               "Find Profitable Topics", "Optimal Upload Time", 
                                               "Grow Subscribers", "Niche Success Strategies", 
                                               "Search Optimization Tips", "Effective Hashtags"]:
            query = st.text_input("What is your question?")

        elif natural_language_queries_option == "Thumbnail Improvement":
            video_url = st.text_input("Enter Video URL")

        # Button to perform the natural language query
        if st.button("Generate Data"):
            # Check if the required inputs are provided
            if not query and not video_url:
                st.error("Please provide the required input (either a question or video URL).")
            else:
                with st.spinner("Generating data..."):
                    try:
                        # Execute the appropriate function based on the selected option
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
                            result = thumbnail_improvement(video_url)
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

                    except Exception as e:
                        st.error(f"Generation failed: {str(e)}")




    elif page == "Competition Analysis":
        st.title("Competition Analysis")
        st.write("Analyze competition within YouTube content.")

        # Competition Analysis options
        competition_analysis_option = st.selectbox(
          "Competition Analysis:",
          [
            "Competitor Video Strategy", 
            "Title Comparison", 
            "Upload Pattern Insights"
          ]
        )
    
        # Input field for Channel URL (common for all analysis options)
        channel_url = st.text_input("Enter Channel URL")
    
        # Check if the channel URL is provided
        if not channel_url:
         st.warning("Please enter a valid Channel URL to proceed.")
    
        # Button to analyze competitors
        if st.button("Analyze Competitors") and channel_url:
         with st.spinner("Analyzing..."):
            try:
                # Execute the appropriate analysis based on the selected option
                if competition_analysis_option == "Competitor Video Strategy":
                    result = competitor_video_strategy(channel_url)
                    st.success("Analysis complete!")
                    st.write(result)

                elif competition_analysis_option == "Title Comparison":
                    result = title_comparison(channel_url)
                    st.success("Analysis complete!")
                    st.write(result)

                elif competition_analysis_option == "Upload Pattern Insights":
                    result = upload_pattern_insights(channel_url)
                    st.success("Analysis complete!")
                    st.write(result)

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")




    elif page == "Local Competition Analysis":
    # st.write("Analyze local competition within YouTube content.")
    
    # Select competition analysis option
       local_competition_analysis_option = st.selectbox(
        "Local Competition Analysis:",
        [
            "Country Top Creators", 
            "Country Competitor Analysis", 
            "Cross-Country Creator Comparison", 
            "Industry Leader Identification", 
            "Local Channel Strategy Insights", 
            "Regional Performance Benchmarks", 
            "Market Share Analysis"
        ]
       ) 

       # Common input fields for all analysis options
       country = None
       country1 = None
       country2 = None
       industry = None
       niche = None
       region = None
       market = None

       # Display country input if needed
       if local_competition_analysis_option in ["Country Top Creators", "Country Competitor Analysis", 
                                            "Local Channel Strategy Insights"]:
          country = st.text_input("Enter the Country")
        
       # Display cross-country comparison inputs
       if local_competition_analysis_option == "Cross-Country Creator Comparison":
          country1 = st.text_input("Enter the first country")
          country2 = st.text_input("Enter the second country")
        
       # Display industry selection for industry analysis
       if local_competition_analysis_option == "Industry Leader Identification":
          industry = st.selectbox("Select industry", 
            [
            "Health and Wellness",
            "Finance",
            "Gaming",
            "Technology",
            "Marketing",
            "Travel",
            "Food",
            "Fitness",
            "Other (specify)"
            ]
          )
          if industry == "Other (specify)":
            industry = st.text_input("Enter custom industry")
            
          # Display niche selection for niche analysis
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
             niche = st.selectbox("Select a niche", niches)

             # Display region selection for regional performance benchmarks
             if local_competition_analysis_option == "Regional Performance Benchmarks":
                region = st.selectbox("Select region", [
            "North America",
            "South America",
            "Europe",
            "Asia",
            "Africa",
            "Australia/Oceania"
                 ]
                )

             # Display market selection for market share analysis
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
                 ]
                )
                if market == "Other (specify)":
                   market = st.text_input("Enter custom market(s) (comma-separated)")

    # Button to perform the local competition analysis
    if st.button("Perform Local Competition Analysis"):
        # Check for missing inputs
        if local_competition_analysis_option in ["Country Top Creators", "Country Competitor Analysis", 
                                                "Local Channel Strategy Insights"] and not country:
            st.error("Please enter a country.")
        elif local_competition_analysis_option == "Cross-Country Creator Comparison" and (not country1 or not country2):
            st.error("Please enter both countries for comparison.")
        elif local_competition_analysis_option == "Industry Leader Identification" and not industry:
            st.error("Please select an industry.")
        elif local_competition_analysis_option == "Niche Competitor Finder" and not niche:
            st.error("Please select a niche.")
        elif local_competition_analysis_option == "Regional Performance Benchmarks" and not region:
            st.error("Please select a region.")
        elif local_competition_analysis_option == "Market Share Analysis" and not market:
            st.error("Please select a market.")
        else:
            with st.spinner("Analysis ongoing..."):
                try:
                    # Execute the appropriate analysis logic
                    if local_competition_analysis_option == "Country Top Creators":
                        result = country_top_creators(country)
                        st.success("Top creators analysis completed!")
                        st.write(result)

                    elif local_competition_analysis_option == "Country Competitor Analysis":
                        result = country_competitor_analysis(country)
                        st.success("Competitor analysis completed!")
                        st.write(result)

                    elif local_competition_analysis_option == "Cross-Country Creator Comparison":
                        result = cross_country_creator_comparison(country1, country2)
                        st.success("Cross-country comparison completed!")
                        st.write(result)

                    elif local_competition_analysis_option == "Industry Leader Identification":
                        result = industry_leader_identification(industry)
                        st.success("Industry leader identification completed!")
                        st.write(result)

                    elif local_competition_analysis_option == "Local Channel Strategy Insights":
                        result = local_channel_strategy_insights(country)
                        st.success("Local channel strategy insights completed!")
                        st.write(result)

                    elif local_competition_analysis_option == "Regional Performance Benchmarks":
                        result = regional_performance_benchmarks(region)
                        st.success("Regional performance benchmarks completed!")
                        st.write(result)

                    elif local_competition_analysis_option == "Market Share Analysis":
                        result = market_share_analysis(market)
                        st.success("Market share analysis completed!")
                        st.write(result)

                except Exception as e:
                    st.error(f"Local competition analysis failed: {str(e)}")



# Custom Styling
st.markdown("""
    <style>
        .main { background-color: #F0F8FF; padding: 20px; }
        .section-title { color: #2E8B57; font-size: 30px; font-weight: bold; }
        .text { font-size: 18px; color: #333; }
        .stButton>button { background-color: #4CAF50; color: white; padding: 10px 24px; border-radius: 12px; }
        .stButton>button:hover { background-color: #45a049; }
        .stDataFrame { border: 2px solid #2E8B57; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)


# Mine Ends
