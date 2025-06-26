import csv
from googleapiclient.discovery import build

API_KEY = 'AIzaSyDrtBA5cVa-HkAy91VjiN_GdXNG3L_uynw'
#VIDEO_ID='k2DbnzJa9fk'
def get_youtube_client():
    return build('youtube', 'v3', developerKey=API_KEY)

def get_all_comments(video_id):
    youtube = get_youtube_client()
    all_comments = []
    next_page_token = None

    while True:
        response = youtube.commentThreads().list(
            part="snippet,replies",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token,
            textFormat="plainText"
        ).execute()

        for item in response['items']:
            top_comment = item['snippet']['topLevelComment']['snippet']
            all_comments.append({
                'type': 'top-level',
                'parent_comment_id': '',
                'comment_id': item['snippet']['topLevelComment']['id'],
                'author': top_comment['authorDisplayName'],
                'comment': top_comment['textDisplay'],
                'published_at': top_comment['publishedAt'],
                'like_count': top_comment['likeCount']
            })

            # Get replies if they exist
            if 'replies' in item:
                for reply in item['replies']['comments']:
                    reply_snippet = reply['snippet']
                    all_comments.append({
                        'type': 'reply',
                        'parent_comment_id': item['snippet']['topLevelComment']['id'],
                        'comment_id': reply['id'],
                        'author': reply_snippet['authorDisplayName'],
                        'comment': reply_snippet['textDisplay'],
                        'published_at': reply_snippet['publishedAt'],
                        'like_count': reply_snippet['likeCount']
                    })

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return all_comments

def save_to_csv(comments, filename='youtube_comments_with_replies.csv'):
    keys = ['type', 'comment_id', 'parent_comment_id', 'author', 'comment', 'published_at', 'like_count']
    with open(filename, 'w', newline='', encoding='utf-8') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(comments)

"""
if __name__ == "__main__":
    print("Fetching comments and replies...")
    comments = get_all_comments(VIDEO_ID)
    print(f"Fetched {len(comments)} total comments and replies.")

    save_to_csv(comments)
    print("Saved to youtube_comments_with_replies.csv")
    print("Done.") """
    