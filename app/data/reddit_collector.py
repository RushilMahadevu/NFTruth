import requests
import base64
from typing import List, Dict, Optional
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()

# Load environment variables for Reddit API credentials
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")


class RedditDataCollector:
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.access_token = None
        
        # Define subreddit categories for targeted collection
        self.subreddit_categories = {
            'crypto_general': ['cryptocurrency', 'CryptoCurrency', 'crypto', 'CryptoMarkets', 'altcoin'],
            'nft_specific': ['NFT', 'NFTs', 'opensea', 'NFTsMarketplace', 'CryptoPunks'],
            'ethereum': ['ethereum', 'ethtrader', 'ethfinance', 'EthereumClassic'],
            'trading_focused': ['wallstreetbets', 'CryptoMoonShots', 'SatoshiStreetBets'],
            'tech_analysis': ['CryptoTechnology', 'cryptodevs', 'ethdev']
        }

    def get_access_token(self) -> bool:
        """Get OAuth access token from Reddit API"""
        auth_url = "https://www.reddit.com/api/v1/access_token"
        
        auth_string = f"{self.client_id}:{self.client_secret}"
        auth_bytes = auth_string.encode('ascii')
        auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
        
        headers = {
            "Authorization": f"Basic {auth_b64}",
            "User-Agent": self.user_agent
        }
        
        data = {"grant_type": "client_credentials"}
        
        response = requests.post(auth_url, headers=headers, data=data)
        
        if response.status_code == 200:
            token_data = response.json()
            self.access_token = token_data.get("access_token", "")
            return bool(self.access_token)
        return False

    def search_subreddit(self, subreddit: str, query: str, limit: int = 25, 
                        time_filter: str = 'week', sort: str = 'relevance') -> List[Dict]:
        """Search within a specific subreddit and return raw post data"""
        if not self.access_token:
            return []

        url = f"https://oauth.reddit.com/r/{subreddit}/search"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "User-Agent": self.user_agent
        }

        params = {
            "q": f"{query} subreddit:{subreddit}",
            "type": "link",
            "sort": sort,
            "t": time_filter,
            "limit": min(limit, 100),
            "restrict_sr": "true"
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                posts = data.get("data", {}).get("children", [])
                return [self._extract_post_data(post) for post in posts]
            else:
                print(f"Error searching r/{subreddit}: {response.status_code}")
                return []
        except Exception as e:
            print(f"Exception searching r/{subreddit}: {e}")
            return []

    def get_subreddit_posts(self, subreddit: str, sort: str = 'hot', limit: int = 25, 
                           time_filter: str = 'week') -> List[Dict]:
        """Get posts from a specific subreddit by sort type"""
        if not self.access_token:
            return []

        url = f"https://oauth.reddit.com/r/{subreddit}/{sort}"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "User-Agent": self.user_agent
        }

        params = {"limit": min(limit, 100)}
        if sort in ['top', 'controversial']:
            params['t'] = time_filter

        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                posts = data.get("data", {}).get("children", [])
                return [self._extract_post_data(post) for post in posts]
            return []
        except Exception as e:
            print(f"Exception getting {sort} posts from r/{subreddit}: {e}")
            return []

    def get_post_comments(self, subreddit: str, post_id: str, limit: int = 100) -> List[Dict]:
        """Get comments for a specific post"""
        if not self.access_token:
            return []

        url = f"https://oauth.reddit.com/r/{subreddit}/comments/{post_id}"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "User-Agent": self.user_agent
        }

        params = {"limit": min(limit, 500)}

        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                if len(data) > 1:
                    comments = data[1].get("data", {}).get("children", [])
                    return [self._extract_comment_data(comment) for comment in comments]
            return []
        except Exception as e:
            print(f"Exception getting comments for post {post_id}: {e}")
            return []

    def collect_targeted_data(self, query: str, categories: Optional[List[str]] = None,
                             time_filter: str = 'week', posts_per_subreddit: int = 25,
                             include_comments: bool = False, comment_limit: int = 50) -> Dict:
        """
        Collect raw data from targeted subreddits
        
        Args:
            query: Search term
            categories: List of category names to search (None = all categories)
            time_filter: Reddit time filter (hour, day, week, month, year, all)
            posts_per_subreddit: Max posts to fetch per subreddit
            include_comments: Whether to fetch comments for each post
            comment_limit: Max comments to fetch per post
        """
        if not self.get_access_token():
            return {"error": "Failed to authenticate with Reddit API"}

        if categories is None:
            categories = list(self.subreddit_categories.keys())

        collected_data = {
            "query": query,
            "time_filter": time_filter,
            "collection_timestamp": datetime.utcnow().isoformat(),
            "categories": {},
            "all_posts": [],
            "metadata": {
                "total_posts": 0,
                "total_comments": 0,
                "subreddits_searched": []
            }
        }

        for category in categories:
            if category not in self.subreddit_categories:
                continue
                
            subreddits = self.subreddit_categories[category]
            category_data = {
                "subreddits": {},
                "posts": []
            }
            
            print(f"Collecting from {category}: {', '.join(subreddits)}")
            
            for subreddit in subreddits:
                subreddit_posts = []
                
                # Search for query in subreddit
                search_posts = self.search_subreddit(
                    subreddit, query, posts_per_subreddit, time_filter
                )
                
                # Get hot posts and filter for relevance
                hot_posts = self.get_subreddit_posts(
                    subreddit, 'hot', posts_per_subreddit, time_filter
                )
                relevant_hot = [p for p in hot_posts if self._contains_query(p, query)]
                
                # Combine and deduplicate
                all_posts = self._deduplicate_posts(search_posts + relevant_hot)
                
                # Fetch comments if requested
                if include_comments:
                    for post in all_posts:
                        comments = self.get_post_comments(
                            subreddit, post['id'], comment_limit
                        )
                        post['comments'] = comments
                        collected_data["metadata"]["total_comments"] += len(comments)
                
                subreddit_posts = all_posts
                
                category_data["subreddits"][subreddit] = {
                    "post_count": len(subreddit_posts),
                    "posts": subreddit_posts
                }
                category_data["posts"].extend(subreddit_posts)
                collected_data["metadata"]["subreddits_searched"].append(subreddit)
            
            collected_data["categories"][category] = category_data
            collected_data["all_posts"].extend(category_data["posts"])

        collected_data["metadata"]["total_posts"] = len(collected_data["all_posts"])
        return collected_data

    def _extract_post_data(self, post: Dict) -> Dict:
        """Extract relevant fields from Reddit post data"""
        post_data = post.get("data", {})
        
        return {
            "id": post_data.get("id"),
            "title": post_data.get("title", ""),
            "selftext": post_data.get("selftext", ""),
            "url": post_data.get("url", ""),
            "subreddit": post_data.get("subreddit", ""),
            "author": post_data.get("author", ""),
            "created_utc": post_data.get("created_utc", 0),
            "ups": post_data.get("ups", 0),
            "downs": post_data.get("downs", 0),
            "upvote_ratio": post_data.get("upvote_ratio", 0),
            "num_comments": post_data.get("num_comments", 0),
            "score": post_data.get("score", 0),
            "permalink": post_data.get("permalink", ""),
            "domain": post_data.get("domain", ""),
            "is_self": post_data.get("is_self", False),
            "stickied": post_data.get("stickied", False),
            "over_18": post_data.get("over_18", False),
            "spoiler": post_data.get("spoiler", False),
            "locked": post_data.get("locked", False),
            "flair_text": post_data.get("link_flair_text", ""),
            "author_flair_text": post_data.get("author_flair_text", ""),
            "gilded": post_data.get("gilded", 0),
            "total_awards_received": post_data.get("total_awards_received", 0)
        }

    def _extract_comment_data(self, comment: Dict) -> Dict:
        """Extract relevant fields from Reddit comment data"""
        comment_data = comment.get("data", {})
        
        # Skip "more" comment objects
        if comment_data.get("kind") == "more":
            return None
            
        return {
            "id": comment_data.get("id"),
            "body": comment_data.get("body", ""),
            "author": comment_data.get("author", ""),
            "created_utc": comment_data.get("created_utc", 0),
            "ups": comment_data.get("ups", 0),
            "downs": comment_data.get("downs", 0),
            "score": comment_data.get("score", 0),
            "permalink": comment_data.get("permalink", ""),
            "parent_id": comment_data.get("parent_id", ""),
            "depth": comment_data.get("depth", 0),
            "gilded": comment_data.get("gilded", 0),
            "total_awards_received": comment_data.get("total_awards_received", 0),
            "stickied": comment_data.get("stickied", False),
            "author_flair_text": comment_data.get("author_flair_text", "")
        }

    def _contains_query(self, post: Dict, query: str) -> bool:
        """Check if post contains query terms"""
        title = post.get("title", "").lower()
        selftext = post.get("selftext", "").lower()
        
        query_terms = query.lower().split()
        combined_text = f"{title} {selftext}"
        
        return any(term in combined_text for term in query_terms)

    def _deduplicate_posts(self, posts: List[Dict]) -> List[Dict]:
        """Remove duplicate posts based on post ID"""
        seen_ids = set()
        unique_posts = []
        
        for post in posts:
            post_id = post.get("id")
            if post_id and post_id not in seen_ids:
                seen_ids.add(post_id)
                unique_posts.append(post)
                
        return unique_posts

    def get_user_data(self, username: str) -> Dict:
        """Get user profile data"""
        if not self.access_token:
            return {}

        url = f"https://oauth.reddit.com/user/{username}/about"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "User-Agent": self.user_agent
        }

        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                user_data = data.get("data", {})
                return {
                    "name": user_data.get("name", ""),
                    "created_utc": user_data.get("created_utc", 0),
                    "link_karma": user_data.get("link_karma", 0),
                    "comment_karma": user_data.get("comment_karma", 0),
                    "total_karma": user_data.get("total_karma", 0),
                    "is_gold": user_data.get("is_gold", False),
                    "is_mod": user_data.get("is_mod", False),
                    "verified": user_data.get("verified", False),
                    "has_verified_email": user_data.get("has_verified_email", False)
                }
            return {}
        except Exception as e:
            print(f"Exception getting user data for {username}: {e}")
            return {}

# # Usage example
# if __name__ == "__main__":
#     collector = RedditDataCollector(
#         client_id=REDDIT_CLIENT_ID,
#         client_secret=REDDIT_CLIENT_SECRET, 
#         user_agent=REDDIT_USER_AGENT
#     )
    
#     # Collect raw data for ML processing
#     raw_data = collector.collect_targeted_data(
#         query="Bored Ape Yacht Club",
#         categories=['nft_specific', 'crypto_general'],
#         time_filter='week',
#         include_comments=True,
#         comment_limit=25
#     )
    
#     print(f"Collected {raw_data['metadata']['total_posts']} posts")
#     print(f"Collected {raw_data['metadata']['total_comments']} comments")
#     print(f"Categories: {list(raw_data['categories'].keys())}")
    
#     # Save raw data for ML pipeline
#     import json
#     with open('reddit_raw_data.json', 'w') as f:
#         json.dump(raw_data, f, indent=2)