"""
URL Link Extractor

This module provides functionality for recursively extracting valid URLs from a given website
up to a specified depth. It filters out unwanted file types and specific URL patterns.

"""

import re 
import logging
import argparse
from typing import Set, List, Optional
from urllib.parse import urlparse

# Third-party imports
import requests
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup 
from urllib3.exceptions import InsecureRequestWarning

# Constants
DEFAULT_MAX_DEPTH = 10
DEFAULT_OUTPUT_FILE = "links.csv"

# File format filters
IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.svg'}
PPT_FORMATS = {'.ppt', '.pptx', '.odp', '.key'}
VIDEO_FORMATS = {'.mp4', '.webm', '.ogg', '.avi', '.flv', '.wmv', '.mov'}
OTHER_FORMATS = {'.zip', '.css', '.js', '.csv', '.ico', '.bz2', '.epub', '.woff2'}
EXCLUDED_KEYWORDS = {'signin', 'signup', 'login', 'search'}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress insecure request warnings
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

class UrlCollector:
    """A class for collecting and extracting URLs from a website recursively.

    This class implements depth-first URL collection from a starting URL (head_url),
    filtering out unwanted file types and patterns, and saving the results to a CSV file.

    Attributes:
        head_url (str): The starting URL for link extraction
        max_depth (int): Maximum depth for recursive URL extraction
        output_file (str): Path to save the extracted URLs
        extracted_urls (set): Set to store unique extracted URLs
    """

    def __init__(self, head_url: str, max_depth: int = DEFAULT_MAX_DEPTH, 
                 output_file: str = DEFAULT_OUTPUT_FILE) -> None:
        """Initialize the UrlCollector.

        Args:
            head_url: Starting URL for link extraction
            max_depth: Maximum depth for recursive extraction
            output_file: Path to save the extracted URLs
        
        Raises:
            ValueError: If head_url is empty or invalid
        """
        if not head_url:
            raise ValueError("head_url cannot be empty")
        
        try:
            result = urlparse(head_url)
            if not all([result.scheme, result.netloc]):
                raise ValueError
        except ValueError:
            raise ValueError(f"Invalid URL format: {head_url}")

        self.head_url = head_url
        self.max_depth = max(1, max_depth)  # Ensure at least depth of 1
        self.output_file = output_file
        self.extracted_urls: Set[str] = set()

    def extract_main_domain(self) -> str:
        """Extract the main domain from the head URL.

        Returns:
            The main domain name
        """
        parsed_url = urlparse(self.head_url)
        domain_parts = parsed_url.netloc.split('.')
        
        # Handle subdomains
        if len(domain_parts) > 2:
            return '.'.join(domain_parts[-2:])
        return parsed_url.netloc

    def is_valid_link(self, url: str, domain_name: str) -> bool:
        """Check if a URL is valid according to defined criteria.

        Args:
            url: URL to validate
            domain_name: Domain name to check against

        Returns:
            True if the URL is valid, False otherwise
        """
        if not url or not domain_name:
            return False

        url_lower = url.lower()
        
        # Check if URL belongs to the specified domain
        if domain_name not in url_lower:
            return False

        # Check for excluded file formats
        if any(url_lower.endswith(fmt) for fmt in 
               IMAGE_FORMATS | PPT_FORMATS | VIDEO_FORMATS | OTHER_FORMATS):
            return False

        # Check for excluded keywords
        if any(keyword in url_lower for keyword in EXCLUDED_KEYWORDS):
            return False

        return True

    def extract_sub_links(self, url: str, domain_name: str) -> None:
        """Extract all valid sub-links from a given URL.

        Args:
            url: URL to extract links from
            domain_name: Domain name for validation
        """
        try:
            response = requests.get(
                url, 
                verify=False, 
                timeout=30,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all links that start with https://
            for link in soup.find_all('a', attrs={'href': re.compile("^https://")}):
                href = link.get('href')
                if href and self.is_valid_link(href, domain_name):
                    self.extracted_urls.add(href)
                    
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout while fetching URL: {url}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error fetching URL {url}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error processing {url}: {e}")

    def save_extracted_links(self) -> None:
        """Save extracted URLs to a CSV file."""
        try:
            # Ensure head_url is included
            self.extracted_urls.add(self.head_url)
            
            df = pd.DataFrame({"doc_urls": sorted(self.extracted_urls)})
            df.to_csv(self.output_file, index=False)
            logger.info(f"Saved {len(self.extracted_urls)} URLs to {self.output_file}")
            
        except Exception as e:
            logger.error(f"Error saving links to {self.output_file}: {e}")
            raise

    def links_extractor(self) -> Optional[Set[str]]:
        """Extract links recursively up to the specified depth.

        Returns:
            Set of extracted URLs if successful, None if an error occurs
        """
        try:
            domain_name = self.extract_main_domain()
            urls_to_collect = {self.head_url}
            
            with tqdm(total=self.max_depth, desc="Collecting URLs", unit="depth") as pbar:
                for depth in range(self.max_depth):
                    new_urls = set()
                    
                    for url in urls_to_collect:
                        self.extract_sub_links(url, domain_name)
                        new_urls.update(self.extracted_urls - urls_to_collect)
                    
                    if not new_urls:
                        logger.info(f"No new URLs found at depth {depth + 1}")
                        break
                        
                    urls_to_collect = new_urls
                    pbar.update(1)

            self.save_extracted_links()
            return self.extracted_urls
            
        except Exception as e:
            logger.error(f"Error in link extraction: {e}")
            return None

def main() -> None:
    """Main entry point for the URL collector."""
    parser = argparse.ArgumentParser(
        description="Recursively collect URLs from a website",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--head_url",
        type=str,
        required=True,
        help="Starting URL for link extraction"
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        help="Maximum depth for recursive extraction"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help="Path to save the extracted links (CSV format)"
    )
    
    args = parser.parse_args()
    
    try:
        collector = UrlCollector(args.head_url, args.max_depth, args.output_file)
        if collector.links_extractor():
            logger.info("URL extraction completed successfully")
        else:
            logger.error("URL extraction failed")
            
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()


"""
example usage to run the code
python3 link_extractor.py --head_url "https://example.com" --max_depth 5 --output_file links.csv
"""