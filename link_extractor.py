import re 
import logging
import argparse
import requests
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup 
from urllib.parse import urlparse
from urllib3.exceptions import InsecureRequestWarning

# Suppress the warning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class UrlCollector:

    def __init__(self, head_url: str, max_depth: int, output_file: str):
        self.head_url = head_url
        self.max_depth = max_depth  
        self.output_file = output_file
        self.extracted_urls = set()

    def extract_main_domain(self):
        url = self.head_url
        parsed_url = urlparse(url)
        domain_parts = parsed_url.netloc.split('.')
        if len(domain_parts) > 1:
            main_domain = domain_parts[-2] + '.' + domain_parts[-1]
        else:
            main_domain = domain_parts[0]
        return main_domain

    def is_valid_link(self,url: str, domain_name: str) -> bool:
        # Define lists of image, video, ppt, and other formats
        image_formats = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.svg']
        ppt_formats = ['.ppt', '.odp', '.key']
        video_formats = ['.mp4', '.webm', '.ogg', '.avi', '.flv', '.wmv', '.mov']
        other_formats = ['.zip', '.css', '.js', '.csv', '.ico', '.bz2', '.epub', '.woff2']

        # Check if the URL belongs to the specified domain, and if it is not an image, video, ppt, or other format
        if domain_name in url and \
        not any(keyword in url for keyword in image_formats + video_formats + ppt_formats + other_formats) and \
        not any(substring in url.lower() for substring in ["signin", "signup", "login", "search"]):
            return True
        return False    

    def extract_sub_links(self, url, domain_name):
        try:
            response = requests.get(url, verify=False)
            response.raise_for_status()
            html_document = response.text             
            if html_document:
                soup = BeautifulSoup(html_document, 'html.parser')      
                for link in soup.find_all('a', attrs={'href': re.compile("^https://")}): 
                    if self.is_valid_link(link.get('href'), domain_name): 
                        self.extracted_urls.add(link.get('href'))
        except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
            logger.warning(f"Error occurred while fetching URL {url}: {e}")

    def save_extracted_links(self):
        self.extracted_urls.add(self.head_url)
        data = {"doc_urls": list(self.extracted_urls)}
        df = pd.DataFrame(data)
        df.to_csv(self.output_file, index=False)
        logger.info(f"Extracted links saved to {self.output_file}")

    def links_extractor(self):
        domain_name = self.extract_main_domain()
        
        with tqdm(total=self.max_depth, desc="Collecting URLs", unit="Depth") as pbar:
            urls_to_collect = set([self.head_url])
            for _ in range(self.max_depth):
                # urls_collected = len(self.extracted_urls)
                for i in urls_to_collect:
                    self.extract_sub_links(i, domain_name)
                
                urls_to_collect.update(self.extracted_urls - urls_to_collect)
                pbar.update(1)

        self.save_extracted_links()
        logger.info(f"Total links collected: {len(self.extracted_urls)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="URL Collector")
    parser.add_argument("--head_url", type=str, help="Provide a Head URL")
    parser.add_argument("--max_depth", type=int, default=10, help="Maximum number of depth")
    parser.add_argument("--output_file", type=str, default="link.csv", help="Path to save the extracted links")
    args = parser.parse_args()

    collector = UrlCollector(args.head_url, args.max_depth, args.output_file)
    collector.links_extractor()
