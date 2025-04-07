import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging
from typing import List, Dict, Optional
import time
from dataclasses import dataclass
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

@dataclass
class PageContent:
    url: str
    title: str
    content: str
    links: List[str]

class WebCrawler:
    def __init__(self, base_url: str, max_pages: int = 50, timeout: int = 300):
        self.base_url = base_url
        self.max_pages = max_pages
        self.timeout = timeout
        self.visited = set()
        self.pages: List[PageContent] = []
        self.logger = logging.getLogger(__name__)
        self.start_time = datetime.now()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def _get_page_content(self, url: str) -> Optional[PageContent]:
        """Get content from a single page."""
        try:
            self.logger.info(f"Fetching content from {url}")
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.find_all(['script', 'style']):
                element.decompose()
            
            # Get title
            title = soup.title.string.strip() if soup.title else url
            
            # Extract structured content
            content_parts = []
            
            # Get headings
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                level = int(heading.name[1])
                text = heading.get_text(strip=True)
                if text:
                    content_parts.append(f"{'#' * level} {text}\n")
            
            # Get paragraphs and lists
            for element in soup.find_all(['p', 'li', 'div']):
                # Skip if element is a heading (already processed)
                if element.name.startswith('h'):
                    continue
                    
                # Get text content
                text = element.get_text(strip=True)
                if not text:
                    continue
                    
                # Format based on element type
                if element.name == 'li':
                    content_parts.append(f"- {text}\n")
                elif element.name == 'p':
                    content_parts.append(f"{text}\n\n")
                else:
                    # For divs, check if they contain meaningful content
                    if len(text.split()) > 5:  # Only include if it has meaningful content
                        content_parts.append(f"{text}\n\n")
            
            # Get tables
            for table in soup.find_all('table'):
                rows = []
                for row in table.find_all('tr'):
                    cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
                    if cells:
                        rows.append(' | '.join(cells))
                if rows:
                    content_parts.append('\n'.join(rows) + '\n\n')
            
            # Combine all content
            content = ''.join(content_parts)
            
            # Clean content
            content = re.sub(r'\n\s*\n', '\n\n', content)
            content = re.sub(r'\s+', ' ', content).strip()
            
            # Extract links
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('/'):
                    href = urljoin(url, href)
                if href not in self.visited:
                    links.append(href)
            
            return PageContent(url=url, title=title, content=content, links=links)
            
        except Exception as e:
            self.logger.error(f"Error processing {url}: {str(e)}")
            return None

    def crawl(self) -> List[Dict]:
        """Crawl the website and extract content."""
        try:
            self.logger.info(f"Starting crawl of {self.base_url}")
            queue = [self.base_url]
            processed_count = 0
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=10) as executor:
                while queue and len(self.visited) < self.max_pages:
                    # Check timeout
                    elapsed_time = (datetime.now() - self.start_time).seconds
                    if elapsed_time > self.timeout:
                        self.logger.warning(f"Crawling timeout reached after {elapsed_time} seconds")
                        break
                    
                    # Process multiple URLs in parallel
                    current_urls = queue[:10]
                    queue = queue[10:]
                    
                    # Filter out already visited URLs
                    current_urls = [url for url in current_urls if url not in self.visited]
                    if not current_urls:
                        continue
                        
                    self.logger.info(f"Processing {len(current_urls)} URLs (Total processed: {processed_count})")
                    
                    # Process URLs in parallel
                    futures = [executor.submit(self._get_page_content, url) for url in current_urls]
                    
                    for future in futures:
                        page_content = future.result()
                        if page_content:
                            self.visited.add(page_content.url)
                            self.pages.append(page_content)
                            processed_count += 1
                            
                            # Add ALL new links to queue
                            new_links = [link for link in page_content.links if link not in self.visited]
                            queue.extend(new_links)
                            if new_links:
                                self.logger.info(f"Found {len(new_links)} new links to process")
                    
                    # Minimal rate limiting
                    time.sleep(0.1)
            
            self.logger.info(f"Crawling completed. Processed {len(self.pages)} pages in {(datetime.now() - self.start_time).seconds} seconds")
            
            # Convert to list of dictionaries for compatibility
            result = [
                {
                    "url": page.url,
                    "title": page.title,
                    "content": page.content
                }
                for page in self.pages
            ]
            
            self.logger.info(f"Successfully converted {len(result)} pages to dictionary format")
            return result
            
        except Exception as e:
            self.logger.error(f"Crawling failed: {str(e)}")
            return []  # Return empty list instead of raising error 