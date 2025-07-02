import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import trafilatura
from datetime import datetime
import time
import logging
import json

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    webdriver = None
    Service = None
    Options = None
    By = None
    WebDriverWait = None
    EC = None
    ChromeDriverManager = None
    logging.warning("Selenium not available, will use HTML-only scraping")

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AllsvenskanScraper:
    def __init__(self):
        self.base_url = "https://allsvenskan.se/matcher"
        self.api_url = "https://allsvenskan.se/wp-json/sef-leagues/v1/matches"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'application/json,text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'sv-SE,sv;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
    
    def save_debug_content(self, content, year, stage):
        """Save content at different processing stages"""
        try:
            filename = f"debug_{stage}_{year}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(str(content))
            logger.debug(f"Saved debug content to {filename}")
        except Exception as e:
            logger.error(f"Failed to save debug content: {e}")
    
    def _parse_swedish_date(self, day, month_str, year=2025):
        """Improved Swedish date parsing"""
        swedish_months = {
            'januari': 1, 'februari': 2, 'mars': 3, 'april': 4,
            'maj': 5, 'juni': 6, 'juli': 7, 'augusti': 8,
            'september': 9, 'oktober': 10, 'november': 11, 'december': 12
        }

        try:
            month_num = swedish_months.get(month_str.lower())
            if month_num is None:
                logger.warning(f"Unknown month: {month_str}")
                return f"{year}-07-01"  # Default fallback

            day_num = int(day)
            date_obj = datetime(year, month_num, day_num)
            return date_obj.strftime('%Y-%m-%d')

        except (ValueError, TypeError) as e:
            logger.error(f"Date parsing error: {e}, day={day}, month={month_str}")
            return f"{year}-07-01"
    
    def _extract_score_from_line(self, line):
        """Extract score from a single line"""
        # Pattern for "X - Y" or "X-Y"
        score_pattern = r'(\d+)\s*-\s*(\d+)'
        match = re.search(score_pattern, line.strip())

        if match:
            return int(match.group(1)), int(match.group(2))
        return None, None

    def _extract_teams_from_line(self, line):
        """Extract team names from a match line"""
        # Clean the line first
        line = line.strip()

        # Remove common prefixes/suffixes
        line = re.sub(r'^(OMGÅNG\s+\d+|MÅNDAG|TISDAG|ONSDAG|TORSDAG|FREDAG|LÖRDAG|SÖNDAG)', '', line, flags=re.IGNORECASE)
        line = re.sub(r'\d{1,2}\s+(januari|februari|mars|april|maj|juni|juli|augusti|september|oktober|november|december)', '', line, flags=re.IGNORECASE)

        # Look for team vs team pattern
        if ' - ' in line:
            parts = line.split(' - ', 1)
            if len(parts) == 2:
                home = parts[0].strip()
                away = parts[1].strip()

                # Remove score if present in team name
                home = re.sub(r'\s+\d+\s*$', '', home)
                away = re.sub(r'^\s*\d+\s+', '', away)

                return home, away

        return None, None
    
    def validate_match_data(self, match_data):
        """Validate extracted match data"""
        errors = []

        if not match_data.get('HomeTeam'):
            errors.append("Missing home team")
        if not match_data.get('AwayTeam'):
            errors.append("Missing away team")

        # Check for reasonable team names
        home = match_data.get('HomeTeam', '')
        away = match_data.get('AwayTeam', '')

        if len(home) < 2 or len(away) < 2:
            errors.append("Team names too short")

        if home == away:
            errors.append("Same team playing itself")

        # Check date format
        date_str = match_data.get('Date', '')
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            errors.append(f"Invalid date format: {date_str}")

        # Check scores if present
        home_goals = match_data.get('HomeGoals')
        away_goals = match_data.get('AwayGoals')

        if home_goals is not None:
            if not isinstance(home_goals, int) or home_goals < 0:
                errors.append(f"Invalid home goals: {home_goals}")

        if away_goals is not None:
            if not isinstance(away_goals, int) or away_goals < 0:
                errors.append(f"Invalid away goals: {away_goals}")

        return errors
    
    def debug_content_structure(self, year=2025):
        """Debug function to examine content structure"""
        logger.info(f"Debugging content structure for year {year}")
        
        # Always use base URL since year-specific paths cause 404
        year_url = self.base_url

        try:
            # Get raw HTML
            response = requests.get(year_url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all potential match containers
            potential_matches = soup.find_all(['div', 'span', 'td'], 
                                             text=re.compile(r'\d+\s*-\s*\d+'))

            logger.info(f"Found {len(potential_matches)} potential score elements")
            for i, elem in enumerate(potential_matches[:5]):  # Show first 5
                logger.info(f"Element {i}: {elem.get_text().strip()}")
                logger.info(f"Parent: {elem.parent.name if elem.parent else 'None'}")
                logger.info(f"Context: {elem.parent.get_text()[:100] if elem.parent else 'None'}")
                logger.info("---")
                
            # Save full HTML for inspection
            self.save_debug_content(response.text, year, "full_html")
            
        except Exception as e:
            logger.error(f"Error debugging content structure: {e}")
            logger.debug(f"Full traceback: ", exc_info=True)
    
    def _extract_matches_from_html(self, soup, year=2025):
        """Extract matches directly from HTML structure"""
        matches = []
        
        try:
            logger.debug(f"Looking for embedded data in HTML for year {year}")
            
            # Look for script tags with embedded JSON data
            script_tags = soup.find_all('script')
            for script in script_tags:
                if script.string and 'match' in script.string.lower():
                    logger.debug(f"Found potential match data in script tag")
                    # Try to extract JSON from script content
                    # This would need to be customized based on actual website structure
                    
            # Look for specific CSS selectors that might contain match data
            # These would need to be determined by examining the actual website
            potential_containers = [
                '[class*="match"]',
                '[class*="fixture"]', 
                '[class*="game"]',
                '[data-*="match"]',
                'article',
                '.match-container',
                '.fixture-container'
            ]
            
            for selector in potential_containers:
                elements = soup.select(selector)
                if elements:
                    logger.debug(f"Found {len(elements)} elements with selector: {selector}")
                    for elem in elements[:5]:  # Check first 5
                        text = elem.get_text().strip()
                        if text and len(text) > 10:
                            logger.debug(f"Sample content: {text[:100]}")
                            
                            # Try to extract match info using our parsing methods
                            home_team, away_team = self._extract_teams_from_line(text)
                            if home_team and away_team:
                                home_goals, away_goals = self._extract_score_from_line(text)
                                
                                match_data = {
                                    'Date': f'{year}-07-01',  # Default, would need better date extraction
                                    'Venue': 'Stadium',
                                    'Match': f"{home_team} - {away_team}",
                                    'HomeTeam': home_team,
                                    'AwayTeam': away_team,
                                    'HomeGoals': home_goals,
                                    'AwayGoals': away_goals,
                                    'FTHG': home_goals,
                                    'FTAG': away_goals,
                                    'Summary': 'Result' if home_goals is not None else 'Fixture'
                                }
                                
                                validation_errors = self.validate_match_data(match_data)
                                if not validation_errors:
                                    matches.append(match_data)
                                    logger.debug(f"Extracted match from HTML: {home_team} vs {away_team}")
            
            logger.info(f"Extracted {len(matches)} matches from HTML structure")
            
        except Exception as e:
            logger.error(f"Error extracting matches from HTML: {e}")
            logger.debug(f"Full traceback: ", exc_info=True)
            
        return matches
    
    def scrape_matches(self, years=None):
        """Scrape match data from allsvenskan.se for specified years"""
        if years is None:
            years = [2025]  # Default to current year
        
        all_matches = []
        
        for year in years:
            print(f"Scraping data for year {year}...")
            year_matches = self._scrape_year(year)
            if not year_matches.empty:
                all_matches.append(year_matches)
            
            # Add delay between requests to be respectful
            time.sleep(1)
        
        if all_matches:
            combined_df = pd.concat(all_matches, ignore_index=True)
            print(f"Successfully scraped {len(combined_df)} matches from {len(years)} years")
            return combined_df
        else:
            print("No data found, returning sample data")
            return self._create_sample_data()
    
    def _discover_api_endpoint(self, year):
        """Try to discover the correct API endpoint"""
        logger.info(f"Attempting to discover API endpoint for year {year}")
        
        # Try multiple possible API endpoints based on common WordPress structures
        possible_apis = [
            f"https://allsvenskan.se/wp-json/sef-leagues/v1/matches?league=allsvenskan&season={year}",
            f"https://allsvenskan.se/wp-json/wp/v2/matches?season={year}",
            f"https://allsvenskan.se/wp-json/sef/v1/matches?season={year}",
            f"https://allsvenskan.se/api/matches?season={year}",
            f"https://allsvenskan.se/wp-json/sef-leagues/v1/fixtures?league=allsvenskan&season={year}",
            f"https://allsvenskan.se/wp-json/wp/v2/posts?per_page=100&search=allsvenskan+{year}+match",
            f"https://allsvenskan.se/wp-json/custom/v1/matches?year={year}",
        ]
        
        for api_url in possible_apis:
            try:
                session = requests.Session()
                session.headers.update(self.headers)
                response = session.get(api_url, timeout=10)
                
                logger.debug(f"Testing API: {api_url} - Status: {response.status_code}")
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        # Check if the response contains meaningful match data
                        if self._validate_api_response(data, year):
                            logger.info(f"Found working API endpoint: {api_url}")
                            return api_url, data
                        else:
                            logger.debug(f"API response doesn't contain match data: {api_url}")
                    except json.JSONDecodeError:
                        logger.debug(f"Invalid JSON response from: {api_url}")
                        continue
                        
            except Exception as e:
                logger.debug(f"API endpoint {api_url} failed: {e}")
                continue
        
        logger.warning(f"No working API endpoint found for year {year}")
        return None, None

    def _validate_api_response(self, data, year):
        """Validate if API response contains meaningful match data"""
        try:
            # Check for various possible response structures
            if isinstance(data, list):
                # Direct list of matches
                if len(data) > 0 and isinstance(data[0], dict):
                    first_item = data[0]
                    # Look for match-like fields
                    match_fields = ['home', 'away', 'match', 'fixture', 'team']
                    if any(field in str(first_item).lower() for field in match_fields):
                        return True
            
            elif isinstance(data, dict):
                # Object with matches array
                for key in ['matches', 'fixtures', 'games', 'data']:
                    if key in data and isinstance(data[key], list) and len(data[key]) > 0:
                        return True
                
                # WordPress posts with match content
                if 'title' in data or 'content' in data:
                    content = str(data).lower()
                    if any(word in content for word in ['allsvenskan', 'match', 'vs', 'against']):
                        return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Error validating API response: {e}")
            return False

    def _scrape_year_api(self, year):
        """Scrape match data using the JSON API - more reliable approach"""
        try:
            # First try to discover the working API endpoint
            api_url, data = self._discover_api_endpoint(year)
            
            if api_url and data:
                logger.info(f"Successfully fetched JSON data for {year} from {api_url}")
                # Parse JSON data into match format
                matches = self._parse_json_data(data, year)
                return pd.DataFrame(matches) if matches else pd.DataFrame()
            else:
                logger.info(f"No working API endpoint found for {year}, falling back to HTML scraping")
                return self._scrape_year_html(year)
            
        except Exception as e:
            logger.error(f"Error in API scraping for year {year}: {e}")
            logger.debug(f"Full traceback: ", exc_info=True)
            return self._scrape_year_html(year)

    def _parse_json_data(self, data, year):
        """Parse JSON response from the API into match data"""
        matches = []
        
        try:
            # The API should return match data in a structured format
            if 'matches' in data:
                match_list = data['matches']
            elif isinstance(data, list):
                match_list = data
            else:
                logger.warning(f"Unexpected JSON structure for {year}: {list(data.keys())}")
                return matches
            
            for match in match_list:
                try:
                    # Extract match information from JSON
                    match_data = self._extract_match_from_json(match, year)
                    if match_data:
                        validation_errors = self.validate_match_data(match_data)
                        if not validation_errors:
                            matches.append(match_data)
                        else:
                            logger.debug(f"Match validation failed: {validation_errors}")
                            
                except Exception as e:
                    logger.error(f"Error parsing individual match: {e}")
                    continue
            
            logger.info(f"Parsed {len(matches)} matches from JSON data for {year}")
            
        except Exception as e:
            logger.error(f"Error parsing JSON data for {year}: {e}")
            
        return matches

    def _extract_match_from_json(self, match, year):
        """Extract match data from a single JSON match object"""
        try:
            # Map JSON fields to our match data structure
            # This will need to be adjusted based on actual API response format
            
            home_team = match.get('home_team', {}).get('name', '') or match.get('home', '')
            away_team = match.get('away_team', {}).get('name', '') or match.get('away', '')
            
            # Handle date parsing
            match_date = match.get('date', '') or match.get('match_date', '')
            if match_date:
                # Parse the date from various possible formats
                parsed_date = self._parse_api_date(match_date, year)
            else:
                parsed_date = f"{year}-01-01"  # Fallback date
            
            # Handle scores
            home_goals = match.get('home_goals')
            away_goals = match.get('away_goals')
            
            # Handle different score formats
            if home_goals is None and 'score' in match:
                score = match['score']
                if isinstance(score, str) and '-' in score:
                    parts = score.split('-')
                    if len(parts) == 2:
                        try:
                            home_goals = int(parts[0].strip())
                            away_goals = int(parts[1].strip())
                        except ValueError:
                            pass
            
            # Determine if this is a result or fixture
            is_result = home_goals is not None and away_goals is not None
            
            match_data = {
                'Date': parsed_date,
                'Venue': match.get('venue', 'Stadium'),
                'Match': f"{home_team} - {away_team}",
                'HomeTeam': home_team,
                'AwayTeam': away_team,
                'HomeGoals': home_goals,
                'AwayGoals': away_goals,
                'FTHG': home_goals,
                'FTAG': away_goals,
                'Summary': 'Result' if is_result else 'Fixture'
            }
            
            return match_data
            
        except Exception as e:
            logger.error(f"Error extracting match from JSON: {e}")
            return None

    def _parse_api_date(self, date_str, year):
        """Parse date from API response"""
        try:
            # Try different date formats that might be returned by the API
            date_formats = [
                '%Y-%m-%d',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M:%S',
                '%d/%m/%Y',
                '%m/%d/%Y',
                '%d.%m.%Y'
            ]
            
            for fmt in date_formats:
                try:
                    parsed = datetime.strptime(date_str[:len(fmt)], fmt)
                    return parsed.strftime('%Y-%m-%d')
                except ValueError:
                    continue
            
            # If no format works, return a default
            logger.warning(f"Could not parse date: {date_str}")
            return f"{year}-01-01"
            
        except Exception as e:
            logger.error(f"Error parsing API date {date_str}: {e}")
            return f"{year}-01-01"

    def _scrape_year_selenium(self, year):
        """Use Selenium to render JavaScript and extract match data"""
        if not SELENIUM_AVAILABLE:
            logger.warning("Selenium not available, falling back to static HTML scraping")
            return self._scrape_year_static_html(year)
        
        try:
            logger.info(f"Using Selenium to scrape year {year}")
            
            # Setup Chrome options for headless mode
            chrome_options = webdriver.ChromeOptions()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            
            # Initialize driver
            try:
                service = Service(ChromeDriverManager().install())
                driver = webdriver.Chrome(service=service, options=chrome_options)
            except Exception as e:
                logger.error(f"Failed to initialize Chrome driver: {e}")
                return self._scrape_year_static_html(year)
            
            try:
                # Navigate to the matches page
                url = self.base_url
                logger.info(f"Navigating to: {url}")
                driver.get(url)
                
                # Wait for page to load
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                # If we need a specific year other than current, try to set it
                if year != 2025:
                    try:
                        # Try to find year selector or modify URL parameters
                        current_url = driver.current_url
                        if "season" not in current_url:
                            new_url = f"{current_url}?season={year}"
                            driver.get(new_url)
                            time.sleep(3)  # Wait for new data to load
                    except Exception as e:
                        logger.debug(f"Could not set year {year}: {e}")
                
                # Wait for match data to load (adjust selector based on actual page structure)
                time.sleep(5)  # Give JavaScript time to load data
                
                # Get the rendered HTML
                page_source = driver.page_source
                soup = BeautifulSoup(page_source, 'html.parser')
                
                # Save debug content
                self.save_debug_content(page_source, year, "selenium_html")
                
                # Look for match data in the rendered page
                matches = self._extract_matches_from_selenium_html(soup, year)
                
                if not matches:
                    # Try to find match elements using different strategies
                    matches = self._extract_matches_with_javascript(driver, year)
                
                logger.info(f"Selenium extracted {len(matches)} matches for {year}")
                return pd.DataFrame(matches) if matches else pd.DataFrame()
                
            finally:
                driver.quit()
                
        except Exception as e:
            logger.error(f"Error in Selenium scraping for year {year}: {e}")
            logger.debug(f"Full traceback: ", exc_info=True)
            return self._scrape_year_static_html(year)

    def _extract_matches_from_selenium_html(self, soup, year):
        """Extract matches from Selenium-rendered HTML"""
        matches = []
        
        try:
            # Look for match elements that would be populated by JavaScript
            match_selectors = [
                'div[data-match]',
                'div[data-fixture]',
                'div[class*="match"]',
                'div[class*="fixture"]',
                'tr[class*="match"]',
                'li[class*="match"]',
                '.match-row',
                '.fixture-row',
                '.game-row',
                '.round-game',
                'article[class*="match"]'
            ]
            
            for selector in match_selectors:
                elements = soup.select(selector)
                if elements:
                    logger.debug(f"Found {len(elements)} elements with selector: {selector}")
                    
                    for elem in elements:
                        try:
                            # Try to extract match data from the element
                            match_data = self._parse_match_element_selenium(elem, year)
                            if match_data:
                                validation_errors = self.validate_match_data(match_data)
                                if not validation_errors:
                                    matches.append(match_data)
                                    logger.debug(f"Extracted match: {match_data['HomeTeam']} vs {match_data['AwayTeam']}")
                        except Exception as e:
                            logger.debug(f"Error parsing match element: {e}")
                            continue
                    
                    if matches:
                        break  # Found matches, no need to try other selectors
            
            # Also look for JSON data in script tags
            if not matches:
                scripts = soup.find_all('script')
                for script in scripts:
                    if script.string and any(keyword in script.string.lower() for keyword in ['match', 'fixture', 'game']):
                        try:
                            # Try to extract structured data
                            import json
                            script_content = script.string
                            
                            # Look for JSON arrays or objects
                            json_patterns = [
                                r'"matches":\s*\[.*?\]',
                                r'"fixtures":\s*\[.*?\]',
                                r'"games":\s*\[.*?\]',
                                r'\[.*?"homeTeam".*?\]'
                            ]
                            
                            for pattern in json_patterns:
                                json_match = re.search(pattern, script_content, re.DOTALL)
                                if json_match:
                                    try:
                                        json_str = json_match.group(0)
                                        if json_str.startswith('"'):
                                            # Extract just the array part
                                            array_match = re.search(r'\[.*?\]', json_str, re.DOTALL)
                                            if array_match:
                                                json_str = array_match.group(0)
                                        
                                        data = json.loads(json_str)
                                        for item in data:
                                            if isinstance(item, dict):
                                                processed_match = self._extract_match_from_json(item, year)
                                                if processed_match:
                                                    validation_errors = self.validate_match_data(processed_match)
                                                    if not validation_errors:
                                                        matches.append(processed_match)
                                        
                                        if matches:
                                            break
                                            
                                    except Exception as e:
                                        logger.debug(f"Error parsing JSON from script: {e}")
                                        continue
                        except Exception as e:
                            logger.debug(f"Error processing script tag: {e}")
                            continue
            
        except Exception as e:
            logger.error(f"Error extracting matches from Selenium HTML: {e}")
        
        return matches

    def _parse_match_element_selenium(self, element, year):
        """Parse a match element from Selenium-rendered HTML"""
        try:
            # Get all text content
            text = element.get_text(strip=True)
            
            # Look for data attributes
            home_team = element.get('data-home-team', '') or element.get('data-home', '')
            away_team = element.get('data-away-team', '') or element.get('data-away', '')
            match_date = element.get('data-date', '') or element.get('data-match-date', '')
            score = element.get('data-score', '')
            
            # If no data attributes, try to parse from text
            if not (home_team and away_team):
                home_team, away_team = self._extract_teams_from_line(text)
            
            if not score:
                home_goals, away_goals = self._extract_score_from_line(text)
            else:
                home_goals, away_goals = self._parse_score_string(score)
            
            # Parse date if available
            if match_date:
                parsed_date = self._parse_api_date(match_date, year)
            else:
                parsed_date = f"{year}-06-01"  # Default date
            
            if home_team and away_team:
                return {
                    'Date': parsed_date,
                    'Venue': 'Stadium',
                    'Match': f"{home_team} - {away_team}",
                    'HomeTeam': home_team,
                    'AwayTeam': away_team,
                    'HomeGoals': home_goals,
                    'AwayGoals': away_goals,
                    'FTHG': home_goals,
                    'FTAG': away_goals,
                    'Summary': 'Result' if home_goals is not None else 'Fixture'
                }
        except Exception as e:
            logger.debug(f"Error parsing match element: {e}")
            
        return None

    def _extract_matches_with_javascript(self, driver, year):
        """Try to extract matches using JavaScript execution"""
        matches = []
        
        try:
            # Try to execute JavaScript to get match data
            js_scripts = [
                "return window.matchData || window.fixtures || window.games || [];",
                "return document.querySelectorAll('[data-match]').length;",
                "return Array.from(document.querySelectorAll('.match, .fixture')).map(el => el.textContent);",
            ]
            
            for script in js_scripts:
                try:
                    result = driver.execute_script(script)
                    if result:
                        logger.info(f"JavaScript execution returned: {type(result)} with {len(result) if hasattr(result, '__len__') else 'unknown'} items")
                        
                        if isinstance(result, list):
                            for item in result[:10]:  # Limit to first 10 items
                                if isinstance(item, dict):
                                    processed_match = self._extract_match_from_json(item, year)
                                    if processed_match:
                                        validation_errors = self.validate_match_data(processed_match)
                                        if not validation_errors:
                                            matches.append(processed_match)
                                elif isinstance(item, str):
                                    # Try to parse text content
                                    home_team, away_team = self._extract_teams_from_line(item)
                                    if home_team and away_team:
                                        home_goals, away_goals = self._extract_score_from_line(item)
                                        match_data = {
                                            'Date': f"{year}-06-01",
                                            'Venue': 'Stadium',
                                            'Match': f"{home_team} - {away_team}",
                                            'HomeTeam': home_team,
                                            'AwayTeam': away_team,
                                            'HomeGoals': home_goals,
                                            'AwayGoals': away_goals,
                                            'FTHG': home_goals,
                                            'FTAG': away_goals,
                                            'Summary': 'Result' if home_goals is not None else 'Fixture'
                                        }
                                        validation_errors = self.validate_match_data(match_data)
                                        if not validation_errors:
                                            matches.append(match_data)
                        
                        if matches:
                            break
                            
                except Exception as e:
                    logger.debug(f"JavaScript execution failed: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error extracting matches with JavaScript: {e}")
        
        return matches

    def _parse_score_string(self, score):
        """Parse score string like '2-1' into home and away goals"""
        try:
            if '-' in score:
                parts = score.split('-')
                if len(parts) == 2:
                    home_goals = int(parts[0].strip())
                    away_goals = int(parts[1].strip())
                    return home_goals, away_goals
        except (ValueError, AttributeError):
            pass
        return None, None

    def _scrape_year_static_html(self, year):
        """Static HTML scraping method (original approach)"""
        try:
            # Use the base URL without year path since it causes 404
            year_url = self.base_url
            
            logger.info(f"Fetching static HTML from: {year_url}")
            
            # Try trafilatura first (bypasses some blocking)
            downloaded = None
            try:
                downloaded = trafilatura.fetch_url(year_url)
                if downloaded:
                    logger.info(f"Successfully fetched content using trafilatura for {year}")
            except Exception as e:
                logger.error(f"Trafilatura failed for {year}: {e}")
            
            # Fallback to requests with retry logic
            if not downloaded:
                for attempt in range(3):
                    try:
                        print(f"Attempt {attempt + 1} with requests for {year}")
                        session = requests.Session()
                        session.headers.update(self.headers)
                        response = session.get(year_url, timeout=30)
                        
                        if response.status_code == 200:
                            downloaded = response.text
                            print(f"Successfully fetched content using requests for {year}")
                            break
                        else:
                            print(f"HTTP {response.status_code} for {year}")
                            
                    except Exception as e:
                        print(f"Request attempt {attempt + 1} failed for {year}: {e}")
                        if attempt < 2:
                            time.sleep(2 ** attempt)  # Exponential backoff
            
            if not downloaded:
                print(f"Failed to download content for {year}")
                return pd.DataFrame()
            
            # Extract text content and parse
            text_content = trafilatura.extract(downloaded)
            
            # Log what trafilatura extracted
            if text_content:
                logger.debug(f"Trafilatura extracted {len(text_content)} characters")
            else:
                logger.warning("Trafilatura extracted no content")
            
            # Fallback: Parse HTML directly with BeautifulSoup
            soup = BeautifulSoup(downloaded, 'html.parser')
            
            # Look for embedded JSON data or structured content
            matches = self._extract_matches_from_html(soup, year)
            
            if not matches:
                # Last resort: use text extraction
                if not text_content:
                    text_content = soup.get_text()
                logger.info(f"Fallback to text parsing with {len(text_content)} characters")
                matches = self._parse_text_content(text_content, year)
            
            year_df = pd.DataFrame(matches) if matches else pd.DataFrame()
            print(f"Found {len(year_df)} matches for {year}")
            return year_df
            
        except Exception as e:
            logger.error(f"Error scraping static HTML for year {year}: {e}")
            logger.debug(f"Full traceback: ", exc_info=True)
            return pd.DataFrame()

    def _scrape_year_html(self, year):
        """Main HTML scraping method - tries Selenium first, then static HTML"""
        # Try Selenium first for JavaScript-rendered content
        if SELENIUM_AVAILABLE:
            selenium_result = self._scrape_year_selenium(year)
            if not selenium_result.empty:
                return selenium_result
        
        # Fallback to static HTML scraping
        return self._scrape_year_static_html(year)

    def _scrape_year(self, year):
        """Main scraping method with clear error handling for data source issues"""
        logger.info(f"Attempting to scrape match data for year {year}")
        
        # First try API discovery
        try:
            api_result = self._scrape_year_api(year)
            if not api_result.empty:
                logger.info(f"Successfully obtained {len(api_result)} matches from API for {year}")
                return api_result
        except Exception as e:
            logger.error(f"API scraping failed for {year}: {e}")
        
        # Try HTML/Selenium fallback
        try:
            html_result = self._scrape_year_html(year)
            if not html_result.empty:
                logger.info(f"Successfully obtained {len(html_result)} matches from HTML for {year}")
                return html_result
        except Exception as e:
            logger.error(f"HTML scraping failed for {year}: {e}")
        
        # If all methods fail, return empty DataFrame with clear error message
        logger.error(f"""
        ===== DATA SOURCE ISSUE =====
        Unable to extract real match data for {year} from allsvenskan.se
        
        REASON: The website uses JavaScript to load match data dynamically, 
        and the specific API endpoint structure is not publicly documented.
        
        ATTEMPTED METHODS:
        - API endpoint discovery (multiple endpoints tested)
        - Static HTML parsing
        - JavaScript rendering with Selenium
        
        SOLUTION NEEDED: 
        - Use browser developer tools to find the correct API endpoint
        - Or implement proper headless browser with correct selectors
        - Or use alternative data sources
        
        For now, the application will use sample data to demonstrate functionality.
        =============================
        """)
        
        return pd.DataFrame()  # Return empty DataFrame to trigger sample data
    
    def _parse_match_element(self, element, year=2025):
        """Parse individual match element"""
        try:
            # Extract text content
            text = element.get_text(strip=True) if hasattr(element, 'get_text') else str(element)
            
            # Look for score pattern (e.g., "3-1", "0-0")
            score_pattern = r'(\d+)\s*-\s*(\d+)'
            score_match = re.search(score_pattern, text)
            
            # Look for team names pattern
            team_pattern = r'([A-Za-zÅÄÖåäö\s]+?)\s+(?:\d+\s*-\s*\d+|\-)\s+([A-Za-zÅÄÖåäö\s]+)'
            team_match = re.search(team_pattern, text)
            
            if team_match:
                home_team = team_match.group(1).strip()
                away_team = team_match.group(2).strip()
                
                # Extract date if available
                date_pattern = r'(\d{1,2})\s+(\w+)'
                date_match = re.search(date_pattern, text)
                
                match_date = f"{year}-07-01"  # Default date with correct year
                if date_match:
                    day = date_match.group(1)
                    month = date_match.group(2)
                    match_date = self._parse_swedish_date(day, month, year)
                
                match_data = {
                    'Date': match_date,
                    'Venue': 'Unknown',
                    'Match': f"{home_team} - {away_team}",
                    'HomeGoals': None,
                    'AwayGoals': None,
                    'Summary': 'Match'
                }
                
                if score_match:
                    match_data['HomeGoals'] = int(score_match.group(1))
                    match_data['AwayGoals'] = int(score_match.group(2))
                
                return match_data
                
        except Exception as e:
            logger.error(f"Error in _parse_match_element: {e}")
            logger.debug(f"Full traceback: ", exc_info=True)
        
        return None
    
    def _parse_text_content(self, text_content, year=2025):
        """Parse matches from extracted text content with detailed logging"""
        logger.debug(f"Parsing text content of {len(text_content)} characters for year {year}")
        
        # Save raw content for inspection
        self.save_debug_content(text_content, year, "raw_content")
        
        matches = []
        
        if not text_content:
            logger.warning(f"No text content provided for year {year}")
            return matches
        
        lines = text_content.split('\n')
        current_round = None
        current_date = None
        current_venue = None
        
        logger.debug(f"Processing {len(lines)} lines of content")
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Check for round header (OMGÅNG X)
            round_match = re.search(r'OMGÅNG\s+(\d+)', line, re.IGNORECASE)
            if round_match:
                current_round = round_match.group(1)
                logger.debug(f"Found round: {current_round}")
                i += 1
                continue
            
            # Check for date (MÅNDAG 31 MARS, SÖNDAG 2 JUNI, etc.)
            date_match = re.search(r'(MÅNDAG|TISDAG|ONSDAG|TORSDAG|FREDAG|LÖRDAG|SÖNDAG)\s+(\d+)\s+(\w+)', line, re.IGNORECASE)
            if date_match:
                day = date_match.group(2)
                month_str = date_match.group(3)
                current_date = self._parse_swedish_date(day, month_str, year)
                logger.debug(f"Found date: {current_date} from '{line}'")
                i += 1
                continue
            
            # Check for venue (simple text line before match)
            if line and not re.search(r'\d', line) and '-' not in line and len(line) < 50 and not line.startswith('http'):
                current_venue = line
                logger.debug(f"Found venue: {current_venue}")
                i += 1
                continue
            
            # Check for match line with teams
            home_team, away_team = self._extract_teams_from_line(line)
            if home_team and away_team:
                logger.debug(f"Found teams: {home_team} vs {away_team}")
                
                # Look for scores in the same line first
                home_goals, away_goals = self._extract_score_from_line(line)
                
                # If no score in same line, check next few lines
                if home_goals is None and away_goals is None:
                    for j in range(i + 1, min(i + 4, len(lines))):
                        score_line = lines[j].strip()
                        home_goals, away_goals = self._extract_score_from_line(score_line)
                        if home_goals is not None:
                            break
                        
                        # Look for two separate numbers (home and away goals)
                        if re.match(r'^\d+$', score_line):
                            if home_goals is None:
                                home_goals = int(score_line)
                            elif away_goals is None:
                                away_goals = int(score_line)
                                break
                
                # Create match data
                match_data = {
                    'Date': current_date or f'{year}-07-01',
                    'Venue': current_venue or 'Stadium',
                    'Match': f"{home_team} - {away_team}",
                    'HomeTeam': home_team,
                    'AwayTeam': away_team,
                    'HomeGoals': home_goals,
                    'AwayGoals': away_goals,
                    'FTHG': home_goals,
                    'FTAG': away_goals,
                    'Round': current_round,
                    'Summary': 'Result' if home_goals is not None else 'Fixture'
                }
                
                # Validate match data
                validation_errors = self.validate_match_data(match_data)
                if validation_errors:
                    logger.warning(f"Validation errors for match {home_team} vs {away_team}: {validation_errors}")
                else:
                    matches.append(match_data)
                    logger.debug(f"Added match: {home_team} {home_goals if home_goals is not None else '?'}-{away_goals if away_goals is not None else '?'} {away_team}")
            
            i += 1
        
        logger.info(f"Parsed {len(matches)} matches for year {year}")
        self.save_debug_content(f"Final matches: {matches}", year, "parsed_matches")
        
        return matches
    

    
    def _create_sample_data(self):
        """Create sample data structure when scraping fails"""
        # This creates realistic sample data based on actual Allsvenskan teams
        teams = [
            "Malmö FF", "AIK", "Djurgårdens IF", "Hammarby IF", "IFK Göteborg",
            "BK Häcken", "IF Elfsborg", "IFK Norrköping", "Kalmar FF", "Halmstads BK",
            "Mjällby AIF", "IK Sirius FK", "Värnamo", "Degerfors IF", "Varbergs BoIS", "Östersunds FK"
        ]
        
        import random
        random.seed(42)
        
        matches = []
        
        # Generate some completed matches (results)
        for round_num in range(1, 16):  # 15 rounds completed
            round_matches = []
            team_pairs = teams.copy()
            random.shuffle(team_pairs)
            
            for i in range(0, len(team_pairs), 2):
                if i + 1 < len(team_pairs):
                    home_team = team_pairs[i]
                    away_team = team_pairs[i + 1]
                    
                    # Generate realistic scores
                    home_goals = random.choices([0, 1, 2, 3, 4], weights=[20, 35, 30, 12, 3])[0]
                    away_goals = random.choices([0, 1, 2, 3, 4], weights=[25, 40, 25, 8, 2])[0]
                    
                    match_date = f"2024-{3 + (round_num // 4):02d}-{(round_num % 4 * 7 + 1):02d}"
                    
                    matches.append({
                        'Date': match_date,
                        'Venue': f"{home_team} Stadium",
                        'Match': f"{home_team} - {away_team}",
                        'HomeGoals': home_goals,
                        'AwayGoals': away_goals,
                        'Summary': 'Result'
                    })
        
        # Generate remaining fixtures (without results)
        for round_num in range(16, 31):  # Remaining rounds
            team_pairs = teams.copy()
            random.shuffle(team_pairs)
            
            for i in range(0, len(team_pairs), 2):
                if i + 1 < len(team_pairs):
                    home_team = team_pairs[i]
                    away_team = team_pairs[i + 1]
                    
                    match_date = f"2024-{7 + (round_num // 4):02d}-{(round_num % 4 * 7 + 1):02d}"
                    
                    matches.append({
                        'Date': match_date,
                        'Venue': f"{home_team} Stadium",
                        'Match': f"{home_team} - {away_team}",
                        'HomeGoals': None,
                        'AwayGoals': None,
                        'Summary': 'Fixture'
                    })
        
        return pd.DataFrame(matches)
