import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import trafilatura
from datetime import datetime
import time
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AllsvenskanScraper:
    def __init__(self):
        self.base_url = "https://allsvenskan.se/matcher"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
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
        
        # Check if year is current year (use default matcher page)
        if year == 2025:
            year_url = self.base_url
        else:
            # For past years, try different URL patterns
            year_url = f"{self.base_url}?season={year}"

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
    
    def _scrape_year(self, year):
        """Scrape match data for a specific year with better error handling"""
        try:
            # Check if year is current year (use default matcher page)
            if year == 2025:
                year_url = self.base_url
            else:
                # For past years, try different URL patterns
                year_url = f"{self.base_url}?season={year}"
            
            logger.info(f"Fetching from: {year_url}")
            
            # Save debug content
            self.save_debug_content(f"Attempting to scrape year {year} from {year_url}", year, "start")
            
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
            logger.error(f"Error scraping data for year {year}: {e}")
            logger.debug(f"Full traceback: ", exc_info=True)
            return pd.DataFrame()
    
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
