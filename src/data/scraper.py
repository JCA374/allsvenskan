import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import trafilatura
from datetime import datetime
import time

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
            year_url = f"{self.base_url}/{year}/"
            print(f"Fetching from: {year_url}")
            
            # Try trafilatura first (bypasses some blocking)
            downloaded = None
            try:
                downloaded = trafilatura.fetch_url(year_url, include_comments=False, include_tables=True)
                if downloaded:
                    print(f"Successfully fetched content using trafilatura for {year}")
            except Exception as e:
                print(f"Trafilatura failed for {year}: {e}")
            
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
            text_content = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
            if not text_content:
                # Fallback to BeautifulSoup extraction
                soup = BeautifulSoup(downloaded, 'html.parser')
                text_content = soup.get_text()
            
            print(f"Extracted {len(text_content)} characters of text for {year}")
            
            # Parse the Swedish format content
            matches = self._parse_text_content(text_content, year)
            
            year_df = pd.DataFrame(matches) if matches else pd.DataFrame()
            print(f"Found {len(year_df)} matches for {year}")
            return year_df
            
        except Exception as e:
            print(f"Error scraping data for year {year}: {e}")
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
            pass
        
        return None
    
    def _parse_text_content(self, text_content, year=2025):
        """Parse matches from extracted text content in Swedish format"""
        matches = []
        
        if not text_content:
            return matches
        
        lines = text_content.split('\n')
        current_round = None
        current_date = None
        current_venue = None
        
        # Swedish month mapping
        swedish_months = {
            'januari': '01', 'februari': '02', 'mars': '03', 'april': '04',
            'maj': '05', 'juni': '06', 'juli': '07', 'augusti': '08',
            'september': '09', 'oktober': '10', 'november': '11', 'december': '12'
        }
        
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
                i += 1
                continue
            
            # Check for date (MÅNDAG 31 MARS, SÖNDAG 2 JUNI, etc.)
            date_match = re.search(r'(MÅNDAG|TISDAG|ONSDAG|TORSDAG|FREDAG|LÖRDAG|SÖNDAG)\s+(\d+)\s+(\w+)', line, re.IGNORECASE)
            if date_match:
                day = date_match.group(2)
                month_str = date_match.group(3).lower()
                month = swedish_months.get(month_str, '07')
                current_date = f"{year}-{month}-{day.zfill(2)}"
                i += 1
                continue
            
            # Check for venue (simple text line before match)
            if line and not re.search(r'\d', line) and '-' not in line and len(line) < 50:
                current_venue = line
                i += 1
                continue
            
            # Check for match line with teams and score
            # Format: "Team1 - Team2" followed by score on next lines
            if ' - ' in line and not line.startswith('http'):
                teams_line = line
                
                # Extract teams
                teams_parts = teams_line.split(' - ')
                if len(teams_parts) == 2:
                    home_team = teams_parts[0].strip()
                    away_team = teams_parts[1].strip()
                    
                    # Look for scores in next few lines
                    home_goals = None
                    away_goals = None
                    
                    # Check next few lines for score pattern
                    for j in range(i + 1, min(i + 4, len(lines))):
                        score_line = lines[j].strip()
                        
                        # Look for two separate numbers (home and away goals)
                        if re.match(r'^\d+$', score_line):
                            if home_goals is None:
                                home_goals = int(score_line)
                            elif away_goals is None:
                                away_goals = int(score_line)
                                break
                        
                        # Look for "X Y" pattern on same line
                        score_match = re.search(r'^(\d+)\s+(\d+)$', score_line)
                        if score_match:
                            home_goals = int(score_match.group(1))
                            away_goals = int(score_match.group(2))
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
                        'Summary': 'Match' if home_goals is not None else 'Fixture'
                    }
                    
                    matches.append(match_data)
            
            i += 1
        
        return matches
    
    def _parse_swedish_date(self, day, month_str, year=2025):
        """Convert Swedish month names to dates"""
        swedish_months = {
            'januari': '01', 'februari': '02', 'mars': '03', 'april': '04',
            'maj': '05', 'juni': '06', 'juli': '07', 'augusti': '08',
            'september': '09', 'oktober': '10', 'november': '11', 'december': '12'
        }
        
        month = swedish_months.get(month_str.lower(), '07')
        return f"{year}-{month}-{day.zfill(2)}"
    
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
