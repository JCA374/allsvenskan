# Web Scraper Fix Guide

## Issues Identified

### 1. **Date Parsing Problems**
- **Issue**: Swedish date parsing is inconsistent and has multiple fallback patterns
- **Problem**: Default dates (`{year}-07-01`) are being used too often
- **Impact**: All matches getting assigned the same default date

### 2. **Score Extraction Issues**
- **Issue**: Score parsing logic is overly complex and unreliable
- **Problem**: Looking for scores in multiple formats across multiple lines
- **Impact**: Missing scores or extracting wrong values

### 3. **Regex Pattern Issues**
- **Issue**: Multiple conflicting regex patterns for team names and scores
- **Problem**: Patterns are too broad or too narrow
- **Impact**: False positives or missed matches

### 4. **Error Handling Problems**
- **Issue**: Silent failures with `except Exception as e: pass`
- **Problem**: Errors are ignored without logging
- **Impact**: Can't debug what's going wrong

## Debugging Steps

### Step 1: Enable Debug Logging

Add comprehensive logging to see what's happening:

```python
import logging

# Add at top of scraper.py
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def _parse_text_content(self, text_content, year=2025):
    """Parse matches with detailed logging"""
    logger.debug(f"Parsing text content of {len(text_content)} characters")

    # Save raw content for inspection
    with open(f'debug_content_{year}.txt', 'w', encoding='utf-8') as f:
        f.write(text_content)
    logger.debug(f"Raw content saved to debug_content_{year}.txt")
```

### Step 2: Add Content Inspection

Create a debug function to examine the actual HTML structure:

```python
def debug_content_structure(self, year=2025):
    """Debug function to examine content structure"""
    year_url = f"{self.base_url}/{year}/"

    # Get raw HTML
    response = requests.get(year_url, headers=self.headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all potential match containers
    potential_matches = soup.find_all(['div', 'span', 'td'], 
                                     text=re.compile(r'\d+\s*-\s*\d+'))

    print(f"Found {len(potential_matches)} potential score elements")
    for i, elem in enumerate(potential_matches[:5]):  # Show first 5
        print(f"Element {i}: {elem.get_text().strip()}")
        print(f"Parent: {elem.parent.name if elem.parent else 'None'}")
        print(f"Context: {elem.parent.get_text()[:100] if elem.parent else 'None'}")
        print("---")
```

### Step 3: Fix Date Parsing

Replace the complex date parsing with a simpler, more robust approach:

```python
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
```

### Step 4: Simplify Score Extraction

Create a more focused approach to finding scores:

```python
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
```

### Step 5: Add Validation Functions

Create validation to catch errors early:

```python
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
```

## Quick Fixes

### Fix 1: Replace Silent Exception Handling

Replace all instances of:
```python
except Exception as e:
    pass
```

With:
```python
except Exception as e:
    logger.error(f"Error in function_name: {e}")
    logger.debug(f"Full traceback: ", exc_info=True)
```

### Fix 2: Add Content Saving for Debugging

```python
def save_debug_content(self, content, year, stage):
    """Save content at different processing stages"""
    filename = f"debug_{stage}_{year}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(str(content))
    logger.debug(f"Saved debug content to {filename}")
```

### Fix 3: Implement Fallback Strategies

```python
def scrape_with_fallbacks(self, year):
    """Try multiple scraping strategies"""
    strategies = [
        self._scrape_with_trafilatura,
        self._scrape_with_requests,
        self._scrape_with_selenium,  # If available
    ]

    for strategy in strategies:
        try:
            result = strategy(year)
            if not result.empty:
                logger.info(f"Successfully scraped {year} with {strategy.__name__}")
                return result
        except Exception as e:
            logger.warning(f"{strategy.__name__} failed for {year}: {e}")
            continue

    logger.error(f"All scraping strategies failed for {year}")
    return pd.DataFrame()
```

## Testing Recommendations

### 1. Test with Known Data
Create a test with hardcoded Swedish match data to verify parsing logic.

### 2. Test Individual Functions
Test each parsing function separately:
- Date parsing with various Swedish date formats
- Team name extraction with different input formats  
- Score extraction with different score presentations

### 3. Validate Output
Add assertions to check:
- All dates are valid and in correct year
- No duplicate matches
- Team names are reasonable (not empty, not too long)
- Scores are non-negative integers

## Immediate Action Items

1. **Enable debug logging** and run scraper to see what content is actually being extracted
2. **Save raw content** to files for manual inspection
3. **Test date parsing** with sample Swedish dates
4. **Validate all extracted data** before saving
5. **Implement gradual fallbacks** rather than all-or-nothing approaches

Run these debugging steps first, then we can create more specific fixes based on what the logs reveal.