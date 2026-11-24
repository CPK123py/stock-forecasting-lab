import csv
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

def scrape_column_content(url, output_file):
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")  # Start with browser maximized
    chrome_options.add_argument("--disable-notifications")  # Disable notifications
    
    # Initialize the WebDriver
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        # Navigate to the URL
        driver.get(url)
        
        # Wait for the page to load completely
        time.sleep(5)  # Give some time for JavaScript to load content
        
        # Wait for elements to be present
        wait = WebDriverWait(driver, 10)
        
        # Try multiple possible selectors for the news items
        try:
            # First attempt with your original selector
            news_elements = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.sc-krqipW.ifgLUz")))
        except TimeoutException:
            try:
                # Try a more general selector that might capture news items
                news_elements = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.news-item")))
            except TimeoutException:
                # Try to find elements by examining the page structure
                print("Attempting to find elements by examining page structure...")
                # Print the page source to debug
                print(driver.page_source[:1000])  # Print first 1000 chars for debugging
                
                # Try to find any div that might contain the news items
                news_elements = driver.find_elements(By.TAG_NAME, "div")
                # Filter to those that might be news items (having certain text or structure)
                news_elements = [elem for elem in news_elements if elem.text and len(elem.text) > 10]
        
        # Extract the content
        column_content = []
        print(f"Found {len(news_elements)} potential news elements")
        
        for element in news_elements:
            try:
                # Try to get the text directly
                text = element.text
                if text:
                    column_content.append(text)
                    print(f"Extracted: {text[:50]}...")  # Print first 50 chars for debugging
                
                # If that doesn't work well, try to find child elements
                if not text:
                    children = element.find_elements(By.XPATH, "./*")
                    if len(children) > 1:
                        second_child = children[1]
                        text = second_child.text
                        if text:
                            column_content.append(text)
                            print(f"Extracted from child: {text[:50]}...")
            except Exception as e:
                print(f"Error extracting text: {e}")
        
        # Save to CSV
        with open(output_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Scraped Content"])  # Column header
            for row in column_content:
                writer.writerow([row])
        
        print(f"✅ Data saved to '{output_file}'")
        print(f"Total items extracted: {len(column_content)}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.quit()

# Example usage
url = "https://www.zomato.com/investor-relations/news-and-events"
output_file = "zomato_news.csv"
scrape_column_content(url, output_file)