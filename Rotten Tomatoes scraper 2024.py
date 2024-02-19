from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time

# Set up the WebDriver (assuming you have Chrome WebDriver installed)
driver = webdriver.Chrome()

# Navigate to the desired webpage
driver.get("https://www.rottentomatoes.com/m/the_boy_and_the_heron/reviews")
driver.implicitly_wait(10)


    # Click the "Load More" button until it disappears
try:
    while True:
        # find the button:
        load_more_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "rt-button")))
        # click button:
        driver.execute_script("arguments[0].click();", load_more_button)
        time.sleep(1)
except:
    pass
    
# Find all review elements
review_elements = driver.find_element(By.CLASS_NAME, 'reviews-container')

data = []

# extracting lists of reviews and states
reviews_text = review_elements.find_elements(By.CLASS_NAME, 'review-text')
reviews_state = review_elements.find_elements(By.CSS_SELECTOR, 'score-icon-critic-deprecated')
reviews_date = review_elements.find_elements(By.XPATH, '//*[@id="reviews"]/div[1]/div[*]/div[2]/p[2]/span')

t = (reviews_text, reviews_state, reviews_date)

for rev, st, dt in zip(*t):
    data.append((rev.text, st.get_attribute("state"), dt.text))

df = pd.DataFrame(data, columns = ['Review', 'State', 'Date'])
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year

df.to_excel("Movie reviews.xlsx")

# Close the browser = web driver
driver.quit()

