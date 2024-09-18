from httpcore import TimeoutException
import requests
from bs4 import BeautifulSoup
import os
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from src.modules.paths_reference import ROOT_PATH
import time
import random

def esperar_aleatoriamente(min_segundos=2, max_segundos=5):
    time.sleep(random.uniform(min_segundos, max_segundos))

path_articulos_directory = ROOT_PATH /"data" / "articulos"
def extraer_articulos(driver,user_agents):
    total_articles = 0
    current_page = 1

    while True:
        print(f"Procesando página {current_page}")

        # Esperar a que los artículos se carguen
        try:
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'ultimate-layouts-item')))
        except TimeoutException:
            print(f"Tiempo de espera agotado en la página {current_page}. Finalizando la extracción.")
            break

        # Obtener el contenido de la página
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        articles = soup.find_all('article', class_='ultimate-layouts-item')

        print(f"Se encontraron {len(articles)} publicaciones en la página {current_page}.\n")

        for article in articles:
            total_articles += 1
            link = article.find('a', class_='ultimate-layouts-title-link')['href']
            print(f"Enlace a la publicación: {link}")

            esperar_aleatoriamente(1, 3)

            headers = {'User-Agent': random.choice(user_agents)}
            article_response = requests.get(link, headers=headers)
            article_soup = BeautifulSoup(article_response.content, 'html.parser')

            title = article_soup.find('div', class_='entry-title').get_text(strip=True) if article_soup.find('div',
                                                                                                             class_='entry-title') else "Título no encontrado"

            content_div = article_soup.find('div', class_='entry-content')
            if content_div:
                for script in content_div(["script", "style"]):
                    script.decompose()
                content = content_div.get_text(separator="\n", strip=True)
            else:
                content = "Contenido no encontrado"

            filename = re.sub(r'[^\w\-_\. ]', '_', title)
            filename = f"articulo_{total_articles}_{filename[:50]}.txt"

            with open(path_articulos_directory/filename, "w", encoding='utf-8') as file:
                file.write(f"Título: {title}\n\n")
                file.write(f"Enlace: {link}\n\n")
                file.write(f"Contenido:\n{content}")

            print(f"Artículo guardado: {filename}")
            print("=" * 80)

        # Buscar la siguiente página
        try:
            next_page = current_page + 1
            next_page_link = driver.find_element(By.CSS_SELECTOR, f'.paginationjs-page[data-num="{next_page}"]')
            if next_page_link:
                next_page_link.click()
                current_page = next_page
                time.sleep(2)  # Esperar a que la nueva página cargue
            else:
                print("No se encontró el enlace a la siguiente página. Finalizando la extracción.")
                break
        except NoSuchElementException:  # type: ignore
            print("No hay más páginas para navegar. Finalizando la extracción.")
            break

    print(f"Total de artículos extraídos: {total_articles}")

