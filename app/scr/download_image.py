from selenium.webdriver.common.keys import Keys
from sklearn.datasets import load_files
from selenium import webdriver
from bs4 import BeautifulSoup
import requests
import certifi
import time
import sys
import os


def read_data(file_name):
    """Convert input file with dog breeds into list.

    :param file_name: (str) Name of file with list of dog breeds.
    :return: lines: (arr) List with dog breeds.
    """

    with open(file_name, 'r') as file:
        lines = file.readlines()

    lines = [row.strip().lower() for row in lines]
    return lines


def search_google(word, n_images, directory, chromedriver):
    """Search for keywords in Google.

    :param word: (str) Keyword to search for.
    :param n_images: (int) Maximum number of images to save.
    :param directory: (str) Directory where images are saved.
    :param chromedriver:
    :return:
    """

    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    ca_certs = certifi.where()

    output_directory = '{}/{}'.format(directory, word.replace(' ', '_'))
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    image_name = get_image_name(n_images - 1, output_directory)
    if os.path.exists(image_name):
        return
    # sys.exit('STOP')

    searchurl = 'https://www.google.com/search?q=\"{}\"&source=lnms&tbm=isch'.format(word.replace(' ', '+'))
    # print(searchurl)

    try:
        browser = webdriver.Chrome(chromedriver, options=options)
    except Exception as e:
        print(f'chromedriver not found in this environment.')
        print(f'Install on your machine. exception: {e}')
        sys.exit()

    browser.set_window_size(224, 224)
    browser.get(searchurl)
    time.sleep(1)

    print(f'Getting you a lot of images. This may take a few moments...')

    element = browser.find_element_by_tag_name('body')

    # Scroll down
    n_scroll = 10
    scroll_down(n_scroll, n_scroll, element, browser)

    print(f'Reached end of page.')
    time.sleep(0.5)
    print(f'Retry')
    time.sleep(0.5)

    # Scroll down 2
    scroll_down(n_scroll, n_scroll, element, browser)

    page_source = browser.page_source

    soup = BeautifulSoup(page_source, 'lxml')
    images = soup.find_all('img')

    urls = []
    for image in images:
        try:
            url = image['data-src']
            if not url.find('https://'):
                urls.append(url)
        except:
            try:
                url = image['src']
                if not url.find('https://'):
                    urls.append(image['src'])
            except Exception as e:
                print(f'No found image sources.')
                print(e)

    count = 0
    if urls:
        for url in urls:
            try:
                res = requests.get(url, stream=True, verify=ca_certs)
                rawdata = res.raw.read()

                image_name = get_image_name(count, output_directory)
                with open(image_name, 'wb') as f:
                    f.write(rawdata)
                    count += 1

                    if count == n_images:
                        break

            except Exception as e:
                print('Failed to write rawdata.')
                print(e)

    browser.close()


def scroll_down(n1, n2, element, browser):
    """Scroll down web page."""

    for i in range(n1):
        element.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.3)

    try:
        browser.find_element_by_id('smb').click()
        for i in range(n1):
            element.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.3)
    except:
        for i in range(n2):
            element.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.3)


def get_image_name(count, output_directory):
    """Define name of image according to count index."""

    image_name = os.path.join(output_directory, 'img_' + str(count) + '.jpg')
    return image_name


def download(keyword_list, n_images, output_directory, chromedriver):
    """Download images of dogs.

    :param keyword_list: (arr) List with dog breeds.
    :param n_images: (int) Maximum number of images to save.
    :param output_directory: (str) Directory where images are saved.
    :param chromedriver:
    :return:
    """

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    # word = keyword_list[0]
    for word in keyword_list:
        search_google(word, n_images, output_directory, chromedriver)


def test():
    dog_breed_file_name = '../data/inp/dog_breed.txt'
    dog_breed_list = read_data(dog_breed_file_name)

    n_images = 10
    output_directory = '../data/dogs/train'
    chromedriver = '../data/exe/chromedriver'

    download(dog_breed_list[:1], n_images, output_directory, chromedriver)


if __name__ == '__main__':
    test()
