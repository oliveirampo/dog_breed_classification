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
    # print(image_name)
    # sys.exit(123)

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
    # user a larger number if you want to download more images
    n_scroll = 20
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


def download(keyword_list, n_images, train_directory, validation_directory, test_directory, chromedriver):
    """Download images of dogs.

    :param keyword_list: (arr) List with dog breeds.
    :param n_images: (int) Maximum number of images to save.
    :param train_directory: (str) Directory for training images.
    :param validation_directory: (str) Directory for validation images.
    :param test_directory: (str) Directory for testing images.
    :param chromedriver:
    :return:
    """

    if not os.path.exists(train_directory):
        os.mkdir(train_directory)

    # word = keyword_list[0]
    for word in keyword_list:
        search_google(word, n_images, train_directory, chromedriver)

    # create test and validation directories
    if not os.path.exists(test_directory):
        os.mkdir(test_directory)
    if not os.path.exists(validation_directory):
        os.mkdir(validation_directory)

    for word in keyword_list:
        get_test_and_validation_images(word, n_images, train_directory, validation_directory, test_directory)


def get_test_and_validation_images(word, n_images, train_directory, validation_directory, test_directory):
    """Move some of the images downloaded to train directory into test and validation directories."""

    src_directory = '{}/{}'.format(train_directory, word.replace(' ', '_'))
    image_name = get_image_name(n_images - 1, src_directory)
    if not os.path.exists(image_name):
        print('Missing image: {}'.format(image_name))
        sys.exit(1)

    n_train = n_images // 2
    n_test = n_images // 6

    # the first n_test images will be used to test the model
    test_images = ['img_{}.jpg'.format(i) for i in range(n_test)]
    validation_images = ['img_{}.jpg'.format(i) for i in range(n_test, n_train)]

    move_images(word.replace(' ', '_'), train_directory, test_directory, test_images)
    # move_images(word.replace(' ', '_'), train_directory, validation_directory, validation_images)


def move_images(word, train_directory, directory, image_list):
    """Move image in list from train to test/validation directory."""

    src_directory = '{}/{}'.format(train_directory, word)
    trg_directory = '{}/{}'.format(directory, word)

    if not os.path.exists(trg_directory):
        os.mkdir(trg_directory)

    for image in image_list:
        src_image = '{}/{}'.format(src_directory, image)
        trg_image = '{}/{}'.format(trg_directory, image)

        if not os.path.exists(src_image):
            continue

        os.rename(src_image, trg_image)


def test():
    dog_breed_file_name = '../data/inp/dog_breed.txt'
    dog_breed_list = read_data(dog_breed_file_name)

    n_images = 10
    train_directory = '../data/dogs/train'
    validation_directory = '../data/dogs/validation'
    test_directory = '../data/dogs/test'
    chromedriver = '../data/exe/chromedriver_linux64'

    download(dog_breed_list[:1], n_images, train_directory, validation_directory, test_directory, chromedriver)


if __name__ == '__main__':
    test()
