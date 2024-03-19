import cv2
import numpy as np
path_to_image = 'sample_data/'
path_to_dataset = '/home/uy/Tensorflow/scripts/generate_data/vietnamese_currency/generated_data/test/'

# generate a white plain of size plain_height x plain_width
plain_height = 1080
plain_width = 1920
currency = ['200','500','1000', '2000', '5000', '10000', '20000', '50000', '100000', '200000', '500000']
test_size = 75

def xml_initiate(file, name, path):
    file.write('<annotation>\n')
    file.write('\t<folder>generated_data</folder>\n')
    file.write('\t<filename>' + name + '</filename>\n')
    file.write('\t<path>' + path + '</path>\n')
    file.write('\t<source>\n')
    file.write('\t\t<database>Unknown</database>\n')
    file.write('\t</source>\n')
    file.write('\t<size>\n')
    file.write('\t\t<width>' + str(plain_width) + '</width>\n')
    file.write('\t\t<height>' + str(plain_height) + '</height>\n')
    file.write('\t\t<depth>3</depth>\n')
    file.write('\t</size>\n')
    file.write('\t<segmented>0</segmented>\n')

def xml_object(file, x, y, w, h, name):
    file.write('\t<object>\n')
    file.write('\t\t<name>' + name + '</name>\n')
    file.write('\t\t<pose>Unspecified</pose>\n')
    file.write('\t\t<truncated>0</truncated>\n')
    file.write('\t\t<difficult>0</difficult>\n')
    file.write('\t\t<bndbox>\n')
    file.write('\t\t\t<xmin>' + str(x) + '</xmin>\n')
    file.write('\t\t\t<ymin>' + str(y) + '</ymin>\n')
    file.write('\t\t\t<xmax>' + str(x+w) + '</xmax>\n')
    file.write('\t\t\t<ymax>' + str(y+h) + '</ymax>\n')
    file.write('\t\t</bndbox>\n')
    file.write('\t</object>\n')

def close_xml(file):
    file.write('</annotation>\n')
    file.close()


for index in range(0, test_size):
    path = path_to_dataset + str(index) + ".xml"
    label = open(path, "w")
    name = str(index) + ".png"
    xml_initiate(label, name, path)

    img = 255 * np.ones(shape=[plain_height, plain_width, 3], dtype=np.uint8)

    # iterate through the list of currency
    for i in currency:
        id = np.random.randint(1, 3)
        row = np.random.randint(0, plain_height)
        col = np.random.randint(0, plain_width)
        # read the image
        img1 = cv2.imread(path_to_image + i + '/' + str(id) + '.png')

        # overwrite the image into the white plain in the random position
        height, width, channels = img1.shape

        # if the image is too big, crop it
        height = min(height, plain_height - row)
        width = min(width, plain_width - col)
        # crop the image
        img1 = img1[0:height, 0:width]

        img[row:row+height, col:col+width] = img1

        # write the xml file
        xml_object(label, col, row, width, height, i)
    # save the image
    close_xml(label)
    cv2.imwrite(path_to_dataset + name, img)
