from tkinter import * #tkinterx library
import tkinter.font as tkFont
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import filedialog
import os
import cv2  # opencv-python
from sklearn.cluster import KMeans  # scikit-learn
import glob
import numpy as np
import pandas as pd  # Pandas requires numpy to run so modules must be imported in this order.
import matplotlib.pyplot as plt
import re
import warnings
warnings.filterwarnings('ignore')  # Avoid printing Python warnings, set ignore as a parameter

def Spacer():
    txt.insert(END, "\n------------------------------------------------------------------------------------------------------------------------------------------------\n")
    txt.update()
    txt.see("end")
    return

def rgb_to_hsv(dominant_color):
    # This function converts the dominant colour from RGB to HSV.
    r = dominant_color[0]/255
    g = dominant_color[1]/255
    b = dominant_color[2]/255
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100

    dominant_color = [h, s, v]
    return dominant_color

def Image_Dimensions_Check():
    global crop_top_on # To change a global varaible inside another function you must re-declare the variable as global first
    global crop_bottom_on
    global crop_left_on
    global crop_right_on

    # Check whether the crop settings of light ON and OFF are different.
    if (crop_top_off != crop_top_on) or (crop_bottom_off != crop_bottom_on) or (crop_left_on != crop_left_on) or (crop_right_off != crop_right_on):
        crop_top_on = crop_top_off
        crop_bottom_on = crop_bottom_off
        crop_left_on = crop_left_off
        crop_right_on = crop_right_off

        txt.insert(END, "\nWARNING: the dimension of the images taken with light OFF and ON are different. This makes it difficult to create a video.\n"
                        "Only the cropping parameters used for the images taken with light OFF will be applied.\n", "warning")
        txt.update()
        txt.see("end")

    return

def Find_Dominant_Color(filename, img, results_list):
    # reshape the image data into a 2D array, where each row represents a pixel and the columns represent the red, green, and blue channels.
    pixels = img.reshape(-1, 3)  # -1 is to transform the 64x64 list into a single list 64x64 = 4096 long

    # Use K-Means clustering to identify the most dominant colors. The number of clusters will be the number of colors we want to identify.
    kmeans = KMeans(n_clusters=5) # 5 clusters: it is just because it works well.
    kmeans.fit(pixels)

    # The most dominant color is the centroid of the cluster with the most points (pixels). We’ll identify this color and convert it to an 8-bit RGB value.
    counts = np.bincount(kmeans.labels_)
    dominant_color = kmeans.cluster_centers_[np.argmax(counts)]
    dominant_color = dominant_color.round(0).astype(int)

    # Convert RGB colour to HSV. In order ot identify if the photo was taken with light on or off, the HUE is more sensitive than the green value of RGB and the kmeans process will more robust.
    dominant_color = rgb_to_hsv(dominant_color)

    # Generate results and append to a list
    result = [filename, dominant_color]
    results_list.append(result)
    return results_list

def Identify_Light_OnOff_Kmeans(results_list):
    # Save all dominant colors in a list to apply KMeans
    dominant_color_values = []
    for result in results_list:
        dominant_color_values.append(result[1]) # Use the HSV color

    dominant_color_values = np.array(dominant_color_values)  # transform list into numpy array

    # KMeans on list of resutls to identify images with light on or off
    kmeans = KMeans(n_clusters=2, algorithm='elkan')  # We try to identify 2 clusters, on and off.
    kmeans.fit(dominant_color_values)

    # Scatter plot to see each cluster points visually
    labels = kmeans.predict(dominant_color_values)  # Getting the cluster labels
    centroids = kmeans.cluster_centers_
    fig = plt.figure(figsize=(10, 10))
    #ax = fig.gca(projection='3d') # Line OK with Python 3.7
    ax = fig.add_subplot(projection='3d') # Line OK with Python 3.11
    x = np.array(labels == 0)
    y = np.array(labels == 1)
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c="black", s=150, label="Centers", alpha=1)
    ax.scatter(dominant_color_values[x, 0], dominant_color_values[x, 1], dominant_color_values[x, 2], c="blue", s=40, label="C1")
    ax.scatter(dominant_color_values[y, 0], dominant_color_values[y, 1], dominant_color_values[y, 2], c="green", s=40, label="C2")
    ax.set(xlabel="H channel", ylabel="S channel", zlabel="V channel")
    plt.savefig("Output/Results KMeans plot.png", format="png", dpi=300)
    #plt.show()
    plt.close() # Always close figure otherwise you run out of RAM memory!

    cluster_results = kmeans.labels_[:]  # Returns a 1D array with the results of the 2 clusters numered 0 (light off) and 1 (light on). NOTE: 0 and 1 labels switch randomly!
    # Merge all results in a single array and print it.
    i = 0
    for result in results_list:
        result.append(cluster_results[i])
        i += 1

    # Create 2 arrays of hue channels based on KMeans clustering results
    green_channel_array0 = []
    green_channel_array1 = []
    for result in results_list:
        if result[2] == 0:  # This checks the clustering result of KMeans
            green_channel_array0.append(result[1][0]) # Append the hue value
        else:
            green_channel_array1.append(result[1][0]) # Append the hue value

    # Get the average of the values of the 2 arrays and determine which one is the greeneset, and attribute to the greenest the "light on" label
    green_channel_array0 = np.array(green_channel_array0)
    green_channel_array1 = np.array(green_channel_array1)
    num_images_on = 0
    num_images_off = 0
    if np.mean(green_channel_array0) < np.mean(green_channel_array1):  # 0 = light on
        for result in results_list:
            if result[2] == 0:
                result[2] = "on"
                num_images_on +=1
            else:
                result[2] = "off"
                num_images_off +=1
    else:  # 1 = light on
        for result in results_list:
            if result[2] == 1:
                result[2] = "on"
                num_images_on +=1
            else:
                result[2] = "off"
                num_images_off +=1

    return results_list, num_images_on, num_images_off

def PCM_Analysis(i, filename, img, mm_per_pixel, results_list, results_PCM_Areas):
    #Read if image was carried out with light on or off and perform analysis with proper parameters.
    # ---------------------GREEN LIGHT---------------------#
    if results_list[i][2] == "on":
        # Crop image
        img = img[crop_top_on:crop_bottom_on, crop_left_on:crop_right_on]  # First pair of numbers is for vertical starting from teh top; second pair of numbers is for horizontal starting from the left

        # Adjust contrast (alpha) and brightness (beta)
        # define the alpha and beta
        alpha = alpha_on  # Contrast control
        beta = beta_on  # Brightness control
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # Transform in B&W
        grey = cv2.cvtColor(adjusted, cv2.COLOR_RGB2GRAY)

        # Threshold image
        _, thresh = cv2.threshold(grey, threshold_on, 255, cv2.THRESH_BINARY_INV)  # _ = 30

        # Fill in gaps
        kernel = np.ones((kernel_num_on, kernel_num_on), np.uint8)
        morf = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=morpho_open_iter_on)
        morf = cv2.morphologyEx(morf, cv2.MORPH_CLOSE, kernel, iterations=morpho_close_iter_on)

        #Print that image was taken with light on
        text1 = "Light ON"
        cv2.putText(img, text1, (5, 20), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 255, 255), 2)

    # ---------------------NO LIGHT---------------------#
    else:
        # Crop image
        img = img[crop_top_off:crop_bottom_off, crop_left_off:crop_right_off]  #First pair of numbers is for vertical starting from teh top; second pair of numbers is for horizontal starting from the left

        #Adjust contrast (alpha) and brightness (beta)
        # define the alpha and beta
        alpha = alpha_off # Contrast control
        beta = beta_off # Brightness control Contrast?
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        #Transform in B&W
        grey = cv2.cvtColor(adjusted, cv2.COLOR_RGB2GRAY)

        #Threshold image
        _, thresh = cv2.threshold(grey, threshold_off, 255, cv2.THRESH_BINARY_INV)  # _ = 30

        #Fill in gaps
        kernel = np.ones((kernel_num_off,kernel_num_off),np.uint8)
        morf = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = morpho_open_iter_off)
        morf = cv2.morphologyEx(morf, cv2.MORPH_CLOSE, kernel, iterations = morpho_close_iter_off)

        #Print that image was taken with light off
        text1 = "Light OFF"
        cv2.putText(img, text1, (5, 20), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 255, 255), 2)


    #Find contours of objects in image and print them
    #Reference material: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
    cnts, _ = cv2.findContours(morf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #Make a list of the lenght of each contour
    list_cnts = []
    for cnt in cnts:
        list_cnts.append(len(cnt))
    #print(list_cnts)
    #print(max(list_cnts))

    #if no contours are identified because white or black image was given in input, then return area=0
    if len(list_cnts) == 0:
        results_PCM_Areas.append(0)
        text2 = "Area = No shape detected."
        cv2.putText(img, text2, (5, 45), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 0, 255), 2)  # 5 is the space form teh left edge of teh image, 20 from the top.

    for cnt in cnts:
        if len(cnt) == max(list_cnts):
            # Draw contour on original image
            cv2.drawContours(img, cnt, -1, (8, 163, 253), 3)
            cnt_area = cv2.contourArea(cnt) #Get area in pixels
            # Print results on final image
            text2 = "Area = {} mm^2".format(round(cnt_area*(mm_per_pixel**2)))
            cv2.putText(img, text2, (5, 45), cv2.FONT_HERSHEY_COMPLEX, 0.75, (8, 163, 253), 2) #5 is the space form teh left edge of teh image, 20 from the top.
            results_PCM_Areas.append(cnt_area*(mm_per_pixel**2))

    filename_noext, ext = filename.split('.')
    cv2.imwrite('Output/{}_Result.png'.format(filename_noext), img)

    return results_PCM_Areas
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#RUN SOFTWARE
root  = Tk()
root.wm_title("LAMPAD - Legit Automated Method for Prototissue Area Detection")
root.geometry("950x850")


def Open_File():
    global path_filename, path_file, filename #Define variable as global so that they can be used in other functions.
    global filename_noext, extension
    try:
        path_filename = filedialog.askopenfilename(initialdir=path_file,filetypes=(("Image files","*.png"),("Image files","*.jpeg"),("Image files","*.tif"),))
        path_file, filename = os.path.split(path_filename)
        filename_noext, extension = os.path.splitext(filename)

    except:
        path_filename = filedialog.askopenfilename(initialdir=os.getcwd(),filetypes=(("Image files","*.png"),("Image files","*.jpeg"),("Image files","*.tif"),))
        path_file, filename = os.path.split(path_filename)
        filename_noext, extension = os.path.splitext(filename)
        pass

    # Open image
    try:
        img = cv2.imread(path_filename)
        height, width, channels = img.shape
        Spacer()
        txt.insert(END, "\nOpening file: {}\n".format(filename))
        txt.insert(END, "Dimensions of original image: {}, {}, {}\n".format(height, width, channels))
        txt.update()
        txt.see("end")
        Spacer()
    except:
        return

    return

def Crop():
    crop_top = (int(e7.get()))
    crop_bottom = (int(e8.get()))
    crop_left = (int(e9.get()))
    crop_right = (int(e10.get()))

    # Open image
    try:
        img = cv2.imread(path_filename)
    except:
        txt.insert(END, "\nError: first, open a file!\n")
        return

    height, width, channels = img.shape

    if (crop_top < 0) or (crop_top > height) or (crop_top > crop_bottom):
        txt.insert(END, "\nError, image dimensions: {} x {}.\n".format(height, width))
        txt.update()
        txt.see("end")
        return

    if (crop_bottom < 0) or (crop_bottom > height):
        txt.insert(END, "\nError, image dimensions: {} x {}.\n".format(height, width))
        txt.update()
        txt.see("end")
        return

    if (crop_left < 0) or (crop_left > width) or (crop_left > crop_right):
        txt.insert(END, "\nError, image dimensions: {} x {}.\n".format(height, width))
        txt.update()
        txt.see("end")
        return

    if (crop_right < 0) or (crop_right > width):
        txt.insert(END, "\nError, image dimensions: {} x {}.\n".format(height, width))
        txt.update()
        txt.see("end")
        return


    # Crop image
    img = img[crop_top:crop_bottom, crop_left:crop_right]  # First pair of numbers is for vertical starting from teh top; second pair of numbers is for horizontal starting from the left
    height, width, channels = img.shape
    txt.insert(END, "\nCropping done on image: {}\n".format(filename))
    txt.update()
    txt.see("end")

    cv2.imshow("Cropped", cv2.resize(img, dsize=(0, 0), fx=1, fy=1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def Contrast_BaW():
    crop_top = (int(e7.get()))
    crop_bottom = (int(e8.get()))
    crop_left = (int(e9.get()))
    crop_right = (int(e10.get()))
    alpha = (float(e12.get()))
    beta = (float(e13.get()))


    # Open image
    try:
        img = cv2.imread(path_filename)
    except:
        txt.insert(END, "\nError: first, open a file!\n")
        return

    height, width, channels = img.shape

    if (crop_top < 0) or (crop_top > height) or (crop_top > crop_bottom):
        txt.insert(END, "\nError, image dimensions: {} x {}.\n".format(height, width))
        txt.update()
        txt.see("end")
        return

    if (crop_bottom < 0) or (crop_bottom > height):
        txt.insert(END, "\nError, image dimensions: {} x {}.\n".format(height, width))
        txt.update()
        txt.see("end")
        return

    if (crop_left < 0) or (crop_left > width) or (crop_left > crop_right):
        txt.insert(END, "\nError, image dimensions: {} x {}.\n".format(height, width))
        txt.update()
        txt.see("end")
        return

    if (crop_right < 0) or (crop_right > width):
        txt.insert(END, "\nError, image dimensions: {} x {}.\n".format(height, width))
        txt.update()
        txt.see("end")
        return

    # Crop image
    img = img[crop_top:crop_bottom, crop_left:crop_right]  # First pair of numbers is for vertical starting from teh top; second pair of numbers is for horizontal starting from the left

    # Adjust contrast (alpha) and brightness (beta)
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Transform in B&W
    grey = cv2.cvtColor(adjusted, cv2.COLOR_RGB2GRAY)
    txt.insert(END, "\nContrast correction and B&W conversion done on image: {}\n".format(filename))
    txt.update()
    txt.see("end")

    cv2.imshow('Grey', cv2.resize(grey, dsize=(0, 0), fx=1, fy=1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return

def Threshold():
    crop_top = (int(e7.get()))
    crop_bottom = (int(e8.get()))
    crop_left = (int(e9.get()))
    crop_right = (int(e10.get()))
    alpha = (float(e12.get()))
    beta = (float(e13.get()))
    threshold = (int(e15.get()))

    # Open image
    try:
        img = cv2.imread(path_filename)
    except:
        txt.insert(END, "\nError: first, open a file!\n")
        return

    height, width, channels = img.shape

    if (crop_top < 0) or (crop_top > height) or (crop_top > crop_bottom):
        txt.insert(END, "\nError, image dimensions: {} x {}.\n".format(height, width))
        txt.update()
        txt.see("end")
        return

    if (crop_bottom < 0) or (crop_bottom > height):
        txt.insert(END, "\nError, image dimensions: {} x {}.\n".format(height, width))
        txt.update()
        txt.see("end")
        return

    if (crop_left < 0) or (crop_left > width) or (crop_left > crop_right):
        txt.insert(END, "\nError, image dimensions: {} x {}.\n".format(height, width))
        txt.update()
        txt.see("end")
        return

    if (crop_right < 0) or (crop_right > width):
        txt.insert(END, "\nError, image dimensions: {} x {}.\n".format(height, width))
        txt.update()
        txt.see("end")
        return

    # Crop image
    img = img[crop_top:crop_bottom, crop_left:crop_right]  # First pair of numbers is for vertical starting from teh top; second pair of numbers is for horizontal starting from the left

    # Adjust contrast (alpha) and brightness (beta)
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Transform in B&W
    grey = cv2.cvtColor(adjusted, cv2.COLOR_RGB2GRAY)

    # Threshold image
    _, thresh = cv2.threshold(grey, threshold, 255, cv2.THRESH_BINARY_INV)  # _ = 30

    txt.insert(END, "\nThresholding done on image: {}\n".format(filename))
    txt.update()
    txt.see("end")

    cv2.imshow('Thresh', cv2.resize(thresh, dsize=(0, 0), fx=1, fy=1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return

def Find_PCM():
    try:
        crop_top = (int(e7.get()))
        crop_bottom = (int(e8.get()))
        crop_left = (int(e9.get()))
        crop_right = (int(e10.get()))
        alpha = (float(e12.get()))
        beta = (float(e13.get()))
        threshold = (int(e15.get()))
        kernel_num = (int(e17.get()))
        morpho_open_iter = (int(e18.get()))
        morpho_close_iter = (int(e19.get()))
        distance_pixel = (float(e2.get()))
        distance_mm = (float(e3.get()))
        mm_per_pixel = distance_mm/distance_pixel

        # Open image
        try:
            img = cv2.imread(path_filename)
        except:
            txt.insert(END, "\nError: first, open a file!\n")
            return

        height, width, channels = img.shape

        if (crop_top < 0) or (crop_top > height) or (crop_top > crop_bottom):
            txt.insert(END, "\nError, image dimensions: {} x {}.\n".format(height, width))
            txt.update()
            txt.see("end")
            return

        if (crop_bottom < 0) or (crop_bottom > height):
            txt.insert(END, "\nError, image dimensions: {} x {}.\n".format(height, width))
            txt.update()
            txt.see("end")
            return

        if (crop_left < 0) or (crop_left > width) or (crop_left > crop_right):
            txt.insert(END, "\nError, image dimensions: {} x {}.\n".format(height, width))
            txt.update()
            txt.see("end")
            return

        if (crop_right < 0) or (crop_right > width):
            txt.insert(END, "\nError, image dimensions: {} x {}.\n".format(height, width))
            txt.update()
            txt.see("end")
            return

        # Crop image
        img = img[crop_top:crop_bottom, crop_left:crop_right]  # First pair of numbers is for vertical starting from teh top; second pair of numbers is for horizontal starting from the left

        # Adjust contrast (alpha) and brightness (beta)
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # Transform in B&W
        grey = cv2.cvtColor(adjusted, cv2.COLOR_RGB2GRAY)

        # Threshold image
        _, thresh = cv2.threshold(grey, threshold, 255, cv2.THRESH_BINARY_INV)  # _ = 30

        # Fill in gaps
        kernel = np.ones((kernel_num, kernel_num), np.uint8)
        morf = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=morpho_open_iter)
        morf = cv2.morphologyEx(morf, cv2.MORPH_CLOSE, kernel, iterations=morpho_close_iter)
        #cv2.imshow('Morpho transf', cv2.resize(morf, dsize=(0, 0), fx=1, fy=1))

        # Find contours of objects in image and print them
        # Reference material: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
        cnts, _ = cv2.findContours(morf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Make a list of the lenght of each contour
        list_cnts = []
        for cnt in cnts:
            list_cnts.append(len(cnt))

        for cnt in cnts:
            if len(cnt) == max(list_cnts):

                # Method simple detected contour
                cv2.drawContours(img, cnt, -1, (8, 163, 253), 3)
                cnt_area = cv2.contourArea(cnt)  # Get area in pixels
                # Print results on final image
                text3 = "Area = {} mm^2".format(round(cnt_area * (mm_per_pixel ** 2)))
                cv2.putText(img, text3, (5, 25), cv2.FONT_HERSHEY_COMPLEX, 0.9, (8, 163, 253), 2)

        # Check if Output folder exists otherwise create it
        output_folder_path = "{}/Output".format(os.getcwd())
        isExist = os.path.exists(output_folder_path)
        Spacer()
        if not isExist:
            os.makedirs(output_folder_path)
            txt.insert(END, "WARNING: Output folder does not exist and now has been created.\n", "warning")
            txt.update()
            txt.see("end")

        txt.insert(END, "\nPCM found on image: {}\n".format(filename))
        cv2.imwrite("Output/Test Result.png", img)
        #cv2.imshow('img', cv2.resize(img, dsize=(0, 0), fx=1, fy=1))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        return
    except:
        txt.insert(END, "WARNING: Something went wrong...\n", "warning")
        txt.update()
        txt.see("end")

def Store_Light_Off():
    global crop_top_off
    global crop_bottom_off
    global crop_left_off
    global crop_right_off
    global alpha_off
    global beta_off
    global threshold_off
    global kernel_num_off
    global morpho_open_iter_off
    global morpho_close_iter_off

    crop_top_off = (int(e7.get()))
    crop_bottom_off = (int(e8.get()))
    crop_left_off = (int(e9.get()))
    crop_right_off = (int(e10.get()))
    alpha_off = (float(e12.get()))
    beta_off = (float(e13.get()))
    threshold_off = (int(e15.get()))
    kernel_num_off = (int(e17.get()))
    morpho_open_iter_off = (int(e18.get()))
    morpho_close_iter_off = (int(e19.get()))

    # Check if Output folder exists otherwise create it
    output_folder_path = "{}/Output".format(os.getcwd())
    isExist = os.path.exists(output_folder_path)
    Spacer()
    if not isExist:
        os.makedirs(output_folder_path)
        txt.insert(END, "WARNING: Output folder does not exist and now has been created.\n", "warning")
        txt.update()
        txt.see("end")

    txt.insert(END, "\nLight OFF settings:\n")
    txt.insert(END, "Pixels from the top: {}\n".format(crop_top_off))
    txt.insert(END, "Pixels from the bottom: {}\n".format(crop_bottom_off))
    txt.insert(END, "Pixels from the left: {}\n".format(crop_left_off))
    txt.insert(END, "Pixels from the right: {}\n".format(crop_right_off))
    txt.insert(END, "Alpha: {}\n".format(alpha_off))
    txt.insert(END, "Beta: {}\n".format(beta_off))
    txt.insert(END, "Threshold: {}\n".format(threshold_off))
    txt.insert(END, "Kernel: {}\n".format(kernel_num_off))
    txt.insert(END, "Morphology open iterations: {}\n".format(morpho_open_iter_off))
    txt.insert(END, "Morphology close iterations: {}\n".format(morpho_close_iter_off))
    txt.update()
    txt.see("end")
    Spacer()

    # Save parameters for later
    file = open("Output/Parameters OFF.txt", "w")
    file.write("Pixels from the top: {}\n".format(crop_top_off))
    file.write("Pixels from the bottom: {}\n".format(crop_bottom_off))
    file.write("Pixels from the left: {}\n".format(crop_left_off))
    file.write("Pixels from the right: {}\n".format(crop_right_off))
    file.write("Alpha: {}\n".format(alpha_off))
    file.write("Beta: {}\n".format(beta_off))
    file.write("Threshold: {}\n".format(threshold_off))
    file.write("Kernel: {}\n".format(kernel_num_off))
    file.write("Morphology open iterations: {}\n".format(morpho_open_iter_off))
    file.write("Morphology close iterations: {}\n".format(morpho_close_iter_off))
    file.close()

    return

def Store_Light_On():
    global crop_top_on
    global crop_bottom_on
    global crop_left_on
    global crop_right_on
    global alpha_on
    global beta_on
    global threshold_on
    global kernel_num_on
    global morpho_open_iter_on
    global morpho_close_iter_on

    crop_top_on = (int(e7.get()))
    crop_bottom_on = (int(e8.get()))
    crop_left_on = (int(e9.get()))
    crop_right_on = (int(e10.get()))
    alpha_on = (float(e12.get()))
    beta_on = (float(e13.get()))
    threshold_on = (int(e15.get()))
    kernel_num_on = (int(e17.get()))
    morpho_open_iter_on = (int(e18.get()))
    morpho_close_iter_on = (int(e19.get()))

    # Check if Output folder exists otherwise create it
    output_folder_path = "{}/Output".format(os.getcwd())
    isExist = os.path.exists(output_folder_path)
    Spacer()
    if not isExist:
        os.makedirs(output_folder_path)
        txt.insert(END, "WARNING: Output folder does not exist and now has been created.\n", "warning")
        txt.update()
        txt.see("end")

    txt.insert(END, "\nLight ON settings:\n")
    txt.insert(END, "Pixels from the top: {}\n".format(crop_top_on))
    txt.insert(END, "Pixels from the bottom: {}\n".format(crop_bottom_on))
    txt.insert(END, "Pixels from the left: {}\n".format(crop_left_on))
    txt.insert(END, "Pixels from the right: {}\n".format(crop_right_on))
    txt.insert(END, "Alpha: {}\n".format(alpha_on))
    txt.insert(END, "Beta: {}\n".format(beta_on))
    txt.insert(END, "Threshold: {}\n".format(threshold_on))
    txt.insert(END, "Kernel: {}\n".format(kernel_num_on))
    txt.insert(END, "Morphology open iterations: {}\n".format(morpho_open_iter_on))
    txt.insert(END, "Morphology close iterations: {}\n".format(morpho_close_iter_on))
    txt.update()
    txt.see("end")
    Spacer()

    # Save parameters for later
    file = open("Output/Parameters ON.txt", "w")
    file.write("Pixels from the top: {}\n".format(crop_top_on))
    file.write("Pixels from the bottom: {}\n".format(crop_bottom_on))
    file.write("Pixels from the left: {}\n".format(crop_left_on))
    file.write("Pixels from the right: {}\n".format(crop_right_on))
    file.write("Alpha: {}\n".format(alpha_on))
    file.write("Beta: {}\n".format(beta_on))
    file.write("Threshold: {}\n".format(threshold_on))
    file.write("Kernel: {}\n".format(kernel_num_on))
    file.write("Morphology open iterations: {}\n".format(morpho_open_iter_on))
    file.write("Morphology close iterations: {}\n".format(morpho_close_iter_on))
    file.close()

    return

def Load_Light_Off():
    output_folder_path = "{}/Output".format(os.getcwd())
    path_filename_param = filedialog.askopenfilename(initialdir=output_folder_path,filetypes=(("Text files","*.txt"),))
    path_param, filename_param = os.path.split(path_filename_param)

    data = []
    try:
        with open(path_filename_param) as file:
            for line in file:
                _, param = line.split(":")
                size = len(param)
                data.append(param[1:size - 1]) # To remove "/n" (new line)
        file.close()
    except:
        return

    e7.delete(0, END)
    e8.delete(0, END)
    e9.delete(0, END)
    e10.delete(0, END)
    e12.delete(0, END)
    e13.delete(0, END)
    e15.delete(0, END)
    e17.delete(0, END)
    e18.delete(0, END)
    e19.delete(0, END)

    e7.insert(END, data[0])
    e8.insert(END, data[1])
    e9.insert(END, data[2])
    e10.insert(END, data[3])
    e12.insert(END, data[4])
    e13.insert(END, data[5])
    e15.insert(END, data[6])
    e17.insert(END, data[7])
    e18.insert(END, data[8])
    e19.insert(END, data[9])

    global crop_top_off
    global crop_bottom_off
    global crop_left_off
    global crop_right_off
    global alpha_off
    global beta_off
    global threshold_off
    global kernel_num_off
    global morpho_open_iter_off
    global morpho_close_iter_off

    crop_top_off = int(data[0])
    crop_bottom_off = int(data[1])
    crop_left_off = int(data[2])
    crop_right_off = int(data[3])
    alpha_off = float(data[4])
    beta_off = float(data[5])
    threshold_off = int(data[6])
    kernel_num_off = int(data[7])
    morpho_open_iter_off = int(data[8])
    morpho_close_iter_off = int(data[9])

    Spacer()
    txt.insert(END, "\nFile opened: {}\n\n".format(filename_param))
    txt.insert(END, "\nLight OFF loaded settings:\n")
    txt.insert(END, "Pixels from the top: {}\n".format(crop_top_off))
    txt.insert(END, "Pixels from the bottom: {}\n".format(crop_bottom_off))
    txt.insert(END, "Pixels from the left: {}\n".format(crop_left_off))
    txt.insert(END, "Pixels from the right: {}\n".format(crop_right_off))
    txt.insert(END, "Alpha: {}\n".format(alpha_off))
    txt.insert(END, "Beta: {}\n".format(beta_off))
    txt.insert(END, "Threshold: {}\n".format(threshold_off))
    txt.insert(END, "Kernel: {}\n".format(kernel_num_off))
    txt.insert(END, "Morphology open iterations: {}\n".format(morpho_open_iter_off))
    txt.insert(END, "Morphology close iterations: {}\n".format(morpho_close_iter_off))
    txt.update()
    txt.see("end")
    Spacer()

    return

def Load_Light_On():
    output_folder_path = "{}/Output".format(os.getcwd())
    path_filename_param = filedialog.askopenfilename(initialdir=output_folder_path,filetypes=(("Text files","*.txt"),))
    path_param, filename_param = os.path.split(path_filename_param)

    data = []
    try:
        with open(path_filename_param) as file:
            for line in file:
                _, param = line.split(":")
                size = len(param)
                data.append(param[1:size - 1])  # To remove "/n" (new line)
        file.close()
    except:
        return

    e7.delete(0, END)
    e8.delete(0, END)
    e9.delete(0, END)
    e10.delete(0, END)
    e12.delete(0, END)
    e13.delete(0, END)
    e15.delete(0, END)
    e17.delete(0, END)
    e18.delete(0, END)
    e19.delete(0, END)

    e7.insert(END, data[0])
    e8.insert(END, data[1])
    e9.insert(END, data[2])
    e10.insert(END, data[3])
    e12.insert(END, data[4])
    e13.insert(END, data[5])
    e15.insert(END, data[6])
    e17.insert(END, data[7])
    e18.insert(END, data[8])
    e19.insert(END, data[9])

    global crop_top_on
    global crop_bottom_on
    global crop_left_on
    global crop_right_on
    global alpha_on
    global beta_on
    global threshold_on
    global kernel_num_on
    global morpho_open_iter_on
    global morpho_close_iter_on

    crop_top_on = int(data[0])
    crop_bottom_on = int(data[1])
    crop_left_on = int(data[2])
    crop_right_on = int(data[3])
    alpha_on = float(data[4])
    beta_on = float(data[5])
    threshold_on = int(data[6])
    kernel_num_on = int(data[7])
    morpho_open_iter_on = int(data[8])
    morpho_close_iter_on = int(data[9])

    Spacer()
    txt.insert(END, "\nFile opened: {}\n\n".format(filename_param))
    txt.insert(END, "\nLight ON loaded settings:\n")
    txt.insert(END, "Pixels from the top: {}\n".format(crop_top_on))
    txt.insert(END, "Pixels from the bottom: {}\n".format(crop_bottom_on))
    txt.insert(END, "Pixels from the left: {}\n".format(crop_left_on))
    txt.insert(END, "Pixels from the right: {}\n".format(crop_right_on))
    txt.insert(END, "Alpha: {}\n".format(alpha_on))
    txt.insert(END, "Beta: {}\n".format(beta_on))
    txt.insert(END, "Threshold: {}\n".format(threshold_on))
    txt.insert(END, "Kernel: {}\n".format(kernel_num_on))
    txt.insert(END, "Morphology open iterations: {}\n".format(morpho_open_iter_on))
    txt.insert(END, "Morphology close iterations: {}\n".format(morpho_close_iter_on))
    txt.update()
    txt.see("end")
    Spacer()

    return

def Batch_find_PCM():
    # OPEN FOLDER
    path_foldername = filedialog.askdirectory(initialdir=path_file)

    # GET PARAMETERS
    distance_pixel = (float(e2.get()))
    distance_mm = (float(e3.get()))
    mm_per_pixel = distance_mm/distance_pixel
    PCMFijiArea_mm2 = (float(e4.get()))
    time_interval_sec = (float(e5.get()))

    # Check if all corpping etc. parameters were stored or loaded through the appropriate buttons, otherwise warn to check them and save them.
    try:
        crop_top_off
        crop_bottom_off
        crop_left_off
        crop_right_off
        alpha_off
        beta_off
        threshold_off
        kernel_num_off
        morpho_open_iter_off
        morpho_close_iter_off
        crop_top_on
        crop_bottom_on
        crop_left_on
        crop_right_on
        alpha_on
        beta_on
        threshold_on
        kernel_num_on
        morpho_open_iter_on
        morpho_close_iter_on
    except:
        txt.insert(END, "\nWARNING: Parameters not saved. Please, check all parameters for both light ON and OFF and save them using the specific buttons."
                        "Alternatively, load parameters for both images taken with light OFF and ON.\n", "warning")
        txt.update()
        txt.see("end")
        return

    #Check whether the croppoing parameters are consistent between light OFF and ON images
    Image_Dimensions_Check()


    # IDENTIFY ALL IMAGES TAKEN WITH LIGHT ON OR OFF
    # Get all files
    paths_filenames = sorted(glob.glob("{}/*.tif".format(path_foldername)))  # Get all .tif files and put them in alphabetical order
    paths_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))  # This gets them in the correct numerical order.
    txt.insert(END, "\nNumber of files found: {}.\n".format(len(paths_filenames)))
    txt.insert(END, "Determining dominant colour.\nProcessing files...\n")
    txt.update()
    txt.see("end")


    # Get all paths and filenames and memorise them, then show all filenames found adn find the dominant color in each file.
    paths = []
    filenames = []
    results_list = []
    counter = 0
    for path_filename in paths_filenames:  # You can select here the images to analyse
        path, filename = os.path.split(path_filename)  # Splits path and file name
        img = cv2.imread(path_filename)
        if counter == 10:
            txt.insert(END, "File name: {}\n".format(filename))
            height, width, channels = img.shape
            txt.insert(END, "Dimensions of original image: {}, {}, {}\n".format(height, width, channels))
            txt.update()
            txt.see("end")
            counter = 0
        # Resize the image to 64x64 pixels to speed up processing and convert it to the RGB color space.
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        Find_Dominant_Color(filename, img, results_list)
        counter += 1

    txt.insert(END, "\nDominant colours found for {} images.\n".format(len(results_list))) # Print the results showing the dominant color for each file
    txt.update()
    txt.see("end")

    # Identify light on/off using KMeans unsupervised machine learning algorithm
    results_list, num_images_on, num_images_off = Identify_Light_OnOff_Kmeans(results_list)
    txt.insert(END, "\n{} images with light on/off identified:".format(len(results_list)))  # Print the results showing the dominant color and the cluster for each file
    txt.insert(END, " {} images with light on;".format(num_images_on))
    txt.insert(END, " {} images with light off.\n".format(num_images_off))
    txt.update()
    txt.see("end")


    # --------------------------------------------------------------------------------------------------------------------#
    # --------------------------------------------------------------------------------------------------------------------#
    # DETECT PCM AREA
    # Get all paths and filenames and memorise them, then show all filenames found.
    txt.insert(END, "\nDetecting PCM area from {} images.\nProcessing files...\n".format(len(results_list)))
    txt.update()
    txt.see("end")

    paths = []
    filenames = []
    results_PCM_Areas = []
    i = 0
    counter = 0
    for path_filename in paths_filenames:  # You can select here the images to analyse
        path, filename = os.path.split(path_filename)  # Splits path and file name
        img = cv2.imread(path_filename)
        if counter == 10:
            txt.insert(END, "File name: {}\n".format(filename))
            height, width, channels = img.shape
            txt.insert(END, "Dimensions of original image: {}, {}, {}\n".format(height, width, channels))
            txt.update()
            txt.see("end")
            counter = 0

        results_PCM_Areas = PCM_Analysis(i, filename, img, mm_per_pixel, results_list, results_PCM_Areas)
        i += 1
        counter += 1

    # Add PCM areas and time to the list of results
    txt.insert(END, "\nNumber of images processed: {}".format(len(results_list)))
    txt.insert(END, "\nNumber of PCM areas detected: {}\n".format(len(results_PCM_Areas)))
    txt.update()
    txt.see("end")

    i = 0
    for result in results_list:
        result.append(time_interval_sec * i)
        result.append(results_PCM_Areas[i])
        result.append(results_PCM_Areas[i] * (PCMFijiArea_mm2 / results_PCM_Areas[0]))
        i += 1

    # --------------------------------------------------------------------------------------------------------------------#
    # --------------------------------------------------------------------------------------------------------------------#
    # EXPORT DATA IN CSV AND PLOT RESULTS
    # Save CSV file
    header = ["Filename", "Dominant HSV color", "Light on/off", "Time [s]", "Area [mm^2]", "Corrected area [mm^2]"]
    df = pd.DataFrame(results_list, columns=header)  # Create new dataframe (df)
    df.to_csv("Output/Results.csv")

    txt.insert(END, "\nProcess done, results saved in Output folder.\nSummary of final results below.\n\n{}".format(df.head(10)))
    Spacer()
    txt.insert(END, "END OF PROGRAM")
    Spacer()
    Spacer()
    txt.update()
    txt.see("end")


    # Plot results and save them
    x = df["Time [s]"].to_numpy()
    y1 = df["Area [mm^2]"].to_numpy()
    y2 = df["Corrected area [mm^2]"].to_numpy()
    fig, ax = plt.subplots()
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_xlabel("Time (s)", size=18)
    ax.set_ylabel("Area ($\mathregular{mm^2}$)", size=18)  # To make a superscript the $...$ symbols are required. However, this may change the font, so you use \mathregular{...} to use the same font as the rest of the axis label.
    ax.grid(True, color="grey", linewidth="0.5", linestyle="--", zorder=1)
    plt.scatter(x, y1, s=15, zorder=2, label="Original data")
    plt.scatter(x, y2, s=15, zorder=3, label="Corrected data")
    ax.legend()
    plt.savefig("Output/Results plot.png", format="png", bbox_inches='tight', dpi=300)
    plt.close() # Always close figure otherwise you run out of RAM memory!

    #Save video from final images
    images = []
    for result  in results_list:
        filename_noext, extension = os.path.splitext(result[0])
        new_filename = "Output/{}_Result.png".format(filename_noext)
        images.append(new_filename) # Finds all the path names matching a specified pattern and put them in order (sorted)

    # Get the size in pixels of the first image to be used for the dimensions of the video.
    height, width, layers = cv2.imread(images[0]).shape

    # Create the video file. cv2.VideoWriter(file name, codec, frames ps, ...)
    video = cv2.VideoWriter("Output/Result video.mp4", 0, 10, (width, height))

    # Write video according specifics above
    for image in images:
        video.write(cv2.imread(image))  # Requests file name and not actual file?

    cv2.destroyAllWindows()
    video.release()  # Save the final video file
    return

# --------------------------------------------------------------------------------------------------------------------#

column2 = 200
column3 = 350
column4 = 530
column5 = 620

button = Button(root, text = "Open test file", command = Open_File).grid(row = 1, column = 0, sticky = W, padx = 360, pady = 2)

separator = ttk.Separator(root, orient='horizontal').grid(row = 2, column = 0, sticky = EW, pady = 2)

l1 = Label(root, text="Parameters:", font="Helvetica 16 bold").grid(row = 3, column = 0, sticky = W, pady = 2)

l2 = Label(root, text="Distance in pixels:").grid(row = 4, column = 0, sticky = W, pady = 2)
e2 = Entry(root, width=7)
e2.insert(END, "145.5")
e2.grid(row = 4, column = 0, sticky = W, padx = column2, pady = 2)

l3 = Label(root, text="Distance in mm:").grid(row = 4, column = 0, sticky = W, padx = column3, pady = 2)
e3 = Entry(root, width=5)
e3.insert(END, "1")
e3.grid(row = 4, column = 0, sticky = W, padx = column4, pady = 2)

l4 = Label(root, text="PCM area from Fiji:").grid(row = 5, column = 0, sticky = W, pady = 2)
e4 = Entry(root, width=7)
e4.insert(END, "18")
e4.grid(row = 5, column = 0, sticky = W, padx = column2, pady = 2)

l5 = Label(root, text="Time interval in seconds:").grid(row = 5, column = 0, padx = column3, sticky = W, pady = 2)
e5 = Entry(root, width=5)
e5.insert(END, "1")
e5.grid(row = 5, column = 0, sticky = W, padx = column4, pady = 2)

separator = ttk.Separator(root, orient='horizontal').grid(row = 6, column = 0, sticky = EW, pady = 2)

l6 = Label(root, text="Cropping area:", font="Helvetica 16 bold").grid(row = 7, column = 0, sticky = W, pady = 2)

l7 = Label(root, text="Pixels from top:").grid(row = 8, column = 0, sticky = W, pady = 2)
e7 = Entry(root, width=5)
e7.insert(END, "600")
e7.grid(row = 8, column = 0, sticky = W, padx = column2, pady = 2)

l8 = Label(root, text="Pixels from bottom:").grid(row = 8, column = 0, padx = column3, sticky = W, pady = 2)
e8 = Entry(root, width=5)
e8.insert(END, "1500")
e8.grid(row = 8, column = 0, sticky = W, padx = column4, pady = 2)

l9 = Label(root, text="Pixels from left:").grid(row = 9, column = 0, sticky = W, pady = 2)
e9 = Entry(root, width=5)
e9.insert(END, "400")
e9.grid(row = 9, column = 0, sticky = W, padx = column2, pady = 2)

l10 = Label(root, text="Pixels from right:").grid(row = 9, column = 0, padx = column3, sticky = W, pady = 2)
e10 = Entry(root, width=5)
e10.insert(END, "1400")
e10.grid(row = 9, column = 0, sticky = W, padx = column4, pady = 2)

button2 = Button(root, text = "Crop", command = Crop).grid(row = 9, column = 0, sticky = W, padx = column5, pady = 2)

l11 = Label(root, text="Adjust contrast and make B&W:", font="Helvetica 16 bold").grid(row = 10, column = 0, sticky = W, pady = 2)

l12 = Label(root, text="Alpha contrast:").grid(row = 11, column = 0, sticky = W, pady = 2)
e12 = Entry(root, width=5)
e12.insert(END, "1.25")
e12.grid(row = 11, column = 0, sticky = W, padx = column2, pady = 2)

l13 = Label(root, text="Beta contrast:").grid(row = 11, column = 0, padx = column3, sticky = W, pady = 2)
e13 = Entry(root, width=5)
e13.insert(END, "50")
e13.grid(row = 11, column = 0, sticky = W, padx = column4, pady = 2)

button3 = Button(root, text = "Adjust contrast and make B&W", command = Contrast_BaW).grid(row = 11, column = 0, sticky = W, padx = column5, pady = 2)

l14 = Label(root, text="Threshold:", font="Helvetica 16 bold").grid(row = 12, column = 0, sticky = W, pady = 2)

l15 = Label(root, text="Threshold:").grid(row = 13, column = 0, sticky = W, pady = 2)
e15 = Entry(root, width=5)
e15.insert(END, "220")
e15.grid(row = 13, column = 0, sticky = W, padx = column2, pady = 2)

button4 = Button(root, text = "Threshold", command = Threshold).grid(row = 13, column = 0, sticky = W, padx = column5, pady = 2)

l16 = Label(root, text="Morphological transformations:", font="Helvetica 16 bold").grid(row = 14, column = 0, sticky = W, pady = 2)

l17 = Label(root, text="Kernel (n x n):").grid(row = 15, column = 0, sticky = W, pady = 2)
e17 = Entry(root, width=5)
e17.insert(END, "5")
e17.grid(row = 15, column = 0, sticky = W, padx = column2, pady = 2)

l18 = Label(root, text="Morphology open iterations:").grid(row = 16, column = 0, sticky = W, pady = 2)
e18 = Entry(root, width=5)
e18.insert(END, "2")
e18.grid(row = 16, column = 0, sticky = W, padx = column2, pady = 2)

l19 = Label(root, text="Morphology close iterations:").grid(row = 16, column = 0, padx = column3, sticky = W, pady = 2)
e19 = Entry(root, width=5)
e19.insert(END, "4")
e19.grid(row = 16, column = 0, sticky = W, padx = column4, pady = 2)

button5 = Button(root, text = "Find PCM", command = Find_PCM).grid(row = 16, column = 0, sticky = W, padx = column5, pady = 2)

button6 = Button(root, text = "Save param for light OFF", command = Store_Light_Off).grid(row = 17, column = 0, sticky = W, padx = 5, pady = 2)

button7 = Button(root, text = "Load param for light OFF", command = Load_Light_Off).grid(row = 17, column = 0, sticky = W, padx = 220, pady = 2)

button9 = Button(root, text = "Save param for light ON", command = Store_Light_On).grid(row = 17, column = 0, sticky = W, padx = 450, pady = 2)

button10 = Button(root, text = "Load param for light ON", command = Load_Light_On).grid(row = 17, column = 0, sticky = W, padx = 660, pady = 2)


separator = ttk.Separator(root, orient='horizontal').grid(row = 18, column = 0, sticky = EW, pady = 2)

button11 = Button(root, text = "Batch processing", command = Batch_find_PCM).grid(row = 19, column = 0, sticky = W, padx = 360, pady = 2)


#Create and write inside a text box
txt = scrolledtext.ScrolledText(root, height=19, width=102)
txt.configure(font=("TkDefaultFont", 12, "normal"))
txt.tag_config('warning', foreground="red")
txt.grid(row=20, column = 0, columnspan = 2, sticky=W, padx = 5) #W=allign to left
txt.see("end")


root.mainloop()