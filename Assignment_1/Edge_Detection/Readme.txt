
Question 2: Edge detection

Description: Write a function that finds edge intensity and orientation in an image. Display the output of your function
            for one of the given images in the handout.

File structure:
    Edge_Detection
        |___Input_Image
        |___Output_Image
        |            |___cat2
        |            |___img0
        |            |___littledog
        |___EdgeDetection.py
        |___Readme.txt
        
Instruction how to run the code
        In the Edge_Detection root folder, run:     python3 EdgeDetection.py 
        The code will iterate over all image (cat2, img0 and littledog) from Input_Image folder and generate coresponding images (gradient magnitude, gradient orientation and edge detection) to Output_Image folder 