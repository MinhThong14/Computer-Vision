
Question 3: Sticks filter 

Description: Create sticks filter then apply the gradient magnitude image before finding the edges to reduce speckle noise and preserve linear stuctures 

File structure:
    Stick_Filter
        |___Input_Image
        |___Output_Image
        |            |___cat2
        |            |___img0
        |            |___littledog
        |___StickFilter.py
        |___Readme.txt
        
Instruction how to run the code
        In the Stick_Filter root folder, run:     python3 StickFilter.py 
        The code will iterate over all images (cat2, img0 and littledog) from Input_Image folder and generate coresponding images (sticks filters image and edge detection image after applying sticks filter) to Output_Image folder 
        