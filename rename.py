# a simple script to rename files from 1 to +++N
import os
import sys

#update dir to your location
path = "R:\\Temp\\training\\data\\images"
new_filename= ""
i=0

filename=os.listdir(path)
for dir,subdir,listfilename in os.walk(path):
    for filename in listfilename:
        i += 1
        new_filename = str(i) + '.jpg' #adjust to your specific extension
        src=os.path.join(dir, filename)
        dst=os.path.join(dir, new_filename)
        os.rename(src, dst)
