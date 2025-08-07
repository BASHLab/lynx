# lynx
# NLMS Filter Testing
In order to collect physiological data like breathing, chewing etc. using the earbud, run the **LMS_Filter_Test.py** file. 

The microphones will start recording signals, while the filter will dynamically use the external mic as reference to isolate the internal signals like breathing. When done, hit CTRL-C to stop recording. 

The resulting output will be three files: **raw external mic signal, raw internal mic signal, and the filtered LMS output**. The files will be saved with data and timestamp in the directory the file was run.
