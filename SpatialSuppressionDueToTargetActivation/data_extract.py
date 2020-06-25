#### The viewing distance was set to 61 cm. With a 21 inch CRT monitor, 1 deg ~= 0.94 cm ~= 28 pixel
# However, the Eyelink is using a view-distance of 75 to calculate saccade amplitudes, 1 deg ~= 0.76 cm ~= 34.4 pixel 
import os
from math import sqrt, atan, sin, cos, pi 

cwd    = os.getcwd() # get the current working directory
d_dir  = cwd + '/Exp1_data/' # get the path to the data directory
f_name = os.listdir(d_dir)     # get all the data file names
f_name.sort();

all_f  = open('AC.csv', 'w') # open a file to save cleaned data

header = ['subj', 'name', 'gender', 'age', 'run', 'shape', 'dist', 'td_dif', 'resp', 'resp_key', 'd_loc', 't_loc',
			'gotKey','startTime','respTime','resp', 'fix_loc', 'prac']
all_f.write('\t'.join(header)+'\n')

## The location of targets
scn_size = (1680,1050)
deg2pix  = int(840/(atan(24.0/71)*180/pi))
Decc = 4  # the radius of the invisible circle
tar_locs = []
for j in range(8):
    sti_x = int(scn_size[0]/2 + cos(j*pi/4)*Decc*deg2pix)
    sti_y = int(scn_size[1]/2 - sin(j*pi/4)*Decc*deg2pix)
    tar_locs.append((sti_x,sti_y))
    
for f in f_name:
    if f[0] <> '.':
        d_f = open(d_dir + f)

        for line in d_f:
            msg = line.split('\t')
            if msg[0] == 'conf':
                all_f.write('\t'.join(msg))
            else:
                if msg[0] != 'subj':                    

                    if int(msg[0]) < 9:
                        if int(msg[0]) > 5:
                            new_id = int(msg[0])-9
                        else:
                            new_id = int(msg[0])-1
                            
                    if int(msg[0]) >8 and int(msg[0]) < 17:
                        if int(msg[0]) > 13:
                            new_id = int(msg[0])-17
                        else:
                            new_id = int(msg[0])-9
                            
                    if int(msg[0]) >16 and int(msg[0]) < 25:
                        if int(msg[0]) > 21:
                            new_id = int(msg[0])-25
                        else:
                            new_id = int(msg[0])-17
                    
                    if int(msg[0]) >24 and int(msg[0]) < 33:
                        if int(msg[0]) > 29:
                            new_id = int(msg[0])-33
                        else:
                            new_id = int(msg[0])-25
                    
                            
                    if msg[11] == str(tar_locs[new_id]):
                        msg[11] = 'fd'
                    if msg[11] == str(tar_locs[new_id+1]) or msg[11] == str(tar_locs[new_id-1]):
                        msg[11] = 'fd1'
                    if msg[11] == str(tar_locs[new_id+2]) or msg[11] == str(tar_locs[new_id-2]):
                        msg[11] = 'fd2'
                    if msg[11] == str(tar_locs[new_id+3]) or msg[11] == str(tar_locs[new_id-3]):
                        msg[11] = 'fd3'
                    if msg[11] == str(tar_locs[new_id-4]):
                        msg[11] = 'fd4'
                            
                    if msg[10] == str(tar_locs[new_id]):
                        msg[10] = 'fd'
                    if msg[10] == str(tar_locs[new_id+1]) or msg[10] == str(tar_locs[new_id-1]):
                        msg[10] = 'fd1'
                    if msg[10] == str(tar_locs[new_id+2]) or msg[10] == str(tar_locs[new_id-2]):
                        msg[10] = 'fd2'
                    if msg[10] == str(tar_locs[new_id+3]) or msg[10] == str(tar_locs[new_id-3]):
                        msg[10] = 'fd3'
                    if msg[10] == str(tar_locs[new_id-4]):
                        msg[10] = 'fd4'
                    
                    if msg[-1] != 'practice\n':
                        all_f.write('\t'.join(msg))

all_f.close()
