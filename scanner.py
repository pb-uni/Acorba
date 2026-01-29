'''
nbcs, scanner.py script.
Loaded buy the ACORBA GUI to perform scanner root segmentation and angle
measurements.
'''
#Load dependencies
import os
import sys
import gc
import acorba
import tifffile
import cv2
import numpy as np
import pandas as pd
import PySimpleGUI as sg
from time import sleep, time
from skimage.io import imread
import matplotlib.pyplot as plt

sg.theme('Light Brown 8')
start_time = time()#Start measuring time, so the user knows how long took his session
#To silence non-essential tensorflow informations and non-consequential warnings


print("ALL LIBRARIES ARE IMPORTED, ACORBA WILL START NOW\
      n##################################################")

plt.ion()#Shuts off matplotlib interactive mode

#For debugging
#args=acorba.debug.scanner_debug()
#args.save_lenghts='False'
save_lenghts='False'

#retrieve user parameter from the parser
args=acorba.folder.parser()

#Set random seed to "The Answer to the Ultimate Question of Life, The Universe, and Everything."
np.random.seed(42)

#Set the broken skeleton repair treshold (in pixels)
#if two skeleton end and origin are spaced < 100 pixels they will be stitched
brokenfactor=int(args.broken)

#set the number of pixel to set the origin of the angle vector
originvector=int(args.vector)

#Create a folder for segmentation tif stack in case it was ticked by the user
if args.savesegmentation=="True":
    exp_dir = args.input_folder+"/Saved Segmentations/"
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

#Retrieve images list
listimage=acorba.folder.list_tif(folder=args.input_folder)

stack2=0#Increment for the progress bar

#Load Deep Machine Learning libraries and models/weights
if args.method=='Deep Machine Learning':
    from tensorflow.autograph import set_verbosity
    set_verbosity(0)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    if len(args.custom)>0:
        from keras.models import model_from_json
        print('Custom models and weights loading:')
        model_weights=os.listdir(args.custom)
        for i in model_weights:
            if i.endswith(".json"):
                with open(args.custom+"/"+i, 'r') as json:#check if its working
                    loaded_model_json = json.read()
                model = model_from_json(loaded_model_json)
            if i.endswith('.h5'):
                model.load_weights(args.custom+"/"+i)
    else:
        from keras_unet.models import satellite_unet
        model = satellite_unet(input_shape=(256, 256, 1))
        if args.superaccuracy=="True":
            model.load_weights("models/unetnocontrast_lessaugm_09092021_22.h5")
        else:
            model.load_weights("models/unetnocontrast_lessaugm_09092021_87.h5")
    print("Scanner prediction model and weights loaded")

#Setting the progress bar total length by opening every image lenght info
#and summing their timeframe numbers
looplenght=acorba.folder.looplenght_determinator(
    list_image=listimage,folder=args.input_folder)


listimage2=[]
#Starting analysis
for images in listimage:#loop for .tif stacks
    if args.savesegmentation=="True":
        seg_list=[]#create segmentation list to transform into an array at the
        #end and imwrite, save padded picture if machine learning regular
        if args.method=="Deep Machine Learning" and args.superaccuracy=="False":
            im_list=[]
    lene=imread(args.input_folder+'/'+images)#Stack import
    if min(lene.shape)==lene.shape[2]:#be sure the shape is (height,width,timeframes)
        lene=np.rollaxis(lene,2,0)
    ntf=int(min(lene.shape))#Finding the number of timeframes
    imgname=images#Store image name in a different object as images part of
    #the for loop it was creating problem with the try/except functions
    if args.prediction in ("First","All"):
        #Create empty list for prediction export
        predsexp=[]
        imgsexp=[]
    try:#it slows down the code but allows to pass a stack that would be
    #problematic and crashes the whole code
        imnb=0
        angletot=[]
        lengthtot=[]
        lxtot=[]
        lytot=[]
        while imnb < ntf:#Loop for timeframe
############################################################Segmentation block
            
            im=lene[imnb]
            if args.method=="Deep Machine Learning":
                #Set the minimum area to be detected by the detection of contour
                areatresh=80
                if args.superaccuracy=="False":
                    #Deep machine learning segmentation with padding
                    mega, new_im=acorba.scanner.DML_default(image=im,model=model)
                else:
                    #Super accuracy mode for scanner >> smooth tiling,
                    #requires tons of processing power and RAM!
                    mega=acorba.scanner.DML_default_superaccu(image=im,model=model)
                    new_im=im

                if args.savesegmentation=="True":
                    seg_list.append(mega)
                    if args.method=="Deep Machine Learning" and args.superaccuracy=="False":
                        im_list.append(new_im)
                if args.prediction in ("First","All"):
                    mega2=mega.copy()
                    #resize down by 40% for speeding up analysis
                    #without loosing to much spatial infos
                mega=cv2.resize(mega, (int(im.shape[1] * 60 / 100),
                                       int(im.shape[0] * 60 / 100)))

            elif args.method=="Traditional":
                areatresh=600
                if args.tradmethod=="Entropy":
                    mega=acorba.scanner.trad_scanner(im)
                else:
                    mega=acorba.scanner.trad_scanner_threshold(im)

                if args.savesegmentation=="True":
                    seg_list.append(mega)

                if args.prediction in ("First","All"):
                    mega2=mega.copy()

                new_im=im
                #resize down by 40% for speeding up analysis
                #without loosing to much spatial infos
                mega=cv2.resize(mega, (int(mega.shape[1] * 60 / 100),
                                       int(mega.shape[0] * 60 / 100)))

            elif args.method=="Own masks":
                areatresh=80
                mega=imread(args.binary_folder+'/'+images,key=imnb)
                #transform any binary into 0/1 image instead of example 0/255
                if np.max(mega>1):
                    mega=mega/np.max(mega)
                mega=mega.astype('uint8')
                #resize down by 40% for speeding up analysis
                #without loosing to much spatial infos
                mega=cv2.resize(mega, (int(mega.shape[1] * 60 / 100),int(mega.shape[0] * 60 / 100)))

            if args.prediction in ("First","All") and args.method != "Own masks":
                predsexp.append(cv2.resize(mega2, (int(mega2.shape[1] * 50 / 100),
                                                   int(mega2.shape[0] * 50 / 100))))
                imgsexp.append(cv2.resize(new_im, (int(new_im.shape[1] * 50 / 100),
                                                   int(new_im.shape[0] * 50 / 100))))

####################################################batch root detection block
            listskel,listxy=acorba.scanner.root_skeleton(mega, areatresh)

            #Saving skeletons lenght in pixels
            if save_lenghts=="True":
                lenghts_list=[len(skeleton[skeleton==1]) for skeleton in listskel]
                ly=[]
                lx=[]

            #Create empty plot
            if imnb>0:
                plt.close()
            fig, axs = plt.subplots(len(listskel), 1, sharex = True)
            fig.suptitle(images+" - tf "+str(imnb))
            plt.show()
####################################################Angle measurements block
            #if  original_number_skeletons==len(listskel):
            i=0
            listorigins=[]
            listends=[]
            listangles=[]
            while i<len(listskel):
                #setup progress bar for each timeframe
                '''if not sg.OneLineProgressMeter('Calculation in progress',
                                               stack2, looplenght, 'single'):
                    sys.exit("User stop!!!!!!!!!!!!!!")'''

                #calculate angle of one skeleton tip and retrieve other infos along
                anglee, origins, ends,newskely,newskelx,start,endminus=(acorba.scanner.root_angle(listskel[i],originvector))
                
                listends.append(ends)
                listorigins.append(origins)
                listangles.append(anglee)


                #Plot the angle analysis plot one root at a time, if i==0 create the blank plot
                if i ==0:
                    for axe in axs:
                        axe.clear()
                axs[len(listskel)-i-1].plot(newskelx,newskely,
                                            color="grey")
                axs[len(listskel)-i-1].scatter(start[0],start[1],
                                               color="red",s=15)
                axs[len(listskel)-i-1].scatter(endminus[0],endminus[1],
                                               color='purple',s=15)
                axs[len(listskel)-i-1].axis('off')
                axs[len(listskel)-i-1].invert_yaxis()
                
                #save ly and lx to calculate Vertical and Horizontal growth indexes
                if save_lenghts=="True" and i==0:
                    listorigins_zero=origins
                    listends_zero=ends
                if save_lenghts=="True":
                    ly.append(listends[i][0]-listorigins_zero[0])
                    lx.append(listends[i][1]-listorigins_zero[1])
                plt.pause(0.001)
                i=i+1
            
            plt.show()
            
            listorigins_real=[]
            for (ori,xy) in zip(listorigins,listxy):
                listorigins_real.append(np.array(ori)+np.array(xy))
            listends_real=[]
            for (ori,xy) in zip(listends,listxy):
                listends_real.append(np.array(ori)+np.array(xy))

            if args.rootplot=='True':
                fig.savefig(args.input_folder+'/'+images+'_'+'plateplot_'+str(imnb+1)+'.png', 
                            bbox_inches = 'tight',pad_inches = 0)

            print('Angle list for plate_'+images+"_timeframe_"+str(imnb+1)+':')
            print(listangles)

            if save_lenghts=='True':
                print("Length list for plate_"+images+"_timeframe_"+str(imnb+1))
                print(lenghts_list)
            #else:
                #print("Broken skeleton has been activated, nb skeleton dif. than tf0")
##############################################Check for broken skeleton block
            mini,d_piece=acorba.scanner.check_broken_skeleton(listends_real,listorigins_real,brokenfactor)

################if broken skeleton, link pieces and redetect skeletons
            if np.any(mini):
                print("Broken skeleton detector activated, recalculating angles")
                for broken in mini:
                    broken_piece=d_piece[broken]
                    broken_end=int(broken_piece[1])
                    broken_origin=int(broken_piece[0])
                    cv2.line(mega, pt1=(listorigins_real[broken_end][0],
                                            listorigins_real[broken_end][1]), 
                                            pt2=(listends[broken_origin][0],
                                            listends_real[broken_origin][1]), 
                                            color=1, thickness=10)

                listskel,listxy=acorba.scanner.root_skeleton(mega, areatresh)

                #Saving skeletons lenght in pixels
                if save_lenghts=="True":
                    lenghts_list=[len(skeleton[skeleton==1]) for skeleton in listskel]
                    ly=[]
                    lx=[]

#################Recalculating angles with the repaired skeletons#############
                #Create empty plot
                plt.close()
                fig2, axs2 = plt.subplots(len(listskel), 1, sharex = True)
                fig2.suptitle(images+" - tf "+str(imnb)+" repaired skeleton(s)")
                
                i=0
                listorigins=[]
                listends=[]
                listangles=[]
                while i<len(listskel):
                    #setup progress bar for each timeframe
                    if not sg.OneLineProgressMeter('Calculation in progress',
                                                   stack2, looplenght, 'single'):
                        sys.exit("User stop!!!!!!!!!!!!!!")

                    #calculate angle of one skeleton tip and retrieve other infos along
                    anglee, origins, ends,newskely,newskelx,start,endminus=acorba.scanner.root_angle(listskel[i],originvector)
                    listends.append(ends)
                    listorigins.append(origins)
                    listangles.append(anglee)

                    #Plot the angle analysis plot one root at a time, if i==0 create the blank plot
                    if i ==0:
                        for axe in axs:
                            axe.clear()
                    axs2[len(listskel)-i-1].plot(newskelx,newskely,
                                                color="grey")
                    axs2[len(listskel)-i-1].scatter(start[0],start[1],
                                                   color="red",s=15)
                    axs2[len(listskel)-i-1].scatter(endminus[0],endminus[1],
                                                   color='purple',s=15)
                    axs2[len(listskel)-i-1].axis('off')
                    axs2[len(listskel)-i-1].invert_yaxis()

                    #save ly and lx to calculate Vertical and Horizontal growth indexes
                    if save_lenghts=="True" and i==0:
                        listorigins_zero=origins
                        listends_zero=ends
                    if save_lenghts=="True":
                        ly.append(listends[i][0]-listorigins_zero[0])
                        lx.append(listends[i][1]-listorigins_zero[1])
                    i=i+1
                    plt.pause(0.001)
                
                plt.show()
                

                if args.rootplot=='True':
                    fig2.savefig(args.input_folder+'/'+images+'_'+'plateplot_'+str(imnb+1)+'_recalculated.png',
                                bbox_inches = 'tight',pad_inches = 0)

                print('Angle list for plate_'+images+"_timeframe_"+str(imnb+1)+':'+"recalculated angle")
                print(listangles)
                if save_lenghts=="True":
                    print("Length list for plate_"+images+"_timeframe_"+str(imnb+1))
                    print(lenghts_list)
            angletot.append([listangles])
            if save_lenghts=="True":
                lengthtot.append([lenghts_list])
                lytot.append([ly])
                lxtot.append([lx])
            imnb=imnb+1
            stack2=stack2+1
        if args.savesegmentation=="True":
            seg_list=np.array(seg_list)
            if args.superaccuracy=="True":
                seg_list=np.squeeze(seg_list, axis=(3,))
            path=args.input_folder+'//Saved Segmentations/'+images+'_'+'segmented.tif'
            tifffile.imwrite(path, seg_list,imagej=True,metadata={'axes': 'TYX'},shape=(seg_list.shape))
            if args.method=="Deep Machine Learning" and args.superaccuracy=="False":
                im_list=np.array(im_list)
                if args.superaccuracy=="True":
                    im_list=np.squeeze(im_list, axis=(3,))
                path=args.input_folder+'/Saved Segmentations/'+images+'_'+'original_padded.tif'
                tifffile.imwrite(path, im_list,imagej=True,metadata={'axes': 'TYX'},shape=(im_list.shape))
                del im_list
            del seg_list
    
        export_folder=args.input_folder

    #Output export as 1 xlsx per input

        po=0
        outangles=[]
        if args.normalization=='True':
            outanglesnorm=[]
        while po <len(listskel):
            temp=acorba.utils.Extract(angletot,po) 
            temp=pd.DataFrame(temp)
            if args.smooth=='False':
                temp=temp.rolling(2, min_periods=1).mean()
            temp=np.array(temp)
            if args.normalization=='True':
                tempnorm=temp-temp[0]
                tempplot=tempnorm
            else:
                tempplot=temp
            outangles.append(temp)
            if args.normalization=='True':
                outanglesnorm.append(tempnorm)
            po=po+1
        outangles=np.array(outangles)
        outangles=np.transpose(outangles)
        outangles.shape=(outangles.shape[1],outangles.shape[2])
        df = pd.DataFrame(outangles)
        if args.normalization=='True':
            outanglesnorm=np.array(outanglesnorm)
            outanglesnorm=np.transpose(outanglesnorm)
            outanglesnorm.shape=(outanglesnorm.shape[1],outanglesnorm.shape[2])
            dfnorm = pd.DataFrame(outanglesnorm)
            col_nb=0
            col_list=[]
            while col_nb<df.shape[1]:
                col_list.append("Root_"+str(col_nb+1))
                col_nb=col_nb+1
            col_list[0]=col_list[0]+" (Bottom)"
            col_list[len(col_list)-1]=col_list[len(col_list)-1]+" (Top)"
            dfnorm.columns=col_list
            df.columns=col_list
            if save_lenghts=='True':
                outlengths=np.array(lengthtot)
                lxtot=np.array(lxtot)
                lytot=np.array(lytot)
                outvgi=np.array(lytot/outlengths)
                outhgi=np.array(lxtot/outlengths)
                #The image was resize to 60% of its original size
                outlengths=outlengths+(outlengths/0.4)
                if int(args.scale)>1:
                    outlengths=outlengths*int(args.scale)
                outlengths.shape=(outlengths.shape[0],outlengths.shape[2])
                df_length = pd.DataFrame(outlengths)
                df_length.columns=col_list
                outvgi.shape=(outvgi.shape[0],outvgi.shape[2])
                df_vgi = pd.DataFrame(outvgi)
                df_vgi.columns=col_list
                outhgi.shape=(outhgi.shape[0],outhgi.shape[2])
                df_hgi = pd.DataFrame(outhgi)
                df_hgi.columns=col_list
                with pd.ExcelWriter(
                        export_folder+'/'+images+'__output.xlsx',
                        engine="xlsxwriter") as writer:
                    df.to_excel(writer,"Raw data")
                    dfnorm.to_excel(writer,"Normalized data")
                    df_length.to_excel(writer,"Partial lengths")
                    df_vgi.to_excel(writer,"VGI")
                    df_hgi.to_excel(writer,"HGI")

            else:
                with pd.ExcelWriter(
                        export_folder+'/'+images+'__output.xlsx',
                        engine="xlsxwriter") as writer:
                    df.to_excel(writer,"Raw data")
                    dfnorm.to_excel(writer,"Normalized data")

        else:
            with pd.ExcelWriter(
                    export_folder+'/'+images+'__output.xlsx',
                    engine="xlsxwriter") as writer:
                df.to_excel(writer,"Raw data")

    except Exception as e:
        print('Oups something went wrong!')
        print(e," Error on line {}".format(sys.exc_info()[-1].tb_lineno))
        listimage2.append(images)#Add the name of the stacks that crash to a second list for printing it at the end
        stack2=stack2+ntf-imnb#increment the loop to continue to the other images
    finally:
        if args.prediction == "First":
            imgsexp=np.array(imgsexp)
            predsexp=np.array(predsexp)
            acorba.plot.plot_imgs2(org_imgs=imgsexp, folder=args.input_folder, inc=imgname, pred_imgs=predsexp,nm_img_to_plot=1,color="green",kind="root")
        elif args.prediction == 'All':
            imgsexp=np.array(imgsexp)
            predsexp=np.array(predsexp)
            acorba.plot.plot_imgs2(org_imgs=imgsexp, folder=args.input_folder, inc=imgname, pred_imgs=predsexp,nm_img_to_plot=len(imgsexp),color="green",kind="root")            
        sleep(0.01)
        plt.close('all')
            #retrieve non processed image list
listremoved=listimage2
plt.close()
print("I'm done")
if len(listremoved)==0:
    print("everything went well")
else:
    print("The following files raised errors and were not analyzed fully: ",listremoved)
print("it took me",time() - start_time, "seconds to do my job!")
print("Um9vdCByb2NrISB6bG9vcDI3")
#collect garbage files
gc.collect()
