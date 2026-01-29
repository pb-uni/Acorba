# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 19:43:40 2021

@author: Nelson
"""
import dask
@dask.delayed
def full_scanner(images,args,looplenght,model,brokenfactor,stack2,listimage2,originvector):
    import os
    import sys
    import gc
    import acorba
    from time import sleep, time
    import tifffile
    import cv2
    from skimage.io import imread
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import PySimpleGUI as sg
    import dask
    if args.savesegmentation=="True":
        seg_list=[]#create segmentation list to transform into an array at the end and imwrite, save padded picture if machine learning regular
        if args.method=="Deep Machine Learning" and args.superaccuracy=="False":
            im_list=[]
    lene=imread(args.input_folder+'/'+images)#Stack import
    if min(lene.shape)==lene.shape[2]:#be sure the shape is (height,width,timeframes)
        lene=np.rollaxis(lene,2,0)
    ntf=int(min(lene.shape))#Finding the number of timeframes
    imgname=images#Store image name in a different object as images part of the for loop it was creating problem with the try/except functions
    if args.prediction in ("First","All"):
        #Create empty list for prediction export
        predsexp=[]
        imgsexp=[]
    try:#it slows down the code but allows to pass a stack that would be problematic and crashes the whole code
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
                predsexp.append(cv2.resize(mega2, (int(mega2.shape[1] * 50 / 100),int(mega2.shape[0] * 50 / 100))))
                imgsexp.append(cv2.resize(new_im, (int(new_im.shape[1] * 50 / 100),int(new_im.shape[0] * 50 / 100))))

####################################################batch root detection block
            listskel=acorba.scanner.root_skeleton(mega, areatresh)
            
            #Saving skeletons lenght in pixels
            if args.save_lenghts=="True":
                lenghts_list=[len(skeleton[skeleton==1]) for skeleton in listskel]
                ly=[]
                lx=[]
                
            #Create empty plot
            if imnb==0:
                fig, axs = plt.subplots(len(listskel), 1, sharex = True)
                fig.suptitle(images)

####################################################Angle measurements block
            i=0
            listorigins=[]
            listends=[]
            listangles=[]
            while i<len(listskel):
                #setup progress bar for each timeframe
                if not sg.OneLineProgressMeter('Calculation in progress', stack2, looplenght, 'single'):
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
                axs[len(listskel)-i-1].plot(newskelx,newskely,color="grey")
                axs[len(listskel)-i-1].scatter(start[0],start[1],color="red",s=15)
                axs[len(listskel)-i-1].scatter(endminus[0],endminus[1],color='purple',s=15)
                axs[len(listskel)-i-1].axis('off')
                axs[len(listskel)-i-1].invert_yaxis()
                
                #save ly and lx to calculate Vertical and Horizontal growth indexes
                if args.save_lenghts=="True" and i==0:
                    listorigins_zero=origins
                    listends_zero=ends
                if args.save_lenghts=="True":
                    ly.append(listends[i][0]-listorigins_zero[0])
                    lx.append(listends[i][1]-listorigins_zero[1])
                i=i+1

            plt.pause(.001)
            plt.show(block=False)
            
            if args.rootplot=='True':
                fig.savefig(args.input_folder+'/'+images+'_'+'plateplot_'+str(imnb+1)+'.png', bbox_inches = 'tight',pad_inches = 0)
            
            print('Angle list for plate_'+images+"_timeframe_"+str(imnb+1)+':')
            print(listangles)
            
            if args.save_lenghts=='True':
                print("Length list for plate_"+images+"_timeframe_"+str(imnb+1))
                print(lenghts_list)

##############################################Check for broken skeleton block
            mini,d_piece=acorba.scanner.check_broken_skeleton(listends,listorigins,brokenfactor)

################if broken skeleton, link pieces and redetect skeletons
            if np.any(mini):
                print("Broken skeleton detector activated, recalculating angles")
                for broken in mini:
                    broken_piece=d_piece[broken]
                    broken_end=int(broken_piece[1])
                    broken_origin=int(broken_piece[0])
                    cv2.line(mega, pt1=(listorigins[broken_end][0],listorigins[broken_end][1]), pt2=(listends[broken_origin][0],listends[broken_origin][1]), color=1, thickness=10)
                
                listskel=acorba.scanner.root_skeleton(mega, areatresh)
                
                #Saving skeletons lenght in pixels
                if args.save_lenghts=="True":
                    lenghts_list=[len(skeleton[skeleton==1]) for skeleton in listskel]
                    ly=[]
                    lx=[]

#################Recalculating angles with the repaired skeletons#############
                i=0
                listorigins=[]
                listends=[]
                listangles=[]
                while i<len(listskel):
                    #setup progress bar for each timeframe
                    if not sg.OneLineProgressMeter('Calculation in progress', stack2, looplenght, 'single'):
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
                    axs[len(listskel)-i-1].plot(newskelx,newskely,color="grey")
                    axs[len(listskel)-i-1].scatter(start[0],start[1],color="red",s=15)
                    axs[len(listskel)-i-1].scatter(endminus[0],endminus[1],color='purple',s=15)
                    axs[len(listskel)-i-1].axis('off')
                    axs[len(listskel)-i-1].invert_yaxis()
                    
                    #save ly and lx to calculate Vertical and Horizontal growth indexes
                    if args.save_lenghts=="True" and i==0:
                        listorigins_zero=origins
                        listends_zero=ends
                    if args.save_lenghts=="True":
                        ly.append(listends[i][0]-listorigins_zero[0])
                        lx.append(listends[i][1]-listorigins_zero[1])
                    i=i+1
                    
                plt.pause(.001)
                plt.show(block=False)
                
                if args.rootplot=='True':
                    fig.savefig(args.input_folder+'/'+images+'_'+'plateplot_'+str(imnb+1)+'_recalculated.png', bbox_inches = 'tight',pad_inches = 0)
                
                print('Angle list for plate_'+images+"_timeframe_"+str(imnb+1)+':'+"recalculated angle")
                print(listangles)
                if args.save_lenghts=="True":
                    print("Length list for plate_"+images+"_timeframe_"+str(imnb+1))
                    print(lenghts_list)
            angletot.append([listangles])
            if args.save_lenghts=="True":
                lengthtot.append([lenghts_list])
                lytot.append([ly])
                lxtot.append([lx])
            imnb=imnb+1
            stack2=stack2+1
        if args.savesegmentation=="True":
            seg_list=np.array(seg_list)
            path=args.input_folder+'//Saved Segmentations/'+images+'_'+'segmented.tif'
            tifffile.imwrite(path, seg_list)
            if args.method=="Deep Machine Learning" and args.superaccuracy=="False":
                im_list=np.array(im_list)
                path=args.input_folder+'/Saved Segmentations/'+images+'_'+'original_padded.tif'
                tifffile.imwrite(path, im_list)
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
            if args.save_lenghts=='True':
                outlengths=np.array(lengthtot)
                lxtot=np.array(lxtot)
                lytot=np.array(lytot)
                outvgi=np.array(lytot/outlengths)
                outhgi=np.array(lxtot/outlengths)
                outlengths=outlengths+(outlengths/0.4)#The image was resize to 60% of its original size
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
                        export_folder+'/'+images+'__output.xlsx',engine="xlsxwriter") as writer:
                    df.to_excel(writer,"Raw data")
                    dfnorm.to_excel(writer,"Normalized data")
                    df_length.to_excel(writer,"Partial lengths")
                    df_vgi.to_excel(writer,"VGI")
                    df_hgi.to_excel(writer,"HGI")

            else:
                with pd.ExcelWriter(
                        export_folder+'/'+images+'__output.xlsx',engine="xlsxwriter") as writer:
                    df.to_excel(writer,"Raw data")
                    dfnorm.to_excel(writer,"Normalized data")

        else:
            with pd.ExcelWriter(
                    export_folder+'/'+images+'__output.xlsx',engine="xlsxwriter") as writer:
                df.to_excel(writer,"Raw data")

    except Exception as e:
        print('Oups something went wrong!')
        print(e)
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
    return listimage2