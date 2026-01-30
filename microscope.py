'''
nbcs, scanner.py script.
Loaded buy the ACORBA GUI to perform microscope root segmentation and angle
measurements.
'''
#Load libraries
import os
import cv2
import sys
import gc
import tifffile
import tifffile as tiff
from time import sleep, time
import numpy as np
from skimage.transform import resize
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import FreeSimpleGUI as sg
from pathlib import PurePath
import acorba

plt.ioff()

start_time = time()
print("ALL LIBRARIES ARE IMPORTED, ACORBA WILL START NOW\n##################################################")

#retrieve user inputs from the GUI
args=acorba.folder.parser()

#for debugging
#args=acorba.debug.micro_debug()
#save_lenghts='False'
save_lenghts='False'

#Set random seed to "The Answer to the Ultimate Question of Life, The Universe, and Everything."
np.random.seed(42)
#size of the model input
size=256

#Create a separate folder for segmentations so that they are not confused with the originals
if args.savesegmentation=="True":
    directory = args.input_folder+"/Saved Segmentations/"
    if not os.path.exists(directory):
        os.mkdir(directory)

#Load DML models and weights
if len(args.binary_folder)==0:
    from tensorflow.autograph import set_verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
    set_verbosity(0)
    if len(args.custom)>0:
        from keras.models import model_from_json
        print('Custom models and weights loading:')
        model_weights=os.listdir(args.custom)
        custom_models_list=[]
        custom_weights_list=[]
        for i in model_weights:
            if i.endswith(".json"):
                custom_models_list.append(i)
            if i.endswith('.h5'):
                custom_weights_list.append(i)
        ans_surface = 'surface'
        ans_tip = 'tip'
        for title in custom_models_list:
            if ans_surface in title:
                with open(args.custom+"/"+title, 'r') as json_file:
                    loaded_model_json = json_file.read()
                loaded_model = model_from_json(loaded_model_json)
                print('root surface model loaded')
            elif ans_tip in title:
                with open(args.custom+"/"+title, 'r') as json_file:
                    loaded_model_json = json_file.read()
                loaded_model_tip = model_from_json(loaded_model_json)
                print('root tip model loaded')
        for title in custom_weights_list:
            if ans_surface in title:
                loaded_model.load_weights(args.custom+"/"+title)
                print('root surface weights loaded')
            elif ans_tip in title:
                loaded_model_tip.load_weights(args.custom+"/"+title)
                print('root tip weights loaded')
    else:
        
        if args.exp_type=='Microscopy Through':
            from keras_unet.models import satellite_unet
            print("Default models and weights loading")
            loaded_model = satellite_unet(input_shape=(size, size, 1))
                #load root tip model
            loaded_model_tip = satellite_unet(input_shape=(size, size, 1))
                # load weights into new model
            loaded_model.load_weights("models/model_through_surface_256_sat_dice_update22092021_138.h5")
            loaded_model_tip.load_weights("models/model_through_tip_256_sat_dice_update22092021_193.h5")
        else:
            from keras_unet.models import custom_unet
            loaded_model=custom_unet(
                input_shape=(256, 256, 1),
                use_batch_norm=True,
                num_classes=1,
                filters=64,
                dropout=0.2,
                output_activation='sigmoid')
            loaded_model_tip=custom_unet(
                input_shape=(256, 256, 1),
                use_batch_norm=True,
                num_classes=1,
                filters=64,
                dropout=0.2,
                output_activation='sigmoid')
            loaded_model.load_weights("models/Microscopy_sandwich_256_dicecoef_customunet_01102021.h5")
            loaded_model_tip.load_weights("models/Microscopy_sandwich_TIP_256_dicecoef_customunet_01102021.h5")
    print("Models and weights are loaded")
     

#retrieve images list
listimage_temp=os.listdir(args.input_folder)
rootlist=[i for i in listimage_temp if i.endswith(".tif")]
del listimage_temp
print("The following files will be processed")
print(rootlist)

#list for root that crash
rootlist2=[]

#Setting the progress bar total length by opening every image lenght info and summing their timeframe numbers
looplenght=acorba.folder.looplenght_determinator(
    list_image=rootlist,folder=args.input_folder)

#Start for loop for all the file in the rootlist
angletotal=[]

if save_lenghts=="True":
    coord_tip_total=[]
inc_stack2=0

for inc_root in rootlist:
    sg.theme('Light Brown 8')
    print("Processing file "+inc_root)
    test= tiff.imread(args.input_folder+'/'+inc_root)
    test_shape=test.shape
    tradlist=[]
    print("Converting stack to an array and resizing/padding to 256x256")
    X_test = np.zeros((len(test),size, size), dtype=np.float32)
    try:
        n=0
        while n<len(test):
            
            sg.OneLineProgressMeter(inc_root+' import, padding, resizing', n+1, len(test), 'Import')
            
            im=(test[n]/255).astype('uint8')
            tradlist.append(im)
            
            if im.shape[0]!=im.shape[1]:
                X_test[n]=acorba.microscope.padding_resizing(image_input=im,size=size)/255
            else:
                X_test[n] = resize(im, (size, size), mode='constant', preserve_range=True)/255

            n=n+1

        if args.savesegmentation=="True":    
            path=args.input_folder+'//Saved Segmentations/'+inc_root+'_'+'original_resized256.tif'
            tifffile.imwrite(path, X_test)
        tradlist=np.array(tradlist)

        #Predictions
        if args.method=="Deep Machine Learning":
            X_test=np.expand_dims(X_test,3)
            print("Prediction of root surface by deep machine learning")
            preds_test = loaded_model.predict(X_test, verbose=2,batch_size=1)
            prediction =(preds_test > 0.5).astype(np.uint8)
            print("Prediction of root tip by deep machine learning")
            preds_test = loaded_model_tip.predict(X_test, verbose=2,batch_size=1)
            prediction_tip =(preds_test > 0.5).astype(np.uint8)
            if args.prediction =="First":
                print("Exporting first timeframe prediction")
                acorba.microscope.plot_imgs(org_imgs=X_test, folder=args.input_folder, inc=inc_root, pred_imgs=prediction,nm_img_to_plot=1,color="green",kind="root") #Need to be in the exp directory
                acorba.microscope.plot_imgs(org_imgs=X_test,folder=args.input_folder, inc=inc_root, pred_imgs=prediction_tip,nm_img_to_plot=1,color="red",kind="tip") #SAME
            elif args.prediction =='All':
                print("Exporting all timeframe predictions")
                acorba.microscope.plot_imgs(org_imgs=X_test, folder=args.input_folder, inc=inc_root, pred_imgs=prediction,nm_img_to_plot=len(X_test),color="green",kind="root") #Need to be in the exp directory
                acorba.microscope.plot_imgs(org_imgs=X_test,folder=args.input_folder, inc=inc_root, pred_imgs=prediction_tip,nm_img_to_plot=len(X_test),color="red",kind="tip") #SAME
        elif args.method=='Own masks':
            prediction=tiff.imread(args.binary_folder+'/surface/'+inc_root)
            prediction=prediction.astype('uint8')
            prediction_tip=tiff.imread(args.binary_folder+'/tip/'+inc_root)
            prediction_tip=prediction_tip.astype('uint8')
            if np.max(prediction.shape)>size:
                prediction= resize(prediction, (len(prediction),size, size), mode='constant', preserve_range=True)
                prediction_tip= resize(prediction_tip, (len(prediction_tip),size, size), mode='constant', preserve_range=True)
            if np.max(prediction>1):
                prediction=prediction/np.max(prediction)
            if np.max(prediction_tip>1):
                prediction_tip=prediction_tip/np.max(prediction_tip)
        else:
            print('Traditionnal segmentation of root surface')
            prediction=[]
            if args.exp_type=='Microscopy Through':
                for img_pred in X_test:
                    prediction.append(acorba.microscope.trad_micro_through(img_pred))
            else:
                for img_pred in tradlist:
                    prediction.append(acorba.microscope.trad_micro_sandwich(img_pred))
            prediction=np.array(prediction)
            X_test=np.expand_dims(X_test,3)
            print("Prediction of root tip by deep machine learning")
            preds_test = loaded_model_tip.predict(X_test, verbose=2,batch_size=1)
            prediction_tip =(preds_test > 0.5).astype(np.uint8)
            if args.prediction =="First":
                print("Exporting first timeframe prediction")
                acorba.microscope.plot_imgs(org_imgs=X_test, folder=args.input_folder+'/', inc=inc_root, pred_imgs=prediction,nm_img_to_plot=1,color="green",kind="root") #Need to be in the exp directory
                acorba.microscope.plot_imgs(org_imgs=X_test,folder=args.input_folder+'/', inc=inc_root, pred_imgs=prediction_tip,nm_img_to_plot=1,color="red",kind="tip") #SAME
            elif args.prediction =='All':
                print("Exporting all timeframe predictions")
                acorba.microscope.plot_imgs(org_imgs=X_test, folder=args.input_folder+'/', inc=inc_root, pred_imgs=prediction,nm_img_to_plot=len(X_test),color="green",kind="root") #Need to be in the exp directory
                acorba.microscope.plot_imgs(org_imgs=X_test,folder=args.input_folder+'/', inc=inc_root, pred_imgs=prediction_tip,nm_img_to_plot=len(X_test),color="red",kind="tip") #SAME
        if args.savesegmentation=="True":    
            path=args.input_folder+'//Saved Segmentations/'+inc_root+'_'+'root_tip_256.tif'
            tifffile.imwrite(path, prediction_tip)
            path=args.input_folder+'//Saved Segmentations/'+inc_root+'_'+'root_surface_256.tif'
            tifffile.imwrite(path, prediction)

        #Angle calculation
        print("Calculation of the angles")
        
        slopetip=[]
        inter=[]
        angleslist2=[]
        flipping=0
        inc_stack=0
        if save_lenghts=="True":
                coord_tip=[]
        while inc_stack<len(X_test):
            if not sg.OneLineProgressMeter('Calculation in progress', inc_stack2, looplenght, 'Angles calculation'):
                sys.exit("User stop!!!!!!!!!!!!!!")

            #import binary masks
            ol=prediction[inc_stack]
            oltip=prediction_tip[inc_stack]
            
            ############Binary preprocessessing#############
            #Fill holes
            oltip=acorba.microscope.fill_holes(binary_image=oltip)
            ol=acorba.microscope.fill_holes(binary_image=ol)
            
            #Flip the tip on the left side if on the right side (horizontal flip)
            if inc_stack==0 and np.sum(ol[:,250])<2:
                flipping=1
            if flipping==1:
                ol=cv2.flip(ol,1)
                oltip=cv2.flip(oltip,1)
                
            #Cleaning
            oltip=acorba.microscope.binary_cleaner(oltip)
            ol=acorba.microscope.binary_cleaner(ol)
            
            #Find contour and clean it
            kernel = np.ones((2,2),np.uint8)
            contour=cv2.morphologyEx(oltip, cv2.MORPH_GRADIENT, kernel)
            perimeter=np.where(contour==1)
            contour_root=cv2.morphologyEx(ol, cv2.MORPH_GRADIENT, kernel)
            perimeter_root=np.where(contour_root==1)
            
            #Define area of the root tip on the root surface prediction
            M = cv2.moments(oltip)
            # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            circle=cv2.circle(ol.copy(), (cX, cY), int(args.circlepix), 2, -1)
            circle[circle==2]=0
            ol2=ol.copy()
            ol2=ol2-circle
            contour_tip=cv2.morphologyEx(ol2, cv2.MORPH_GRADIENT, kernel)
            perimeter_tip=np.where(contour_tip==1)
            contours, hierarchy = cv2.findContours(ol2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            (xd,yd),radius = cv2.minEnclosingCircle(np.stack((perimeter_tip[1],perimeter_tip[0]),axis=-1))
            center = (int(xd),int(yd))
            radius = int(radius)
            mask2=cv2.drawContours(ol2.copy(), contours[0],-1, color=3)
            rr=cv2.circle(mask2,center,radius,4,2)
            peri_circle=np.where(cv2.circle(mask2,center,radius,4,2)==4)
            coord_circle=np.stack((peri_circle[1],peri_circle[0]),axis=-1)
            coord_peri=np.stack((perimeter_tip[1],perimeter_tip[0]),axis=-1)
            intcp=list(set(map(tuple,coord_peri)).intersection(set(map(tuple,coord_circle))))
            intercp=list(zip(*intcp))
            intercpx=np.array(intercp[0])
            intercpy=np.array(intercp[1])
            intercp2=np.stack((intercpx,intercpy),-1)

            km = KMeans(
            n_clusters=2, init='random',
            n_init=10, max_iter=300, 
            tol=1e-04, random_state=0
            )   
            y_km = km.fit_predict(intercp2)
            g1=len(np.where(y_km==0)[0])
            g2=len(np.where(y_km==1)[0])
            indd=y_km.argsort()
            intercp2=intercp2[indd]
            
            if g1<g2:
                tipx=intercp2[0][0]
                tipy=intercp2[0][1]
            else:
                m=len(intercp2)
                tipx=intercp2[m-1][0]
                tipy=intercp2[m-1][1]

            #Skeletonize root tip
            skel = skeletonize(ol2,method='lee')
            skel=np.array(skel)
            skel=np.uint8(skel)
            cfit=np.where(skel==1)
            endpoints_skel=acorba.scanner.skeleton_endpoints(skel)
            p1temp=(endpoints_skel[1][0],endpoints_skel[0][0])
            p2temp=(endpoints_skel[1][1],endpoints_skel[0][1])
            dtip_P1=np.sqrt((p1temp[0]-tipx)**2+(p1temp[1]-tipy)**2)
            dtip_P2=np.sqrt((p2temp[0]-tipx)**2+(p2temp[1]-tipy)**2)
            
            if dtip_P1<dtip_P2:
                origin=p1temp
                end=p2temp
            else:
                origin=p2temp
                end=p1temp

            points = cfit[1],cfit[0]

            pp=0
            newskelx=[]
            newskely=[]
            while pp<len(cfit[1])-1:
                if pp==0:
                    listpoint=(cfit[1],cfit[0])
                    start=origin
                    indextormx=np.where(listpoint[0]==start[0])
                    indextormy=np.where(listpoint[1]==start[1])
                    indextorm=int(np.intersect1d(indextormx, indextormy)[0])
                    listpoint=[np.delete(listpoint[0],indextorm),np.delete(listpoint[1],indextorm)]
                    newskelx.append(start[0])
                    newskely.append(start[1])
                else:
                    listpoint=[np.delete(listpoint[0],closestindex),np.delete(listpoint[1],closestindex)]

                uu=0
                distance=[]
                while uu<len(listpoint[1]):
                    distance.append(np.sqrt((start[0] - listpoint[0][uu]) ** 2 + (start[1] - listpoint[1][uu]) ** 2))
                    uu=uu+1
                distancear=np.array(distance)
                closestindex=int(np.where(distance==min(distance))[0])
                newskelx.append(listpoint[0][closestindex])
                newskely.append(listpoint[1][closestindex])
                pp=pp+1
            idx=int(len(newskelx)/2)    

            #Fit root tip skeleton (avoid branches)
            splines=np.polyfit(newskelx,newskely,1)
            poly1d = np.poly1d(splines)
            xnew=newskelx
            ynew= poly1d(xnew)
            Tipmax_x=[origin[0],newskelx[idx]]
            Tipmax_y=[origin[1],ynew[idx]]

            #adapt fitting line x to either go left of right (<90° or >90°) (only one first timeframe, avoid root tip detection jumping one side to another during analysis)
            if inc_stack==0:
                xnew0=np.linspace(0,newskelx[idx],500)
                ynew0= poly1d(xnew0)
                xnew1=np.linspace(newskelx[idx],size,500)
                ynew1= poly1d(xnew1)
                blank = np.zeros([ol.shape[0],ol.shape[0]],dtype=np.uint8)
                start0=(int(xnew0[0]),int(ynew0[0]))
                end0=(int(xnew0[len(xnew0)-1]),int(ynew0[len(ynew0)-1]))
                mask0=cv2.line(blank, start0,end0, color=(1))
                sum0=np.sum(cv2.bitwise_and(ol,mask0 ))
                blank = np.zeros([ol.shape[0],ol.shape[0]],dtype=np.uint8)
                start1=(int(xnew1[0]),int(ynew1[0]))
                end1=(int(xnew1[len(xnew1)-1]),int(ynew1[len(ynew1)-1]))
                mask1=cv2.line(blank, start1,end1, color=(1))
                sum1=np.sum(cv2.bitwise_and(ol,mask1 ))
                
                if origin[0]<newskelx[idx] and sum0<sum1:
                    direction=0
                    xnew=xnew0
                    ynew=ynew0
                elif origin[0]>newskelx[idx] and sum0>sum1:
                    direction=1
                    xnew=xnew1
                    ynew=ynew1
                elif origin[0]<newskelx[idx] and sum0>sum1:
                    direction=1
                    xnew=xnew1
                    ynew=ynew1
                elif origin[0]>newskelx[idx] and sum0<sum1:
                    direction=0
                    xnew=xnew0
                    ynew=ynew0
            else:
                if direction==0:
                    xnew=np.linspace(0,newskelx[idx],500)
                else:
                    xnew=np.linspace(newskelx[idx],size,500)
                ynew=poly1d(xnew)
            '''
            plt.scatter(peri_circle[1],peri_circle[0])
            plt.scatter(perimeter_root[1],perimeter_root[0])
            plt.scatter(perimeter_tip[1],perimeter_tip[0])
            plt.scatter(intercpx,intercpy,color="red")
            plt.scatter(Tipmax_x,Tipmax_y)
            plt.plot(xnew,ynew)'''
        

            #Find the closest pair of coordinates between perimeter and middle model
            listx_fit=xnew.tolist()
            listy_fit=ynew.tolist()
            listx_mask=perimeter_tip[1].tolist()
            listy_mask=perimeter_tip[0].tolist()
            i=0
            distance=[]
            points=[]
            lenghtfit=len(listx_fit)
            lenghtmask=len(listx_mask)
            while i < lenghtmask:
                totestx_mask=listx_mask[i]
                totesty_mask=listy_mask[i]
                yy=0
                while yy < lenghtfit:    
                    totestx_fit=listx_fit[yy]
                    totesty_fit=listy_fit[yy]
                    distance.append(np.sqrt((totestx_fit - totestx_mask) ** 2 + (totesty_fit - totesty_mask) ** 2))
                    points.append([totestx_fit, totesty_fit])
                    yy=yy+1
                i=i+1
            closest_d=min(distance)
            closest_index=distance.index(closest_d)
            closest_coord=points[closest_index]
            Tipmax_x[0]=closest_coord[0]
            Tipmax_y[0]=closest_coord[1]

            #Calculate the angle
            angle2=acorba.scanner.angle_to((Tipmax_x[1],Tipmax_y[1]),(Tipmax_x[0],Tipmax_y[0]), clockwise=False)
            angle2=angle2-180
            angleslist2.append(angle2)
            if save_lenghts=="True":
                coord_tip.append([Tipmax_x[0],Tipmax_y[0]])
                
            #Plot
            plt.clf()
            plt.scatter(perimeter_root[1],perimeter_root[0],color="gainsboro",s=0.5)
            plt.scatter(perimeter_tip[1],perimeter_tip[0],color="red",s=0.5)
            plt.plot(xnew,ynew,linewidth=2,color='orange')
            plt.plot(newskelx,newskely)
            plt.xlim([0, size])
            plt.ylim([size, 0])
            plt.scatter(Tipmax_x[1],Tipmax_y[1],color="purple")
            plt.axis("off")
            stamp=r'File: '+inc_root+' /// Timeframe: '+str(inc_stack+1)+' /// Angle: '+str(angle2)+'°'
            plt.text(1, 1, stamp, fontsize=6)
            plt.pause(.001)
            plt.show(block=False)
            
            if args.rootplot=='True':
                plt.savefig(args.input_folder+'/'+inc_root+'_'+'rootplot_'+str(inc_stack+1)+'.png')
            
            #Increment timeframe loops
            inc_stack=inc_stack+1
            inc_stack2=inc_stack2+1
            sleep(0.02)
        if save_lenghts=="True":
            lenghts=[0]
            i=0
            while i<len(X_test)-1:
                distance=np.sqrt((coord_tip[0][0] - coord_tip[i+1][0]) ** 2 + (coord_tip[0][1] - coord_tip[i][1]) ** 2)
                lenghts.append(distance)
                i+=1
            print("calculated lenghts: ")
            print(lenghts)
        angletotal.append((angleslist2))
        
        if save_lenghts=='True':
            coord_tip_total.append(lenghts)
            
        rootlist2.append(inc_root)
    except Exception as e:
        print('Oups something went wrong!')
        print("Root: "+inc_root+" crashed at timeframe "+str(inc_stack))
        print("Error type: ")
        print(e)

        inc_stack2=inc_stack2+len(test)-inc_stack
        if inc_stack==0:
            angleslist2.append(0)
            angleslist2.append(0)
            angletotal.append((angleslist2))
        else:
            angletotal.append((angleslist2))
        if save_lenghts=='True':
            coord_tip_total.append(lenghts)

listremoved=acorba.utils.Diff(rootlist,rootlist2)
angletotalf=np.array(angletotal,dtype=object)
angletotalf2=angletotalf.copy()
angletoplot=angletotalf2.copy()
if args.normalization=='True':
    i=0
    angletotalf2norm=angletotalf.copy()
    while i<len(angletotalf):
        angletotalf2norm[i]=np.array(angletotalf2[i])-angletotalf2[i][0]
        i=i+1
    angletoplot=angletotalf2norm.copy()
    

#Export output as a xlsx file containing all the angles
print('writing output')

outangles=np.array(angletotalf2)
len_list=[len(a) for a in outangles]
len_list=np.array(len_list)
len_dif=np.unique(len_list)

if len(len_dif)>1:
    #outangles=np.transpose(outangles)
    d = dict(enumerate(outangles.flatten(), 1))
    df = pd.DataFrame.from_dict(d, orient='index')
    df=df.transpose()
    df.columns=rootlist
else:
    outangles=np.transpose(outangles)
    df = pd.DataFrame(outangles)
    df.columns=rootlist
    
if args.normalization=='True':
    outanglenorm=np.array(angletotalf2norm)
    if len(len_dif)>1:
        d2 = dict(enumerate(outanglenorm.flatten(), 1))
        df2 = pd.DataFrame.from_dict(d2, orient='index')
        df2=df2.transpose()
        df2.columns=rootlist
    else:
        outanglenorm=np.transpose(outanglenorm)
        df2 = pd.DataFrame(outanglenorm)
        df2.columns=rootlist

pathout = PurePath(args.input_folder)
foldername=pathout.name

if save_lenghts=='True':
    out_distance=np.array(coord_tip_total)
    out_distance=(out_distance*np.max(test_shape))/256#Everything was calculated on reduced picture
    if int(args.scale)>1:
        out_distance=out_distance*int(args.scale)
    out_distance=np.transpose(out_distance)
    df3 = pd.DataFrame(out_distance)
    df3.columns=rootlist

export_folder=args.input_folder
# write dataframe to excel
with pd.ExcelWriter(export_folder+'/'+foldername+'_output.xlsx',engine="xlsxwriter") as writer:
    if args.normalization=='True' and save_lenghts=='True':
        df.to_excel(writer,'Raw data')
        df2.to_excel(writer,'Normalized data')
        df3.to_excel(writer,'Lenghts')
    
    elif args.normalization=='False' and save_lenghts=='True':
        df.to_excel(writer,'Raw data')
        df3.to_excel(writer,'Lenghts')
    
    elif args.normalization=='True' and save_lenghts=='False':
        df.to_excel(writer,'Raw data')
        df2.to_excel(writer,'Normalized data')
    
    elif args.normalization=='False' and save_lenghts=='False':
        df.to_excel(writer,'Raw data')
        
    #writer.save()

print("I'm done")
if len(listremoved)==0:
    print("everything went well")
else:
    print("The following files raised errors and were not or only partially analyzed: ",listremoved)
print("it took me",time() - start_time, "seconds to do my job!")
print("Um9vdCByb2NrISB6bG9vcDI3")
gc.collect()