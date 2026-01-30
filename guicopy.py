import subprocess
import sys
import FreeSimpleGUI as sg
import webbrowser
import os

version = '1.3'
date="Feb. 2021"

def runCommand(cmd, timeout=None, window=None):
    """ run shell command

	@param cmd: command to execute
	@param timeout: timeout for command execution

	@return: (return code from command, command output)
	"""

    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ''
    for line in p.stdout:
        line = line.decode(errors='replace' if (sys.version_info) < (3, 5)
                           else 'backslashreplace').rstrip()
        output += line
        print(line)
        window.Refresh()

    retval = p.wait(timeout)

    return (retval, output)
def main():
    input_defintion = {
        '-FOLDER-' : ('--input_folder'),
        '-EXP-' : ('--exp_type'),#menu deroulant
        '-NORM-' : ('--normalization'),#tick
        '-PRED-' : ('--prediction'),#none, all, first
        '-BINARY-' : ('--binary_folder'),#folder input "left blank machine learning
        '-ROOTPLOT-' : ('--rootplot'),
        '-METHOD-' : ('--method'),
        '-CUSTOM-' : ('--custom'),
        '-SMOOTH-' : ('--smooth'),
        '-ACCU-' : ('--superaccuracy'),
        '-SEG-' : ('--savesegmentation'),
        '-TRADMETH-' : ('--tradmethod'),
        '-LENGHTS-' : ('--save_lenghts'),
        '-SCALE-' : ('--scale')
        }
    #user input
    col1=sg.Column([[sg.Frame(layout=[
        [sg.Text('Experiment type', size=(30, 1), justification='l', tooltip="Experiment type"),
         sg.Combo(('Microscopy Through','Microscopy Sandwich','Scanner'),default_value='Microscopy Through',size=(30, 1),key='-EXP-')],
        [sg.Text('Segmentation method', size=(30, 1), justification='l', tooltip="Segmentation method"),
         sg.Combo(('Deep Machine Learning','Traditional','Own masks'),default_value='Deep Machine Learning',size=(30, 1),key='-METHOD-')],
        [sg.Text('Input Folder (containing .tif stacks)', size=(30, 1),justification='l', tooltip="The folder containing your images"),
         sg.Input(default_text='', key='-FOLDER-', size=(30,1), tooltip="The folder containing your images"),sg.FolderBrowse()],
        [sg.Text('Use your own masks? (left blank if not)',justification='l', size=(30, 1), tooltip="The folder containing your masks"),
         sg.Input(default_text='', key='-BINARY-', size=(30,1), tooltip="The folder containing your masks"),sg.FolderBrowse()],
        [sg.Button("Create your own masks with napari")],
        [sg.Text('Use your own models? (left blank if not)',justification='l', size=(30, 1), tooltip="The folder containing your models"),
         sg.Input(default_text='', key='-CUSTOM-', size=(30,1), tooltip="The folder containing your models"),sg.FolderBrowse()],
        ], title='User input',vertical_alignment="center")]])
    
    col2=sg.Column([[sg.Frame(layout=[
        #Analysis option
        [sg.CBox('Normalized the data to the first angle',default=True,key='-NORM-',size=(26,1))],
         [sg.CBox('Save lengths:', size=(13, 1),key='-LENGHTS-',default=False),
         sg.Text('1 pixels =', size=(7, 1), tooltip="Scale for lengths"),
         sg.Spin([i for i in range(1,20000)], initial_value=1, k='-SCALE-'),
         sg.Text('µm', size=(2, 1))]
        ], title='Analysis options', relief=sg.RELIEF_SUNKEN,vertical_alignment="center")],
        #scanner option
        [sg.Frame(layout=[
        [sg.CBox('Deactivate smoothing', size=(15, 1),key='-SMOOTH-',default=True)],
         [sg.CBox('Super accuracy (High RAM/GPU required)', size=(30, 1),key='-ACCU-',default=False)],
         [sg.Text('Traditional method', size=(14, 1), justification='r', tooltip="Traditional method"),
         sg.Combo(('Entropy','Threshold'),default_value='Entropy',size=(10, 1),key='-TRADMETH-')],
        ], title='Scanner options', relief=sg.RELIEF_SUNKEN,vertical_alignment="center")]])
    
    layout = [[sg.Text('Hello World! Let\'s measure some angles', font='Any 10')]]
    layout +=[

        [col1, col2],

        #Fake console (delayed)
        [sg.Text('Command Line Output:')],
        [sg.Multiline(size=(120,22), reroute_stdout=True, reroute_stderr=False, reroute_cprint=True,  write_only=True, font='Courier 8', autoscroll=True, key='-ML-')],
        
        #Buttons
        [sg.Button('Full Analysis'),sg.Button("Segmentation only"),sg.Button('Test segmentation'),sg.Button('Exit'),sg.Button("Report a bug"),sg.Button("Check for updates")],
        
        #Soft infos
        [sg.Text('Want to create a library for another specie? Share you images? Report a bug? Check for updates? Develop a new feature? see https://sourceforge.net/projects/acorba/', font='Any 8', text_color='brown')]]

    layout_advanced=[[sg.Button('Full Analysis')]]
    layout_export=        [
        [sg.CBox('Save analysis plots', size=(20, 1),key='-ROOTPLOT-'),
         sg.CBox('Save raw segmentations (tif)', size=(20, 1),key='-SEG-')],
        [sg.Text('Save root surface prediction/traditional segmentation of ')
        ,sg.Combo(('None','First','All'),default_value='None',size=(10, 1),key='-PRED-'),
        sg.Text(' timeframe(s) (Does not work with Own Masks)',size=(50,1))]]
    
    layout_infos=[[sg.Text('Charles University, Faculty of Sciences, Dpt. of Experimental Plant Biology, Cell Growth Lab, Prague, Czech Republic (NBC Serre and M Fendrych)', font='Any 8', text_color='brown')],
                  [sg.Button('Open Release file')]]
    layout2 =[[sg.TabGroup([[  sg.Tab('Analysis', layout),sg.Tab('Export', layout_export),
                               sg.Tab('Advanced settings', layout_advanced),sg.Tab('Release infos', layout_infos)]], key='-TAB GROUP-', expand_x=True, expand_y=True),]]
    window = sg.Window('ACORBA: Automatic Calculation Of Root Bending Angle____v'+version+" "+date, layout2, finalize=True,keep_on_top=False,
                       icon='acorbalogo2..ico')   # adding finalize in case a print is added later before read
    window.BringToFront()   
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):        # if window was closed
            break
        elif event == 'Full Analysis':                      # if start button
            parms = ''
            for key in input_defintion:
                parms += f'{input_defintion[key]} "{values[key]}" '
                if values['-EXP-']=='Scanner':
                        # the command that will be invoked with the parameters
                    command_to_run = r'scanner.py '
                else:
                        # the command that will be invoked with the parameters
                    command_to_run = r'microscope.py '
            command=command_to_run + parms
            thread=runCommand(cmd=command, window=window)
            thread.start()
        elif event == 'Test segmentation':
            parms = ''
            for key in input_defintion:
                parms += f'{input_defintion[key]} "{values[key]}" '
                if values['-EXP-']=='Scanner':
                    command_to_run=r'testsegmentation_scanner.py '
                else:
                    command_to_run = r'testsegmentation_micro.py '
            command2=command_to_run + parms
            runCommand(cmd=command2, window=window)
        elif event == 'Segmentation only':
            parms = ''
            for key in input_defintion:
                parms += f'{input_defintion[key]} "{values[key]}" '
                if values['-EXP-']=='Scanner':
                    command_to_run=r'Segmentation_only.py '
                else:
                    command_to_run = r'Segmentation_onlymicro.py '
            command2=command_to_run + parms
            runCommand(cmd=command2, window=window)
        elif event=="Report a bug":
            webbrowser.open('https://sourceforge.net/p/acorba/tickets/')
        elif event=="Create your own masks with napari":
            parms = ''
            for key in input_defintion:
                parms += f'{input_defintion[key]} "{values[key]}" '
                command_to_run=r'test.py '
            command2=command_to_run + parms
            runCommand(cmd=command2, window=window)
        elif event=="Check for updates":
            link = "https://sourceforge.net/projects/acorba/files/"
            import requests
            from utils import find_all
            try:
                with requests.get(link) as f:
                    file=f.text
                a=file.find("ACORBA_setup")
                b=file.find(".exe")
                a=list(find_all(file, "ACORBA_setup")) # [0, 5, 10, 15]
                b=list(find_all(file, ".exe"))
                aa=[]
                for (i,y) in zip(a,b):
                   aa.append(file[i+len("ACORBA_setup_v"):y])
                version_list = list(set(aa))
                final=[i for i in version_list if float(i)>float(version)]
            except Exception as e:
                print("oups I can't connect to that")
                print(e)
        
            if len(final)>0:
                answer=sg.popup_yes_no('New version available. Do you want to download it? version'+version, keep_on_top=True,title="version check")
                if answer == 'Yes':
                    webbrowser.open('https://sourceforge.net/projects/acorba/files/latest/download')                    
            else:
                sg.Popup('All good, you are currently running the latest version', keep_on_top=True,title="version check")
        elif event=="Open Release file":
            os.system("README.txt")
    window.close()



if __name__ == '__main__':
    sg.theme('Light Brown 8')
    main()
