
#!/usr/bin/env python3
import os
import sys
import subprocess
import FreeSimpleGUI as sg

# macOS cosmetic: silence the "system Tk is deprecated" warning (harmless on other OSes)
os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")

version = 'v1.2 Jan. 2026'

# ---------- Subprocess helpers -------------------------------------------------

def run_command(cmd_list, window=None):
    """
    Execute a command as a list (no shell), stream output to the Multiline if provided.
    Works reliably across macOS/Linux/Windows and handles spaces/quoting safely.
    """
    try:
        proc = subprocess.Popen(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
    except Exception as e:
        if window:
            window['-ML-'].write(f"[ERROR] Failed to start: {' '.join(cmd_list)}\n{e}\n")
        return 1, str(e)

    lines = []
    for line in proc.stdout:
        lines.append(line.rstrip('\n'))
        if window:
            window['-ML-'].write(line)
            window.refresh()

    rc = proc.wait()
    return rc, "\n".join(lines)


def build_param_list(values, mapping):
    """
    Convert GUI values into a flat list of CLI args:
    ['--flag', 'value', '--flag2', 'value2', ...]
    Checkboxes become '1' or '0' to preserve original behavior.
    Empty optional fields become '' (your downstream scripts decide what to do).
    """
    args = []
    for key, flag in mapping.items():
        val = values.get(key, '')
        if isinstance(val, bool):
            args.extend([flag, '1' if val else '0'])
        else:
            args.extend([flag, str(val or '')])
    return args


def script_for(exp_type, action):
    """
    Choose the worker script based on experiment type and the button pressed.
    """
    exp = (exp_type or '').strip()
    if action == 'Full Analysis':
        return 'scanner.py' if exp == 'Scanner' else 'microscope.py'
    if action == 'Test segmentation':
        return 'testsegmentation_scanner.py' if exp == 'Scanner' else 'testsegmentation_micro.py'
    if action == 'Segmentation only':
        return 'Segmentation_only.py' if exp == 'Scanner' else 'Segmentation_onlymicro.py'
    return None

# ---------- GUI ---------------------------------------------------------------

def main():
    # (Fixed typo) input_definition instead of input_defintion
    input_definition = {
        '-FOLDER-'   : '--input_folder',
        '-EXP-'      : '--exp_type',           # combo
        '-SAVEPLOT-' : '--saveplot',           # checkbox
        '-NORM-'     : '--normalization',      # checkbox
        '-PRED-'     : '--prediction',         # None / First / All
        '-BINARY-'   : '--binary_folder',      # optional folder (own masks)
        '-ROOTPLOT-' : '--rootplot',           # checkbox
        '-METHOD-'   : '--method',             # combo
        '-CUSTOM-'   : '--custom',             # optional folder (own models)
        '-SMOOTH-'   : '--smooth',             # checkbox
        '-ACCU-'     : '--superaccuracy',      # checkbox
        '-SEG-'      : '--savesegmentation',   # checkbox
        '-TRADMETH-' : '--tradmethod'          # combo
    }

    layout = [[sg.Text("Hello World! Let's measure some angles", font='Any 10')]]

    layout += [
        # General input infos
        [
            sg.Text('Experiment type', size=(35, 1), justification='r', tooltip="Experiment type"),
            sg.Combo(('Microscopy Through', 'Microscopy Sandwich', 'Scanner'),
                     default_value='Microscopy Through', size=(20, 1), key='-EXP-')
        ],
        [
            sg.Text('Segmentation method', size=(35, 1), justification='r', tooltip="Segmentation method"),
            sg.Combo(('Deep Machine Learning', 'Traditional', 'Own masks'),
                     default_value='Deep Machine Learning', size=(25, 1), key='-METHOD-')
        ],
        [
            sg.Text('Input Folder (containing .tif stacks)', size=(35, 1), justification='r',
                    tooltip="The folder containing your images"),
            sg.Input(default_text='', key='-FOLDER-', size=(40, 1)),
            sg.FolderBrowse()
        ],
        [
            sg.Text('Use your own masks? (left blank if not)', size=(35, 1), justification='r',
                    tooltip="The folder containing your masks"),
            sg.Input(default_text='', key='-BINARY-', size=(40, 1)),
            sg.FolderBrowse()
        ],
        [
            sg.Text('Use your own models? (left blank if not)', size=(35, 1), justification='r',
                    tooltip="The folder containing your models"),
            sg.Input(default_text='', key='-CUSTOM-', size=(40, 1)),
            sg.FolderBrowse()
        ],

        # General options
        [sg.Frame(
            layout=[
                [
                    sg.Checkbox('Normalized the data to the first angle', default=True, key='-NORM-', size=(30, 1)),
                    sg.Checkbox('Save angle/time plots', key='-SAVEPLOT-', size=(20, 1)),
                    sg.Checkbox('Save analysis plots', key='-ROOTPLOT-', size=(15, 1)),
                    sg.Checkbox('Save raw segmentations (tif)', key='-SEG-', size=(20, 1)),
                ],
                [
                    sg.Text('Save root surface prediction/traditional segmentation of '),
                    sg.Combo(('None', 'First', 'All'), default_value='None', size=(10, 1), key='-PRED-'),
                    sg.Text(' timeframe(s) (Does not work with Own Masks)')
                ]
            ],
            title='General options',
            relief=sg.RELIEF_SUNKEN,
            vertical_alignment="center"
        )],

        # Scanner options
        [sg.Frame(
            layout=[
                [
                    sg.Checkbox('Deactivate smoothing', size=(20, 1), key='-SMOOTH-', default=True),
                    sg.Checkbox('Super accuracy mode (High RAM/GPU highly recommended)', size=(50, 1),
                                key='-ACCU-', default=False),
                    sg.Text('Traditional method', size=(14, 1), justification='r', tooltip="Traditional method"),
                    sg.Combo(('Entropy', 'Threshold'), default_value='Entropy', size=(10, 1), key='-TRADMETH-'),
                ],
            ],
            title='Scanner options',
            relief=sg.RELIEF_SUNKEN,
            vertical_alignment="center"
        )],

        # Fake console
        [sg.Text('Command Line Output:')],
        [sg.Multiline(size=(125, 25), write_only=True, font='Courier 8',
                      autoscroll=True, key='-ML-')],

        # Buttons
        [sg.Button('Full Analysis'),
         sg.Button('Segmentation only'),
         sg.Button('Test segmentation'),
         sg.Button('Exit')],

        # Footer
        [sg.Text('Charles University, Faculty of Sciences, Dpt. of Experimental Plant Biology, '
                 'Cell Growth Lab, Prague, Czech Republic (NBC Serre and M Fendrych)',
                 font='Any 8', text_color='brown')],
        [sg.Text('MacOS port by Philippe Baumann, University of Fribourg - '
                 'Source code: https://sourceforge.net/projects/acorba/',
                 font='Any 8', text_color='black')]
    ]

    # NOTE: icon=None avoids PNG/ICO decode crashes on macOS/Tk
    window = sg.Window(
        'ACORBA: Automatic Calculation Of Root Bending Angle____' + version,
        layout,
        finalize=True,
        keep_on_top=False,
        icon=None
    )

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
            break

        if event in ('Full Analysis', 'Test segmentation', 'Segmentation only'):
            script = script_for(values.get('-EXP-'), event)
            if not script:
                window['-ML-'].write(f"[ERROR] Could not determine script for action: {event}\n")
                continue

            args = [sys.executable, script] + build_param_list(values, input_definition)
            window['-ML-'].write(f"\n[RUN] {' '.join(args)}\n")
            code, _ = run_command(args, window=window)
            window['-ML-'].write(f"[DONE] Exit code: {code}\n")

    window.close()


if __name__ == '__main__':
    sg.theme('Light Brown 3')
    main()