#!/usr/bin/env python3
"""
ACORBA: Automatic Calculation Of Root Bending Angle
Modern GUI redesign with contemporary aesthetics
"""
import os
import sys
import subprocess
import webbrowser
from datetime import date
import FreeSimpleGUI as sg

# macOS cosmetic: silence the "system Tk is deprecated" warning
os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")

version = 'v1.3 Jan. 2026'
today = date.today()
d = today.strftime("%b-%d-%Y")

# ---------- Modern Color Scheme -------------------------------------------------
COLORS = {
    'bg_primary': '#FAFAFA',
    'bg_secondary': '#FFFFFF',
    'bg_tertiary': '#F5F5F5',
    'accent': '#2E7D32',
    'accent_hover': '#388E3C',
    'text_primary': '#1A1A1A',
    'text_secondary': '#666666',
    'border': '#E0E0E0',
    'console_bg': '#1E1E1E',
    'console_text': '#D4D4D4',
    'success': '#43A047',
    'error': '#E53935',
    'warning': '#FB8C00'
}

# Core arguments for all scripts
core_args = {
    '-FOLDER-'   : '--input_folder',
    '-EXP-'      : '--exp_type',
    '-METHOD-'   : '--method',
    '-BINARY-'   : '--binary_folder',
    '-CUSTOM-'   : '--custom',
}

# Additional arguments for full analysis
analysis_args = {
    '-SAVEPLOT-' : '--saveplot',
    '-NORM-'     : '--normalization',
    '-PRED-'     : '--prediction',
    '-ROOTPLOT-' : '--rootplot',
    '-SEG-'      : '--savesegmentation',
}

# Microscopy-specific arguments
microscopy_args = {
    '-PIX-'      : '--circlepix',
}

# Scanner-specific arguments
scanner_args = {
    '-SMOOTH-'   : '--smooth',
    '-ACCU-'     : '--superaccuracy',
    '-TRADMETH-' : '--tradmethod',
    '-BROKEN-'   : '--broken',
    '-VECTOR-'   : '--vector'
}

# ---------- Subprocess helpers -------------------------------------------------

def run_command(cmd_list, window=None, save_log=True, output_folder=None):
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
        line_stripped = line.rstrip('\n')
        lines.append(line_stripped)
        if window:
            window['-ML-'].write(line)
            window.refresh()

    rc = proc.wait()
    output = "\n".join(lines)
    
    if save_log and output_folder:
        try:
            log_path = os.path.join(output_folder, f'analysis_log-{d}.txt')
            with open(log_path, 'w') as f:
                f.write(output)
        except Exception as e:
            if window:
                window['-ML-'].write(f"[WARNING] Could not save log: {e}\n")
    
    return rc, output


def build_param_list(values, mapping):
    """Convert GUI values into a flat list of CLI args"""
    args = []
    for key, flag in mapping.items():
        val = values.get(key, '')
        if isinstance(val, bool):
            args.extend([flag, '1' if val else '0'])
        else:
            args.extend([flag, str(val or '')])
    return args


def script_for(exp_type, action):
    """Choose the worker script based on experiment type and button pressed"""
    exp = (exp_type or '').strip()
    if action == 'Full Analysis':
        return 'scanner.py' if exp == 'Scanner' else 'microscope.py'
    if action == 'Test segmentation':
        return 'testsegmentation_scanner.py' if exp == 'Scanner' else 'testsegmentation_micro.py'
    if action == 'Segmentation only':
        return 'Segmentation_only.py' if exp == 'Scanner' else 'Segmentation_onlymicro.py'
    if action == 'Binary Mask Assistant':
        return 'test.py'
    return None


def check_for_updates(current_version):
    """Check if newer version is available on SourceForge"""
    link = "https://sourceforge.net/projects/acorba/files/"
    try:
        import requests
        with requests.get(link, timeout=10) as f:
            file = f.text
        
        version_list = []
        search_str = "ACORBA_setup_v"
        pos = 0
        while True:
            start = file.find(search_str, pos)
            if start == -1:
                break
            end = file.find(".exe", start)
            if end == -1:
                break
            ver = file[start + len(search_str):end]
            try:
                float(ver)
                version_list.append(ver)
            except ValueError:
                pass
            pos = end + 1
        
        version_list = list(set(version_list))
        current_ver_num = float(current_version.replace('v', '').split()[0])
        newer = [v for v in version_list if float(v) > current_ver_num]
        
        return newer
    except Exception as e:
        return None, str(e)


# ---------- Modern GUI ---------------------------------------------------------------

def create_modern_gui():
    """Create the modern GUI layout"""
    
    # Core arguments for all scripts
    core_args = {
        '-FOLDER-'   : '--input_folder',
        '-EXP-'      : '--exp_type',
        '-METHOD-'   : '--method',
        '-BINARY-'   : '--binary_folder',
        '-CUSTOM-'   : '--custom',
    }
    
    # Additional arguments for full analysis
    analysis_args = {
        '-SAVEPLOT-' : '--saveplot',
        '-NORM-'     : '--normalization',
        '-PRED-'     : '--prediction',
        '-ROOTPLOT-' : '--rootplot',
        '-SEG-'      : '--savesegmentation',
    }
    
    # Microscopy-specific arguments
    microscopy_args = {
        '-PIX-'      : '--circlepix',
    }
    
    # Scanner-specific arguments
    scanner_args = {
        '-SMOOTH-'   : '--smooth',
        '-ACCU-'     : '--superaccuracy',
        '-TRADMETH-' : '--tradmethod',
        '-BROKEN-'   : '--broken',
        '-VECTOR-'   : '--vector'
    }
    
    # Combined for backward compatibility
    input_definition = {**core_args, **analysis_args, **microscopy_args, **scanner_args}

    # Custom button appearance
    button_config = {
        'size': (18, 1),
        'button_color': (COLORS['bg_secondary'], COLORS['accent']),
        'border_width': 0,
        'font': ('Helvetica', 11)
    }
    
    secondary_button = {
        'size': (15, 1),
        'button_color': (COLORS['text_primary'], COLORS['bg_tertiary']),
        'border_width': 0,
        'font': ('Helvetica', 10)
    }
    
    # Browse button config (FolderBrowse has different supported args)
    browse_button = {
        'size': (10, 1),
        'button_color': (COLORS['text_primary'], COLORS['bg_tertiary']),
        'font': ('Helvetica', 10)
    }

    # Header
    header = [
        [sg.Text('ACORBA', font=('Helvetica', 28, 'bold'), text_color=COLORS['accent'],
                pad=((20, 0)))],
        [sg.Text("Automatic Calculation Of Root Bending Angle, Let's measure some angles!", 
                font=('Helvetica', 11), text_color=COLORS['text_secondary'],
                pad=((20, 0), (0, 8)))]
    ]

    # Configuration Section
    config_frame = sg.Frame('Configuration', [
        [
            sg.Column([
                [sg.Text('Experiment Type', size=(18, 1), font=('Helvetica', 10, 'bold'),
                        text_color=COLORS['text_primary'])],
                [sg.Combo(('Microscopy Through', 'Microscopy Sandwich', 'Scanner'),
                         default_value='Microscopy Through', size=(32, 1), key='-EXP-',
                         font=('Helvetica', 10), readonly=True)],
                
                [sg.Text('Segmentation Method', size=(18, 1), font=('Helvetica', 10, 'bold'),
                        text_color=COLORS['text_primary'], pad=((0, 0), (8, 0)))],
                [sg.Combo(('Deep Machine Learning', 'Traditional', 'Own masks'),
                         default_value='Deep Machine Learning', size=(32, 1), key='-METHOD-',
                         font=('Helvetica', 10), readonly=True)],
            ], pad=(15, 6)),
            
            sg.Column([
                [sg.Text('Input Folder', size=(18, 1), font=('Helvetica', 10, 'bold'),
                        text_color=COLORS['text_primary'])],
                [sg.Input(default_text='', key='-FOLDER-', size=(35, 1), font=('Helvetica', 10)),
                 sg.FolderBrowse(button_text='Browse', **browse_button)],
                
                [sg.Checkbox('Normalize to first angle', default=True, key='-NORM-',
                            font=('Helvetica', 10), text_color=COLORS['text_primary'],
                            pad=((0, 0), (8, 0)))],
            ], pad=(15, 6))
        ]
    ], font=('Helvetica', 11, 'bold'), relief=sg.RELIEF_FLAT, 
    border_width=1, pad=(20, 8))

    # Advanced Options - Collapsible style
    advanced_frame = sg.Frame('Advanced Options', [
        [
            sg.Column([
                [sg.Text('Custom Masks', font=('Helvetica', 10, 'bold'))],
                [sg.Input(default_text='', key='-BINARY-', size=(30, 1), font=('Helvetica', 9)),
                 sg.FolderBrowse(button_text='...', size=(3, 1))],
                
                [sg.Text('Custom Models', font=('Helvetica', 10, 'bold'), pad=((0, 0), (8, 0)))],
                [sg.Input(default_text='', key='-CUSTOM-', size=(30, 1), font=('Helvetica', 9)),
                 sg.FolderBrowse(button_text='...', size=(3, 1))],
            ], pad=(15, 0)),
            
            sg.Column([
                [sg.Text('Microscopy Options', font=('Helvetica', 10, 'bold'))],
                [sg.Text('Circle crop size (px):', size=(18, 1)),
                 sg.Spin([i for i in range(1, 20000)], initial_value=40, key='-PIX-', size=(8, 1))],
                
                [sg.Text('Scanner Options', font=('Helvetica', 10, 'bold'), pad=((0, 0), (8, 0)))],
                [sg.Checkbox('Deactivate smoothing', key='-SMOOTH-', default=True, font=('Helvetica', 9))],
                [sg.Checkbox('Super accuracy mode', key='-ACCU-', default=False, font=('Helvetica', 9))],
            ], pad=(15, 0)),
            
            sg.Column([
                [sg.Text('Traditional Method', font=('Helvetica', 10, 'bold'))],
                [sg.Combo(('Entropy', 'Threshold'), default_value='Entropy', 
                         size=(15, 1), key='-TRADMETH-', readonly=True)],
                
                [sg.Text('Root Parameters', font=('Helvetica', 10, 'bold'), pad=((0, 0), (8, 0)))],
                [sg.Text('Max distance (px):', size=(15, 1)),
                 sg.Spin([i for i in range(1, 20000)], initial_value=50, key='-BROKEN-', size=(8, 1))],
                [sg.Text('Vector length (px):', size=(15, 1)),
                 sg.Spin([i for i in range(1, 20000)], initial_value=10, key='-VECTOR-', size=(8, 1))],
            ], pad=(15, 0))
        ]
    ], font=('Helvetica', 11, 'bold'), relief=sg.RELIEF_FLAT, 
    border_width=1, pad=(20, 3), visible=True, key='-ADVANCED-')

    # Export Options
    export_frame = sg.Frame('Export Settings', [
        [
            sg.Column([
                [sg.Checkbox('Save analysis plots', key='-ROOTPLOT-', font=('Helvetica', 10))],
                [sg.Checkbox('Save segmentations (TIF)', key='-SEG-', font=('Helvetica', 10))],
            ], pad=(15, 0)),
            
            sg.Column([
                [sg.Text('Save predictions for:', font=('Helvetica', 10, 'bold'))],
                [sg.Combo(('None', 'First', 'All'), default_value='None', 
                         size=(12, 1), key='-PRED-', readonly=True, font=('Helvetica', 10))],
                [sg.Text('(Not available with custom masks)', font=('Helvetica', 8), 
                        text_color=COLORS['text_secondary'])],
            ], pad=(15, 0))
        ]
    ], font=('Helvetica', 11, 'bold'), relief=sg.RELIEF_FLAT, 
    border_width=1, pad=(10, 0))

    # Console Output
    console_section = [
        [sg.Text('Analysis Output', font=('Helvetica', 11, 'bold'), 
                text_color=COLORS['text_primary'], pad=((20, 0), (15, 5)))],
        [sg.Multiline(size=(130, 10), write_only=True, font=('Consolas', 9),
                     autoscroll=True, key='-ML-', background_color=COLORS['console_bg'],
                     text_color=COLORS['console_text'], border_width=0, pad=(20, 0))]
    ]

    # Action Buttons
    action_buttons = [
        [
            sg.Push(),
            sg.Button('Full Analysis', **button_config, key='Full Analysis'),
            sg.Button('Segmentation Only', **button_config, key='Segmentation only'),
            sg.Button('Test Segmentation', **button_config, key='Test segmentation'),
            sg.Button('Mask Assistant', **button_config, key='Binary Mask Assistant'),
            sg.Push()
        ]
    ]

    # Utility Buttons
    utility_buttons = [
        [
            sg.Push(),
            sg.Button('Report Bug', **secondary_button, key='Report a bug'),
            sg.Button('Check Updates', **secondary_button, key='Check for updates'),
            sg.Button('Release Notes', **secondary_button, key='Open Release file'),
            sg.Button('Exit', **secondary_button, key='Exit'),
            sg.Push()
        ]
    ]

    # Footer
    footer = [
        [sg.HorizontalSeparator(pad=(20, 3))],
        [sg.Text('Charles University, Faculty of Sciences, Prague, Czech Republic | NBC Serre & M Fendrych | MacOS Port: P Baumann, University of Fribourg',
                font=('Helvetica', 8), text_color=COLORS['text_secondary'], pad=((20, 0), (5, 2)))],
        [sg.Text(f'Version {version}', font=('Helvetica', 8), 
                text_color=COLORS['accent'], pad=((20, 0), (0, 15)))]
    ]

    # Assemble final layout
    layout = [
        *header,
        [config_frame],
        [advanced_frame],
        [export_frame],
        *console_section,
        [sg.Text('')],  # Spacer
        *action_buttons,
        [sg.Text('')],  # Spacer
        *utility_buttons,
        *footer
    ]

    return layout, input_definition


def main():
    layout, input_definition = create_modern_gui()
    
    # Create window with modern styling
    window = sg.Window(
        f'ACORBA {version}',
        layout,
        finalize=True,
        keep_on_top=False,
        icon=None,
        margins=(0, 0),
        element_padding=(5, 5),
        background_color=COLORS['bg_primary'],
        font=('Helvetica', 10),
        resizable=True
    )

    # Welcome message
    window['-ML-'].write("═" * 80 + "\n")
    window['-ML-'].write("  ACORBA - Automatic Calculation Of Root Bending Angle\n")
    window['-ML-'].write("  Ready for analysis...\n")
    window['-ML-'].write("═" * 80 + "\n\n")

    while True:
        event, values = window.read()
        
        if event in (sg.WIN_CLOSED, 'Exit'):
            break

        # Main analysis actions
        if event in ('Full Analysis', 'Test segmentation', 'Segmentation only', 'Binary Mask Assistant'):
            script = script_for(values.get('-EXP-'), event)
            if not script:
                window['-ML-'].write(f"✗ ERROR: Could not determine script for action: {event}\n")
                continue

            # Determine which arguments to pass based on action type
            if event == 'Full Analysis':
                # Full analysis needs all arguments
                param_mapping = input_definition
            elif event == 'Test segmentation':
                # Test segmentation only needs core args + method-specific args
                param_mapping = {**core_args}
                if values.get('-EXP-') == 'Scanner':
                    param_mapping.update(scanner_args)
                else:
                    param_mapping.update(microscopy_args)
            elif event == 'Segmentation only':
                # Segmentation only needs core args + method-specific args
                param_mapping = {**core_args}
                if values.get('-EXP-') == 'Scanner':
                    param_mapping.update(scanner_args)
                else:
                    param_mapping.update(microscopy_args)
            else:  # Binary Mask Assistant
                # Mask assistant only needs core args
                param_mapping = core_args
            
            args = [sys.executable, script] + build_param_list(values, param_mapping)
            
            window['-ML-'].write("\n" + "─" * 80 + "\n")
            window['-ML-'].write(f"▶ Starting: {event}\n")
            window['-ML-'].write(f"  Command: {' '.join(args[:2])}\n")
            window['-ML-'].write("─" * 80 + "\n")
            
            output_folder = values.get('-FOLDER-', '')
            code, _ = run_command(args, window=window, save_log=True, output_folder=output_folder)
            
            status = "✓ COMPLETED" if code == 0 else "✗ FAILED"
            window['-ML-'].write(f"\n{status} (Exit code: {code})\n")
            window['-ML-'].write("═" * 80 + "\n\n")
        
        # Utility actions
        elif event == 'Report a bug':
            window['-ML-'].write("ℹ Opening bug report page...\n")
            webbrowser.open('https://sourceforge.net/p/acorba/tickets/')
        
        elif event == 'Check for updates':
            window['-ML-'].write("ℹ Checking for updates...\n")
            result = check_for_updates(version)
            
            if result is None or (isinstance(result, tuple) and result[0] is None):
                error_msg = result[1] if isinstance(result, tuple) else "Connection failed"
                window['-ML-'].write(f"✗ Could not check for updates: {error_msg}\n")
                sg.popup_ok("Could not connect to update server.\nPlease check your internet connection.",
                           title="Update Check Failed", keep_on_top=True)
            elif len(result) > 0:
                window['-ML-'].write(f"✓ New version available: {result[0]}\n")
                answer = sg.popup_yes_no(
                    f'New version {result[0]} is available!\n\nWould you like to download it?',
                    title="Update Available", 
                    keep_on_top=True
                )
                if answer == 'Yes':
                    webbrowser.open('https://sourceforge.net/projects/acorba/files/latest/download')
            else:
                window['-ML-'].write("✓ You're running the latest version\n")
                sg.popup_ok('You are running the latest version of ACORBA.',
                           title="Up to Date", keep_on_top=True)
        
        elif event == 'Open Release file':
            try:
                if sys.platform == 'darwin':
                    subprocess.run(['open', 'README.txt'])
                elif sys.platform == 'win32':
                    os.startfile('README.txt')
                else:
                    subprocess.run(['xdg-open', 'README.txt'])
                window['-ML-'].write("ℹ Opening release notes...\n")
            except Exception as e:
                window['-ML-'].write(f"✗ Could not open README.txt: {e}\n")

    window.close()


if __name__ == '__main__':
    # Modern theme setup
    sg.theme_add_new('ACORBAModern', {
        'BACKGROUND': COLORS['bg_primary'],
        'TEXT': COLORS['text_primary'],
        'INPUT': COLORS['bg_secondary'],
        'TEXT_INPUT': COLORS['text_primary'],
        'SCROLL': COLORS['border'],
        'BUTTON': (COLORS['bg_secondary'], COLORS['accent']),
        'PROGRESS': (COLORS['accent'], COLORS['bg_tertiary']),
        'BORDER': 1,
        'SLIDER_DEPTH': 0,
        'PROGRESS_DEPTH': 0,
    })
    
    sg.theme('ACORBAModern')
    main()