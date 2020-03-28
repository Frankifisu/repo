#!/usr/bin/env python3

# =========
#  MODULES
# =========
import os #OS interface: os.getcwd(), os.chdir('dir'), os.system('mkdir dir')
import sys #System-specific functions: sys.argv(), sys.exit(), sys.stderr.write()
import glob #Unix pathname expansion: glob.glob('*.txt')
import re #Regex
import argparse # commandline argument parsers
import math #C library float functions
import subprocess #Spawn process: subprocess.run('ls', stdout=subprocess.PIPE)
import numpy #Scientific computing
import typing #Support for type hints

# ==============
#  PROGRAM DATA
# ==============
PROGNAME = os.path.basename(sys.argv[0])
USER = os.getenv('USER')
HOME = os.getenv('HOME')
SHELL = os.getenv('SHELL')

# ==========
#  DEFAULTS
# ==========
Step = 0.001
Half = 0.5
ToAng = 0.529177210920
ToeV = 27.211386245988
HEADER = '{CC} {what} cartesian gradients for excited state {state}'
COORD = ['x', 'y', 'z']
ATOMS = ['N', 'H', 'O', 'C']
NSteps = 2
NAtoms = len(ATOMS)

# =================
#  BASIC FUNCTIONS
# =================
def errore(message=None):
    """
    Error function
    """
    if message != None:
        print('ERROR: ' + message)
    sys.exit(1)

# =================
#  PARSING OPTIONS
# =================
def parseopt():
    """
    Parse options
    """
    # Create parser
    parser = argparse.ArgumentParser(prog=PROGNAME,
        description='Command-line option parser')
    # Mandatory arguments
    parser.add_argument('indat', nargs='+',
        help='File list')
    # Optional arguments
    parser.add_argument('-o', '--outnam',
        dest='out', action='store', default=None,
        help='Set output file name')
    parser.add_argument('-s', '--state', type=int,
        dest='state', action='store', default=1,
        help='Set state')
    parser.add_argument('-v', '--iprint',
        dest='iprt', action='count', default=0,
        help='Set printing level')
    opts = parser.parse_args()
    # Check options
    return opts

# ================
#  WORK FUNCTIONS
# ================
def filparse(input_file, energie, intensi):
    with open(input_file, 'r') as file_obj:
        for nline, line in enumerate(file_obj) :
            if nline == 0:
                continue
            data = [float(dat) for dat in line.split()]
            energie = numpy.append(energie, data[0])
            intensi = numpy.concatenate([intensi, data[1:4]])
    return energie, intensi
def filwrt(input_file, add, toprint, what):
    with open(input_file, 'w') as file_obj:
        for state in range(toprint.shape[0]):
            file_obj.write(HEADER.format(CC=add, what=what, state=state+1))
            for A in range(toprint.shape[1]):
                file_obj.write(str(toprint[state, A, :]))
                print(toprint[state, A, :])
    return None
def terminating(cond):
    """An easy way to break out of a generator"""
    if cond:
        return True
    raise StopIteration
def common_start(sa, sb):
    """Return common starting substring between two strings"""
    return ''.join(a for a, b in zip(sa, sb) if terminating(a == b))

# ==============
#  MAIN PROGRAM
# ==============
def main():
    # PARSE OPTIONS
    opts = parseopt()
    # Energies at each step
    energie = numpy.empty(0)
    intensi = numpy.empty(0)
    # Extract data from files
    commonnam = opts.indat[0]
    for fileout in opts.indat:
        commonnam = common_start(commonnam, fileout)
        energie, intensi = filparse(fileout, energie, intensi)
    energie = numpy.reshape(energie, [NAtoms, 3, NSteps, -1])
    intensi = numpy.reshape(intensi, [NAtoms, 3, NSteps, -1, 3])
    NState = energie.shape[3]
    grad = numpy.zeros([NState, NAtoms, 3])
    dipd = numpy.zeros([NState, NAtoms, 3, 3])
    # Convert dipole strength derivatives to transition dipole moment derivatives
    energie_eq = numpy.empty(0)
    intensi_eq = numpy.empty(0)
    fileeq = os.listdir("equilibrium")
    filelist = [os.path.splitext(os.path.basename(fil))[0] for fil in fileeq]
    for num, filetry in enumerate(filelist):
        if filetry in commonnam and '.out' in filetry:
            fileout = "equilibrium/" + fileeq[num]
    energie_eq, intensi_eq = filparse(fileout, energie_eq, intensi_eq)
    energie_eq = numpy.reshape(energie_eq, [-1])
    intensi_eq = numpy.reshape(intensi_eq, [-1, 3])
    # Compute numerical derivatives
    for s in range(NState):
        for A in range(NAtoms):
            for xyz in range(3):
                grad[s, A, xyz] = (ToAng/ToeV)*Half*(energie[A, xyz, 1, s] - energie[A, xyz, 0, s])/Step
                for coord in range(3):
                    dipd[s, A, xyz, coord] = ToAng*Half*(intensi[A, xyz, 1, s, coord] - intensi[A, xyz, 0, s, coord])/Step
                    if abs(intensi_eq[s, coord]) < 1.e-10:
                        if abs(dipd[s, A, xyz, coord]) > 1.e-10:
                            print(f'state {s}, atom={ATOMS[A]}{COORD[xyz]}, mu{COORD[coord]} = {intensi_eq[s, coord]}, der = {dipd[s, A, xyz, coord]}')
                        dipd[s, A, xyz, coord] = 0.0
                    else:
                        dipd[s, A, xyz, coord] = dipd[s, A, xyz, coord]*Half/numpy.sqrt(intensi_eq[s, coord])
    # Print them out on files
    outnam = opts.out
    if opts.out is None: outnam = commonnam + 'S' + str(opts.state) + '_grad' + '.dat'
    headr = HEADER.format(CC=commonnam, what='energy', state=opts.state)
    numpy.savetxt(outnam, grad[opts.state-1, :, :], delimiter=' ', newline='\n', header=headr, comments='! ', encoding=None)
    if opts.out is None: outnam = commonnam + 'S' + str(opts.state) + '_dipd' + '.dat'
    headr = HEADER.format(CC=commonnam, what='dipole strength component', state=opts.state)
    prt_dipd = numpy.reshape(dipd, [ NState, 3*NAtoms, 3])
    numpy.savetxt(outnam, prt_dipd[opts.state-1, :, :], delimiter=' ', newline='\n', header=headr, comments='! ', encoding=None)
    sys.exit()

# ===========
#  MAIN CALL
# ===========
if __name__ == '__main__':
    main()
