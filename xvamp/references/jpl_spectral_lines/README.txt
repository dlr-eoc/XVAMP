The catalog data files are composed of 80-character card images, with one card 
image per spectral line.  The format of each card image is:
  FREQ, ERR, LGINT, DR,  ELO, GUP, TAG, QNFMT,  QN',  QN" 
(F13.4,F8.4, F8.4,  I2,F10.4,  I3,  I7,    I4,  6I2,  6I2)

FREQ:  Frequency of the line in MHz.
ERR:   Estimated or experimental error of FREQ in MHz.
LGINT: Base 10 logarithm of the integrated intensity in units of nm^2 MHz at 
       300 K. 
DR:    Degrees of freedom in the rotational partition function (0 for atoms, 
       2 for linear molecules, and 3 for nonlinear molecules).
ELO:   Lower state energy in cm^{-1} relative to the ground state.
GUP:   Upper state degeneracy.
TAG:   Species tag or molecular identifier. 
       A negative value flags that the line frequency has 
       been measured in the laboratory.  The absolute value of TAG is then the 
       species tag and ERR is the reported experimental error.  The three most 
       significant digits of the species tag are coded as the mass number of 
       the species.
QNFMT: Identifies the format of the quantum numbers 
QN':   Quantum numbers for the upper state. 
QN":   Quantum numbers for the lower state. 

The on-line version of the catalog contains individual files for each
molecular species.  Line files are designated as ctttttt.cat, where tttttt 
is the zero-filled catalog tag number. For example,
the H atom line list is in file c001001.cat. 

A directory of the catalog is found in a file called 'catdir.cat.'
Each element of this directory is an 80-character record with the following 
format:
TAG,  NAME, NLINE,  QLOG,  VER 
(I6,X, A13,    I6, 7F7.4,  I2)

TAG:   The species tag or molecular identifier.
NAME:  An ASCII name for the species.
NLINE: The number of lines in the catalog.
QLOG:  A seven-element vector containing the base 10 logorithm of the partition
       function for temperatures of 300 K, 225 K, 150 K, 75 K, 37.5 K, 18.75 K,
       and 9.375 K, respectively.
VER:   The version of the calculation for this species in the catalog. 
       The version number is followed by * if the entry is newer than the
       last edition of the catalog.
