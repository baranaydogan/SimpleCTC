SimpleCTC
=========

SimpleCTC is a simple contour tree connectivity (CTC) library.

It uses SimpleCT for contour tree computation. SimpleCTC contains 
seperate MATLAB and C++ routines to compute CTC.

C++ part includes all the functions necessary to compute CTC
of a binary image, including boundary detection and signed Euclidean 
distance transform.

-------------------------------------------------------------------
FAST GUIDE:

If you are familiar with MATLAB or intend to use SimpleCTC in MATLAB,
please go to the "matlab" folder and read the README file. This 
explains and directs you on how to install and use SimpleCTC.

If you are planning to use SimpleCTC in C/C++, please go to "example"
folder and read the README file.


-------------------------------------------------------------------

-------------------------------------------------------------------
DESCRIPTION:

SimpleCTC simply computes the contour tree connectivity (CTC) of 2D/3D 
images.

If you use this code or CTC in your work please cite the following
publication:

Aydogan D.B. & Hyttinen J. “Contour tree connectivity of binary images 
from algebraic graph theory”, IEEE International Conference
on Image Processing (ICIP), 15-22.09.2013, Melbourne, Australia

Known dependencies are:
- SLEPc
- PETSc
- OpenMPI

____________________________________________________________________
LICENSE INFORMATION:

SimpleCTC is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by 
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

SimpleCTC is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
____________________________________________________________________

Dogu Baran Aydogan - baran.aydogan@gmail.com
23.07.2014
____________________________________________________________________