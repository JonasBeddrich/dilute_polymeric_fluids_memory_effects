# Numerical simulation of dilute polymeric fluids with memory effects in the turbulent flow regime

Solver for the time-fractional Navier-Stokes Fokker-Planck system based on the MFEM library.

## Publication in the Journal of Computational Physics 

Article in press: 
Beddrich, Jonas, Stephan B. Lunowa, and Barbara Wohlmuth. "Numerical simulation of dilute polymeric fluids with memory effects in the turbulent flow regime." Journal of Computational Physics ??? (2025): 113955.

## Project dependencies

Core dependencies are:

- [CMake](https://cmake.org/) (3.19 or above)
- [MFEM](https://mfem.org/) (4.5 or above)
- [Hypre](https://computing.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods) (necessary for MFEM)
- [Metis](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) (necessary for MFEM)

## Build instructions
1. Clone the repository
   ```bash
   git clone https://github.com/JonasBeddrich/dilute_polymeric_fluids_memory_effects.git
   ```
2. Create a build directory
   ```bash
   mkdir build && cd build
   ```
3. Run CMake and point it to the source directory
   ```bash
   cmake ..
   ```
4. Build the project
   ```bash
   make -j
   ```

## Usage

Run the executables for solving the Navier-Stokes-Fokker-Planck equation
```bash
cd applications
./ns-fp-spherical [<options>]
```
or the time-fractional Navier-Stokes-Fokker-Planck system
```bash
cd applications
./tf-ns-fp-spherical [<options>]
```
or for the standard Navier-Stokes system as a comparison baseline 
```bash
cd applications
./ns-testing [<options>]
```

## Authors and acknowledgment

### Developers

- [Stephan Lunowa](mailto:stephan.lunowa@tum.de)
- [Jonas Beddrich](mailto:jonas.beddrich@tum.de)

## License

Copyright (c) 2025 Jonas Beddrich 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

See also [online (MIT License)](https://opensource.org/license/mit/)
