Code for the generalized Bar-Hillel construction 

to start run:
```bash
$ git clone git@github.com:rycolab/rayuela.git
$ cd rayuela
$ pip install -e .
```

The code for the generalized construction is
in 'rayuela/cfg/cfg.py' at line 418.

In 'rayuela/test/cfg/test_epsilon_Bar
_Hillel.py' we unit test the construction by
performing ε-removal on the fsa and then 
running the original version.