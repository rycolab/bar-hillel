To start run:
  
  
  cd rayuela
  
  pip install -e .
  
The code for the generalized Bar-Hillel construction is
in 'rayuela/cfg/cfg.py' at line 418.

In 'rayuela/test/cfg/test_epsilon_Bar
_Hillel.py' we unit test the our construction by
performing ε-removal on the fsa and then 
running the original version of the Bar-Hillel construction.
