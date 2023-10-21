# On the Intersection of Context-Free and Regular Languages

This repository includes code supporting [Pasti et al.'s (2023)](https://aclanthology.org/2023.eacl-main.52) efficient algorithm for interesting weighted finite-state automata and weight context-free grammars.

![image](https://github.com/rycolab/bar-hillel/assets/219898/9d3a86c4-2c26-4e83-ba8b-6ba6942fe631)

## Installation and Usage

```  
cd rayuela
pip install -e .
```

The code for the generalized Bar-Hillel construction is the `intersect_fsa_ε` in `rayuela/cfg/cfg.py`. Test cases are provided in `rayuela/test/cfg/test_epsilon_Bar_Hillel.py`.

## Citation
```
@inproceedings{pasti-etal-2023-intersection,
    title = "On the Intersection of Context-Free and Regular Languages",
    author = "Pasti, Clemente  and
      Opedal, Andreas  and
      Pimentel, Tiago  and
      Vieira, Tim  and
      Eisner, Jason  and
      Cotterell, Ryan",
    booktitle = "Proceedings of the Conference of the European Chapter of the Association for Computational Linguistics",
    year = "2023",
    publisher = "Association for Computational Linguistics",
    url = "[https://aclanthology.org/2023.eacl-main.52](https://aclanthology.org/2023.eacl-main.52/)",
    doi = "10.18653/v1/2023.eacl-main.52",
    pages = "737--749",
}
```
