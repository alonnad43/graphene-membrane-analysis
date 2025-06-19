import unittest
import json
import os

class TestForcefieldParams(unittest.TestCase):
    def setUp(self):
        with open('data/forcefield_params.json', 'r') as f:
            self.ff = json.load(f)

    def test_dihedral_types(self):
        self.assertIn('dihedral_types', self.ff)
        for key in ["O_double-C_carb-O_single", "C-O-C_epoxy", "CT-N-CT", "O_S-O_S-S"]:
            self.assertIn(key, self.ff['dihedral_types'])

    def test_coarse_grained_beads(self):
        self.assertIn('coarse_grained_beads', self.ff)
        for bead in ["PL", "RV"]:
            self.assertIn(bead, self.ff['coarse_grained_beads'])

    def test_regeneration_chemistry(self):
        self.assertIn('regeneration_chemistry', self.ff)
        self.assertIn('ClO', self.ff['regeneration_chemistry'])
        self.assertIn('bonds', self.ff['regeneration_chemistry'])
        self.assertIn('O-ClO', self.ff['regeneration_chemistry']['bonds'])

    def test_pfas_cross_terms(self):
        self.assertIn('pfas_cross_terms', self.ff)
        for key in ["PFOA-C_carb", "PFOS-O_double", "PFBS-CT"]:
            self.assertIn(key, self.ff['pfas_cross_terms'])

    def test_antibiotic_parameters(self):
        self.assertIn('antibiotic_parameters', self.ff)
        for key in ["Cipro_C-N", "Cipro_N-C-C", "Tetra_O-C-O"]:
            self.assertIn(key, self.ff['antibiotic_parameters'])

    def test_microplastic_hybrid(self):
        self.assertIn('microplastic_hybrid', self.ff)
        for key in ["PET", "PS"]:
            self.assertIn(key, self.ff['microplastic_hybrid'])

if __name__ == "__main__":
    unittest.main()
