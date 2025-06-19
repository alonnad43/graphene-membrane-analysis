"""
Test deep integration of all new force-field and simulation data sections in forcefield_params.json
"""
import json
import os
import pytest

data_path = os.path.join('data', 'forcefield_params.json')

with open(data_path, 'r') as f:
    params = json.load(f)

def test_dihedral_types():
    assert 'dihedral_types' in params, "Missing dihedral_types section"
    assert isinstance(params['dihedral_types'], dict), "dihedral_types should be a dict"

def test_coarse_grained_beads():
    assert 'coarse_grained_beads' in params, "Missing coarse_grained_beads section"
    assert isinstance(params['coarse_grained_beads'], dict), "coarse_grained_beads should be a dict"

def test_regeneration_chemistry():
    assert 'regeneration_chemistry' in params, "Missing regeneration_chemistry section"
    assert isinstance(params['regeneration_chemistry'], dict), "regeneration_chemistry should be a dict"

def test_pfas_cross_terms():
    assert 'pfas_cross_terms' in params, "Missing pfas_cross_terms section"
    assert isinstance(params['pfas_cross_terms'], dict), "pfas_cross_terms should be a dict"

def test_antibiotic_parameters():
    assert 'antibiotic_parameters' in params, "Missing antibiotic_parameters section"
    assert isinstance(params['antibiotic_parameters'], dict), "antibiotic_parameters should be a dict"

def test_microplastic_hybrid():
    assert 'microplastic_hybrid' in params, "Missing microplastic_hybrid section"
    assert isinstance(params['microplastic_hybrid'], dict), "microplastic_hybrid should be a dict"

def test_pathogen_parameters():
    assert 'pathogen_parameters' in params, "Missing pathogen_parameters section"
    assert isinstance(params['pathogen_parameters'], dict), "pathogen_parameters should be a dict"
