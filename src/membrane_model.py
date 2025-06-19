"""
Defines the Membrane class for all simulation phases.

Each Membrane object holds:
- Phase 1: macroscale traits (thickness, pore_size, pressure)
- Phase 2: structural layout (GO/rGO stacking, interlayer spacing)
- Phase 3: atomic structure definition for LAMMPS simulation
- Microstructure variants: GO_GO, rGO_rGO, GO_rGO, dry_hybrid, wet_hybrid

Microstructure variants affect:
- Interlayer spacing (0.34-1.05 nm)
- Permeability and rejection efficiency
- Mechanical properties based on π-π stacking vs hydration
"""

import numpy as np

class MembraneVariant:
    """
    Represents microstructure variants of GO/rGO membranes based on interlayer spacing.
    
    Variants:
    - GO_GO: 1.05 nm (hydrated, hydrogen bonding)
    - rGO_rGO: 0.34 nm (π-π stacking, dry)
    - GO_rGO: 0.80 nm (hybrid interface)
    - dry_hybrid: 0.6 nm (compressed hybrid)
    - wet_hybrid: 0.85 nm (hydrated hybrid)
    """
    
    VARIANTS = {
        'GO_GO': {
            'interlayer_spacing_nm': 1.05,
            'description': 'Hydrated GO layers with hydrogen bonding',
            'porosity': 0.45,
            'c_o_ratio': 2.1,
            'id_ig_ratio': 0.95,
            'surface_energy_mJ_m2': 62.1,
            'contact_angle_deg': 29.5,
            'flux_multiplier': 1.2,
            'rejection_multiplier': 0.9
        },
        'rGO_rGO': {
            'interlayer_spacing_nm': 0.34,
            'description': 'π-π stacked rGO layers, dry',
            'porosity': 0.25,
            'c_o_ratio': 8.5,
            'id_ig_ratio': 2.8,
            'surface_energy_mJ_m2': 110.0,
            'contact_angle_deg': 122.0,
            'flux_multiplier': 0.7,
            'rejection_multiplier': 1.15
        },
        'GO_rGO': {
            'interlayer_spacing_nm': 0.80,
            'description': 'GO-rGO hybrid interface',
            'porosity': 0.35,
            'c_o_ratio': 4.8,
            'id_ig_ratio': 1.6,
            'surface_energy_mJ_m2': 86.0,
            'contact_angle_deg': 75.8,
            'flux_multiplier': 1.0,
            'rejection_multiplier': 1.05
        },
        'dry_hybrid': {
            'interlayer_spacing_nm': 0.60,
            'description': 'Compressed hybrid structure',
            'porosity': 0.28,
            'c_o_ratio': 5.2,
            'id_ig_ratio': 1.8,
            'surface_energy_mJ_m2': 95.0,
            'contact_angle_deg': 85.0,
            'flux_multiplier': 0.8,
            'rejection_multiplier': 1.1
        },
        'wet_hybrid': {
            'interlayer_spacing_nm': 0.85,
            'description': 'Hydrated hybrid structure',
            'porosity': 0.40,
            'c_o_ratio': 4.2,
            'id_ig_ratio': 1.4,
            'surface_energy_mJ_m2': 78.0,
            'contact_angle_deg': 65.0,
            'flux_multiplier': 1.1,
            'rejection_multiplier': 0.95
        }
    }
    
    @classmethod
    def get_variant_properties(cls, variant_name):
        """Get properties for a specific microstructure variant."""
        return cls.VARIANTS.get(variant_name, cls.VARIANTS['GO_rGO'])
    
    @classmethod
    def infer_variant_from_spacing(cls, interlayer_spacing_nm, tolerance=0.1):
        """
        Infer closest variant based on measured interlayer spacing.
        
        Args:
            interlayer_spacing_nm (float): Measured spacing
            tolerance (float): Tolerance for matching (±nm)
            
        Returns:
            tuple: (variant_name, confidence)
        """
        best_match = None
        min_diff = float('inf')
        
        for variant_name, props in cls.VARIANTS.items():
            diff = abs(props['interlayer_spacing_nm'] - interlayer_spacing_nm)
            if diff < min_diff:
                min_diff = diff
                best_match = variant_name
        
        confidence = max(0, 1 - (min_diff / tolerance))
        return best_match, confidence


class Membrane:
    """
    Represents a membrane with physical and performance properties.

    Args:
        name (str): Membrane name (e.g., 'GO', 'rGO', 'hybrid')
        pore_size_nm (float): Pore size in nanometers
        thickness_nm (float): Thickness in nanometers
        flux_lmh (float): Water flux in L·m⁻²·h⁻¹
        modulus_GPa (float): Young's modulus in GPa
        tensile_strength_MPa (float): Tensile strength in MPa
        contact_angle_deg (float): Contact angle in degrees
        rejection_percent (float, optional): Oil rejection efficiency (%)
        variant (str, optional): Microstructure variant ('GO_GO', 'rGO_rGO', etc.)
    
    Additional Args:
        pei_branches (bool, optional): If True, add PEI branches
        pei_density (float, optional): Fraction of carbons to functionalize with PEI
        ions (dict, optional): Ions to add, e.g. {"Na": 10, "Cl": 10}
        contaminants (dict, optional): Contaminants to add, e.g. {"BPA": 5, "SMX": 2}
    """

    def __init__(self, name, pore_size_nm, thickness_nm, flux_lmh, modulus_GPa,
                 tensile_strength_MPa, contact_angle_deg, rejection_percent=None,
                 variant=None, pei_branches=False, pei_density=0.1, ions=None, contaminants=None):
        self.name = name
        self.pore_size_nm = pore_size_nm
        self.thickness_nm = thickness_nm
        self.flux_lmh = flux_lmh
        self.modulus_GPa = modulus_GPa
        self.tensile_strength_MPa = tensile_strength_MPa
        self.contact_angle_deg = contact_angle_deg
        self.rejection_percent = rejection_percent
        
        # Microstructure variant handling
        self.variant = variant
        self.variant_properties = {}
        if variant:
            self.variant_properties = MembraneVariant.get_variant_properties(variant)
            self._apply_variant_effects()
        
        # Fluid properties for physics-based calculations
        self.water_viscosity = 0.89       # mPa·s (at 25°C)
        self.oil_viscosity = 25.0         # mPa·s
        self.oil_droplet_size = 5.0       # µm

        # New: PEI, ions, contaminants
        self.pei_branches = pei_branches
        self.pei_density = pei_density
        self.ions = ions if ions is not None else {}
        self.contaminants = contaminants if contaminants is not None else {}

    def __repr__(self):
        """
        Returns a string representation of the Membrane object for logging or debugging.
        """
        return (
            f"<Membrane {self.name}: "
            f"pore={self.pore_size_nm} nm, "
            f"thickness={self.thickness_nm} nm, "
            f"flux={self.flux_lmh} L·m⁻²·h⁻¹, "
            f"modulus={self.modulus_GPa} GPa, "
            f"strength={self.tensile_strength_MPa} MPa, "
            f"contact_angle={self.contact_angle_deg}°, "
            f"rejection={self.rejection_percent}%>"
        )


    def to_dict(self):
        """
        Converts the membrane attributes into a dictionary for data export or serialization.
        """
        d = {
            "name": self.name,
            "pore_size_nm": self.pore_size_nm,
            "thickness_nm": self.thickness_nm,
            "flux_lmh": self.flux_lmh,
            "modulus_GPa": self.modulus_GPa,
            "tensile_strength_MPa": self.tensile_strength_MPa,
            "contact_angle_deg": self.contact_angle_deg,
            "rejection_percent": self.rejection_percent,
            "variant": self.variant,
            "pei_branches": self.pei_branches,
            "pei_density": self.pei_density,
            "ions": self.ions,
            "contaminants": self.contaminants
        }
        return d

    def to_lammps_data(self):
        """
        Export atomic structure definition for Phase 3 LAMMPS simulation.
        
        Returns:
            dict: LAMMPS-compatible structure data
        """
        d = {
            "membrane_type": self.name.split()[0] if ' ' in self.name else self.name,
            "thickness_angstrom": self.thickness_nm * 10,  # Convert nm to Angstrom
            "pore_size_angstrom": self.pore_size_nm * 10,
            "contact_angle": self.contact_angle_deg,
            "modulus_gpa": self.modulus_GPa,
            "strength_mpa": self.tensile_strength_MPa,
            "pei_branches": self.pei_branches,
            "pei_density": self.pei_density,
            "ions": self.ions,
            "contaminants": self.contaminants
        }
        return d
    
    @classmethod
    def from_dict(cls, data):
        """
        Create Membrane object from dictionary (for batch generation).
        
        Args:
            data (dict): Membrane properties dictionary
        
        Returns:
            Membrane: New membrane instance
        """
        return cls(
            name=data.get('name', 'Unknown'),
            pore_size_nm=data.get('pore_size_nm', 1.0),
            thickness_nm=data.get('thickness_nm', 100.0),
            flux_lmh=data.get('flux_lmh', 0.0),
            modulus_GPa=data.get('modulus_GPa'),
            tensile_strength_MPa=data.get('tensile_strength_MPa'),
            contact_angle_deg=data.get('contact_angle_deg'),
            rejection_percent=data.get('rejection_percent'),
            variant=data.get('variant'),
            pei_branches=data.get('pei_branches', False),
            pei_density=data.get('pei_density', 0.1),
            ions=data.get('ions', {}),
            contaminants=data.get('contaminants', {})
        )

    def __eq__(self, other):
        """
        Compares two Membrane objects for equality.
        """
        if not isinstance(other, Membrane):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __hash__(self):
        """
        Enables Membrane objects to be used in sets or as dictionary keys.
        """
        return hash((
            self.name,
            self.pore_size_nm,
            self.thickness_nm,
            self.flux_lmh,
            self.modulus_GPa,
            self.tensile_strength_MPa,
            self.contact_angle_deg,
            self.rejection_percent
        ))

    def _apply_variant_effects(self):
        """Apply microstructure variant effects to membrane properties."""
        if not self.variant_properties:
            return
            
        # Update contact angle if variant specifies it
        if 'contact_angle_deg' in self.variant_properties:
            self.contact_angle_deg = self.variant_properties['contact_angle_deg']
        
        # Apply flux and rejection multipliers
        if hasattr(self, '_base_flux_lmh'):
            self.flux_lmh = self._base_flux_lmh * self.variant_properties.get('flux_multiplier', 1.0)
        else:
            self._base_flux_lmh = self.flux_lmh
            self.flux_lmh *= self.variant_properties.get('flux_multiplier', 1.0)
            
        if hasattr(self, '_base_rejection_percent') and self.rejection_percent:
            self.rejection_percent = self._base_rejection_percent * self.variant_properties.get('rejection_multiplier', 1.0)
        elif self.rejection_percent:
            self._base_rejection_percent = self.rejection_percent
            self.rejection_percent *= self.variant_properties.get('rejection_multiplier', 1.0)
    
    def get_interlayer_spacing(self):
        """Get interlayer spacing for this membrane variant."""
        if self.variant_properties:
            return self.variant_properties.get('interlayer_spacing_nm', 0.8)
        # Default values based on membrane type
        defaults = {'GO': 1.05, 'rGO': 0.34, 'hybrid': 0.8}
        return defaults.get(self.name.split()[0], 0.8)
    
    def get_variant_info(self):
        """Get detailed variant information."""
        if not self.variant:
            return {"variant": "default", "properties": {}}
        
        return {
            "variant": self.variant,
            "properties": self.variant_properties,
            "interlayer_spacing_nm": self.get_interlayer_spacing(),
            "description": self.variant_properties.get('description', 'Standard variant')
        }

def compute_interface_penalty(layers):
    """
    Compute performance penalty due to GO/rGO layer transitions in hybrid membranes.
    
    Args:
        layers (list): List of layer types ['GO', 'rGO', 'GO', ...]
    
    Returns:
        float: Penalty factor (0.02 = 2% penalty per transition)
    
    Scientific basis:
        - Interface resistance increases with material transitions
        - Each GO/rGO boundary introduces flow impedance
    """
    penalty = 0
    for i in range(1, len(layers)):
        if layers[i] != layers[i - 1]:
            penalty += 0.02  # 2% penalty per transition
    return penalty

def generate_membrane_variants_with_variability(design_pore_size, design_thickness, literature_CA, n=1):
    """
    Generate membrane variants with realistic variability for Phase 1.
    Args:
        design_pore_size (float): Nominal pore size (nm)
        design_thickness (float): Nominal thickness (nm)
        literature_CA (float): Literature contact angle (deg)
        n (int): Number of variants to generate
    Returns:
        list of dicts: Each dict contains noisy parameters
    """
    variants = []
    for _ in range(n):
        pore_size = np.random.normal(design_pore_size, 0.2)  # ±0.2 nm std dev
        thickness = max(50, np.random.normal(design_thickness, 10))  # ±10 nm, min 50
        contact_angle = np.random.normal(literature_CA, 5)  # ±5°
        variants.append({
            'pore_size_nm': pore_size,
            'thickness_nm': thickness,
            'contact_angle_deg': contact_angle
        })
    return variants
