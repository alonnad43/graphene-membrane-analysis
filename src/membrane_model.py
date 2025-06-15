"""
Defines the Membrane class for all simulation phases.

Each Membrane object holds:
- Phase 1: macroscale traits (thickness, pore_size, pressure)
- Phase 2: structural layout (GO/rGO stacking, interlayer spacing)
- Phase 3: atomic structure definition for LAMMPS simulation

# TODO: Add .to_dict() method for export to JSON
# TODO: Add .to_lammps_data() to export atomic format for Phase 3
# TODO: Add from_dict() constructor for batch membrane generation
"""

class Membrane:
    """
    Represents a membrane with physical and performance properties.

    Args:
        name (str): Membrane name (e.g., 'GO', 'rGO', 'Hybrid')
        pore_size_nm (float): Pore size in nanometers
        thickness_nm (float): Thickness in nanometers
        flux_lmh (float): Water flux in L·m⁻²·h⁻¹
        modulus_GPa (float): Young's modulus in GPa
        tensile_strength_MPa (float): Tensile strength in MPa
        contact_angle_deg (float): Contact angle in degrees        rejection_percent (float, optional): Oil rejection efficiency (%)
    """

    def __init__(self, name, pore_size_nm, thickness_nm, flux_lmh, modulus_GPa,
                 tensile_strength_MPa, contact_angle_deg, rejection_percent=None):
        self.name = name
        self.pore_size_nm = pore_size_nm
        self.thickness_nm = thickness_nm
        self.flux_lmh = flux_lmh
        self.modulus_GPa = modulus_GPa
        self.tensile_strength_MPa = tensile_strength_MPa
        self.contact_angle_deg = contact_angle_deg
        self.rejection_percent = rejection_percent
        
        # Fluid properties for physics-based calculations
        self.water_viscosity = 0.89       # mPa·s (at 25°C)
        self.oil_viscosity = 25.0         # mPa·s
        self.oil_droplet_size = 5.0       # µm

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
        return {
            "name": self.name,
            "pore_size_nm": self.pore_size_nm,
            "thickness_nm": self.thickness_nm,
            "flux_lmh": self.flux_lmh,
            "modulus_GPa": self.modulus_GPa,
            "tensile_strength_MPa": self.tensile_strength_MPa,
            "contact_angle_deg": self.contact_angle_deg,
            "rejection_percent": self.rejection_percent,
        }

    def to_lammps_data(self):
        """
        Export atomic structure definition for Phase 3 LAMMPS simulation.
        
        Returns:
            dict: LAMMPS-compatible structure data
        """
        return {
            "membrane_type": self.name.split()[0] if ' ' in self.name else self.name,
            "thickness_angstrom": self.thickness_nm * 10,  # Convert nm to Angstrom
            "pore_size_angstrom": self.pore_size_nm * 10,
            "contact_angle": self.contact_angle_deg,
            "modulus_gpa": self.modulus_GPa,
            "strength_mpa": self.tensile_strength_MPa
        }
    
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
            rejection_percent=data.get('rejection_percent')
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
