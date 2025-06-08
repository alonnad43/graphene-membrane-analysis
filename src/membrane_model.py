"""
Defines the Membrane class and related data structures for GO/rGO membrane simulation.

The Membrane class stores physical and performance properties for a single membrane type.
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
        contact_angle_deg (float): Contact angle in degrees
        rejection_percent (float, optional): Oil rejection efficiency (%)
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
