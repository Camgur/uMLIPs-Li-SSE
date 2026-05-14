"""utils/models.py: uMLIP loader utilities.

Each entry in CALCULATOR_BLOCKS is a self-contained Python code snippet
that, when executed, defines a local variable named 'calculator' holding a
ready-to-use ASE Calculator. Call get_calculator(model_name) to execute
the relevant block and return that calculator.
"""


CALCULATOR_BLOCKS = {
    'chgnet-2023': '''
from matgl import load_model
from matgl.ext.ase import PESCalculator
mdl = load_model("CHGNet-MPtrj-2023.12.1-2.7M-PES")
calculator = PESCalculator(potential=mdl)
''',
    'chgnet-2024': '''
from matgl import load_model
from matgl.ext.ase import PESCalculator
mdl = load_model("CHGNet-MPtrj-2024.2.13-11M-PES")
calculator = PESCalculator(potential=mdl)
''',
    'chgnet-PBE-2025': '''
from matgl import load_model
from matgl.ext.ase import PESCalculator
mdl = load_model("CHGNet-MatPES-PBE-2025.2.10-2.7M-PES")
calculator = PESCalculator(potential=mdl)
''',
    'chgnet-r2-2025': '''
from matgl import load_model
from matgl.ext.ase import PESCalculator
mdl = load_model("CHGNet-MatPES-r2SCAN-2025.2.10-2.7M-PES")
calculator = PESCalculator(potential=mdl)
''',
    'chgnet-torch': '''
from chgnet.model.dynamics import CHGNetCalculator
calculator = CHGNetCalculator(use_device='cuda')
''',
    'm3gnet-2021': '''
from matgl import load_model
from matgl.ext.ase import PESCalculator
mdl = load_model("M3GNet-MP-2021.2.8-PES")
calculator = PESCalculator(potential=mdl)
''',
    'm3gnet-PBE-2025': '''
from matgl import load_model
from matgl.ext.ase import PESCalculator
mdl = load_model("M3GNet-MatPES-PBE-v2025.1-PES")
calculator = PESCalculator(potential=mdl)
''',
    'm3gnet-r2-2025': '''
from matgl import load_model
from matgl.ext.ase import PESCalculator
mdl = load_model("M3GNet-MatPES-r2SCAN-v2025.1-PES")
calculator = PESCalculator(potential=mdl)
''',
    'mace-0b3': '''
from mace.calculators import mace_mp
calculator = mace_mp(
    model="medium-0b3",
    dispersion=False, device='cuda', default_dtype='float64'
)
''',
    'mace-l2': '''
from mace.calculators import mace_mp
calculator = mace_mp(
    model="large",
    dispersion=False, device='cuda', default_dtype='float64'
)
''',
    'mace-mpa': '''
from mace.calculators import mace_mp
calculator = mace_mp(
    model="medium-mpa-0",
    dispersion=False, device='cuda', default_dtype='float64'
)
''',
    'mace-omat': '''
from mace.calculators import mace_mp
calculator = mace_mp(
    model="medium-omat-0",
    dispersion=False, device='cuda', default_dtype='float64'
)
''',
    'mace-r2': '''
from mace.calculators import mace_mp
calculator = mace_mp(
    model="mace-matpes-r2scan-0",
    dispersion=False, device='cuda', default_dtype='float64'
)
''',
    "orb-v2": """
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
orbff = pretrained.orb_v2(device='cuda')
calculator = ORBCalculator(orbff, device='cuda')
""",
    "orb-v3-cMPA": """
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
orbff = pretrained.orb_v3_conservative_20_mpa(device='cuda')
calculator = ORBCalculator(orbff, device='cuda')
""",
    "orb-v3-cOMAT": """
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
orbff = pretrained.orb_v3_conservative_20_omat(device='cuda')
calculator = ORBCalculator(orbff, device='cuda')
""",
    "orb-v3-iMPA": """
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
orbff = pretrained.orb_v3_conservative_inf_mpa(device='cuda')
calculator = ORBCalculator(orbff, device='cuda')
""",
    "orb-v3-iOMAT": """
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
orbff = pretrained.orb_v3_conservative_inf_omat(device='cuda')
calculator = ORBCalculator(orbff, device='cuda')
""",
}


def get_calculator(model_name):
    """Return an ASE Calculator instance for model_name.
    
    Executes the code block stored in CALCULATOR_BLOCKS[model_name] and
    returns the 'calculator' object it defines.
    
    Parameters:
        model_name: Key in CALCULATOR_BLOCKS (e.g. 'mace-0b3', 'chgnet-2024')
    
    Returns:
        ASE Calculator object, configured and ready for use
    
    Raises:
        KeyError if model_name is not in CALCULATOR_BLOCKS
    """
    if model_name not in CALCULATOR_BLOCKS:
        available = ", ".join(CALCULATOR_BLOCKS)
        raise KeyError(
            f"Model '{model_name}' not found in CALCULATOR_BLOCKS. "
            f"Available models: {available}"
        )

    namespace = {}
    exec(CALCULATOR_BLOCKS[model_name], namespace)  # noqa: S102
    return namespace["calculator"]


def list_models():
    """Return the names of all models defined in CALCULATOR_BLOCKS."""
    return list(CALCULATOR_BLOCKS.keys())
