import sys
from pathlib import Path
this_dir = Path(__file__).resolve().parent
BASE_DIR = this_dir

carbon_ec_module_path = Path(BASE_DIR) / "carbon-ec"
carbon_ec_module_python_path = carbon_ec_module_path / "python"

print("carbon_ec_module_path:", carbon_ec_module_path)
print("carbon_ec_module_python_path:", carbon_ec_module_python_path)

sys.path.append(str(carbon_ec_module_python_path))
# from carbon_ec.python import EC_Carbon
from EC_Carbon import EC_Carbon

ec_carbon = EC_Carbon("COM4")