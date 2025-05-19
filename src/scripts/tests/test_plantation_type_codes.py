import ast
import unittest
from pathlib import Path

CODE_PATH = Path(__file__).resolve().parents[1] / "src/scripts/utilities/constants_and_names.py"


def load_codes():
    source = CODE_PATH.read_text()
    tree = ast.parse(source, filename=str(CODE_PATH))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "plantation_type_codes":
                    return ast.literal_eval(ast.get_source_segment(source, node.value))
    raise RuntimeError("plantation_type_codes not found")


class TestPlantationTypeCodes(unittest.TestCase):
    def test_codes_unique(self):
        codes = load_codes()
        vals = list(codes.values())
        self.assertEqual(len(vals), len(set(vals)))

    def test_rotation_codes_different(self):
        codes = load_codes()
        self.assertNotEqual(codes["long_rotation"], codes["short_rotation"])


if __name__ == "__main__":
    unittest.main()