"""
Test imports
"""

try:
    from classificator import *
    _top_import_error = None
except Exception as e:
    _top_import_error = e

try:
    from classificator import (
        classify, combined, models
        )
    _module_import_error = None
except Exception as e:
    _module_import_error = e

try:
    import classificator
    _package_import_error = None
except Exception as e:
    _package_import_error = e

def test_import_classificator():
    assert _top_import_error == None

def test_import_module():
    assert _module_import_error == None

def test_import_package():
    assert _package_import_error == None
