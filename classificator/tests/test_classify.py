"""
Classificator test script
"""

from classificator.classify import Classificator

CONFIG = "scripts/config.json"

def test_classify():
    clf = Classificator(config_loc=CONFIG)
    clf.choose_model()
