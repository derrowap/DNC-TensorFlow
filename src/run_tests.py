"""Runs all tests for this project.

TO ADD TESTS:
    1) Create a new testing file for a module at 'testing/module_name_test.py'.
    2) Implement the 'suite()' method that returns a TestSuite containing all
       the test cases you want to be tested here.
    3) Import that test with 'from . testing import module_name_test'.
    4) Add it to the list of tests to run in the 'suite()' method of this
       module found at 'suite.addTest([..., module_name_test.suite(), ...])'.
"""

import unittest

from . testing import dnc_test, tape_head_test

def suite():
    suite = unittest.TestSuite()

    # All test suites found in the 'testing/' directory.
    suite.addTests([
        dnc_test.suite(),
        tape_head_test.suite(),
    ])

    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
