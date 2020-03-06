import unittest
from numpy import exp, sqrt, random, mean
from mc import payoffs
from mc import mc

__author__ = "mkapchenko"
__copyright__ = "mkapchenko"
__license__ = "mit"

from fox_toolbox.utils import xml_parser, rates, volatility
from mc import mc
from hw import calibration as hw_calib
import os

import numpy.testing as npt


random.seed(42)

class test_HW_zero_vol(unittest.TestCase):
    
    def setUp(self):
        self.logs_folder = r'D:\WORKSPACE\perso\GitHub\mc\tests\hw_zero_vol'

    def test_hw_bond(self):

        spotstep = 1
        timestep = 10
        
        x0 = 0.
        

        # READ INPUT LOG
        INPUT_5SWO = xml_parser.get_files('zero_vol.xml', self.logs_folder)
        OUTPUT_5SWO = xml_parser.get_files('zero_vol.result', self.logs_folder)
        
        

        main_curve, sprds = xml_parser.get_rate_curves(INPUT_5SWO)
        dsc_curve = main_curve

        try:
            estim_curve = sprds[0]
        except TypeError:
            estim_curve = main_curve
        dsc_curve

        #READ OUT LOG
        _, irsmout = xml_parser.get_xml(OUTPUT_5SWO)

        ref_mr, (hw_buckets, hw_sigma) = xml_parser.get_hw_params(irsmout)

        ref_sigmas = rates.Curve(hw_buckets, hw_sigma, 'PieceWise')
        
        T = 5. # Tp from log
        t = 4. # Tf from log

        hw = mc.HullWhiteProcess( mr = ref_mr, sigma = ref_sigmas, x0 = x0)
        df = hw.generate_paths(until=T, timestep=timestep, spotstep=spotstep)
        

        c = 0
        for x in df.loc[t]:
            c+=hw.fwd_bond(t, T, dsc_curve, ref_mr, ref_sigmas, x)

        mc_undisc_bond = c/spotstep

        undisc_ref = 0.9978351813686
        disc_ref = 0.9910763859623
        

        self.assertAlmostEqual(mc_undisc_bond, undisc_ref, 8, 'with zero vol, the bond price from logs doesnt match hw pirce A*exp(-B*x)')
        self.assertAlmostEqual(disc_ref, mc_undisc_bond * dsc_curve.get_dsc(t), 8, 'with zero vol, the bond price from logs doesnt match hw pirce A*exp(-B*x)')

