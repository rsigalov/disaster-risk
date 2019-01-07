#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 11:00:49 2019

@author: rsigalov
"""

class OptionSmile:
    def __init__(self, date, exdate, secid, strike_list, impl_vol):
        self.date = date
        self.exdate = exdate
        self.secid = secid
        self.strike_list = strike_list
        self.impl_vol = impl_vol
        
        a = date
        b = exdate
        delta = b - a
        self.T = delta.days/252 # 252 trading days in a year (~approximately)
        
    def fit_smile_bdbg(self):
        