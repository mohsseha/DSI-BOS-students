#!/usr/bin/env python

def celsius_to_fahrenheit(temp):
    """ Takes input temperature (in Celsius) and converts to fahrenheit"""
    f_temp = temp * (1.8) + 32
    print "{}".format(f_temp)

def fahrenheit_to_celsius(temp):
    """ Takes input temperature (in Celsius) and converts to fahrenheit"""
    c_temp = (temp - 32) * (5/9.0)
    print "{}".format(c_temp)

if __name__ == "__main__":
    import sys
    celsius_to_fahrenheit(int(sys.argv[1]))
    fahrenheit_to_celsius(float(sys.argv[2]))
