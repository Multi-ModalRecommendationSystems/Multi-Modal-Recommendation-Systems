"""
SpectralFlux.py
Compute the spectral flux between consecutive spectra
This technique can be for onset detection
rectify - only return positive values
"""
import numpy as np

def spectralFlux(spectra, rectify=False):
    """
    Compute the spectral flux between consecutive spectra
    """
    spectralFlux = []

    # Compute flux for zeroth spectrum
    flux = 0
    # for bin in spectra[0]:
    #     flux = flux + abs(bin)

    for bin in spectra:
        flux = flux + abs(bin)

    spectralFlux.append(flux)

    # # Compute flux for subsequent spectra
    # for s in range(1, len(spectra)):
    #     prevSpectrum = spectra[s - 1]
    #     spectrum = spectra[s]
    #
    #     flux = 0
    #     for bin in range(spectrum.shape):
    #         diff = abs(spectrum[bin]) - abs(prevSpectrum[bin])
    #         # If rectify is specified, only return positive values
    #         if rectify and diff < 0:
    #             diff = 0
    #         flux = flux + diff
    #
    #     spectralFlux.append(flux)

    return np.mean(spectralFlux)
