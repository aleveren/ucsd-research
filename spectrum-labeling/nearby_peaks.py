import numpy as np

def nearby_peaks_slow(wavelengths, target_wavelength, delta):
    '''
    Find indices of wavelengths that are within delta of the target wavelength
    '''
    if wavelengths[-1] < target_wavelength - delta:
        i_lo = len(wavelengths)
    else:
        i_lo = min([i for i in xrange(len(wavelengths))
            if wavelengths[i] >= target_wavelength - delta])
    if wavelengths[0] > target_wavelength + delta:
        i_hi = 0
    else:
        i_hi = max([i+1 for i in xrange(len(wavelengths))
            if wavelengths[i] <= target_wavelength + delta])
    return i_lo, i_hi

def nearby_peaks(wavelengths, target_wavelength, delta):
    '''
    Find indices of wavelengths that are within delta of the target wavelength
    '''
    lo = target_wavelength - delta
    hi = target_wavelength + delta

    wav_prev = lambda i: wavelengths[i - 1] if i > 0 else -np.inf
    wav_next = lambda i: wavelengths[i] if i < len(wavelengths) else np.inf
    
    def find_lo_index(i_min, i_max):
        i_pivot = int((i_min + i_max) / 2)
        if i_min == i_max:
            result = i_pivot
        elif wav_next(i_pivot) < lo:
            result = find_lo_index(i_pivot + 1, i_max)
        elif wav_prev(i_pivot) < lo:
            result = i_pivot
        else:
            result = find_lo_index(i_min, i_pivot)
        return result

    def find_hi_index(i_min, i_max):
        i_pivot = int((i_min + i_max) / 2)
        if i_min == i_max:
            result = i_pivot
        elif wav_prev(i_pivot) > hi:
            result = find_hi_index(i_min, i_pivot)
        elif wav_next(i_pivot) > hi:
            result = i_pivot
        else:
            result = find_hi_index(i_pivot + 1, i_max)
        return result
        
    i_lo = find_lo_index(0, len(wavelengths))
    i_hi = find_hi_index(0, len(wavelengths))

    return i_lo, i_hi
