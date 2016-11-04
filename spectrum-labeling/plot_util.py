import matplotlib.pyplot as plt

def show_labeled_spectrum(x, y, px, py, xlo, xhi, p_to_e, ax = None):
    '''
    Given a spectrum (x, y), locations of peaks (px, py),
    and a dict `p_to_e` that maps each peak's x coordinate
    to a list of elements, plot the spectrum for x = xlo to xhi
    and show labels from `p_to_e` next to each peak
    '''
    max_height_in_range = max(y[(x >= xlo) & (x <= xhi)])

    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 6, forward=True)

    ax.plot(x, y)

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(-0.1 * max_height_in_range, 2 * max_height_in_range)
    ax.set_xlabel("Wavelength (nm)")

    # Show peak labels
    for i in xrange(len(px)):
        peak_x = px[i]
        peak_y = py[i]
        if peak_x in p_to_e:
            elts = p_to_e[peak_x]
            peak_label = " / ".join(sorted(elts))
        else:
            peak_label = "?"
        text = "{:.2f}: {}".format(peak_x, peak_label)
        if peak_x >= xlo and peak_x <= xhi:
            ax.text(peak_x, peak_y + (0.05 * max_height_in_range), text,
                rotation = "vertical",
                horizontalalignment = "left",
                rotation_mode = "anchor")

    return ax
