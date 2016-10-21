# Draw Sigmoid with given t
def draw_sigmoid(t):
    '''
    Simple plotting function that draws a sigmoid curve in a xkcd (<3) way.
    For multiple t, subplots are being created.
    @t: list with min 1 and max 4 integer values.
    '''
    if len(t) > 4 or t < 1:
        raise ValueError("Wrong length of t. (has to be smaller than 4 and can't be < 1.")
    import numpy as np
    import pylab
    from scipy.optimize import curve_fit

    pylab.xkcd()

    x = np.linspace(-10, 10, 100)

    pylab.figure(1, figsize=(8,6))

    plotno = 221
    for nom in t:
        def sigmoid(k):
            return 1 / (1 + np.exp(-k/nom))
        
        y = sigmoid(x)
        ax = pylab.subplot(plotno)
        
        # Move left y-axis and bottim x-axis to centre, passing through (0,0)
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')

        # Eliminate upper and right axes
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        # Show ticks in the left and lower axes only
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.title.set_text('Sigmoid (t = -t / ' + str(nom) + ')')
        ax.set_yticks([.05, .95]) 
        ax.set_yticklabels([0, 1])
        ax.set_xticks([-5, 5, 8])
        ax.set_xticklabels([-5, 5, 'Time'])
        
        pylab.plot(x,y)
        
        plotno += 1

    pylab.tight_layout()

    pylab.show()
