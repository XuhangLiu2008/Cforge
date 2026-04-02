import matplotlib.pyplot as plt

MAXWINDOW = 114

FIGURE_NUM_GENERATOR = (i for i in range(1, MAXWINDOW + 1))

def visualmethod(title="", new_figure=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if new_figure:
                fig, ax = plt.subplots(num=next(FIGURE_NUM_GENERATOR))
            else:
                fig, ax = plt.gcf(), plt.gca()
                
            fig.canvas.manager.set_window_title(title if title != "" else func.__name__)
            
            # pass ax (and/or fig) into the function
            result = func(*args, ax=ax, fig=fig, **kwargs)
            
            return result
        return wrapper
    return decorator

def next_fig_ax():
    fig, ax = plt.subplots(num=next(FIGURE_NUM_GENERATOR))
    return fig, ax