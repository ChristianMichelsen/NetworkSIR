import numpy as np
from bokeh.plotting import figure, output_file, show
N = 4000
x = np.random.random(size=N) * 100
y = np.random.random(size=N) * 100
radii = np.random.random(size=N) * 1.5
colors = ["#%02x%02x%02x" %(int(r),int(g),150) for r,g in zip(50+2*x, 30+2*y)]
TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
p = figure(tools=TOOLS)
p.scatter(x, y, radius=radii, fill_color=colors, fill_alpha=0.6, line_color=None)
output_file("Test.html", title="example")
show(p) # open a browser





import xarray as xr
import holoviews as hv
hv.extension('matplotlib')

air = xr.tutorial.open_dataset('air_temperature').load()
ds = hv.Dataset(air.isel(time=range(100)))
images = ds.to(hv.Image, ['lon', 'lat']).options(fig_inches=(10, 5), colorbar=True, cmap='viridis')
hv.save(images, 'hv_anim.mp4', fps=4)




#%%

import xarray as xr
import holoviews as hv
hv.extension('matplotlib')

air = xr.tutorial.open_dataset('air_temperature').load()
hv_ds = hv.Dataset(air.isel(time=range(100)))
hmap = ds.to(hv.Image, ['lon', 'lat'])
hv.renderer('bokeh').save(hmap, 'test', fmt='scrubber')


# %%



from bokeh.plotting import figure, show
from bokeh.io import output_notebook
output_notebook()
plot = figure(plot_width=300, plot_height=300)
plot.line(x=[1,2,3], y=[1,2,3])
show(plot)


# %%


import numpy as np
import holoviews as hv
from holoviews import opts

hv.extension('matplotlib')
hv.output(fig='svg')

python=np.array([2, 3, 7, 5, 26, 221, 44, 233, 254, 265, 266, 267, 120, 111])
pypy=np.array([12, 33, 47, 15, 126, 121, 144, 233, 254, 225, 226, 267, 110, 130])
jython=np.array([22, 43, 10, 25, 26, 101, 114, 203, 194, 215, 201, 227, 139, 160])

dims = dict(kdims='time', vdims='memory')
python = hv.Area(python, label='python', **dims)
pypy   = hv.Area(pypy,   label='pypy',   **dims)
jython = hv.Area(jython, label='jython', **dims)

overlay = (python * pypy * jython).opts(opts.Area(alpha=0.5))
overlay.relabel("Area Chart") + hv.Area.stack(overlay).relabel("Stacked Area Chart")

# %%


import numpy as np
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')


# create some example data
python=np.array([2, 3, 7, 5, 26, 221, 44, 233, 254, 265, 266, 267, 120, 111])
pypy=np.array([12, 33, 47, 15, 126, 121, 144, 233, 254, 225, 226, 267, 110, 130])
jython=np.array([22, 43, 10, 25, 26, 101, 114, 203, 194, 215, 201, 227, 139, 160])

dims = dict(kdims='time', vdims='memory')
python = hv.Area(python, label='python', **dims)
pypy   = hv.Area(pypy,   label='pypy',   **dims)
jython = hv.Area(jython, label='jython', **dims)


opts.defaults(opts.Area(fill_alpha=0.5))
overlay = (python * pypy * jython)
overlay.relabel("Area Chart") + hv.Area.stack(overlay).relabel("Stacked Area Chart")


# %%



# %%


import numpy as np
import holoviews as hv
from holoviews import opts
from holoviews.operation import contours

hv.extension('matplotlib')


x = y = np.arange(-3.0, 3.0, 0.1)
X, Y = np.meshgrid(x, y) 

def g(x,y,c):
    return 2*((x-y)**2/(x**2+y**2)) + np.exp(-(np.sqrt(x**2+y**2)-c)**2)


holomap = hv.HoloMap([(t, hv.Image(g(X,Y, 4 * np.sin(np.pi*t)))) for t in np.linspace(0,1,21)]).opts(
    cmap='fire', colorbar=True, show_title=False, xaxis='bare', yaxis='bare')

contour_hmap = contours(holomap, filled=True)

hv.output(contour_hmap, holomap='gif', fps=5)

#%%

hv.save(contour_hmap, 'holomap.gif', fps=5)
# hv.output(contour_hmap, holomap='mp4', fps=5)
hv.save(contour_hmap, 'holomap.mp4', fps=5)

# %%
