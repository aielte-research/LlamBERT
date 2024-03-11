from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, FreehandDrawTool, PolyDrawTool
from bokeh.io import curdoc
from bokeh.models.ranges import DataRange1d
from matplotlib import pyplot as plt
import numpy as np
import colorcet as cc
import os
import json
import yaml
import math
from functools import reduce
def deep_get(dictionary, keys, default):
    return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split("."), dictionary)

def flatten_list(xss):
    return [x for xs in xss for x in xs]
    #return list(np.array(xss).flatten())

def get_colors(nmbr):
    return cc.palette["glasbey_category10"][:nmbr]

def get_string_val(lst, i):
    try:
        return lst[i]
    except IndexError:
        return ''

def jitter(lst, width):
    return [x + np.random.uniform(low=-width, high=width) for x in lst] 

def json_pretty_print(text, indent=4):
    level = 0
    list_level = 0
    inside_apostrophe = 0
    last_backslash_idx = -2
    ret = ""
    for i,c in enumerate(text):
        if c=="}" and inside_apostrophe % 2 == 0:
            level -= 1
            ret += "\n" + " "*(level*indent)
        ret += c
        if c=="{" and inside_apostrophe % 2 == 0:
            level += 1
            ret += "\n" + " "*(level*indent)
        elif c=="[" and inside_apostrophe % 2 == 0:
            list_level += 1
        elif c=="]" and inside_apostrophe % 2 == 0:
            list_level -= 1
        elif c=='"' and last_backslash_idx != i-1:
            inside_apostrophe += 1
        elif c=="\\":
            last_backslash_idx=i
        elif c=="," and inside_apostrophe % 2 == 0 and list_level==0:
            ret += "\n" + " "*(level*indent)
    return ret

def my_cmap(val, high, low=0, typ="log"):
    if val==0:
        return "black"
    cmap=cc.CET_L8
    if val<0:
        cmap=cc.CET_L6
        val=-val
    val-=low
    high-=low
    if typ=="log":
        val=math.log(val+1)
        high=math.log(high+1)
    elif typ=="sqrt":
        val=math.sqrt(val)
        high=math.sqrt(high)
    elif typ=="square":
        val=val**2
        high=high**2
    return cmap[math.ceil(val/high*(len(cmap)-1))]

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_json(data, fpath):
    with open(fpath, 'w') as outfile:
        #json.dump(data, outfile)#, indent=4)
        outfile.write(json_pretty_print(json.dumps(data,separators=(',', ': '), cls=NpEncoder)))

def save_yaml(data, fpath):
    with open(fpath, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style = False)

def selet_not_None(list1,list2):
    return [val for i,val in enumerate(list1) if list2[i]!=None]

def cyclic_fill(lst,length,default="[UNK]"):
    if len(lst)==0:
        lst=[default]
    return [lst[i%len(lst)] for i in range(length)]

matplotlib_dashes = {
    "solid": '-',
    "dotted": ':',
    "dashed": '--',
    "dotdash": '-.',
    "dashdot": '-.'
}

def matplotlib_setcolors(ax, line_color="black", face_color="white", grid_color="0.9", **kwargs):
    ax.tick_params(axis='x', colors=line_color)
    ax.tick_params(axis='y', colors=line_color)
    ax.yaxis.label.set_color(line_color)
    ax.xaxis.label.set_color(line_color)
    ax.title.set_color(line_color)
    ax.set_facecolor(face_color)
    ax.patch.set_alpha(0)
    ax.spines['bottom'].set_color(line_color)
    ax.spines['left'].set_color(line_color)
    ax.spines['top'].set_color(grid_color) 
    ax.spines['right'].set_color(grid_color)

def matlotlib_legend_loc(s):
    return s.replace("_"," ").replace("top","upper").replace("bottom","lower")

def init_matplotlib_figure(width=16, height=9, style="seaborn-poster", **kwargs):
    plt.style.use(style)
    fig = plt.figure(figsize=(width, height))
    ax = plt.subplot(111)   
    return fig, ax

def init_matplotlib_grid_figure(width=16, height=9, style="seaborn-poster", grid_w=1, grid_h=1, **kwargs):
    plt.style.use(style)
    fig, axs = plt.subplots(grid_h, grid_w, figsize=(grid_w*width, grid_h*height))
    return fig, axs.flatten()

def init_bokeh_figure(bokeh={"width": None, "height": None}, title="", xlabel="", ylabel="", xscale="linear", yscale="linear", theme="caliber", p=None, yrange=None, **kwargs):
    curdoc().theme = theme
    tools = "pan,box_zoom,wheel_zoom,xwheel_zoom,ywheel_zoom,save,reset"
    if p is None:
        fig_params=dict(
            title = title,
            tools = tools,
            x_axis_type = xscale,
            y_axis_type = yscale
        )
        if bokeh["width"] is None or bokeh["height"] is None:
            fig_params["sizing_mode"] = 'stretch_both'
        else:
            fig_params["plot_width"] = bokeh["width"]
            fig_params["plot_height"] = bokeh["height"]
        if not yrange is None:
            fig_params["y_range"] = DataRange1d(start = yrange[0], end = yrange[1], range_padding = 0) # type: ignore

        p = figure(**fig_params) # type: ignore

    p.xaxis.axis_label = xlabel
    p.yaxis.axis_label = ylabel

    renderer = p.multi_line([[]], [[]], line_width=5, alpha=0.4, color='red')
    draw_tool_freehand = FreehandDrawTool(renderers=[renderer])
    draw_tool_poly = PolyDrawTool(renderers=[renderer])
    p.add_tools(draw_tool_freehand, draw_tool_poly) # type: ignore

    return p

def bokeh_line45(p, min_x=0, max_x=1, line45_color=None, **kwargs):
    if not line45_color is None:
        src=ColumnDataSource(data=dict(x=[min_x,max_x], y=[min_x,max_x]))
        p.line("x","y", line_color=line45_color, line_width=2, source=src)
    return p

def matplotlib_line45(ax, min_x=0, max_x=1, line45_color=None, **kwargs):
    if not line45_color is None:
        ax.plot([min_x,max_x],[min_x,max_x], color=line45_color, zorder=10)

class Plotter():
    def __init__(self, fname="", dirname="", neptune_experiment=None):
        self.neptune_experiment = neptune_experiment
        self.dirname = dirname
        self.fname = fname

    def get_full_path(self,extension="html",suffix=""):
        if suffix!="":
            suffix = f"_{suffix}"
        if not os.path.isdir(os.path.join(self.dirname,extension)):
            os.makedirs(os.path.join(self.dirname,extension))
        return os.path.join(self.dirname,extension,f"{self.fname}{suffix}.{extension}")
    
    def export_json(self, params):
        try:
            save_json(params, self.get_full_path("json"))
            if self.neptune_experiment!=None:
                self.neptune_experiment[f"json/{self.fname}"].upload(self.get_full_path("json"))
                self.neptune_experiment.sync()
        except:
            save_yaml(params, self.get_full_path("yaml"))
            if self.neptune_experiment!=None:
                self.neptune_experiment[f"yaml/{self.fname}"].upload(self.get_full_path("yaml"))
                self.neptune_experiment.sync()

    def save_matplotlib_figure(self, dpi=240, bg_transparent=True, png=True, svg=True):
        if png:
            plt.savefig(self.get_full_path("png"), transparent=bg_transparent, dpi=dpi)
            if not self.neptune_experiment is None:
                self.neptune_experiment[f"png/{self.fname}"].upload(self.get_full_path("png"))
        if svg:
            plt.savefig(self.get_full_path("svg"), transparent=bg_transparent)
            if not self.neptune_experiment is None:
                self.neptune_experiment[f"svg/{self.fname}"].upload(self.get_full_path("svg"))
        if not self.neptune_experiment is None:
            self.neptune_experiment.sync()
        plt.close()

    def save_bokeh_figure(self,p,suffix=""):
        output_file(self.get_full_path("html",suffix))
        save(p)
        if self.neptune_experiment!=None:
            self.neptune_experiment[f"html/{self.fname}{suffix}"].upload(self.get_full_path("html",suffix))
            self.neptune_experiment.sync()

class GeneralPlotter(Plotter):
    def __init__(
        self,
        Ys: list=[],
        x=[], # defaults to 1,2,3,4,...
        Xs=None,
        errorbars=None,
        xlabel: str="",
        ylabel: str="",
        xscale: str="linear",
        yscale: str="linear",
        title: str="",
        colors=None, # the bulit in categorical colors go up to 256
        dashes=["solid"], #"solid", "dashed", "dotted", "dotdash", "dashdot"
        markers=["."],
        fname: str="general_plot",
        neptune_experiment=None,
        dirname: str="",
        line45_color=None, #None to turn off line45
        legend={
            "labels": [],
            "location": "top_right", # "top_left", "top_right", "bottom_left", "bottom_right", None
        },
        baselines={
            "labels": [],
            "values": [],
            "colors": ["grey"], # can be shorter than names
            "dashes": ["dotted"] # can be shorter than namesself.colors
        },
        histogram={
            "labels": [],
            "Xs": [],
            "colors": None, # can be shorter than names default: ["grey"]
            "bins": 100,
            "density": True,
            "alpha": 0.5
        },
        matplotlib={ # for png and svg
            "width": 16,
            "height": 9,
            "style": "seaborn-poster", # https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
            "png_dpi": 240, #use 240 for 4k resolution on 16x9 image
            "calc_xtics": True,
        },
        bokeh={
            "width": None, #set this to force bokeh plot to fixed dimensions (not recommended)
            "height": None #set this to force bokeh plot to fixed dimensions (not recommended)
        },
        color_settings={
            "suffix":"",
            "grid_color":"0.9",
            "face_color":"white",
            "line_color": "black",
            "bg_transparent": True
        }
    ):
        if len(Ys)>0 and not isinstance(Ys[0],list):
            Ys=[Ys]
        
        if len(legend.get("labels",[])) + len(baselines.get("labels",[])) + len(histogram.get("labels",[])) == 0:
            legend["location"] = None

        if legend.get("location",None) is None or len(legend.get("labels",[])) == 0:
            legend["labels"] = [None for _ in Ys]
        
        if legend.get("location",None) is None or len(baselines.get("labels",[])) == 0:
            baselines["labels"] = [None for _ in baselines["values"]]

        if legend.get("location",None) is None or len(histogram.get("labels",[])) == 0:
            histogram["labels"] = [None for _ in baselines["values"]]

        if len(x)>0:
            x_len=len(x)
        else:
            if len(Ys)>0:
                x_len=max([len(y) for y in Ys])

        if x==[]:
            x=list(range(1, x_len+1))
        if Xs is None:
            Xs = [x for _ in range(len(Ys))]
        min_x = min([min(x) for x in Xs])
        max_x = max([max(x) for x in Xs])

        if colors is None:
            colors = get_colors(len(Ys))
        else:
            colors = cyclic_fill(colors,len(Ys))
        dashes = cyclic_fill(dashes,len(Ys))
        if markers is None:
            markers=[""]
        markers = cyclic_fill(markers,len(Ys))

        if len(baselines["values"]) > 0:
            baselines["colors"] = cyclic_fill(baselines["colors"], len(baselines["values"]), "grey")
            baselines["dashes"] = cyclic_fill(baselines["dashes"], len(baselines["values"]), "dotted")

        if len(histogram["Xs"]) > 0:
            if not histogram["colors"] is None:
                histogram["colors"] = cyclic_fill(histogram["colors"], len(histogram["Xs"]), "grey")
            else:
                histogram["colors"] = get_colors(len(Ys)+len(histogram["Xs"]))[len(Ys):]

        self.params = locals()
        super().__init__(fname, dirname, neptune_experiment)

    #########
    # bokeh #
    #########
    def make_bokeh_plot(self):
        p = init_bokeh_figure(**self.params)

        bokeh_line45(p, **self.params)

        sources = [ColumnDataSource(data=dict(
            x=selet_not_None(x[:len(y)],y),
            y=selet_not_None(y,y),
            maxim=[max(selet_not_None(y,y)) for _ in selet_not_None(y,y)],
            minim=[min(selet_not_None(y,y)) for _ in selet_not_None(y,y)],
            argmax=[np.argmax(selet_not_None(y,y))+1 for _ in selet_not_None(y,y)],
            argmin=[np.argmin(selet_not_None(y,y))+1 for _ in selet_not_None(y,y)],
            label=[get_string_val(self.params["legend"]["labels"],i) for _ in selet_not_None(y,y)]                                 
        )) for i,(x,y) in enumerate(zip(self.params["Xs"], self.params["Ys"]))]

        for (c, l, source, dash, marker) in zip(self.params["colors"], self.params["legend"]["labels"], sources, self.params["dashes"], self.params["markers"]):
            if l!=None:
                p.line('x', 'y', color=c, line_width=2, source=source, line_dash = dash, legend_label=l)
            else:
                p.line('x', 'y', color=c, line_width=2, source=source, line_dash = dash)
            if marker==".":
                p.circle('x', 'y', color=c, source=source)

        if self.params["errorbars"] is not None:
            for x, y, yerr, c in zip(self.params["Xs"], self.params["Ys"], self.params["errorbars"], self.params["colors"]):
                err_source = ColumnDataSource(data=dict(
                    error_low=[val - e for val, e in zip(selet_not_None(y,y), selet_not_None(yerr,yerr))],
                    error_high=[val + e for val, e in zip(selet_not_None(y,y), selet_not_None(yerr,yerr))],
                    x=selet_not_None(x[:len(y)],y)
                ))
                
                p.segment(
                    source=err_source,
                    x0='x',
                    y0='error_low',
                    x1='x',
                    y1='error_high',
                    line_width=2,
                    color=c
                )   
        
        if len(self.params["histogram"]["Xs"]) > 0:
            for y, color, label in zip(*[self.params["histogram"][k] for k in ["Xs","colors","labels"]]):
                if self.params["xscale"]=="log":
                    bins=np.logspace(np.log10(min(y)),np.log10(max(y)), self.params["histogram"]["bins"])
                else:
                    bins=self.params["histogram"]["bins"]
                    
                hist, edges = np.histogram(y, density=self.params["histogram"]["density"], bins=bins)

                source = ColumnDataSource(data=dict(
                    x=[(left+right)/2 for left,right in zip(edges[:-1],edges[1:])],
                    y=hist,
                    left=edges[:-1],
                    right=edges[1:],
                    maxim=[max(hist) for _ in hist],
                    label=[label for _ in hist]
                ))

                if self.params["yscale"]=="log":
                    bottom=1
                else:
                    bottom=0
                                
                if label!=None:
                    p.quad(top="y", bottom=bottom, left="left", right="right", source=source, fill_color=color, line_color=color, alpha=self.params["histogram"]["alpha"], legend_label=label)
                else:
                    p.quad(top="y", bottom=bottom, left="left", right="right", source=source, fill_color=color, line_color=color, alpha=self.params["histogram"]["alpha"])
        
        if len(self.params["baselines"]["values"]) > 0:
            for name, value, color, dash in zip(*[self.params["baselines"][k] for k in ["labels","values","colors","dashes"]]):
                src = ColumnDataSource(data = {
                    "x": [self.params["min_x"]-1,self.params["max_x"]+1],
                    "y": [value,value],
                    "maxim": [value,value],
                    "label": [name,name]
                })
                if name!=None:
                    p.line("x","y", line_dash = dash, line_color = color, line_width = 2, source = src, legend_label = name)
                else:
                    p.line("x","y", line_dash = dash, line_color = color, line_width = 2, source = src)

        if self.params["legend"]["location"]!=None:
            p.legend.location = self.params["legend"]["location"] 
        
        tooltips = [
            (self.params["xlabel"], "@x"),
            (self.params["ylabel"], "@y"),
            ("name", "@label"),
            ("max", "@maxim"),
            ("argmax", "@argmax"),
            ("min", "@minim"),
            ("argmin", "@argmin")
        ]

        p.add_tools(HoverTool(tooltips = tooltips, mode='vline'))  # type: ignore
        p.add_tools(HoverTool(tooltips = tooltips)) # type: ignore
        return p

    # ##############
    # # matplotlib #
    # ##############
    def make_matplotlib_plot(self, ax):
        matplotlib = self.params["matplotlib"]
        
        ax.set_xlabel(self.params["xlabel"])
        ax.set_ylabel(self.params["ylabel"])
        ax.set_title(self.params["title"])

        ax.set_xscale(self.params["xscale"])
        ax.set_yscale(self.params["yscale"])
        
        matplotlib_line45(ax, **self.params)
        
        if len(self.params["baselines"]["values"]) > 0:
            for label, value, color, dash in zip(*[self.params["baselines"][k] for k in ["labels","values","colors","dashes"]]):
                ax.plot([self.params["min_x"]-1, self.params["max_x"]+1], [value,value], matplotlib_dashes[dash], label = label, color = color, zorder = 20)

        for x,y,dash,color,label,marker in zip(self.params["Xs"],self.params["Ys"],self.params["dashes"],self.params["colors"],self.params["legend"]["labels"],self.params["markers"]):
            ax.plot(x[:len(y)], y, matplotlib_dashes[dash], marker=marker, color=color, label=label, zorder=30)
        
        if self.params["errorbars"] is not None:
            for x,y,yerr in zip(self.params["Xs"],self.params["Ys"],self.params["errorbars"]):
                plt.errorbar(x, y, yerr=yerr, alpha=.5, fmt=':', capsize=3, capthick=1)
                data = {
                    'x': x,
                    'y1': [val - e for val, e in zip(y, yerr)],
                    'y2': [val + e for val, e in zip(y, yerr)]
                }
                plt.fill_between(**data, alpha=.2)

        if len(self.params["histogram"]["Xs"]) > 0:
            for y, color, label in zip(*[self.params["histogram"][k] for k in ["Xs","colors","labels"]]):
                if self.params["xscale"] == "log":
                    bins=np.logspace(np.log10(min(y)),np.log10(max(y)), self.params["histogram"]["bins"])
                else:
                    bins = self.params["histogram"]["bins"]
                plt.hist(y, density = self.params["histogram"]["density"], bins=bins, alpha = self.params["histogram"]["alpha"], color=color, label=label, zorder=10)

        if self.params["legend"]["location"]!=None and (len(self.params["legend"]["labels"])>0 or len(self.params["baselines"]["labels"]) > 0 or len(self.params["histogram"]["labels"]) > 0):
            legend = ax.legend(loc=matlotlib_legend_loc(self.params["legend"]["location"]))
            frame = legend.get_frame()
            frame.set_facecolor(self.params["color_settings"].get("face_color", "white"))
            frame.set_edgecolor(self.params["color_settings"].get("grid_color", "0.9"))
            for text in legend.get_texts():
                text.set_color(self.params["color_settings"].get("line_color", "black"))

        ax.grid(True, color=self.params["color_settings"].get("grid_color", "0.9"), zorder=0)
        matplotlib_setcolors(ax, **self.params["color_settings"])
        
        if matplotlib["calc_xtics"]:
            x_max=max([len(str(x)) for x in x[:self.params["x_len"]]])
            ax.set_xticks([float(str(x[:self.params["x_len"]][i])[:6]) for i in range(self.params["x_len"]) if i % max(int(min(x_max,6) * self.params["x_len"] / (4*matplotlib["width"])),1)==0])

        
def general_plot(params, export_types=["json","html","png","svg"]):
    plotter = GeneralPlotter(**params)
    if "json" in export_types:
        plotter.export_json(params)
    if "html" in export_types:
        p = plotter.make_bokeh_plot()
        plotter.save_bokeh_figure(p)
    fig, ax = init_matplotlib_figure(**params.get("matplotlib",{
        "width": 16,
        "height": 9,
        "style": "seaborn-poster"
    }))
    plotter.make_matplotlib_plot(ax)
    plotter.save_matplotlib_figure(deep_get(params,"matplotlib.png_dpi",240), deep_get(params,"color_settings.bg_transparent",True), png="png" in export_types, svg="svg" in export_types)
    return fig