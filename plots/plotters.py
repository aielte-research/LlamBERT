from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, ColorBar, LinearColorMapper, LogColorMapper, FreehandDrawTool, PolyDrawTool, CustomJS, Select, LabelSet, BasicTicker
from bokeh.transform import transform
from bokeh.io import curdoc
from bokeh.colors import RGB
from bokeh.layouts import column
from bokeh.models.ranges import DataRange1d
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import colorcet as cc
import os
import json
import yaml
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px
import math

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

def matplotlib_setcolors(ax, color_settings):
    ax.tick_params(axis='x', colors=color_settings["line_color"])
    ax.tick_params(axis='y', colors=color_settings["line_color"])
    ax.yaxis.label.set_color(color_settings["line_color"])
    ax.xaxis.label.set_color(color_settings["line_color"])
    ax.title.set_color(color_settings["line_color"])
    ax.set_facecolor(color_settings["face_color"])
    ax.patch.set_alpha(0)
    ax.spines['bottom'].set_color(color_settings["line_color"])
    ax.spines['left'].set_color(color_settings["line_color"])
    ax.spines['top'].set_color(color_settings["grid_color"]) 
    ax.spines['right'].set_color(color_settings["grid_color"])

def matlotlib_legend_loc(s):
    return s.replace("_"," ").replace("top","upper").replace("bottom","lower")

class Plotter():
    rgb = [[int(h.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4)] for h in cc.CET_L4]
    dark_color_settings={
        "suffix": "_dark",
        "grid_color": "#323d4e",
        "face_color": "#293340",
        "line_color": "#b1bdcd",#"#0B98C8",
        "bg_transparent": True,
        "cmap":ListedColormap([[r,g,b,(r+g+b)/2] for r,g,b in rgb])#"cet_CET_L8"
    }
    def init_bokeh_figure(self, theme_suffix="", p=None):
        if theme_suffix=="_dark":
            curdoc().theme = 'dark_minimal'
        else:
            curdoc().theme = 'caliber'
        tools = "pan,box_zoom,wheel_zoom,xwheel_zoom,ywheel_zoom,save,reset"
        if p==None:
            fig_params=dict(
                title = self.title,
                tools = tools,
                x_axis_type=self.xscale,
                y_axis_type=self.yscale
            )
            if self.bokeh["width"]==None or self.bokeh["height"]==None:
                fig_params["sizing_mode"] = 'stretch_both'
            else:
                fig_params["plot_width"] = self.bokeh["width"]
                fig_params["plot_height"] = self.bokeh["height"]
            if hasattr(self,"yrange") and self.yrange!=None:
                fig_params["y_range"]=DataRange1d(start = self.yrange[0], end = self.yrange[1], range_padding = 0)

            p = figure(**fig_params)

        p.xaxis.axis_label = self.xlabel
        p.yaxis.axis_label = self.ylabel

        renderer = p.multi_line([[]], [[]], line_width=5, alpha=0.4, color='red')
        draw_tool_freehand = FreehandDrawTool(renderers=[renderer])
        draw_tool_poly = PolyDrawTool(renderers=[renderer])
        p.add_tools(draw_tool_freehand, draw_tool_poly)

        return p

    def __init__(self, neptune_experiment=None):
        self.neptune_experiment = neptune_experiment

    def save_bokeh_figure(self,p,suffix=""):
        output_file(self.get_full_path("html",suffix))
        save(p)
        if self.neptune_experiment!=None:
            self.neptune_experiment["html/"+self.fname+suffix].upload(self.get_full_path("html",suffix))
            self.neptune_experiment.sync() 

    def init_matplotlib_figure(self):
        plt.figure(figsize=[self.matplotlib["width"], self.matplotlib["height"]])
        ax = plt.subplot(111)

        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)

        ax.set_xscale(self.xscale)
        ax.set_yscale(self.yscale)

        return ax
    
    def save_matplotlib_figure(self,suffix,bg_transparent=True):
        plt.savefig(self.get_full_path("png",suffix), transparent=bg_transparent, dpi=self.matplotlib["png_dpi"])
        plt.savefig(self.get_full_path("svg",suffix), transparent=bg_transparent)
        if self.neptune_experiment!=None:
            self.neptune_experiment["png/"+self.fname+suffix].upload(self.get_full_path("png",suffix))
            self.neptune_experiment["svg/"+self.fname+suffix].upload(self.get_full_path("svg",suffix))
            self.neptune_experiment.sync()
        plt.close()

    def get_full_path(self,extension="html",suffix=""):
        if not os.path.isdir(os.path.join(self.dirname,extension)):
            os.makedirs(os.path.join(self.dirname,extension))
        return os.path.join(self.dirname,extension,self.fname+suffix+"."+extension)
    
    def export_json(self):
        try:
            save_json(self.params, self.get_full_path("json"))
            if self.neptune_experiment!=None:
                self.neptune_experiment["json/"+self.fname].upload(self.get_full_path("json"))
                self.neptune_experiment.sync()
        except:
            save_yaml(self.params, self.get_full_path("yaml"))
            if self.neptune_experiment!=None:
                self.neptune_experiment["yaml/"+self.fname].upload(self.get_full_path("yaml"))
                self.neptune_experiment.sync()

class GeneralPlotter(Plotter):
    def __init__(self, params, neptune_experiment=None):
        super().__init__(neptune_experiment) 
        default_params={
            #Ys <-- no default, list of lists
            "x": [], # defaults to 1,2,3,4,...
            "Xs": None,
            "xlabel": "",
            "ylabel": "",
            "xscale": "linear",
            "yscale": "linear",
            "title": "",
            "colors": None, # the bulit in categorical colors go up to 256
            "dashes": ["solid"], #"solid", "dashed", "dotted", "dotdash", "dashdot"
            "markers": ["."],
            "fname": "general_plot",
            "dirname": "",
            "line45_color": None, #None to turn off line45
            "legend":{
                "labels": [],
                "location": "top_right", # "top_left", "top_right", "bottom_left", "bottom_right", None
            },
            "baselines":{
                "labels": [],
                "values": [],
                "colors": ["grey"], # can be shorter than names
                "dashes": ["dotted"] # can be shorter than namesself.colors
            },
            "histogram":{
                "labels": [],
                "Xs": [],
                "colors": None, # can be shorter than names default: ["grey"]
                "bins": 100,
                "density": True,
                "alpha": 0.5
            },
            "matplotlib":{ # for png and svg
                "width": 16,
                "height": 9,
                "style": "seaborn-poster", #"seaborn-poster", "seaborn-talk"
                "png_dpi": 240, #use 240 for 4k resolution on 16x9 image
                "calc_xtics": True
            },
            "bokeh":{
                "width": None, #set this to force bokeh plot to fixed dimensions (not recommended)
                "height": None #set this to force bokeh plot to fixed dimensions (not recommended)
            }       
        }
        for key in default_params:
            params[key] = params.get(key, default_params[key])

        for key in ["legend","baselines","histogram","matplotlib","bokeh"]:
            for key2 in default_params[key]:
                params[key][key2] = params[key].get(key2, default_params[key][key2])

        self.params = params
        for k, v in params.items():
            setattr(self, k, v)

        if len(self.Ys)>0 and type(self.Ys[0])!=list:
            self.Ys=[self.Ys]
        
        if len(self.legend["labels"]) + len(self.baselines["labels"]) + len(self.histogram["labels"]) == 0:
            self.legend["location"] = None

        if self.legend["location"] == None or len(self.legend["labels"]) == 0:
            self.legend["labels"] = [None for _ in self.Ys]
        
        if self.legend["location"] == None or len(self.baselines["labels"]) == 0:
            self.baselines["labels"] = [None for _ in self.baselines["values"]]

        if self.legend["location"] == None or len(self.histogram["labels"]) == 0:
            self.histogram["labels"] = [None for _ in self.baselines["values"]]

        if len(self.x)>0:
            self.x_len=len(self.x)
        else:
            if len(self.Ys)>0:
                self.x_len=max([len(y) for y in self.Ys])
        if self.x==[]:
            self.x=list(range(1, self.x_len+1))
        if self.Xs is None:
            self.Xs = [self.x for _ in range(len(self.Ys))]
        self.min_x=min([min(x) for x in self.Xs])
        self.max_x=max([max(x) for x in self.Xs])
        if self.colors is None:
            self.colors = get_colors(len(self.Ys))
        else:
            self.colors = cyclic_fill(self.colors,len(self.Ys))
        self.dashes = cyclic_fill(self.dashes,len(self.Ys))
        if self.markers is None:
            self.markers=[""]
        self.markers = cyclic_fill(self.markers,len(self.Ys))

        if len(self.baselines["values"]) > 0:
            self.baselines["colors"] = cyclic_fill(self.baselines["colors"], len(self.baselines["values"]), default_params["baselines"]["colors"][0])
            self.baselines["dashes"] = cyclic_fill(self.baselines["dashes"], len(self.baselines["values"]), default_params["baselines"]["dashes"][0])

        if len(self.histogram["Xs"]) > 0:
            if self.histogram["colors"] != None:
                self.histogram["colors"] = cyclic_fill(self.histogram["colors"], len(self.histogram["Xs"]), "grey")
            else:
                self.histogram["colors"] = get_colors(len(self.Ys)+len(self.histogram["Xs"]))[len(self.Ys):]

        if self.dirname!="" and not os.path.isdir(self.dirname):
            os.makedirs(self.dirname)

    def export_all(self,dark=False):
        super().export_json()
        self.export_bokeh()
        self.export_matplotlib()
        if dark:
            self.export_bokeh(theme_suffix="_dark")
            self.export_matplotlib(super().dark_color_settings)

    #########
    # bokeh #
    #########
    def export_bokeh(self, theme_suffix=""):
        p = super().init_bokeh_figure(theme_suffix) 

        if self.line45_color!=None:    
            src=ColumnDataSource(data=dict(x=[self.min_x,self.max_x], y=[self.min_x,self.max_x]))
            p.line("x","y", line_color=self.line45_color, line_width=2, source=src)

        sources = [ColumnDataSource(data=dict(x=selet_not_None(x[:len(y)],y),
                                              y=selet_not_None(y,y),
                                              maxim=[max(selet_not_None(y,y)) for _ in selet_not_None(y,y)],
                                              minim=[min(selet_not_None(y,y)) for _ in selet_not_None(y,y)],
                                              argmax=[np.argmax(selet_not_None(y,y))+1 for _ in selet_not_None(y,y)],
                                              argmin=[np.argmin(selet_not_None(y,y))+1 for _ in selet_not_None(y,y)],
                                              label=[get_string_val(self.legend["labels"],i) for _ in selet_not_None(y,y)]
                                             )
                                    ) for i,(x,y) in enumerate(zip(self.Xs, self.Ys))]

        for (c, l, source, dash, marker) in zip(self.colors, self.legend["labels"], sources, self.dashes, self.markers):
            if l!=None:
                p.line('x', 'y', color=c, line_width=2, source=source, line_dash = dash, legend_label=l)
            else:
                p.line('x', 'y', color=c, line_width=2, source=source, line_dash = dash)
            if marker==".":
                p.circle('x', 'y', color=c, source=source)           
        
        if len(self.histogram["Xs"]) > 0:
            for y, color, label in zip(*[self.histogram[k] for k in ["Xs","colors","labels"]]):
                if self.xscale=="log":
                    bins=np.logspace(np.log10(min(y)),np.log10(max(y)), self.histogram["bins"])
                else:
                    bins=self.histogram["bins"]
                    
                hist, edges = np.histogram(y, density=self.histogram["density"], bins=bins)

                source = ColumnDataSource(data=dict(x=[(left+right)/2 for left,right in zip(edges[:-1],edges[1:])],
                                                    y=hist,
                                                    left=edges[:-1],
                                                    right=edges[1:],
                                                    maxim=[max(hist) for _ in hist],
                                                    label=[label for _ in hist]
                                                   )
                                         )

                if self.yscale=="log":
                    bottom=1
                else:
                    bottom=0
                                
                if label!=None:
                    p.quad(top="y", bottom=bottom, left="left", right="right", source=source, fill_color=color, line_color=color, alpha=self.histogram["alpha"], legend_label=label)
                else:
                    p.quad(top="y", bottom=bottom, left="left", right="right", source=source, fill_color=color, line_color=color, alpha=self.histogram["alpha"])
        
        if len(self.baselines["values"]) > 0:
            for name, value, color, dash in zip(*[self.baselines[k] for k in ["labels","values","colors","dashes"]]):
                src = ColumnDataSource(data = {"x": [self.min_x-1,self.max_x+1],
                                               "y": [value,value],
                                               "maxim": [value,value],
                                               "label": [name,name]})
                if name!=None:
                    p.line("x","y", line_dash = dash, line_color = color, line_width = 2, source = src, legend_label = name)
                else:
                    p.line("x","y", line_dash = dash, line_color = color, line_width = 2, source = src)

        if self.legend["location"]!=None:
            p.legend.location = self.legend["location"] 
        
        tooltips = [
            (self.xlabel, "@x"),
            (self.ylabel, "@y"),
            ("name", "@label"),
            ("max", "@maxim"),
            ("argmax", "@argmax"),
            ("min", "@minim"),
            ("argmin", "@argmin")
        ]

        p.add_tools(HoverTool(tooltips = tooltips, mode='vline'))
        p.add_tools(HoverTool(tooltips = tooltips))
        
        super().save_bokeh_figure(p,theme_suffix)

    ##############
    # matplotlib #
    ##############
    def export_matplotlib(self, color_settings={"suffix":"", "grid_color":"0.9", "face_color":"white", "line_color": "black", "bg_transparent":True}):
        with plt.style.context((self.matplotlib["style"])):
            ax = super().init_matplotlib_figure()
            
            if self.line45_color!=None:
                plt.plot([self.min_x,self.max_x],[self.min_x,self.max_x], color = self.line45_color, zorder=10)

            if len(self.baselines["values"]) > 0:
                for label, value, color, dash in zip(*[self.baselines[k] for k in ["labels","values","colors","dashes"]]):
                    plt.plot([self.min_x-1, self.max_x+1], [value,value],
                              matplotlib_dashes[dash], label=label, color = color, zorder=20)

            for x,y,dash,color,label,marker in zip(self.Xs,self.Ys,self.dashes,self.colors,self.legend["labels"],self.markers):
                plt.plot(x[:len(y)], y, matplotlib_dashes[dash], marker=marker, color=color, label=label, zorder=30)
            
            if len(self.histogram["Xs"]) > 0:
                for y, color, label in zip(*[self.histogram[k] for k in ["Xs","colors","labels"]]):
                    if self.xscale=="log":
                        bins=np.logspace(np.log10(min(y)),np.log10(max(y)), self.histogram["bins"])
                    else:
                        bins=self.histogram["bins"]
                    plt.hist(y, density=self.histogram["density"], bins=bins, alpha = self.histogram["alpha"], color=color, label=label, zorder=10)

            if self.legend["location"]!=None and (len(self.legend["labels"])>0 or len(self.baselines["labels"]) > 0 or len(self.histogram["labels"]) > 0):
                legend = ax.legend(loc=matlotlib_legend_loc(self.legend["location"]))
                frame = legend.get_frame()
                frame.set_facecolor(color_settings["face_color"])
                frame.set_edgecolor(color_settings["grid_color"])
                for text in legend.get_texts():
                    text.set_color(color_settings["line_color"])

            plt.grid(True, color=color_settings["grid_color"], zorder=0)
            matplotlib_setcolors(ax, color_settings)         
            
            #Feliratok az x tengelyen
            if self.matplotlib["calc_xtics"]:
                x_max=max([len(str(x)) for x in self.x[:self.x_len]])
                plt.xticks([float(str(self.x[:self.x_len][i])[:6]) for i in range(self.x_len) if i % max(int(min(x_max,6) * self.x_len / (4*self.matplotlib["width"])),1)==0])

            super().save_matplotlib_figure(color_settings["suffix"],color_settings["bg_transparent"])

class ScatterPlotter(Plotter):
    def __init__(self, params, neptune_experiment=None):
        super().__init__(neptune_experiment) 
        default_params={
            #Xs <-- no default, list of lists
            #Ys <-- no default, list of lists
            "xlabel": "",
            "ylabel": "",
            "xscale": "linear",
            "yscale": "linear",
            "title": "",
            "line45_color": "red", #None to turn off line45
            "colors": None, # the bulit in categorical colors go up to 256
            "fname": "scatter_plot",
            "dirname": "",
            "circle_size": 10,
            "x_jitter": 0,
            "opacity": 0,
            "heatmap": False,
            "boundary":{
                "functions": [],
                "dashes": ["dashed"], #"solid", "dashed", "dotted", "dotdash", "dashdot"
                "colors": ["red"], #"solid", "dashed", "dotted", "dotdash", "dashdot"
            },
            "legend":{
                "labels": [],
                "location": "bottom_right", # "top_left", "top_right", "bottom_left", "bottom_right", None
                "markerscale": 2.
            },
            "matplotlib":{ # for png and svg
                "width": 16,
                "height": 9,
                "style": "seaborn-poster", #"seaborn-poster", "seaborn-talk"
                "png_dpi": 240 #use 240 for 4k resolution on 16x9 image
            },
            "bokeh":{
                "width": None, #set this to force bokeh plot to fixed dimensions (not recommended)
                "height": None #set this to force bokeh plot to fixed dimensions (not recommended)
            }       
        }
        for key in default_params:
            params[key] = params.get(key, default_params[key])
        for key in ["legend","matplotlib","bokeh","boundary"]:
            for key2 in default_params[key]:
                params[key][key2] = params[key].get(key2, default_params[key][key2])
        self.params = params
        for k, v in params.items():
            setattr(self, k, v)

        if len(self.legend["labels"]) == 0:
            self.legend["location"] = None

        if self.legend["location"] == None:
            self.legend["labels"] = [None for _ in self.Xs]        

        if type(self.Xs[0])!=list:
            self.Xs=[self.Xs]
        if type(self.Ys[0])!=list:
            self.Ys=[self.Ys]

        for x,y in zip(self.Xs,self.Ys):
            nan_idx=[]
            for i in reversed(range(len(y))):
                if np.isnan(y[i]):
                    nan_idx.append(i)
            for idx in nan_idx:
                x.pop(idx)
                y.pop(idx)

        if self.colors == None:
            self.colors = get_colors(len(self.Ys))
        else:
            self.colors = cyclic_fill(self.colors,len(self.Ys))

        self.boundary["dashes"] = cyclic_fill(self.boundary["dashes"],len(self.boundary["functions"]))
        self.boundary["colors"] = cyclic_fill(self.boundary["colors"],len(self.boundary["functions"]))

        if self.dirname!="" and not os.path.isdir(self.dirname):
            os.makedirs(self.dirname)
        if self.line45_color!=None or len(self.boundary["functions"])>0:
            self.min_x=min([min(X) for X in self.Xs])
            self.max_x=max([max(X) for X in self.Xs])

    def export_all(self,dark=False):
        super().export_json()
        self.export_bokeh(self.Xs, self.Ys, self.legend["labels"], self.colors)
        self.export_matplotlib(self.Xs, self.Ys, self.legend["labels"], self.colors)
        if dark:
            self.export_bokeh(self.Xs, self.Ys, self.legend["labels"], self.colors,theme_suffix="_dark")
            self.export_matplotlib(self.Xs, self.Ys, self.legend["labels"], self.colors, super().dark_color_settings)
        if self.heatmap:
            self.circle_size/=2
            for i,(x,y,lbl) in enumerate(zip(self.Xs,self.Ys,self.legend["labels"])):
                if lbl!=None:
                    suffix="_heatmap_"+lbl
                else:
                    suffix="_heatmap_"+str(i+1)

                self.export_bokeh([x],[y],[lbl],["black"],heatmap=True, suffix=suffix)
                self.export_matplotlib([x],[y],[lbl],["black"],heatmap=True, suffix=suffix)
                if dark:
                    self.export_bokeh([x],[y],[lbl],["black"],heatmap=True, suffix=suffix,theme_suffix="_dark",palette=[RGB(255*r,255*g,255*b,(r+g+b)/2) for r,g,b in super().rgb])
                    self.export_matplotlib([x],[y],[lbl],[super().dark_color_settings["line_color"]],super().dark_color_settings, heatmap=True, suffix=suffix)

    #########
    # bokeh #
    #########
    def export_bokeh(self, Xs, Ys, labels, colors, heatmap=False, suffix="", theme_suffix="", palette=cc.CET_L18):
        p = super().init_bokeh_figure(theme_suffix) 
        
        if self.line45_color!=None:
            src=ColumnDataSource(data=dict(x=[self.min_x,self.max_x], y=[self.min_x,self.max_x]))
            p.line("x","y", line_color=self.line45_color, line_width=2, source=src)
        
        if len(self.boundary["functions"])>0:
            x_range=np.linspace(self.min_x,self.max_x,100)
            for bf,dash,color in zip(self.boundary["functions"],self.boundary["dashes"],self.boundary["colors"]):
                src = ColumnDataSource(data=dict(
                    x=x_range,
                    y=[eval(bf) for x in x_range],
                ))
                p.line('x', 'y', color=color, line_width=2, source=src, line_dash = dash)
            
        for x,y,color,label in zip(Xs,Ys,colors,labels):
            if label!=None:
                p.circle(jitter(x,self.x_jitter), y, size=self.circle_size, line_width=0, color=color, alpha=1-self.opacity, legend_label = label)
            else:
                p.circle(jitter(x,self.x_jitter), y, size=self.circle_size, line_width=0, color=color, alpha=1-self.opacity)

            if heatmap:
                xmin = min(x)
                xmax = max(x)
                ymin = min(y)
                ymax = max(y)
                xrange = xmax-xmin
                xmin = min(x)-0.25*xrange
                xmax = max(x)+0.25*xrange
                yrange = ymax-ymin
                ymin = min(y)-0.25*yrange
                ymax = max(y)+0.25*yrange

                X, Y = np.mgrid[xmin:xmax:500j, ymin:ymax:500j]
                positions = np.vstack([X.ravel(), Y.ravel()])
                values = np.vstack([x, y])
                kernel = stats.gaussian_kde(values)
                Z = np.reshape(kernel(positions).T, X.shape)

                p.image(image=[np.transpose(Z)], x=xmin, y=ymin, dw=xmax-xmin, dh=ymax-ymin, palette=palette, level="image")

                color_mapper = LinearColorMapper(palette=palette, low=0, high=np.amax(Z))
                color_bar = ColorBar(color_mapper=color_mapper, location = (0,0))

                p.add_layout(color_bar, 'right')

        if self.legend["location"] != None:
            p.legend.location = self.legend["location"]
            p.legend.glyph_height = int(self.circle_size*self.legend["markerscale"]*2)
            p.legend.glyph_width = int(self.circle_size*self.legend["markerscale"]*2)
        
        p.add_tools(HoverTool(tooltips = [(self.xlabel, "$x"), (self.ylabel, "$y")]))

        

        code_hover = '''
            document.getElementsByClassName('bk-tooltip')[0].style.display = 'none';
            document.getElementsByClassName('bk-tooltip')[1].style.display = 'none';

        '''
        if heatmap:
            code_hover += "document.getElementsByClassName('bk-tooltip')[2].style.display = 'none';"
        p.hover.callback = CustomJS(code = code_hover)

        super().save_bokeh_figure(p,suffix+theme_suffix)

    ##############
    # matplotlib #
    ##############
    def export_matplotlib(self, Xs, Ys, labels, colors, color_settings={"suffix":"", "grid_color":"0.9", "face_color":"white", "line_color": "black", "bg_transparent":False, "cmap":"cet_CET_L18"}, heatmap=False, suffix=""):
        with plt.style.context((self.matplotlib["style"])):
            ax = super().init_matplotlib_figure()          

            if self.line45_color!=None:
                plt.plot([self.min_x,self.max_x],[self.min_x,self.max_x], color = self.line45_color, zorder=10)

            if len(self.boundary["functions"])>0:
                x_range=np.linspace(self.min_x,self.max_x,100)
                for bf,dash,color in zip(self.boundary["functions"],self.boundary["dashes"],self.boundary["colors"]):
                    plt.plot(x_range, [eval(bf) for x in x_range], matplotlib_dashes[dash], color=color, zorder=15)

            plt.grid(True, color=color_settings["grid_color"], zorder=5, alpha=0.5)

            for x,y,color,label in zip(Xs, Ys, colors, labels):
                plt.scatter(x, y, marker='.', color=color, zorder=30, alpha=1-self.opacity, linewidth=0, s=self.circle_size**2, label=label)
                
                if heatmap:
                    xmin = min(x)
                    xmax = max(x)
                    ymin = min(y)
                    ymax = max(y)

                    X, Y = np.mgrid[xmin:xmax:500j, ymin:ymax:500j]
                    positions = np.vstack([X.ravel(), Y.ravel()])
                    values = np.vstack([x, y])
                    kernel = stats.gaussian_kde(values)
                    Z = np.reshape(kernel(positions).T, X.shape)

                    im = ax.imshow(np.rot90(Z), cmap=color_settings["cmap"], extent=[xmin, xmax, ymin, ymax],zorder=0)

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.1)

                    cb = plt.colorbar(im, cax=cax)

                    cb.ax.yaxis.set_tick_params(color=color_settings["line_color"])
                    cb.outline.set_edgecolor(color_settings["grid_color"])
                    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=color_settings["line_color"])
                    #cb.patch.set_alpha(0)

            if self.legend["location"]!=None and len(labels)>0:
                legend = ax.legend(loc=matlotlib_legend_loc(self.legend["location"]), markerscale=self.legend["markerscale"])
                legend.set_zorder(40)
                frame = legend.get_frame()
                frame.set_facecolor(color_settings["face_color"])
                frame.set_edgecolor(color_settings["grid_color"])
                for text in legend.get_texts():
                    text.set_color(color_settings["line_color"])              

            matplotlib_setcolors(ax, color_settings)

            super().save_matplotlib_figure(suffix+color_settings["suffix"],color_settings["bg_transparent"])

class SpectrumPlotter(Plotter):
    def __init__(self, params, neptune_experiment=None):
        super().__init__(neptune_experiment) 
        default_params={
            #spectrum <-- no default, list of lists
            "fname": "spectrogram",
            "dirname": "",
            "title": "",
            "xlabel": "Window",
            "ylabel": "Spectrum",
            "xscale": "linear",
            "yscale": "linear",
            "matplotlib":{ # for png and svg
                "width": 16,
                "height": 9,
                "style": "seaborn-poster", #"seaborn-poster", "seaborn-talk"
                "png_dpi": 240 #use 240 for 4k resolution on 16x9 image
            }     
        }
        for key in default_params:
            params[key] = params.get(key, default_params[key])
        for key in ["matplotlib"]:
            for key2 in default_params[key]:
                params[key][key2] = params[key].get(key2, default_params[key][key2])
        self.params = params
        for k, v in params.items():
            setattr(self, k, v)

        if self.dirname!="" and not os.path.isdir(self.dirname):
            os.makedirs(self.dirname)

    def export_all(self,dark=False):
        super().export_json()
        self.export_matplotlib(self.spectrum)

    ##############
    # matplotlib #
    ##############
    def export_matplotlib(self, spectrum):
        ax = super().init_matplotlib_figure()
        img = ax.imshow(spectrum, cmap="viridis", origin="lower", aspect="auto")
        
        super().save_matplotlib_figure("")


class PCAPlotter(Plotter):
    def __init__(self, params, neptune_experiment=None):
        super().__init__(neptune_experiment) 
        default_params={
            #vectors <-- no default, list of lists
            "labels": None,
            "xlabel": "x",
            "ylabel": "y",
            "xscale": "linear",
            "yscale": "linear",
            "title": "",
            "2D": True,
            "3D": False,
            "colors": None, # the bulit in categorical colors go up to 256
            "fname": "PCA_plot",
            "dirname": "",
            "circle_size": 15,
            "opacity": 0.2,
            "heatmap": False,
            "legend":{
                "labels": [],
                "location": "bottom_right", # "top_left", "top_right", "bottom_left", "bottom_right", None
            },
            "matplotlib":{ # for png and svg
                "width": 16,
                "height": 9,
                "style": "seaborn-poster", #"seaborn-poster", "seaborn-talk"
                "png_dpi": 240 #use 240 for 4k resolution on 16x9 image
            },
            "bokeh":{
                "width": None, #set this to force bokeh plot to fixed dimensions (not recommended)
                "height": None, #set this to force bokeh plot to fixed dimensions (not recommended)
                "annotate": True
            }       
        }
        for key in default_params:
            params[key] = params.get(key, default_params[key])
        for key in ["legend","matplotlib","bokeh"]:
            for key2 in default_params[key]:
                params[key][key2] = params[key].get(key2, default_params[key][key2])
        try:
            params["vectors"][0][0][0]
        except:    
            params["vectors"] = [params["vectors"]]

        params["vectors"] = [[list(v.astype(float)) if type(v)!=list else v for v in V] for V in params["vectors"]]

        self.params = params
        for k, v in params.items():
            setattr(self, k, v)

        if len(self.legend["labels"]) == 0:
            self.legend["location"] = None
        
        if self.labels == None:
            self.labels = [[""]*len(v) for v in self.vectors]
        else:
            if type(self.labels[0]) != list:
                self.labels = [self.labels]

        if self.legend["location"] == None:
            self.legend["labels"] = [None for _ in self.vectors]        

        if self.colors == None:
            self.colors = get_colors(len(self.vectors))
        else:
            self.colors = cyclic_fill(self.colors,len(self.vectors))

        if self.dirname!="" and not os.path.isdir(self.dirname):
            os.makedirs(self.dirname)

        self.vectors=np.array(self.vectors)

        pca = PCA()
        self.vectors = pca.fit_transform(self.vectors.reshape(-1, self.vectors.shape[-1])).reshape(self.vectors.shape[0],self.vectors.shape[1],-1)

    def export_tsv(self): #https://projector.tensorflow.org/
        df = pd.DataFrame({
            'label': flatten_list(self.labels),
            'category': flatten_list([[self.legend["labels"][i]]*len(v) for i,v in enumerate(self.vectors)])
        })
        df.to_csv(self.get_full_path("tsv","_labels"), sep="\t", index=False)

        with open(self.get_full_path("tsv","_vectors"), 'w') as f:
            f.write("\n".join(["\t".join(str(val) for val in v) for V in self.vectors for v in V]))

        if self.neptune_experiment!=None:
            self.neptune_experiment[self.fname].upload(self.get_full_path("tsv","_labels"))
            self.neptune_experiment[self.fname].upload(self.get_full_path("tsv","_vectors"))
            self.neptune_experiment.sync()

    def export_all(self):
        self.export_tsv()
        super().export_json()
        if self.params["2D"]:
            self.export_bokeh()
            self.export_bokeh(theme_suffix="_dark")
            self.export_matplotlib()
            self.export_matplotlib(super().dark_color_settings)
        if self.params["3D"]:
            self.export_plotly()

    #########
    # bokeh #
    #########
    def export_bokeh(self, theme_suffix=""): 
        p = super().init_bokeh_figure(theme_suffix)        
            
        vectors_all = self.vectors.reshape(-1, self.vectors.shape[-1])
        
        src = ColumnDataSource(data = {"x": vectors_all[:,0],
                                        "y": vectors_all[:,1],
                                        "color": flatten_list([[self.colors[i]]*len(v) for i,v in enumerate(self.vectors)]),
                                        "label": flatten_list(self.labels),
                                        "legend_label": flatten_list([[self.legend["labels"][i]]*len(v) for i,v in enumerate(self.vectors)])})
        if self.legend["location"] != None:
            p.circle("x", "y", size=self.circle_size, line_width=0, color="color", alpha=1-self.opacity, source = src, legend_group = "legend_label")
            p.legend.location = self.legend["location"] 
        else:
            p.circle("x", "y", size=self.circle_size, line_width=0, color="color", alpha=1-self.opacity, source = src)

        if self.bokeh["annotate"]:
            labels = LabelSet(x='x', y='y', text='label', x_offset=5, y_offset=5, source=src, render_mode='canvas')
            p.add_layout(labels)
        
        p.add_tools(HoverTool(tooltips = [("", "@label")])) #,(self.xlabel, "@x"),(self.ylabel, "@y")]))

        super().save_bokeh_figure(p,theme_suffix) 
    
    ##########
    # plotly #
    ##########
    def export_plotly(self, theme_suffix=""): 
        vectors_all = self.vectors.reshape(-1, self.vectors.shape[-1])
        
        df = pd.DataFrame({
            "x": vectors_all[:,0],
            "y": vectors_all[:,1],
            "z": vectors_all[:,2], 
            "label": np.array(self.labels).flatten(),
            "color": np.array([[self.colors[i]]*len(v) for i,v in enumerate(self.vectors)]).flatten(),
            "legend_label": np.array([[self.legend["labels"][i]]*len(v) for i,v in enumerate(self.vectors)]).flatten()
        })

        fig = px.scatter_3d(df, x='x', y='y', z='z', color_discrete_map={l:c for l,c in zip(self.legend["labels"], self.colors)}, color="legend_label", text="label")

        if self.legend["location"] == None:
            fig.update_layout(showlegend=False)

        fig.show()

        fig.write_html(super().get_full_path("html","_3D"))
        if self.neptune_experiment!=None:
            self.neptune_experiment[self.fname].upload(super().get_full_path("html","_3D"))
            self.neptune_experiment.sync() 
        

    ##############
    # matplotlib #
    ##############
    def export_matplotlib(self, color_settings={"suffix":"", "grid_color":"0.9", "face_color":"white", "line_color": "black", "bg_transparent":False, "cmap":"cet_CET_L18"}):
        with plt.style.context((self.matplotlib["style"])):
            ax = super().init_matplotlib_figure()          

            plt.grid(True, color=color_settings["grid_color"], zorder=5, alpha=0.5)

            for vectors,color,label,lbls in zip(self.vectors, self.colors, self.legend["labels"], self.labels):
                plt.scatter(vectors[:,0], vectors[:,1], marker='.', color=color, zorder=30, alpha=1-self.opacity, linewidth=0, s=self.circle_size**2, label=label)
                if lbls!=None:
                    for i, word in enumerate(lbls):
                        plt.annotate(word, xy=(vectors[i,0],vectors[i,1]), xytext=(self.circle_size/4,3), textcoords='offset points', color=color_settings["line_color"])  

            if self.legend["location"]!=None and len(self.legend["labels"])>0:
                legend = ax.legend(loc=matlotlib_legend_loc(self.legend["location"]))
                frame = legend.get_frame()
                frame.set_facecolor(color_settings["face_color"])
                frame.set_edgecolor(color_settings["grid_color"])
                for text in legend.get_texts():
                    text.set_color(color_settings["line_color"])     

            matplotlib_setcolors(ax, color_settings)

            super().save_matplotlib_figure(color_settings["suffix"],color_settings["bg_transparent"])

class ConfMtxPlotter(Plotter):
    def __init__(self, params, neptune_experiment=None):
        super().__init__(neptune_experiment) 
        default_params={
            #mtx <-- no default
            "xlabel": "Predicted",
            "ylabel": "True",
            "xscale": "linear",
            "yscale": "linear",
            "title": "",
            "fname": "confmtx_plot",
            "dirname": "",
            "show_counts": True,
            "sort_by_sum": False,
            "matplotlib":{ # for png and svg
                "width": 16,
                "height": 9,
                "style": "seaborn-poster", #"seaborn-poster", "seaborn-talk"
                "png_dpi": 240 #use 240 for 4k resolution on 16x9 image
            },
            "bokeh":{
                "width": None, #set this to force bokeh plot to fixed dimensions (not recommended)
                "height": None, #set this to force bokeh plot to fixed dimensions (not recommended)
                "annotate": True
            }       
        }
        for key in default_params:
            params[key] = params.get(key, default_params[key])
        for key in ["matplotlib","bokeh"]:
            for key2 in default_params[key]:
                params[key][key2] = params[key].get(key2, default_params[key][key2])
        
        self.params = params
        for k, v in params.items():
            setattr(self, k, v)    

        if self.dirname!="" and not os.path.isdir(self.dirname):
            os.makedirs(self.dirname)

    def export_all(self):
        super().export_json()
        self.export_bokeh()
        #self.export_bokeh(theme_suffix="_dark")

    def get_LinearColorMapper(self,column):
        low = min(column)
        high = max(column)
        mpr = LinearColorMapper(palette = ['#000000']+cc.CET_L8, low = low, high = high)
        log_mpr = LogColorMapper(palette = ['#000000']+cc.CET_L8, low = low, high = high)
        txt_mpr = LinearColorMapper(palette = ['#ffffff','#000000'], low = low, high = high)
        return mpr, log_mpr,txt_mpr

    def calc_df(self):
        rows_all=list(self.mtx.keys())
        cols_all=list(list(self.mtx.values())[0].keys())
            
        rows=[]
        row_sums=[]
        for row in rows_all:
            sum_reg=0
            for col in cols_all:
                sum_reg+=self.mtx[row][col]
            if sum_reg>0:
                rows.append(row)
                row_sums.append(sum_reg)
        
        cols=[]
        col_sums=[]
        for col in cols_all:
            sum_fn=0
            for row in rows_all:
                sum_fn+=self.mtx[row][col]
            if sum_fn>0:
                cols.append(col)
                col_sums.append(sum_fn)
            
        if self.sort_by_sum:
            row_sums,rows = zip(*[(x,y) for x,y in sorted(zip(row_sums,rows), reverse=True)])
            
            col_sums,cols = zip(*[(x,y) for x,y in sorted(zip(col_sums,cols), reverse=True)])
        
        dic = {}
        dic["row_labels"] = []
        dic["row_sums"] = []
        dic["col_labels"] = []
        dic["col_sums"] = []
        dic["value"] = []
        dic["color"] = []
        for typ in ["abs","log","sqrt","square"]:
            dic["color_"+typ]=[]
        for i,row in enumerate(rows):
            for j,col in enumerate(cols):
                dic["row_labels"].append(row)
                dic["row_sums"].append(row_sums[i])
                dic["col_labels"].append(col)
                dic["col_sums"].append(col_sums[j])
                dic["value"].append(self.mtx[row][col])
                
        low=min(dic["value"])
        high=max(dic["value"])
        for row in rows:
            for col in cols:
                for typ in ["abs","log","sqrt","square"]:
                    dic["color_"+typ].append(my_cmap(self.mtx[row][col], high, low, typ=typ))
                dic["color"].append(my_cmap(self.mtx[row][col], high, low, typ="abs"))
            
        df = pd.DataFrame.from_dict(dic)
        
        return df, rows, cols


    #########
    # bokeh #
    #########
    def export_bokeh(self, theme_suffix=""):
        df, row_labels, col_labels  = self.calc_df()

        size_multiplier = 100
        if len(col_labels)>25:
            size_multiplier = 50
        if len(col_labels)>100:
            size_multiplier = 30
        if self.bokeh["height"] == None:
            self.bokeh["height"] = size_multiplier*len(row_labels)
        if self.bokeh["width"] == None:
            self.bokeh["width"] = size_multiplier*len(col_labels)+90

        p = figure(
            plot_height = self.bokeh["height"],
            plot_width = self.bokeh["width"],
            y_range = list(reversed(row_labels)),
            x_range = list(col_labels),#[:len(row_labels)]),
            x_axis_location = "above",
            tools = "pan,box_zoom,wheel_zoom,save,reset"
        )

        p = super().init_bokeh_figure(theme_suffix, p)
        
        source = ColumnDataSource(df)

        mapper, log_mapper, text_mapper = self.get_LinearColorMapper(df['value'])        

        rect = p.rect(
            x = "col_labels",
            y = "row_labels",
            width = 1,
            height = 1,
            source = source,
            line_color = "color",
            fill_color = "color"
        )
        if self.show_counts:
            p.text(
                x = "col_labels",
                y = "row_labels",
                text = "value",
                source = source,
                text_font_style = 'bold',
                text_align = 'center',
                text_baseline = 'middle',
                text_color = transform('value', text_mapper),
                text_font_size = "10pt"
            )

        color_bar = ColorBar(
            color_mapper = mapper,
            location = (0, 0),
            ticker = BasicTicker(desired_num_ticks=10)
        )

        p.add_layout(color_bar, 'right')

        p.axis.axis_label_text_font_size = "5pt"
        p.xaxis.major_label_orientation = "vertical"
        p.xaxis.axis_label_text_font_size = "12pt"
        p.xaxis.major_label_text_font_size = "8pt"
        p.yaxis.axis_label_text_font_size = "12pt"
        p.yaxis.major_label_text_font_size = "8pt"

        p.add_tools(HoverTool(tooltips = [
            (self.xlabel, '@col_labels'),
            ('Col sum'.format(self.xlabel), '@col_sums'),
            (self.ylabel, '@row_labels'),
            ('Row sum'.format(self.ylabel), '@row_sums'),
            ('count', '@value')
        ], renderers = [rect]))

        select = Select(value='Linear', options=['Linear', 'Logarithmic', 'Square', 'Square root'])
        args =dict(
            source = source,
            select = select,
            color_bar = color_bar,
            rect = rect,
            mapper = mapper,
            log_mapper = log_mapper,
            text_mapper = text_mapper,
        )
        select.js_on_change('value', CustomJS(args=args, code="""
            // make a shallow copy of the current data dict
            const new_data = Object.assign({}, source.data)

            switch(select.value) {
              case 'Linear':
                new_data.color = source.data['color_abs']
                var color_mapper = mapper
                break;
              case 'Logarithmic':
                new_data.color = source.data['color_log']
                var color_mapper = log_mapper
                break;
              case 'Square':
                new_data.color = source.data['color_square']
                break;
              case 'Square root':
                new_data.color = source.data['color_sqrt']
                break;
            } 

            rect.glyph.fill_color = {field: 'color'};
            rect.glyph.line_color = {field: 'color'};
            color_bar.color_mapper = color_mapper
                        
            // set the new data on source, BokehJS will pick this up automatically
            source.data = new_data
            source.change.emit();
            // rect.change.emit();
        """))

        super().save_bokeh_figure(column(p, select),theme_suffix) 
   
