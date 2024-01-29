from plotters import GeneralPlotter
import bokeh

gp_params={
    "Ys": [
        [92.676, 94.032, 94.082, 94.332, 94.44, 94.738, 94.802, 94.846, 94.774],
        [90.236, 94.072, 94.378, 94.764, 95.212, 95.488, 95.864, 96.172, 96.576]
    ],
    "x": [100,200,400,800,1600,3200,6400,12800,25000],
    "errorbars": [
        [1.025, 0.279, 0.367, 0.447, 0.306, 0.102, 0.0686, 0.0686, 0.0395],
        [2.251, 0.514, 0.517, 0.288, 0.349, 0.0573, 0.0981, 0.0497, 0.191]
    ],
    "xlabel": "Training dataset size",
    "ylabel": "Accuracy %",
    "title": "",
    "fname": "IMDB_data_size",
    "dirname": "./",
    "markers": None,
    "legend": {
        "location": "bottom_right",
        "labels": ["llambert","gold"]
    },
    "matplotlib": {
        "calc_xtics": False,
        "width": 9,
        "height": 9,
        "style": "seaborn-poster",
        "png_dpi": 240
    },
    "colors": [bokeh.palettes.Category10[10][0], bokeh.palettes.Category10[10][1]], 
    "dashes": ["solid","solid"],
    "line45_color": None,
    "baselines": {
        "labels": [],
        "values": [],
        "colors": ["grey"],
        "dashes": ["dotted"]
    },
    "histogram": {
        "labels": [],
        "Xs": [],
        "colors": None
    },
    "bokeh": {
        "width": None,
        "height": None
    }
}
#GeneralPlotter(gp_params).export_all()
# gp_params["xscale"]="log"
# gp_params["yscale"]="log"
# gp_params["fname"]+="_loglog"
GeneralPlotter(gp_params).export_all()