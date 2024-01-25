from plotters import GeneralPlotter
import bokeh

gp_params={
    "Ys": [
        [96.54, 96.50, 96.47, 96.35, 96.23, 95.90, 95.76, 94.82, 92.99, 75.67, 70.92, 66.86, 65.52, 47.13],
        [76.27, 77.84, 78.93, 78.27, 78.14, 77.04, 74.29, 68.12, 60.36, 54.72, 54.16, 53.72, 53.57, 51.96],
        #[95.84, 95.70, 95.70, 95.75, 95.65, 95.46, 95.53, 94.74, 91.87, 60.36, 67.14, 52.14, 52.06, 49.42]
    ],
    "x": [0,1,2,5,10,20,30,40,45,48,48.5,48.8,49,50],
    "xlabel": "Mislabel %",
    "ylabel": "Accuracy %",
    "title": "",
    "fname": "IMDB_data_mislabel",
    "dirname": "./",
    "markers": None,
    "legend": {
        "location": "bottom_left",
        "labels": ["finetuned","frozen"] #,"supergold"]
    },
    "matplotlib": {
        "calc_xtics": False,
        "width": 9,
        "height": 9,
        "style": "seaborn-poster",
        "png_dpi": 240
    },
    "colors": [bokeh.palettes.Category10[10][0], bokeh.palettes.Category10[10][1]], #, bokeh.palettes.Category10[10][2]],
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