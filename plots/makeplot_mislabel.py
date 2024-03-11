from plotters import general_plot

gp_params={
    "Ys": [
        [96.54, 96.50, 96.47, 96.35, 96.23, 95.90, 95.76, 94.82, 92.99, 75.67, 70.92, 66.86, 65.52, 47.13],
        [76.27, 77.84, 78.93, 78.27, 78.14, 77.04, 74.29, 68.12, 60.36, 54.72, 54.16, 53.72, 53.57, 51.96],
        #[95.84, 95.70, 95.70, 95.75, 95.65, 95.46, 95.53, 94.74, 91.87, 60.36, 67.14, 52.14, 52.06, 49.42]
    ],
    "x": [0,1,2,5,10,20,30,40,45,48,48.5,48.8,49,50],
    "xlabel": "Mislabel %",
    "ylabel": "Accuracy %",
    "fname": "IMDb_data_mislabel",
    "dirname": "./",
    "legend": {
        "location": "bottom_left",
        "labels": ["End-to-end","Frozen encoder"] #,"supergold"]
    },
    "matplotlib": {
        "calc_xtics": False,
        "ytics": [50,55,60,65,70,75,80,85,90,95,],
        "width": 9,
        "height": 6,
        "style": "seaborn-poster",
        "png_dpi": 240
    }
}
general_plot(gp_params)