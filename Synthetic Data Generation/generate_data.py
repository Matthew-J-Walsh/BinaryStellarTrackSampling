##########################################################
## generate_data.py By Kyle Akira Rocha 8/19
## This is a modified version of run2.py by Milena Healy
##
## use -h for help
##
## EXAMPLES:
## defaults      >>> python generate_data.py
## other options >>> python generate_data.py -g circles -n 500 --file_str 'myfile.dat' -cwr 3
##               >>> python generate_data.py -rf linear quadratic -m 0 1 0 1 0.8 0.75 -show True -v True
#########################################################
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
from datetime import date
import datetime as datetime

parser = argparse.ArgumentParser(description='Generate synthetic data sets for CRIS.')

parser.add_argument('-g',  '--geometry' , dest='geometry'  , default = 'squares', type=str  , help='classification geometry - "y", "circles", "squares"')
parser.add_argument('-n',  '--n_points' , dest='n_points'  , default = 100        , type=float,  help='total number of data points to generate')

parser.add_argument('-cwr', '--classes_w_regr' , dest='classes_w_regr'  , default = 0 , type=float,  help='num classes with regression data')
parser.add_argument('-rf',  '--regr_funcs' , dest='regr_funcs'  , default = ["linear", "quadratic"], nargs = '+' , type=str,  help='regression function(s)')

parser.add_argument('--file_str' , dest='file_str'  , default = 'synth_data.dat' , type=str, help='name of data file output')
parser.add_argument('-show','--show_plots', dest='show_plots', default = False, type = bool, help='show classification and regression plots')
parser.add_argument('-v', '--verbose', dest='verbose', default = False, type = bool, help='print statements')
parser.add_argument('-m', '--misc', dest='misc', nargs='+', default = [], help='misc args for a given geometry')
parser.add_argument('-i', '--save_info', dest='save_info', default = False, type=bool, help='save an info file along with the data')


#args = parser.parse_args( args = [] )
args = parser.parse_args()

if args.verbose:
    print("Initial Conditions:\n", args)

args.geometry = ( args.geometry ).lower()
num_points = int( args.n_points )
args.classes_w_regr = abs( int( args.classes_w_regr ) )
args.regr_funcs = [name.lower() for name in args.regr_funcs]
num_regr_outputs = len(args.regr_funcs)


# OTHER CLS ARGS
if args.geometry == 'squares':
    # colors: purple (TL), orangle (TR), green (BL), black (BR)
    x_rng_low = 0
    x_rng_high = 1
    y_rng_low = 0
    y_rng_high = 1
    cen_x = 0.5  # intersection of squares
    cen_y = 0.5  # intersection of squares
    if args.misc:
        print("MISC ARGS: {0}".format(args.misc))
        x_rng_low, x_rng_high, y_rng_low, y_rng_high, cen_x, cen_y = [float(val) for val in args.misc]

        # info
    other_cls_args = [x_rng_low, x_rng_high, y_rng_low, y_rng_high, cen_x, cen_y]
    other_cls_args_str = "[x_rng_low, x_rng_high, y_rng_low, y_rng_high, cen_x, cen_y]"
if args.geometry == 'y':
    # colors: purple (TL), green (BL), orange (R)
    x_rng_low = 0
    x_rng_high = 1
    y_rng_low = 0
    y_rng_high = 1
    cen_x = 0.5  # intersection of sideways y
    cen_y = 0.5  # intersection of sideways y
    slope_up = 1  # slope of upper line
    slope_down = -3  # slope of lower line
    b = cen_y - slope_down * cen_x   # intercept
    if args.misc:
        print("MISC ARGS: {0}".format(args.misc))
        x_rng_low, x_rng_high, y_rng_low, y_rng_high, \
        cen_x, cen_y, slope_up, slope_down = [float(val) for val in args.misc]

    # info
    other_cls_args = [x_rng_low, x_rng_high, y_rng_low, y_rng_high, \
                            cen_x, cen_y, slope_up, slope_down]
    other_cls_args_str = "[x_rng_low, x_rng_high, y_rng_low, y_rng_high, cen_x, cen_y, slope_up, slope_down]"
if args.geometry == 'circles':
    # colors: black (small), purple (large), green (outside)
    x_rng_low = 0
    x_rng_high = 1
    y_rng_low = 0
    y_rng_high = 1
    r = 0.5  # radius of larger circle
    h = 0.5  # center x
    k = 0.5  # center y
    r2 = 0.2  # radius of smaller circle
    h2 = 0.5  # center x
    k2 = 0.5  # center y
    if args.misc:
        print("MISC ARGS: {0}".format(args.misc))
        x_rng_low, x_rng_high, y_rng_low, y_rng_high, \
        r, h, k, r2, h2, k2 = [float(val) for val in args.misc]

    # info
    other_cls_args = [x_rng_low, x_rng_high, y_rng_low, y_rng_high, \
                        r, h, k, r2, h2, k2]
    other_cls_args_str = "[x_rng_low, x_rng_high, y_rng_low, y_rng_high, r, h, k, r2, h2, k2]"

# OTHER REGR ARGS
if num_regr_outputs > 0:
    # linear
    lin_m = 1; lin_b = 0; lin_sigma = 2
    # quadratic
    quad_a = 1; quad_b = 1; quad_c = 1; quad_sigma = 3
    # Can set exactly which colors to get regr data for
    # see the above for colors / locations
    use_my_colors = False
    my_colors = ['purple']

    # info
    other_rgr_args = [lin_m, lin_b, lin_sigma, quad_a, quad_b, \
                    quad_c, quad_sigma, use_my_colors, my_colors ]
    other_rgr_args_str = "[lin_m, lin_b, lin_sigma, quad_a, quad_b, quad_c, quad_sigma, use_my_colors, my_colors ]"

#--------------------- CLASSIFICATION-------------------------------
def do_cls_squares():
    linev = cen_x
    lineh = cen_y

    plt.axvline(linev,color="k")
    plt.axhline(lineh,color="k")

    plt.fill_between([x_rng_low,cen_x], cen_y, y_rng_high, alpha=0.3, color='purple')
    plt.fill_between([cen_x,x_rng_high], y_rng_low, cen_y, alpha=0.3, color='black')
    plt.fill_between([cen_x,x_rng_high], cen_y, y_rng_high, alpha=0.3, color='orange')
    plt.fill_between([x_rng_low,cen_x], y_rng_low, cen_y, alpha=0.3, color='green')

    xrandom=np.random.uniform(x_rng_low,x_rng_high,size=num_points)
    yrandom=np.random.uniform(x_rng_low,x_rng_high,size=num_points)

    between_colors=[]
    for i in range(num_points):
        if yrandom[i]>lineh and xrandom[i]<linev:
            between_colors.append("purple")
        elif yrandom[i]<lineh and xrandom[i]<linev:
            between_colors.append("green")
        elif yrandom[i]>lineh and xrandom[i]>linev:
            between_colors.append("orange")
        else:
            between_colors.append("black")

    plt.scatter(xrandom,yrandom,c=between_colors,s=100)
    plt.ylim(y_rng_low,y_rng_high); plt.xlim(x_rng_low,x_rng_high)
    plt.title( "Squares" , fontsize=26)
    if args.show_plots:
        plt.show()

    return xrandom, yrandom, np.array(between_colors)
# end do_squares

def do_cls_y():
    x = np.linspace(x_rng_low, x_rng_high, 1e3)
    def f(x):
        if(x<cen_x): return cen_y
        else: return x * slope_up

    def f2(x):
        if(x>=cen_x): return slope_down * x + b
        else: return cen_y

    y=[]
    for i in range(len(x)):
        y.append(f(x[i]))

    y2=[]
    for i in range(len(x)):
        y2.append(f2(x[i]))

    plt.plot(x,y,color="purple")
    plt.plot(x,y2,color="purple")

    plt.fill_betweenx(y,x, alpha=0.3, color="purple")
    plt.fill_between(x,y,y2,alpha=0.3,color="orange")
    plt.fill_between(x,y2,alpha=0.3,color="g")

    xrandom=np.random.uniform(x_rng_low,x_rng_high,size=num_points)
    yrandom=np.random.uniform(y_rng_low,y_rng_high,size=num_points)

    between_colors=[]
    for i in range(num_points):
        if yrandom[i]>f(xrandom[i]):
            between_colors.append("purple")
        elif yrandom[i]<f2(xrandom[i]):
            between_colors.append("green")
        else:
            between_colors.append("orange")

    plt.scatter(xrandom,yrandom,c=between_colors,s=100)
    plt.ylim(y_rng_low,y_rng_high); plt.xlim(x_rng_low,x_rng_high)
    plt.title("Piecewise", fontsize=26)
    if args.show_plots:
        plt.show()
    return xrandom, yrandom, np.array(between_colors)
# end do_y

def do_cls_circles():
    large_circle = plt.Circle((h,k), r, facecolor='#d8b2d8', edgecolor="purple", zorder=0)
    small_circle = plt.Circle((h2,k2), r2, facecolor="#b2b2b2",edgecolor="purple",zorder=1)

    fig, ax1 = plt.subplots(figsize=(7, 7))
    ax1.set_title("Concentric Circles", fontsize=26)
    ax1.set_facecolor('#b2d8b2')

    ax1.set_ylim([y_rng_low,y_rng_high])
    ax1.set_xlim([x_rng_low,x_rng_high])
    ax1.add_artist(large_circle)
    ax1.add_artist(small_circle)

    xrandom=np.random.uniform(x_rng_low,x_rng_high,size=num_points)
    yrandom=np.random.uniform(y_rng_low,y_rng_high,size=num_points)

    def incircle(x,y):
        dist=np.sqrt(((x-h)**2)+((y-k)**2))
        return dist<r
    def incirclesmaller(x,y):
        dist=np.sqrt(((x-h2)**2)+((y-k2)**2))
        return dist<r2

    func_incircle=incircle(xrandom,yrandom)
    func_incirclesmaller=incirclesmaller(xrandom,yrandom)

    between_colors=[]
    for n in range(num_points):
        if func_incircle[n] and func_incirclesmaller[n]:
            between_colors.append("black")
        elif func_incircle[n]:
            between_colors.append("purple")
        else:
            between_colors.append("green")

    ax1.scatter(xrandom,yrandom,c=between_colors,s=100,zorder=2)
    if args.show_plots:
        plt.show()

    return xrandom, yrandom, np.array(between_colors)
# end do_circles

#--------------------- REGRESSION-------------------------------
def do_regr_outputs( main_df ):

    other_cols = abs( len(main_df.keys()) - num_regr_outputs )

    if use_my_colors:
        classes_to_add_data = my_colors
    else:
        classes_to_add_data = []
        for j in range( args.classes_w_regr ):
            classes_to_add_data.append( unique_classes[j] )

    for cls in classes_to_add_data:
        where_this_class = np.where( main_df['class'] == cls )[0]
        how_many = len(where_this_class)
        ct = 0

        if 'linear' in args.regr_funcs:
            # Using input_1 as the 'x' values for linear
            data1 = get_linear_data( main_df['input_1'].iloc[where_this_class] )
            main_df.iloc[where_this_class, [other_cols+ct] ] = data1
            ct += 1
        if 'quadratic' in args.regr_funcs:
            # Using input_2 as the 'x' values for quadratic
            data2 = get_quadratic_data( main_df['input_2'].iloc[where_this_class] )
            main_df.iloc[where_this_class, [other_cols+ct] ] = data2
            ct += 1

        if ct == 0 :
            print("Names not recognized in {0}".format(args.regr_funcs))

    return main_df
# end do_regr_outputs

def get_linear_data( df ):
    random_error_lin = np.random.normal(loc=0,scale=lin_sigma,size= len(df) )
    return lin_m * df.values + lin_b + random_error_lin
# end get_linear_data

def get_quadratic_data( df ):
    random_error_quad = np.random.normal(loc=0,scale=quad_sigma,size=len(df))
    return quad_a * (df.values)**2 + quad_b * (df.values) + quad_c + random_error_quad
#end get_quadratic_data
#----------------------------------------------------------------

if __name__ == "__main__":
    # classification:
    if (args.geometry == "y"):
        x_pts, y_pts, classification = do_cls_y()
    elif ( args.geometry == "squares" ):
        x_pts, y_pts, classification = do_cls_squares()
    elif ( args.geometry == "circles" ):
        x_pts, y_pts, classification = do_cls_circles()
    else:
        raise Exception("\n\n\t '{0}' not an option. \n\t Try 'squares', 'y', 'cirlces'.".format(args.geometry))

    num_unique_classes = int( len( np.unique(classification) ) )
    unique_classes = np.unique( classification )

    if args.verbose:
        print("Num unique classes: {0}".format(num_unique_classes) )
    if args.classes_w_regr > num_unique_classes:
        print("\t classes_w_regr > num_unique_classes \n\t setting classes_w_regr = num_unique_classes")
        args.classes_w_regr = num_unique_classes

    # make table
    main_df = pd.DataFrame()
    main_df["input_1"] = x_pts
    main_df["input_2"] = y_pts
    main_df["class"] = classification

    # regression
    if num_regr_outputs > 0 and args.classes_w_regr > 0:
        for i in range(num_regr_outputs):
            main_df["output_"+str(i+1)] = 'nan'
        main_df = do_regr_outputs(main_df)

    if args.verbose:
        print(main_df)

    # save df to file
    main_df.to_csv( path_or_buf = args.file_str, sep = ' ' )
    print( "SAVED DATA: {0}".format(args.file_str) )

    if args.save_info:
        with open( "info_" + args.file_str, "w+" ) as f:
            now = datetime.datetime.now()
            f.write( "File generated on: {0}\n".format( now.strftime("%d/%m/%Y %H:%M:%S") ) )
            f.write( "Initial Conditions: \n{0}\n".format(args) )
            f.write( "Other classification args: \n{0}\n{1}\n".format( other_cls_args_str, other_cls_args))
            f.write( "Other regression args: \n{0}\n{1}\n".format( other_rgr_args_str, other_rgr_args))
        print( "SAVED INFO FILE" )
    else:
        if args.verbose:
            print("Did not save info file...")
