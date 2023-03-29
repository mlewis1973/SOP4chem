# Matthew.Lewis@gmail.com
# c2023
import numpy as np
import pandas as pd
import argparse
import sys

def bounds_check(vertex, bounds): # (numpy.ndarray, pd.df) 
    upper = bounds.loc[ bounds['bound'] == 'U'].drop(columns='bound').to_numpy().flatten()
    lower = bounds.loc[ bounds['bound'] == 'L'].drop(columns='bound').to_numpy().flatten()
    
    inside = np.logical_and.reduce(vertex>lower) and np.logical_and.reduce(vertex<upper)
    return inside

def simplex_step(df,bounds,scale=1.,code=-9999.0):
    Np = len(df.columns)-2 # number of parameters
    Ns = Np + 1          # dim of simplex
    iter_no = len(df)    # df.tail(1)['iteration'].iloc[0]
    
    simplex = df.tail(Ns).sort_values('score')
    worst = simplex.head(1).to_numpy().flatten()[1:(Np+1)]
    centroid = simplex.tail(Ns-1).to_numpy()[:,1:(Np+1)].mean(axis=0)
    
    new_vertex = centroid + scale*(centroid - worst)
    i = 1
    if not isinstance(bounds,type(None)):
        while not bounds_check(new_vertex, bounds): # take a step back from boundary
            new_vertex = centroid + ( (1.0/3.0)**i  )*scale*(centroid - worst)
            i += 1
    
    df = pd.concat( [df[:-Ns] , simplex] )
    
    df.loc[iter_no] = [iter_no+1] + new_vertex.tolist()+[code]
    return df  

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]",
        description="Generate next experimental parameters via simplex optimization procedure. \n CSV file is updated each time. \n Note: iterations will become sorted due to simplex progression. https://pubs.acs.org/doi/10.1021/ed056p307"
    )
    parser.add_argument(
        "-v","--version", action="version",
        version = f"{parser.prog} version 1.0.0"
    )
    parser.add_argument('-bounds',action='store',default=None, help='Optional CSV file with bounds for experimental parameters. Format of column labels should be: bound, same experimental parameter labels as ordered in data CSV file. Row labels should be: U, L for upper and lower bounds respectively. Defaults to NoneType. ')
    parser.add_argument('file', nargs="?",help='CSV file containing history of experimental parameters.  Format of column labels should be: iteration, 2 or more experimental parameter labels, score. ') 
    return parser

parser = init_argparse()
args = parser.parse_args()

try:
    df = pd.read_csv(args.file)
except:
    print("ERROR: Can not read CSV data file.",file=sys.stderr)
    sys.exit(1)

bounds = None    
if not isinstance(args.bounds, type(None)):
    try:
        bounds = pd.read_csv(args.bounds)
        print(bounds)
    except:
        print("ERROR: Can not read CSV bounds file.",file=sys.stderr)
        sys.exit(1)

scale = 1.
code = -9999.0

last_score = df.at[ df.index[-1],'score']
if last_score < 0 :    # waiting for last experiment to be complete
    print(df)
    try:
        new_score = float(input('If you have performed last experiment in table above, please enter new positive score (negative value will exit program): '))
    except ValueError:
        print("ERROR: Not a valid score.", file=sys.stderr)
        sys.exit(1)

    if new_score < 0:
        print("Exiting without making any changes...\n")
        sys.exit(0)

    previous_scores = df.tail(len(df.columns)).head(len(df.columns) - 1)['score'].to_numpy()
    previous_scores = np.sort(previous_scores)
    if new_score > previous_scores[-1]: # expansion rule 1
        print('RULE 1')
        if last_score < code: # handle previous expansion properly
            scale = 1.0
            df.at[ df.index[-1],'score'] = new_score
        else: 
            scale = 2.0 
            code = -11111.0
            df = pd.concat( [ df.tail(1) , df.head( len(df)-1 ) ] ) 
            #df = df.head( len(df)-1 )
            df.at[ df.index[0],'score'] = new_score
            #df = simplex_step(df,bounds,scale)
    elif new_score >= previous_scores[-2]: # accept simplex rule 2
        print('RULE 2')
        scale = 1.0
        df.at[ df.index[-1],'score'] = new_score
        #df = simplex_step(df,bounds,scale)
    elif new_score > previous_scores[0]: # mini-expansion rule 3a
        print('RULE 3a')
        scale = 0.5
        df = pd.concat( [ df.tail(1) , df.head( len(df)-1 ) ] )
        #df = df.head( len(df)-1 )
        df.at[ df.index[0],'score'] = new_score
        #df = simplex_step(df,bounds,scale)
    else: # mini-contraction rule 3b
        print('RULE 3b')
        scale = -0.5
        #df = df.head( len(df)-1 )
        df = pd.concat( [ df.tail(1) , df.head( len(df)-1 ) ] )
        df.at[ df.index[0],'score'] = new_score
        #df = simplex_step(df,bounds,scale)

#    df.at[ df.index[-1],'score'] = new_score
#else: 
#    df = simplex_step(df,bounds)
df = simplex_step(df,bounds,scale,code)
print(df)
print('Please perform the experiment in the latest iteration above.\nThen re-run this program and enter new score.\n')
df.to_csv(args.file,index=False)




