import pandas as pd
import argparse


def score(TP, FP, FN):
    prc = TP/(TP+FP)
    rcl = TP/(TP+FN)
    return prc, rcl, 2*(prc*rcl)/(prc+rcl) if prc+rcl != 0 else 0


def end(ans):
    global args
    # Calculate metrics
    tp = ans.count('tp')
    tn = ans.count('tn')
    fp = ans.count('fp')
    fn = ans.count('fn')
    prc,rcl,f1 = score(tp,fp,fn)
    # Save to the output
    if args.log_file is not None:
        f = open(args.log_file, "w")
    else:
        f = open(args.input_file, "a")
        f.write('\n')
    f.write(f'! TP: {tp}\n')
    f.write(f'! TN: {tn}\n')
    f.write(f'! FP: {fp}\n')
    f.write(f'! FN: {fn}\n')
    f.write('\n! Metrics:\n')
    f.write(f'! Precision: {prc:.3f}\n')
    f.write(f'! Recall   : {rcl:.3f}\n')
    f.write(f'! F1       : {f1:.3f}\n')
    f.close()
    # Kill
    exit()


def print_help():
    global solved
    solved = False
    print('\nType "tp"/"tn"/"fp"/"fn" to evaluate each row.')
    print('Type "done" to stop the program and save the results before the end of the file.')
    print('Type "correct N <tp/tn/fp/fn>" to correct the row N with the new answer. Example: >correct 1 tp')
    print('Type "." to ignore a row.')
    print('Type "exit" to kill the program without saving.')
    print('Type "help" to display this message.')


def handle_correction(inp, ans):
    global solved
    try:
        row_i = int(inp.split(' ')[1])
        label = (inp.split(' ')[2])
        handle_input(label, row_i, ans)
        print(f'Corrected row {row_i} to {label}')
        solved = False # To make the program not skip the row being evaluated
    except:
        raise ValueError(f'Invalid correction: {inp}. Please try again.')


def handle_input(inp, i, ans):
    if   inp in ['tp','tn','fp','fn']: 
        ans[i] = inp
    elif inp == 'done': 
        end(ans)
    elif inp == 'help':
        print_help()
    elif inp == 'exit':
        exit()
    elif inp.startswith('correct'):
        handle_correction(inp, ans)
    elif inp == '.':
        ans[i] = None
    else:
        raise ValueError(f'Invalid input: {inp}. Please try again.')


parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str)
parser.add_argument('--log_file', type=str, default=None, required=False, help='The path to the log file that will contain the evaluation metrics. If not provided, the metrics will be appended in the end of the input file.')
args = parser.parse_args()

data = pd.read_csv(args.input_file, sep='\t')

print_help()
ans = [None for _ in range(len(data))]
for i,row in data.iterrows():
    solved = False
    while not solved:
        try:
            solved = True
            print(f'\n{row}')
            handle_input(input('>'), i, ans)
        except ValueError as e:
            print(e)
            solved = False
            continue
end(ans)