SG_PATH = '/work/awilf/Standard-Grid'
import sys
sys.path.append(SG_PATH)
import standard_grid
import pickle

if __name__=="__main__":
    hash_in = sys.argv[1]
    grid=pickle.load(open(f'.{hash_in}.pkl',"rb"))
    csv_path=f"results/{hash_in}/csv_results.csv"
    grid.json_interpret("output/results.txt",csv_path)
