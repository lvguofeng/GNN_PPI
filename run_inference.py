import os


def run_func(description, ppi_path, pseq_path, vec_path, gnn_model, result_path):
    os.system("python interface.py \
            --description={} \
            --ppi_path={} \
            --pseq_path={} \
            --vec_path={} \
            --gnn_model={} \
            --result_path={} \
            ".format(description, ppi_path, pseq_path, vec_path, gnn_model, result_path))

if __name__ == "__main__":
    description = "test"

    ppi_path = "./data/9606.protein.actions.all_connected.txt"
    pseq_path = "./data/protein.STRING_all_connected.sequences.dictionary.tsv"
    vec_path = "./data/vec5_CTC.txt"

    gnn_model = "./save_model/gnn_string_bfs/gnn_model_train.ckpt"

    result_path = "./result_save.csv"  
    # test test

    run_func(description, ppi_path, pseq_path, vec_path, gnn_model, result_path)