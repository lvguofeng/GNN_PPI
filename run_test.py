import os


def run_func(description, ppi_path, pseq_path, vec_path,
            index_path, gnn_model, test_all):
    os.system("python gnn_test.py \
            --description={} \
            --ppi_path={} \
            --pseq_path={} \
            --vec_path={} \
            --index_path={} \
            --gnn_model={} \
            --test_all={} \
            ".format(description, ppi_path, pseq_path, vec_path, 
                    index_path, gnn_model, test_all))

if __name__ == "__main__":
    description = "test"

    ppi_path = "./data/9606.protein.actions.all_connected.txt"
    pseq_path = "./data/protein.STRING_all_connected.sequences.dictionary.tsv"
    vec_path = "./data/vec5_CTC.txt"

    index_path = "./train_valid_index_json/string.bfs.fold1.json"
    gnn_model = "./save_model/gnn_string_bfs/gnn_model_train.ckpt"

    test_all = "False"

    # test test

    run_func(description, ppi_path, pseq_path, vec_path, index_path, gnn_model, test_all)
