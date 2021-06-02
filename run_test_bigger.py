import os


def run_func(description, ppi_path, pseq_path, vec_path,
            index_path, gnn_model, bigger_ppi_path, bigger_pseq_path):
    os.system("python gnn_test_bigger.py \
            --description={} \
            --ppi_path={} \
            --pseq_path={} \
            --vec_path={} \
            --index_path={} \
            --gnn_model={} \
            --bigger_ppi_path={} \
            --bigger_pseq_path={} \
            ".format(description, ppi_path, pseq_path, vec_path, 
                    index_path, gnn_model, bigger_ppi_path, bigger_pseq_path))

if __name__ == "__main__":
    description = "test"

    ppi_path = "./data/protein.actions.SHS148k.STRING.txt"
    pseq_path = "./data/protein.SHS148k.sequences.dictionary.tsv"
    vec_path = "./data/vec5_CTC.txt"

    index_path = "./train_valid_index_json/shs148k.bfs.fold1.json"
    gnn_model = "./save_model/gnn_string_bfs/gnn_model_train.ckpt"

    bigger_ppi_path = "./data/9606.protein.actions.all_connected.txt"
    bigger_pseq_path = "./data/protein.STRING_all_connected.sequences.dictionary.tsv"


    run_func(description, ppi_path, pseq_path, vec_path, index_path, gnn_model, bigger_ppi_path, bigger_pseq_path)
