def read_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            list_ = [[decimal for decimal in line.split(",")] for line in line.split("]")]
    return list_


def string_to_float(X_tst_mat):
    while [''] in X_tst_mat:
        X_tst_mat.remove([''])
    for x in range(0, len(X_tst_mat)):
        while '' in X_tst_mat[x]:
            X_tst_mat[x].remove('')
    X_tst = [[89 for i in range(40)] for j in range(len(X_tst_mat))]
    for i in range(0, len(X_tst_mat) - 1):
        if X_tst_mat[i] != ['']:
            for j in range(0, 40):
                if X_tst_mat[i][j][0:2] == '[[':
                    X_tst[i][j] = float(X_tst_mat[i][j][2:])
                elif j == 0 and X_tst_mat[i][j][0:1] == '[':
                    X_tst[i][j] = float(X_tst_mat[i][j][1:])
                elif X_tst_mat[i][j] != '':
                    X_tst[i][j] = float(X_tst_mat[i][j])
                else:
                    continue
        else:
            continue
    return (X_tst)
