import numpy as np

def eval_wmape(y_true, y_pred):

    #Função de avaliação customizada de WMAPE para LightGBM.

    sum_of_absolute_errors = np.sum(np.abs(y_true - y_pred))
    sum_of_actuals = np.sum(np.abs(y_true))
    
    # Prevenção de divisão por zero
    if sum_of_actuals == 0:
        wmape = 0.0
    else:
        wmape = sum_of_absolute_errors / sum_of_actuals
    
    # O LightGBM espera este formato: (nome, valor, se_maior_melhor)
    return 'wmape', wmape, False # False porque um WMAPE menor é melhor