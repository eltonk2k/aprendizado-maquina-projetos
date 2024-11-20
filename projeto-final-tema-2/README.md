# Relatório de Classificação - Modelo MLP para Sobrevivência do Titanic

Este relatório descreve os resultados obtidos na aplicação de um modelo de rede neural do tipo MLP (Multi-Layer Perceptron) para a tarefa de classificação da sobrevivência de passageiros do Titanic.

## Descrição Geral da Solução

O modelo foi treinado para classificar os passageiros em duas classes:
- **Classe 0**: Não Sobreviveu
- **Classe 1**: Sobreviveu

A avaliação do desempenho do modelo foi realizada utilizando métricas padrão de classificação, como precision, recall, e F1-score, tanto para cada classe individualmente quanto para o desempenho geral.

## Principais Resultados Obtidos

### Matriz de Confusão

| Predição/Real         | Sobreviveu (1) | Não Sobreviveu (0) |
|-----------------------|----------------|--------------------|
| **Sobreviveu (1)**     | 74             | 37                 |
| **Não Sobreviveu (0)** | 13             | 144                |

### Interpretação:

- **Verdadeiros Positivos (74)**:  
  O modelo classificou os 74 passageiros que sobreviveram.
  
- **Falsos Negativos (37)**:  
  Estes são passageiros que sobreviveram, mas foram classificados como não sobreviventes.
  
- **Falsos Positivos (13)**:  
  Passageiros que não sobreviveram, mas foram classificados como sobreviventes.
  
- **Verdadeiros Negativos (144)**:  
  O modelo classificou 144 passageiros que não sobreviveram.


### Classe 0 (Não Sobreviveu)
- **Precisão**: 80%  
  O modelo acertou 80% das predições de "não sobreviveu".
  
- **Recall**: 92%  
  O modelo identificou que 92% dos passageiros que realmente não sobreviveram.
  
- **F1-Score**: 85%  
  A combinação entre precisão e recall está bem equilibrada para esta classe.

### Classe 1 (Sobreviveu)
- **Precisão**: 85%  
  O modelo acertou 85% das predições de "sobreviveu".
  
- **Recall**: 67%  
  O modelo identificou que 67% dos passageiros que realmente sobreviveram.
  
- **F1-Score**: 75%  
  Apesar de a precisão seja alta, o recall mais baixo sugere que o modelo tem dificuldades em identificar todos os sobreviventes.

### Métricas Gerais
- **Acurácia Geral**: 81%  
  O modelo acertou 81% das predições totais.
  
- **Média Ponderada (Weighted Avg)**:
  - Precisão: 82%
  - Recall: 81%
  - F1-Score: 81%

## Tendência do Erro e Acurácia por Época

### Erro (Loss)
- Durante as primeiras épocas, tanto o erro de treinamento quanto o de validação diminuem, indicando que o modelo está aprendendo.
- Após aproximadamente 30 épocas, os valores de erro de validação estabilizam, enquanto o erro de treinamento continua caindo levemente. Isso sugere overfitting, ou seja, o modelo se ajusta excessivamente aos dados de treinamento, mas não melhora significativamente no conjunto de validação.

### Acurácia
- A acurácia de treinamento cresce gradualmente e estabiliza entre 83% e 85%.
- A acurácia de validação permanece entre 81% e 83%, demonstrando uma boa consistência entre os dados de treinamento e validação.

## Conclusão

- O modelo alcançou um desempenho estável com uma acurácia geral de 81%.
- O desempenho foi particularmente bom na identificação de passageiros que **não sobreviveram** (Classe 0), com um recall de 92%.
- No entanto, o modelo apresentou dificuldades em identificar todos os passageiros que **sobreviveram** (Classe 1), com recall de 67%.
- O modelo apresentou uma leve tendência ao **overfitting**, o que é esperado em redes neurais com um número moderado de parâmetros e um conjunto de dados relativamente pequeno, como o caso do Titanic.

