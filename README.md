# Automatic Number Plate Recognition (ANPR)

Esse repositório contém um projeto de reconhecimento automático de placas veiculares, desenvolvido na disciplina "Processamento de Imagens" do curso de Ciência da Computação da UFRPE (2022.2).

## Descrição

- **Tema**: Detecção de Placas de Carro
- **Objetivo do Projeto**: reconhecimento automático de placas veiculares, sendo capaz de determinar a presença/ausência de uma placa e seu conteúdo;
- **Algoritmos de Pré-Processamento**: 
  - *Redução de Ruído* (Non-Local Means, Filtros passa-baixa);
  - *Ajuste de Brilho e Contraste* (Equalização de Histograma, Mudança em Brilho, Mudança em Contraste); 
  - *Interpolação* (Vizinho Mais Próxima, Bilinear, Bicúbica, Lanczos); =
  - *Conversão para Escala de Cinza*;
- **Algoritmos de Segmentação/Detecção de Borda**: 
  - *Segmentação* (Watershed, K-Means, Baseados em Região);
  - *Binarização* (Otsu, Limiar Global, Kittler, Redes Neurais Generativas);
  - *Detecção de Borda* (Canny, LOG, Difference of Gaussian, Redes Neurais Generativas);

