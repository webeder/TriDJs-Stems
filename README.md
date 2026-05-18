# TriDJs Stems Separator Engine

Motor ultra-rápido de separação de Stems (Voz, Bateria, Baixo e Outros) em C++17 puro, projetado como add-on em background para software de DJ (TriDJs). Utiliza a biblioteca oficial da Meta **LibTorch** (PyTorch C++) com inferência sob o modelo **HTDemucs v4 (Hybrid Transformer)**.

## 🚀 Funcionalidades

- **Nativo e Standalone:** Sem dependências externas de Python no runtime. Rodado diretamente via linha de comando.
- **Suporte a MP3 e WAV:** Lê arquivos de entrada comprimidos ou PCM sem compressão de forma nativa.
- **Saída de Alta Qualidade:** Exporta 4 stems independentes em formato WAV PCM 32-bit de altíssima fidelidade.
- **Otimizado para CPU e Memória:**
  - Desativação completa de gradientes (`torch::NoGradGuard`) para evitar vazamentos de memória (OOM).
  - Controle exato de threads (`torch::set_num_threads`) para máxima estabilidade e performance no Windows.
  - Processamento em fatias temporais de **7.8 segundos** (`343980` samples), respeitando o limite interno do bloco Transformer do Demucs.

---

## 🛠️ Pré-requisitos para Desenvolvimento

1. **Compilador MSVC x64:** Requer Visual Studio 2019/2022 com suporte a C++ e CMake.
2. **LibTorch (C++):** Baixe a versão estável da [PyTorch](https://pytorch.org/) (CPU ou CUDA) e extraia no diretório `C:\TridjsStems\libtorch`.
3. **Python (Apenas para exportação do modelo):** Python 3.8+ com os pacotes `torch`, `torchaudio` e `demucs`.

---

## 📦 Como Compilar

Recomendamos utilizar o **x64 Native Tools Command Prompt** do Visual Studio para garantir compatibilidade total de 64-bits:

```cmd
# 1. Configurar o projeto com o CMake
mkdir build
cd build
cmake ..

# 2. Compilar em modo de alta performance (Release)
cmake --build . --config Release
```

---

## 🧠 Como Exportar o Modelo de IA

O executável espera encontrar o arquivo compilado da rede neural (`htdemucs_compilado.pt`) na mesma pasta do executável. Para gerar esse arquivo a partir do modelo pré-treinado do Facebook/Meta:

```bash
# Instalar dependências de exportação
pip install torch torchaudio demucs

# Executar script de tracing (TorchScript)
python exportar_modelo.py
```

O script irá baixar o modelo **HTDemucs v4** oficial e compilá-lo para ser lido nativamente pelo nosso binário C++.

---

## ⚡ Como Rodar o Separador

Após compilar e gerar o modelo:
1. Copie o `htdemucs_compilado.pt` para `build/Release/`.
2. Copie as DLLs de `libtorch/lib/` (`c10.dll`, `torch.dll`, `torch_cpu.dll`, `uv.dll`, etc.) para a pasta `build/Release/`.
3. Rode o executável:

```cmd
SeparadorStems.exe "C:\Caminho\Música.mp3" "C:\Pasta_De_Saida"
```

O programa exibirá uma barra de progresso no terminal e salvará 4 arquivos na pasta de destino:
- `Música_vocals.wav` (Voz)
- `Música_drums.wav` (Bateria)
- `Música_bass.wav` (Baixo)
- `Música_other.wav` (Melodia/Outros)

---

## 📄 Licença

Este projeto é desenvolvido para uso interno integrado à suíte de DJ **TriDJs**.
