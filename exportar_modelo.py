import torch
from demucs.pretrained import get_model
import warnings

def exportar_modelo():
    # Ignorar warnings do tracer no terminal para não poluir
    warnings.filterwarnings("ignore")
    
    print("[1/4] Baixando e carregando o modelo oficial HTDemucs v4...")
    # 'htdemucs' retorna um "BagOfModels" (Uma cápsula com vários modelos ou 1 só).
    # O TorchScript exige a rede neural pura. Então extraímos o ".models[0]":
    bag_of_models = get_model('htdemucs')
    model = bag_of_models.models[0]
    
    # É estritamente necessário colocar o modelo em modo de avaliação
    model.eval()

    print("[2/4] Criando tensor de exemplo (dummy input)...")
    # O HTDemucs usa um bloco de Transformer (training_length) restrito exatamente a 343980 samples (7.8s)
    # Precisamos compilar o TorchScript com EXATAMENTE esse tamanho para evitar crashes na inferência.
    chunk_size = 343980
    dummy_input = torch.rand(1, 2, chunk_size)

    print("[3/4] Executando o tracing (Compilação TorchScript)...")
    print("      (Aguarde, este processo mapeia toda a rede neural e pode demorar alguns minutos)")
    
    with torch.no_grad():
        # strict=False e check_trace=False são obrigatórios porque o Demucs tem IFs
        # e condicionais dinâmicas (como cálculo de padding) que confundem o compilador padrão.
        traced_model = torch.jit.trace(model, dummy_input, strict=False, check_trace=False)

    print("[4/4] Salvando o arquivo binário...")
    output_filename = "htdemucs_compilado.pt"
    traced_model.save(output_filename)

    print(f"\n>>> SUCESSO! O modelo '{output_filename}' foi gerado e está pronto para o C++! <<<")

if __name__ == "__main__":
    exportar_modelo()
