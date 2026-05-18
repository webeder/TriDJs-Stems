#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>

// Correções essenciais para MSVC no Windows
#ifdef _MSC_VER
#include <intrin.h> // Resolve '_addcarry_u64': identificador não encontrado
#endif
#define NOMINMAX    // Previne conflitos de macros min/max

// LibTorch Headers
#include <torch/script.h>
#include <torch/torch.h>

// dr_wav: Biblioteca leve para ler e gravar áudio WAV
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

// dr_mp3: Biblioteca leve para ler áudio MP3
#define DR_MP3_IMPLEMENTATION
#include "dr_mp3.h"

namespace fs = std::filesystem;

// A ordem padrão de saída do modelo HTDemucs
const std::vector<std::string> STEM_NAMES = {"drums", "bass", "other", "vocals"};

int main(int argc, char* argv[]) {
    // 1. Validação dos Argumentos de Linha de Comando
    if (argc < 3) {
        std::cerr << "Uso: " << argv[0] << " <arquivo_entrada.wav> <pasta_saida/>" << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_dir = argv[2];
    std::string model_path = "htdemucs_compilado.pt";

    if (!fs::exists(input_path)) {
        std::cerr << "Erro: Arquivo de entrada nao encontrado: " << input_path << std::endl;
        return 1;
    }

    if (!fs::exists(model_path)) {
        std::cerr << "Erro: Modelo '" << model_path << "' nao encontrado na mesma pasta do executavel." << std::endl;
        return 1;
    }

    if (!fs::exists(output_dir)) {
        fs::create_directories(output_dir);
    }

    try {
        // 2. Inicializar ambiente LibTorch e carregar o modelo de IA
        std::cout << "[1/4] Carregando modelo Demucs TorchScript..." << std::endl;
        
        // [CRÍTICO] Desativa o cálculo de gradientes. Sem isso, a RAM acumula lixo até o PC travar!
        torch::NoGradGuard no_grad;
        
        // [CRÍTICO] Limita as threads da CPU para evitar que o processador "engasgue" no Windows
        torch::set_num_threads(4);
        torch::set_num_interop_threads(4);

        torch::jit::script::Module model = torch::jit::load(model_path);
        
        torch::Device device(torch::kCPU);
        if (torch::cuda::is_available()) {
            std::cout << "[INFO] CUDA detectado. Acelerando por GPU." << std::endl;
            device = torch::Device(torch::kCUDA);
            model.to(device);
        } else {
            std::cout << "[INFO] CUDA nao detectado. Executando em CPU." << std::endl;
        }

        // 3. Decodificar o arquivo (.wav ou .mp3)
        std::cout << "[2/4] Lendo arquivo de audio..." << std::endl;
        unsigned int channels;
        unsigned int sample_rate;
        drwav_uint64 total_pcm_frame_count;
        float* sample_data = nullptr;

        std::string ext = fs::path(input_path).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        bool is_mp3 = (ext == ".mp3");

        if (is_mp3) {
            drmp3_config config;
            drmp3_uint64 mp3_frames;
            sample_data = drmp3_open_file_and_read_pcm_frames_f32(
                input_path.c_str(), &config, &mp3_frames, NULL);
            if (sample_data) {
                channels = config.channels;
                sample_rate = config.sampleRate;
                total_pcm_frame_count = mp3_frames;
            }
        } else {
            // Assume WAV
            sample_data = drwav_open_file_and_read_pcm_frames_f32(
                input_path.c_str(), &channels, &sample_rate, &total_pcm_frame_count, NULL);
        }

        if (sample_data == nullptr) {
            std::cerr << "Erro fatal: falha ao decodificar arquivo WAV/MP3." << std::endl;
            return 1;
        }

        // Converter para Tensor
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor audio_tensor = torch::from_blob(
            sample_data, 
            {static_cast<long>(total_pcm_frame_count), static_cast<long>(channels)}, 
            options
        ).clone();
        
        if (is_mp3) {
            drmp3_free(sample_data, nullptr);
        } else {
            drwav_free(sample_data, nullptr);
        }

        // Formato [Batch, Canais, Frames]
        audio_tensor = audio_tensor.transpose(0, 1);
        if (channels == 1) {
            audio_tensor = audio_tensor.repeat({2, 1}); // Força estéreo
        }
        audio_tensor = audio_tensor.unsqueeze(0).to(device);

        // 4. Executar a Inferência (Processamento da IA em Chunks)
        std::cout << "[3/4] Separando stems via inferencia (Aguarde)..." << std::endl;
        
        // HTDemucs é restrito a chunks exatos de 343980 samples (7.8 segundos a 44100Hz)
        const int chunk_size = 343980;
        int total_frames = audio_tensor.size(2); // Dimensão 2 é Frames
        
        // Criar tensor de saída final para a música inteira (4 stems, 2 canais, total_frames)
        torch::Tensor output_tensor = torch::zeros({4, 2, total_frames}, torch::kFloat32).to(device);
        
        for (int start = 0; start < total_frames; start += chunk_size) {
            int end = std::min(start + chunk_size, total_frames);
            int current_chunk_size = end - start;
            
            // Recorta o pedaço atual (2 canais, tamanho atual)
            torch::Tensor chunk = audio_tensor.slice(2, start, end);
            
            // Se for o último pedaço e for menor que o limite, fazemos padding com zeros no final
            if (current_chunk_size < chunk_size) {
                chunk = torch::nn::functional::pad(chunk, torch::nn::functional::PadFuncOptions({0, chunk_size - current_chunk_size}).mode(torch::kConstant).value(0.0));
            }
            
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(chunk);
            
            // Roda a IA no chunk (vai retornar [1, 4, 2, 343980])
            torch::Tensor chunk_out = model.forward(inputs).toTensor();
            chunk_out = chunk_out.squeeze(0); // [4, 2, 343980]
            
            // Salva o pedaço processado no tensor de saída original
            output_tensor.slice(2, start, end) = chunk_out.slice(2, 0, current_chunk_size);
            
            std::cout << "      Progresso: " << (end * 100 / total_frames) << "% concluido...\r" << std::flush;
        }
        std::cout << std::endl;
        
        // Traz o áudio finalizado de volta para a Memória RAM
        output_tensor = output_tensor.cpu();

        // 5. Gravar os 4 arquivos independentes
        std::cout << "[4/4] Gravando stems isolados em: " << output_dir << std::endl;
        std::string base_name = fs::path(input_path).stem().string();

        for (int i = 0; i < 4; ++i) {
            torch::Tensor stem_tensor = output_tensor[i].transpose(0, 1).contiguous();
            
            std::string out_file = base_name + "_" + STEM_NAMES[i] + ".wav";
            std::string full_out_path = (fs::path(output_dir) / out_file).string();

            drwav_data_format format;
            format.container = drwav_container_riff;
            format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
            format.channels = 2;
            format.sampleRate = sample_rate;
            format.bitsPerSample = 32;

            drwav wav;
            if (drwav_init_file_write(&wav, full_out_path.c_str(), &format, NULL)) {
                drwav_write_pcm_frames(&wav, stem_tensor.size(0), stem_tensor.data_ptr<float>());
                drwav_uninit(&wav);
                std::cout << "    -> Salvo: " << out_file << std::endl;
            } else {
                std::cerr << "    [ERRO] Falha ao gravar: " << out_file << std::endl;
            }
        }

        std::cout << "Processo concluido com sucesso!" << std::endl;

    } catch (const c10::Error& e) {
        std::cerr << "Erro fatal na LibTorch: " << e.msg() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Erro inesperado: " << e.what() << std::endl;
        return 1;
    }

    return 0; // Código 0 indica sucesso ao seu software mestre
}
