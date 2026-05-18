cmake_minimum_required(VERSION 3.18)
project(SeparadorStems VERSION 1.0.0 LANGUAGES CXX)

# Exigir C++17 puro
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Gera o arquivo para o IntelliSense do VS Code parar de apontar falsos erros

# Apontar diretamente para a pasta da LibTorch extraída
set(CMAKE_PREFIX_PATH "C:/TridjsStems/libtorch")

# Busca e carrega a LibTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Definir o executável do projeto
add_executable(SeparadorStems main.cpp)

# Otimizações de Release na Linkagem e Includes
target_link_libraries(SeparadorStems PRIVATE "${TORCH_LIBRARIES}")
