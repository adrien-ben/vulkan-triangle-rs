# Vulkan triangle with Rust

![Build badge](https://github.com/adrien-ben/vulkan-triangle-rs/workflows/Cross-platform%20build/badge.svg)

Rendering a simple triangle with Rust and Vulkan. There are two examples. The first using no specific Vulkan features or extentions. The other is using Vulkan 1.3's dynamic rendering feature can be found under `src/bin/dynamic_rendering.rs`.

We use Ash for Vulkan binding. The code is all contained in one file and with no abstraction.

## Run it on Desktop

```sh
# If you want to enable validation layers
export VK_LAYER_PATH=$VULKAN_SDK/Bin
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation

# If you changed the shader code (you'll need glslangValidator on you PATH)
./compile_shaders.sh

# Compile and start the application using cargo
RUST_LOG=info cargo run
# or
RUST_LOG=info cargo run --bin dynamic_rendering
```
