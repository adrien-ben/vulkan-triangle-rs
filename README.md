# Vulkan triangle with Rust

Rendering a simple triangle with Rust and Vulkan.

We use Ash for Vulkan binding. The code is all contained in one file and with no abstraction.

# Run it 

```sh
# If you want to enable validation layers
export VK_LAYER_PATH=$VULKAN_SDK/Bin
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation

# If you changed the shader code (you'll need glslangValidator on you PATH)
./compile_shaders.sh

# Compile and start the application using cargo
RUST_LOG=info cargo run
```
