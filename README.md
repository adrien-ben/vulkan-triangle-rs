# Vulkan triangle with Rust

Rendering a simple triangle with Rust and Vulkan.

We use Ash for Vulkan binding. The code is all contained in one file and with no abstraction.

## Build for Android with Docker

```sh
# Run the following command and the apk will be generated in 'target/android-artifacts/debug/apk'
docker run --rm -v <path-to-local-directory-with-Cargo.toml>:/root/src philipalldredge/cargo-apk cargo apk build
```

For more details or alternatives about the build see the [android_glue][1] project.

To make the app run on Android I had to [fork winit][0] to make it use the master branch of [android_glue][1].
The lastest cargo-apk Docker image is not compatible with the lastest release on crates.io.

You can also download the [released apk][2]

## Run it on Desktop

```sh
# If you want to enable validation layers
export VK_LAYER_PATH=$VULKAN_SDK/Bin
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation

# If you changed the shader code (you'll need glslangValidator on you PATH)
./compile_shaders.sh

# Compile and start the application using cargo
RUST_LOG=info cargo run
```

[0]: https://github.com/adrien-ben/winit/commit/c3e524f7deaa5dbc9a5a59ef0dd37980134cea53
[1]: https://github.com/rust-windowing/android-rs-glue
[2]: https://github.com/adrien-ben/vulkan-triangle-rs/releases
