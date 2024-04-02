use ash::{
    ext::debug_utils,
    khr::{surface, swapchain},
    vk, Device, Entry, Instance,
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use simple_logger::SimpleLogger;
use std::{
    error::Error,
    ffi::{CStr, CString},
};
use vk_triangle_rs::{
    create_vulkan_instance, create_vulkan_swapchain, create_window, read_shader_from_bytes,
};
use winit::{
    event::{Event, WindowEvent},
    window::Window,
};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;
const APP_NAME: &str = "Triangle (dynamic_rendering)";

fn main() -> Result<(), Box<dyn Error>> {
    SimpleLogger::default().env().init()?;

    let (window, event_loop) = create_window(APP_NAME, WIDTH, HEIGHT)?;
    let mut app = Triangle::new(&window)?;
    let mut is_swapchain_dirty = false;

    event_loop.run(move |event, elwt| {
        match event {
            // On resize
            Event::WindowEvent {
                event: WindowEvent::Resized(..),
                ..
            } => {
                log::debug!("Window has been resized");
                is_swapchain_dirty = true;
            }
            // Draw
            Event::AboutToWait => {
                if is_swapchain_dirty {
                    let dim = window.inner_size();
                    if dim.width > 0 && dim.height > 0 {
                        app.recreate_swapchain().expect("Failed to recreate swap");
                    } else {
                        return;
                    }
                }

                is_swapchain_dirty = app.draw().expect("Failed to tick");
            }
            // Exit app on request to close window
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => elwt.exit(),
            // Wait for gpu to finish pending work before closing app
            Event::LoopExiting => app
                .wait_for_gpu()
                .expect("Failed to wait for gpu to finish work"),
            _ => (),
        }
    })?;

    Ok(())
}

struct Triangle {
    _entry: Entry,
    instance: Instance,
    debug_utils: debug_utils::Instance,
    debug_utils_messenger: vk::DebugUtilsMessengerEXT,
    surface: surface::Instance,
    surface_khr: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    graphics_q_index: u32,
    present_q_index: u32,
    device: Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    command_pool: vk::CommandPool,
    swapchain: swapchain::Device,
    swapchain_khr: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    fence: vk::Fence,
}

impl Triangle {
    fn new(window: &Window) -> Result<Self, Box<dyn Error>> {
        log::info!("Create application");

        // Vulkan instance
        let entry = unsafe { Entry::load()? };
        let (instance, debug_utils, debug_utils_messenger) =
            create_vulkan_instance(APP_NAME, &entry, window)?;

        // Vulkan surface
        let surface = surface::Instance::new(&entry, &instance);
        let surface_khr = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None,
            )?
        };

        // Vulkan physical device and queue families indices (graphics and present)
        let (physical_device, graphics_q_index, present_q_index) =
            create_vulkan_physical_device_and_get_graphics_and_present_qs_indices(
                &instance,
                &surface,
                surface_khr,
            )?;

        // Vulkan logical device and queues
        let (device, graphics_queue, present_queue) =
            create_vulkan_device_and_graphics_and_present_qs(
                &instance,
                physical_device,
                graphics_q_index,
                present_q_index,
            )?;

        // Command pool
        let command_pool = {
            let command_pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(graphics_q_index)
                .flags(vk::CommandPoolCreateFlags::empty());
            unsafe { device.create_command_pool(&command_pool_info, None)? }
        };

        // Swapchain
        let (
            swapchain,
            swapchain_khr,
            swapchain_extent,
            swapchain_format,
            swapchain_images,
            swapchain_image_views,
        ) = create_vulkan_swapchain(
            WIDTH,
            HEIGHT,
            &instance,
            &surface,
            surface_khr,
            physical_device,
            graphics_q_index,
            present_q_index,
            &device,
        )?;

        // Pipeline and layout
        let (pipeline, pipeline_layout) =
            create_vulkan_pipeline(&device, swapchain_extent, swapchain_format)?;

        // Create and record command buffers (one per swapchain frame)
        let command_buffers = create_and_record_command_buffers(
            &device,
            command_pool,
            swapchain_images.len(),
            &swapchain_images,
            &swapchain_image_views,
            pipeline,
            swapchain_extent,
        )?;

        // Semaphore use for presentation
        let image_available_semaphore = {
            let semaphore_info = vk::SemaphoreCreateInfo::default();
            unsafe { device.create_semaphore(&semaphore_info, None)? }
        };
        let render_finished_semaphore = {
            let semaphore_info = vk::SemaphoreCreateInfo::default();
            unsafe { device.create_semaphore(&semaphore_info, None)? }
        };
        let fence = {
            let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
            unsafe { device.create_fence(&fence_info, None)? }
        };

        Ok(Self {
            _entry: entry,
            instance,
            debug_utils,
            debug_utils_messenger,
            surface,
            surface_khr,
            physical_device,
            graphics_q_index,
            present_q_index,
            device,
            graphics_queue,
            present_queue,
            command_pool,
            swapchain,
            swapchain_khr,
            swapchain_images,
            swapchain_image_views,
            pipeline,
            pipeline_layout,
            command_buffers,
            image_available_semaphore,
            render_finished_semaphore,
            fence,
        })
    }

    fn recreate_swapchain(&mut self) -> Result<(), Box<dyn Error>> {
        log::debug!("Recreating the swapchain");

        self.wait_for_gpu()?;

        unsafe { self.cleanup_swapchain() };

        // Swapchain
        let (
            swapchain,
            swapchain_khr,
            swapchain_extent,
            swapchain_format,
            swapchain_images,
            swapchain_image_views,
        ) = create_vulkan_swapchain(
            WIDTH,
            HEIGHT,
            &self.instance,
            &self.surface,
            self.surface_khr,
            self.physical_device,
            self.graphics_q_index,
            self.present_q_index,
            &self.device,
        )?;

        // Pipeline and layout
        let (pipeline, pipeline_layout) =
            create_vulkan_pipeline(&self.device, swapchain_extent, swapchain_format)?;

        // Create and record command buffers (one per swapchain frame)
        let command_buffers = create_and_record_command_buffers(
            &self.device,
            self.command_pool,
            swapchain_images.len(),
            &swapchain_images,
            &swapchain_image_views,
            pipeline,
            swapchain_extent,
        )?;

        self.swapchain = swapchain;
        self.swapchain_khr = swapchain_khr;
        self.swapchain_images = swapchain_images;
        self.swapchain_image_views = swapchain_image_views;
        self.pipeline = pipeline;
        self.pipeline_layout = pipeline_layout;
        self.command_buffers = command_buffers;

        Ok(())
    }

    unsafe fn cleanup_swapchain(&mut self) {
        self.device
            .free_command_buffers(self.command_pool, &self.command_buffers);
        self.command_buffers.clear();
        self.device.destroy_pipeline(self.pipeline, None);
        self.device
            .destroy_pipeline_layout(self.pipeline_layout, None);
        self.swapchain_image_views
            .iter()
            .for_each(|v| self.device.destroy_image_view(*v, None));
        self.swapchain_image_views.clear();
        self.swapchain.destroy_swapchain(self.swapchain_khr, None);
    }

    fn draw(&mut self) -> Result<bool, Box<dyn Error>> {
        let fence = self.fence;
        unsafe { self.device.wait_for_fences(&[fence], true, std::u64::MAX)? };

        // Drawing the frame
        let next_image_result = unsafe {
            self.swapchain.acquire_next_image(
                self.swapchain_khr,
                std::u64::MAX,
                self.image_available_semaphore,
                vk::Fence::null(),
            )
        };
        let image_index = match next_image_result {
            Ok((image_index, _)) => image_index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                return Ok(true);
            }
            Err(error) => panic!("Error while acquiring next image. Cause: {}", error),
        };

        unsafe { self.device.reset_fences(&[fence])? };

        let wait_semaphore_submit_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(self.image_available_semaphore)
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);

        let signal_semaphore_submit_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(self.render_finished_semaphore)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);

        let cmd_buffer_submit_info = vk::CommandBufferSubmitInfo::default()
            .command_buffer(self.command_buffers[image_index as usize]);

        let submit_info = vk::SubmitInfo2::default()
            .wait_semaphore_infos(std::slice::from_ref(&wait_semaphore_submit_info))
            .signal_semaphore_infos(std::slice::from_ref(&signal_semaphore_submit_info))
            .command_buffer_infos(std::slice::from_ref(&cmd_buffer_submit_info));

        unsafe {
            self.device.queue_submit2(
                self.graphics_queue,
                std::slice::from_ref(&submit_info),
                fence,
            )?
        };

        let signal_semaphores = [self.render_finished_semaphore];
        let swapchains = [self.swapchain_khr];
        let images_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&images_indices);

        let present_result = unsafe {
            self.swapchain
                .queue_present(self.present_queue, &present_info)
        };
        match present_result {
            Ok(is_suboptimal) if is_suboptimal => {
                return Ok(true);
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                return Ok(true);
            }
            Err(error) => panic!("Failed to present queue. Cause: {}", error),
            _ => {}
        }
        Ok(false)
    }

    pub fn wait_for_gpu(&self) -> Result<(), Box<dyn Error>> {
        unsafe { self.device.device_wait_idle()? };
        Ok(())
    }
}

impl Drop for Triangle {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_fence(self.fence, None);
            self.device
                .destroy_semaphore(self.image_available_semaphore, None);
            self.device
                .destroy_semaphore(self.render_finished_semaphore, None);
            self.cleanup_swapchain();
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.surface.destroy_surface(self.surface_khr, None);
            self.debug_utils
                .destroy_debug_utils_messenger(self.debug_utils_messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}

fn create_vulkan_physical_device_and_get_graphics_and_present_qs_indices(
    instance: &Instance,
    surface: &surface::Instance,
    surface_khr: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32, u32), Box<dyn Error>> {
    log::debug!("Creating vulkan physical device");
    // Get the list of physical devices by prioritizing discrete then integrated gpu
    let devices = {
        let mut devices = unsafe { instance.enumerate_physical_devices()? };
        devices.sort_by_key(|device| {
            let props = unsafe { instance.get_physical_device_properties(*device) };
            match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                _ => 2,
            }
        });
        devices
    };

    let mut graphics = None;
    let mut present = None;
    let device = devices
        .into_iter()
        .find(|device| {
            let device = *device;

            // Does device supports graphics and present queues
            let props = unsafe { instance.get_physical_device_queue_family_properties(device) };
            for (index, family) in props.iter().filter(|f| f.queue_count > 0).enumerate() {
                let index = index as u32;
                graphics = None;
                present = None;

                if family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    && family.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    && graphics.is_none()
                {
                    graphics = Some(index);
                }

                let present_support = unsafe {
                    surface
                        .get_physical_device_surface_support(device, index, surface_khr)
                        .expect("Failed to get device surface support")
                };
                if present_support && present.is_none() {
                    present = Some(index);
                }

                if graphics.is_some() && present.is_some() {
                    break;
                }
            }

            // Does device support desired extensions
            let extension_props = unsafe {
                instance
                    .enumerate_device_extension_properties(device)
                    .expect("Failed to get device ext properties")
            };
            let extention_support = extension_props.iter().any(|ext| {
                let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
                swapchain::NAME == name
            });

            // Does the device have available formats for the given surface
            let formats = unsafe {
                surface
                    .get_physical_device_surface_formats(device, surface_khr)
                    .expect("Failed to get physical device surface formats")
            };

            // Does the device have available present modes for the given surface
            let present_modes = unsafe {
                surface
                    .get_physical_device_surface_present_modes(device, surface_khr)
                    .expect("Failed to get physical device surface present modes")
            };

            // Check 1.3 features
            let mut features13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut features13);
            unsafe { instance.get_physical_device_features2(device, &mut features) };

            graphics.is_some()
                && present.is_some()
                && extention_support
                && !formats.is_empty()
                && !present_modes.is_empty()
                && features13.dynamic_rendering == vk::TRUE
                && features13.synchronization2 == vk::TRUE
        })
        .expect("Could not find a suitable device");

    unsafe {
        let props = instance.get_physical_device_properties(device);
        let device_name = CStr::from_ptr(props.device_name.as_ptr());
        log::info!("Selected physical device: {:?}", device_name);
    }

    Ok((device, graphics.unwrap(), present.unwrap()))
}

fn create_vulkan_device_and_graphics_and_present_qs(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    graphics_q_index: u32,
    present_q_index: u32,
) -> Result<(Device, vk::Queue, vk::Queue), Box<dyn Error>> {
    log::debug!("Creating vulkan device and graphics and present queues");
    let queue_priorities = [1.0f32];
    let queue_create_infos = {
        let mut indices = vec![graphics_q_index, present_q_index];
        indices.dedup();

        indices
            .iter()
            .map(|index| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(*index)
                    .queue_priorities(&queue_priorities)
            })
            .collect::<Vec<_>>()
    };

    let device_extensions_ptrs = [
        swapchain::NAME.as_ptr(),
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        ash::khr::portability_subset::NAME,
    ];

    let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
        .dynamic_rendering(true)
        .synchronization2(true);
    let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut features13);

    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(&device_extensions_ptrs)
        .push_next(&mut features);

    let device = unsafe { instance.create_device(physical_device, &device_create_info, None)? };
    let graphics_queue = unsafe { device.get_device_queue(graphics_q_index, 0) };
    let present_queue = unsafe { device.get_device_queue(present_q_index, 0) };

    Ok((device, graphics_queue, present_queue))
}

fn create_vulkan_pipeline(
    device: &Device,
    extent: vk::Extent2D,
    format: vk::Format,
) -> Result<(vk::Pipeline, vk::PipelineLayout), Box<dyn Error>> {
    log::debug!("Creating vulkan pipeline");
    let layout_info = vk::PipelineLayoutCreateInfo::default();
    let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

    let entry_point_name = CString::new("main")?;

    let vertex_source =
        read_shader_from_bytes(&include_bytes!("../../assets/shaders/shader.vert.spv")[..])?;
    let vertex_create_info = vk::ShaderModuleCreateInfo::default().code(&vertex_source);
    let vertex_module = unsafe { device.create_shader_module(&vertex_create_info, None)? };

    let fragment_source =
        read_shader_from_bytes(&include_bytes!("../../assets/shaders/shader.frag.spv")[..])?;
    let fragment_create_info = vk::ShaderModuleCreateInfo::default().code(&fragment_source);
    let fragment_module = unsafe { device.create_shader_module(&fragment_create_info, None)? };

    let shader_states_infos = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_module)
            .name(&entry_point_name),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_module)
            .name(&entry_point_name),
    ];

    let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();

    let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    let viewports = [vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: extent.width as _,
        height: extent.height as _,
        min_depth: 0.0,
        max_depth: 1.0,
    }];
    let scissors = [vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent,
    }];
    let viewport_info = vk::PipelineViewportStateCreateInfo::default()
        .viewports(&viewports)
        .scissors(&scissors);

    let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false)
        .depth_bias_constant_factor(0.0)
        .depth_bias_clamp(0.0)
        .depth_bias_slope_factor(0.0);

    let multisampling_info = vk::PipelineMultisampleStateCreateInfo::default()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
        .min_sample_shading(1.0)
        .alpha_to_coverage_enable(false)
        .alpha_to_one_enable(false);

    let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(false)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ZERO)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD)];
    let color_blending_info = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(&color_blend_attachments)
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    let color_attachment_formats = [format];
    let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&color_attachment_formats);

    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_states_infos)
        .vertex_input_state(&vertex_input_info)
        .input_assembly_state(&input_assembly_info)
        .viewport_state(&viewport_info)
        .rasterization_state(&rasterizer_info)
        .multisample_state(&multisampling_info)
        .color_blend_state(&color_blending_info)
        .layout(pipeline_layout)
        .push_next(&mut rendering_info);

    let pipeline = unsafe {
        device
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&pipeline_info),
                None,
            )
            .map_err(|e| e.1)?[0]
    };

    unsafe {
        device.destroy_shader_module(vertex_module, None);
        device.destroy_shader_module(fragment_module, None);
    }

    Ok((pipeline, pipeline_layout))
}

fn create_and_record_command_buffers(
    device: &Device,
    pool: vk::CommandPool,
    count: usize,
    images: &[vk::Image],
    image_views: &[vk::ImageView],
    pipeline: vk::Pipeline,
    extent: vk::Extent2D,
) -> Result<Vec<vk::CommandBuffer>, Box<dyn Error>> {
    log::debug!("Creating and recording command buffers");
    let buffers = {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(count as _);

        unsafe { device.allocate_command_buffers(&allocate_info)? }
    };

    for (index, buffer) in buffers.iter().enumerate() {
        let buffer = *buffer;

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default();
        unsafe { device.begin_command_buffer(buffer, &command_buffer_begin_info)? };

        let image_memory_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_READ)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .new_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
            .image(images[index])
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                layer_count: 1,
                level_count: 1,
                ..Default::default()
            });

        let dependency_info = vk::DependencyInfo::default()
            .image_memory_barriers(std::slice::from_ref(&image_memory_barrier));

        unsafe { device.cmd_pipeline_barrier2(buffer, &dependency_info) };

        let color_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(image_views[index])
            .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [1.0, 1.0, 1.0, 1.0],
                },
            });

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&color_attachment_info));

        unsafe { device.cmd_begin_rendering(buffer, &rendering_info) };

        unsafe { device.cmd_bind_pipeline(buffer, vk::PipelineBindPoint::GRAPHICS, pipeline) };

        unsafe { device.cmd_draw(buffer, 3, 1, 0, 0) };

        unsafe { device.cmd_end_rendering(buffer) };

        let image_memory_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::empty())
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .image(images[index])
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                layer_count: 1,
                level_count: 1,
                ..Default::default()
            });

        let dependency_info = vk::DependencyInfo::default()
            .image_memory_barriers(std::slice::from_ref(&image_memory_barrier));

        unsafe { device.cmd_pipeline_barrier2(buffer, &dependency_info) };

        unsafe { device.end_command_buffer(buffer)? };
    }

    Ok(buffers)
}
