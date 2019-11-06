use ash::{
    extensions::{
        ext::DebugReport,
        khr::{Surface, Swapchain},
    },
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
    vk, Device, Entry, Instance,
};
use std::{
    error::Error,
    ffi::{CStr, CString},
    os::raw::{c_char, c_void},
    path::Path,
};
use winit::{dpi::LogicalSize, Event, EventsLoop, Window, WindowBuilder, WindowEvent};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;
const APP_NAME: &str = "Triangle";

fn main() -> Result<(), Box<dyn Error>> {
    simple_logger::init()?;
    Triangle::new()?.run()?;
    Ok(())
}

struct Triangle {
    window: Window,
    events_loop: EventsLoop,
    _entry: Entry,
    instance: Instance,
    debug_report: DebugReport,
    debug_report_callback: vk::DebugReportCallbackEXT,
    surface: Surface,
    surface_khr: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    graphics_q_index: u32,
    present_q_index: u32,
    device: Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    command_pool: vk::CommandPool,
    swapchain: Swapchain,
    swapchain_khr: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
}

impl Triangle {
    fn new() -> Result<Self, Box<dyn Error>> {
        log::info!("Create application");
        // Setup window
        let (window, events_loop) = create_window();

        // Vulkan instance
        let entry = Entry::new()?;
        let (instance, debug_report, debug_report_callback) = create_vulkan_instance(&entry)?;

        // Vulkan surface
        let surface = Surface::new(&entry, &instance);
        let surface_khr = unsafe { surface::create_surface(&entry, &instance, &window)? };

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
            let command_pool_info = vk::CommandPoolCreateInfo::builder()
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
            &instance,
            &surface,
            surface_khr,
            physical_device,
            graphics_q_index,
            present_q_index,
            &device,
        )?;

        // Renderpass
        let render_pass = create_vulkan_render_pass(&device, swapchain_format)?;

        // Framebuffers
        let framebuffers = create_vulkan_framebuffers(
            &device,
            render_pass,
            swapchain_extent,
            &swapchain_image_views,
        )?;

        // Pipeline and layout
        let (pipeline, pipeline_layout) =
            create_vulkan_pipeline(&device, render_pass, swapchain_extent)?;

        // Create and record command buffers (one per swapchain frame)
        let command_buffers = create_and_record_command_buffers(
            &device,
            command_pool,
            swapchain_images.len(),
            &framebuffers,
            render_pass,
            pipeline,
            swapchain_extent,
        )?;

        // Semaphore use for presentation
        let image_available_semaphore = {
            let semaphore_info = vk::SemaphoreCreateInfo::builder();
            unsafe { device.create_semaphore(&semaphore_info, None)? }
        };
        let render_finished_semaphore = {
            let semaphore_info = vk::SemaphoreCreateInfo::builder();
            unsafe { device.create_semaphore(&semaphore_info, None)? }
        };

        Ok(Self {
            window,
            events_loop,
            _entry: entry,
            instance,
            debug_report,
            debug_report_callback,
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
            render_pass,
            framebuffers,
            pipeline,
            pipeline_layout,
            command_buffers,
            image_available_semaphore,
            render_finished_semaphore,
        })
    }

    fn recreate_swapchain(&mut self) -> Result<(), Box<dyn Error>> {
        log::debug!("Recreating the swapchain");
        // Wait for the window to be maximized before recreating the swapchain
        loop {
            if let Some(LogicalSize { width, height }) = self.window.get_inner_size() {
                if width > 0.0 && height > 0.0 {
                    break;
                }
            }
        }

        unsafe { self.device.device_wait_idle()? };

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
            &self.instance,
            &self.surface,
            self.surface_khr,
            self.physical_device,
            self.graphics_q_index,
            self.present_q_index,
            &self.device,
        )?;

        // Renderpass
        let render_pass = create_vulkan_render_pass(&self.device, swapchain_format)?;

        // Framebuffers
        let framebuffers = create_vulkan_framebuffers(
            &self.device,
            render_pass,
            swapchain_extent,
            &swapchain_image_views,
        )?;

        // Pipeline and layout
        let (pipeline, pipeline_layout) =
            create_vulkan_pipeline(&self.device, render_pass, swapchain_extent)?;

        // Create and record command buffers (one per swapchain frame)
        let command_buffers = create_and_record_command_buffers(
            &self.device,
            self.command_pool,
            swapchain_images.len(),
            &framebuffers,
            render_pass,
            pipeline,
            swapchain_extent,
        )?;

        self.swapchain = swapchain;
        self.swapchain_khr = swapchain_khr;
        self.swapchain_images = swapchain_images;
        self.swapchain_image_views = swapchain_image_views;
        self.render_pass = render_pass;
        self.framebuffers = framebuffers;
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
        self.framebuffers
            .iter()
            .for_each(|fb| self.device.destroy_framebuffer(*fb, None));
        self.framebuffers.clear();
        self.device.destroy_render_pass(self.render_pass, None);
        self.swapchain_image_views
            .iter()
            .for_each(|v| self.device.destroy_image_view(*v, None));
        self.swapchain_image_views.clear();
        self.swapchain.destroy_swapchain(self.swapchain_khr, None);
    }

    fn run(&mut self) -> Result<(), Box<dyn Error>> {
        log::info!("Starting application");
        // Main loop
        loop {
            // Processing events
            let mut should_stop = false;
            self.events_loop.poll_events(|event| {
                if let Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } = event
                {
                    should_stop = true;
                }
            });

            // Waiting for gpu to finish. This is not good practice but we just want to keep things simple.
            unsafe { self.device.device_wait_idle()? };

            if should_stop {
                break;
            }

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
                    self.recreate_swapchain()?;
                    continue;
                }
                Err(error) => panic!("Error while acquiring next image. Cause: {}", error),
            };

            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let wait_semaphores = [self.image_available_semaphore];
            let signal_semaphores = [self.render_finished_semaphore];

            let command_buffers = [self.command_buffers[image_index as usize]];
            let submit_info = [vk::SubmitInfo::builder()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores)
                .build()];
            unsafe {
                self.device
                    .queue_submit(self.graphics_queue, &submit_info, vk::Fence::null())?
            };

            let swapchains = [self.swapchain_khr];
            let images_indices = [image_index];
            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&images_indices);

            let present_result = unsafe {
                self.swapchain
                    .queue_present(self.present_queue, &present_info)
            };
            match present_result {
                Ok(is_suboptimal) if is_suboptimal => {
                    self.recreate_swapchain()?;
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.recreate_swapchain()?;
                }
                Err(error) => panic!("Failed to present queue. Cause: {}", error),
                _ => {}
            }
        }

        log::info!("Stopping application");
        Ok(())
    }
}

impl Drop for Triangle {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_semaphore(self.image_available_semaphore, None);
            self.device
                .destroy_semaphore(self.render_finished_semaphore, None);
            self.cleanup_swapchain();
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.surface.destroy_surface(self.surface_khr, None);
            self.debug_report
                .destroy_debug_report_callback(self.debug_report_callback, None);
            self.instance.destroy_instance(None);
        }
    }
}

fn create_window() -> (Window, EventsLoop) {
    log::debug!("Creating window and event loop");
    let events_loop = EventsLoop::new();
    let window = WindowBuilder::new()
        .with_title(APP_NAME)
        .with_dimensions(LogicalSize::new(f64::from(WIDTH), f64::from(HEIGHT)))
        .with_resizable(false)
        .build(&events_loop)
        .unwrap();

    (window, events_loop)
}

fn create_vulkan_instance(
    entry: &Entry,
) -> Result<(Instance, DebugReport, vk::DebugReportCallbackEXT), Box<dyn Error>> {
    log::debug!("Creating vulkan instance");
    // Vulkan instance
    let app_name = CString::new(APP_NAME)?;
    let engine_name = CString::new("No Engine")?;
    let app_info = vk::ApplicationInfo::builder()
        .application_name(app_name.as_c_str())
        .application_version(ash::vk_make_version!(0, 1, 0))
        .engine_name(engine_name.as_c_str())
        .engine_version(ash::vk_make_version!(0, 1, 0))
        .api_version(ash::vk_make_version!(1, 0, 0));

    let mut extension_names = surface::required_extension_names();
    extension_names.push(DebugReport::name().as_ptr());

    let instance_create_info = vk::InstanceCreateInfo::builder()
        .application_info(&app_info)
        .enabled_extension_names(&extension_names);

    let instance = unsafe { entry.create_instance(&instance_create_info, None)? };

    // Vulkan debug report
    let create_info = vk::DebugReportCallbackCreateInfoEXT::builder()
        .flags(vk::DebugReportFlagsEXT::all())
        .pfn_callback(Some(vulkan_debug_callback));
    let debug_report = DebugReport::new(entry, &instance);
    let debug_report_callback =
        unsafe { debug_report.create_debug_report_callback(&create_info, None)? };

    Ok((instance, debug_report, debug_report_callback))
}

unsafe extern "system" fn vulkan_debug_callback(
    flag: vk::DebugReportFlagsEXT,
    typ: vk::DebugReportObjectTypeEXT,
    _: u64,
    _: usize,
    _: i32,
    _: *const c_char,
    p_message: *const c_char,
    _: *mut c_void,
) -> u32 {
    if flag == vk::DebugReportFlagsEXT::DEBUG {
        log::debug!("{:?} - {:?}", typ, CStr::from_ptr(p_message));
    } else if flag == vk::DebugReportFlagsEXT::INFORMATION {
        log::info!("{:?} - {:?}", typ, CStr::from_ptr(p_message));
    } else if flag == vk::DebugReportFlagsEXT::WARNING {
        log::warn!("{:?} - {:?}", typ, CStr::from_ptr(p_message));
    } else if flag == vk::DebugReportFlagsEXT::PERFORMANCE_WARNING {
        log::warn!("{:?} - {:?}", typ, CStr::from_ptr(p_message));
    } else {
        log::error!("{:?} - {:?}", typ, CStr::from_ptr(p_message));
    }
    vk::FALSE
}

fn create_vulkan_physical_device_and_get_graphics_and_present_qs_indices(
    instance: &Instance,
    surface: &Surface,
    surface_khr: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32, u32), Box<dyn Error>> {
    log::debug!("Creating vulkan physical device");
    let devices = unsafe { instance.enumerate_physical_devices()? };
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
                    surface.get_physical_device_surface_support(device, index, surface_khr)
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
                Swapchain::name() == name
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

            graphics.is_some()
                && present.is_some()
                && extention_support
                && !formats.is_empty()
                && !present_modes.is_empty()
        })
        .expect("Could not find a suitable device");

    unsafe {
        let props = instance.get_physical_device_properties(device);
        let device_name = CStr::from_ptr(props.device_name.as_ptr());
        log::debug!("Selected physical device: {:?}", device_name);
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
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(*index)
                    .queue_priorities(&queue_priorities)
                    .build()
            })
            .collect::<Vec<_>>()
    };

    let device_extensions_ptrs = [Swapchain::name().as_ptr()];

    let device_create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(&device_extensions_ptrs);

    let device = unsafe { instance.create_device(physical_device, &device_create_info, None)? };
    let graphics_queue = unsafe { device.get_device_queue(graphics_q_index, 0) };
    let present_queue = unsafe { device.get_device_queue(present_q_index, 0) };

    Ok((device, graphics_queue, present_queue))
}

fn create_vulkan_swapchain(
    instance: &Instance,
    surface: &Surface,
    surface_khr: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    graphics_q_index: u32,
    present_q_index: u32,
    device: &Device,
) -> Result<
    (
        Swapchain,
        vk::SwapchainKHR,
        vk::Extent2D,
        vk::Format,
        Vec<vk::Image>,
        Vec<vk::ImageView>,
    ),
    Box<dyn Error>,
> {
    log::debug!("Creating vulkan swapchain");
    // Swapchain format
    let format = {
        let formats =
            unsafe { surface.get_physical_device_surface_formats(physical_device, surface_khr)? };
        if formats.len() == 1 && formats[0].format == vk::Format::UNDEFINED {
            vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_UNORM,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            }
        } else {
            *formats
                .iter()
                .find(|format| {
                    format.format == vk::Format::B8G8R8A8_UNORM
                        && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                })
                .unwrap_or(&formats[0])
        }
    };
    log::debug!("Swapchain format: {:?}", format);

    // Swapchain present mode
    let present_mode = {
        let present_modes = unsafe {
            surface
                .get_physical_device_surface_present_modes(physical_device, surface_khr)
                .expect("Failed to get physical device surface present modes")
        };
        if present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
            vk::PresentModeKHR::IMMEDIATE
        } else {
            vk::PresentModeKHR::FIFO
        }
    };
    log::debug!("Swapchain present mode: {:?}", present_mode);

    let capabilities =
        unsafe { surface.get_physical_device_surface_capabilities(physical_device, surface_khr)? };

    // Swapchain extent
    let extent = {
        if capabilities.current_extent.width != std::u32::MAX {
            capabilities.current_extent
        } else {
            let min = capabilities.min_image_extent;
            let max = capabilities.max_image_extent;
            let width = WIDTH.min(max.width).max(min.width);
            let height = HEIGHT.min(max.height).max(min.height);
            vk::Extent2D { width, height }
        }
    };
    log::debug!("Swapchain extent: {:?}", extent);

    // Swapchain image count
    let image_count = capabilities.min_image_count;
    log::debug!("Swapchain image count: {:?}", image_count);

    // Swapchain
    let families_indices = [graphics_q_index, present_q_index];
    let create_info = {
        let mut builder = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface_khr)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT);

        builder = if graphics_q_index != present_q_index {
            builder
                .image_sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&families_indices)
        } else {
            builder.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        };

        builder
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
    };

    let swapchain = Swapchain::new(instance, device);
    let swapchain_khr = unsafe { swapchain.create_swapchain(&create_info, None)? };

    // Swapchain images and image views
    let images = unsafe { swapchain.get_swapchain_images(swapchain_khr)? };
    let views = images
        .iter()
        .map(|image| {
            let create_info = vk::ImageViewCreateInfo::builder()
                .image(*image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format.format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            unsafe { device.create_image_view(&create_info, None) }
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok((
        swapchain,
        swapchain_khr,
        extent,
        format.format,
        images,
        views,
    ))
}

fn create_vulkan_render_pass(
    device: &Device,
    format: vk::Format,
) -> Result<vk::RenderPass, Box<dyn Error>> {
    log::debug!("Creating vulkan render pass");
    let attachment_descs = [vk::AttachmentDescription::builder()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .build()];

    let color_attachment_refs = [vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build()];

    let subpass_descs = [vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachment_refs)
        .build()];

    let subpass_deps = [vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        )
        .build()];

    let render_pass_info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachment_descs)
        .subpasses(&subpass_descs)
        .dependencies(&subpass_deps);

    Ok(unsafe { device.create_render_pass(&render_pass_info, None)? })
}

fn create_vulkan_framebuffers(
    device: &Device,
    render_pass: vk::RenderPass,
    extent: vk::Extent2D,
    image_views: &[vk::ImageView],
) -> Result<Vec<vk::Framebuffer>, Box<dyn Error>> {
    log::debug!("Creating vulkan framebuffers");
    Ok(image_views
        .iter()
        .map(|view| [*view])
        .map(|attachments| {
            let framebuffer_info = vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(extent.width)
                .height(extent.height)
                .layers(1);
            unsafe { device.create_framebuffer(&framebuffer_info, None) }
        })
        .collect::<Result<Vec<_>, _>>()?)
}

fn create_vulkan_pipeline(
    device: &Device,
    render_pass: vk::RenderPass,
    extent: vk::Extent2D,
) -> Result<(vk::Pipeline, vk::PipelineLayout), Box<dyn Error>> {
    log::debug!("Creating vulkan pipeline");
    let layout_info = vk::PipelineLayoutCreateInfo::builder();
    let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

    let entry_point_name = CString::new("main")?;

    let vertex_source = read_shader_from_file("shaders/shader.vert.spv")?;
    let vertex_create_info = vk::ShaderModuleCreateInfo::builder().code(&vertex_source);
    let vertex_module = unsafe { device.create_shader_module(&vertex_create_info, None)? };

    let fragment_source = read_shader_from_file("shaders/shader.frag.spv")?;
    let fragment_create_info = vk::ShaderModuleCreateInfo::builder().code(&fragment_source);
    let fragment_module = unsafe { device.create_shader_module(&fragment_create_info, None)? };

    let shader_states_infos = [
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_module)
            .name(&entry_point_name)
            .build(),
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_module)
            .name(&entry_point_name)
            .build(),
    ];

    let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder();

    let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
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
    let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(&viewports)
        .scissors(&scissors);

    let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::builder()
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

    let multisampling_info = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
        .min_sample_shading(1.0)
        .alpha_to_coverage_enable(false)
        .alpha_to_one_enable(false);

    let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::all())
        .blend_enable(false)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ZERO)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD)
        .build()];
    let color_blending_info = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(&color_blend_attachments)
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    let pipeline_info = [vk::GraphicsPipelineCreateInfo::builder()
        .stages(&shader_states_infos)
        .vertex_input_state(&vertex_input_info)
        .input_assembly_state(&input_assembly_info)
        .viewport_state(&viewport_info)
        .rasterization_state(&rasterizer_info)
        .multisample_state(&multisampling_info)
        .color_blend_state(&color_blending_info)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0)
        .build()];

    let pipeline = unsafe {
        device
            .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
            .map_err(|e| e.1)?[0]
    };

    unsafe {
        device.destroy_shader_module(vertex_module, None);
        device.destroy_shader_module(fragment_module, None);
    }

    Ok((pipeline, pipeline_layout))
}

fn read_shader_from_file<P: AsRef<Path>>(path: P) -> Result<Vec<u32>, Box<dyn Error>> {
    log::debug!("Loading shader file {:?}", path.as_ref());
    let mut cursor = fs::load(path);
    Ok(ash::util::read_spv(&mut cursor)?)
}

fn create_and_record_command_buffers(
    device: &Device,
    pool: vk::CommandPool,
    count: usize,
    framebuffers: &[vk::Framebuffer],
    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
    extent: vk::Extent2D,
) -> Result<Vec<vk::CommandBuffer>, Box<dyn Error>> {
    log::debug!("Creating and recording command buffers");
    let buffers = {
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(count as _);

        unsafe { device.allocate_command_buffers(&allocate_info)? }
    };

    for (index, buffer) in buffers.iter().enumerate() {
        let buffer = *buffer;

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);
        unsafe { device.begin_command_buffer(buffer, &command_buffer_begin_info)? };

        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(render_pass)
            .framebuffer(framebuffers[index])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            })
            .clear_values(&[vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [1.0, 1.0, 1.0, 1.0],
                },
            }]);

        unsafe {
            device.cmd_begin_render_pass(
                buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            )
        };

        unsafe { device.cmd_bind_pipeline(buffer, vk::PipelineBindPoint::GRAPHICS, pipeline) };

        unsafe { device.cmd_draw(buffer, 3, 1, 0, 0) };

        unsafe { device.cmd_end_render_pass(buffer) };

        unsafe { device.end_command_buffer(buffer)? };
    }

    Ok(buffers)
}

mod fs {
    use std::io::Cursor;
    use std::path::Path;

    #[cfg(not(target_os = "android"))]
    pub fn load<P: AsRef<Path>>(path: P) -> Cursor<Vec<u8>> {
        use std::fs::File;
        use std::io::Read;

        let mut buf = Vec::new();
        let fullpath = &Path::new("assets").join(&path);
        let mut file = File::open(&fullpath).unwrap();
        file.read_to_end(&mut buf).unwrap();
        Cursor::new(buf)
    }

    #[cfg(target_os = "android")]
    pub fn load<P: AsRef<Path>>(path: P) -> Cursor<Vec<u8>> {

        let filename = path.as_ref().to_str().expect("Can`t convert Path to &str");
        match android_glue::load_asset(filename) {
            Ok(buf) => Cursor::new(buf),
            Err(_) => panic!("Can`t load asset '{}'", filename),
        }
    }
}

mod surface {

    use ash::extensions::khr::Surface;
    use ash::version::{EntryV1_0, InstanceV1_0};
    use ash::vk;
    use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
    use std::os::raw::c_char;
    use winit::Window;

    #[derive(Copy, Clone, Debug)]
    pub enum SurfaceError {
        SurfaceCreationError(vk::Result),
        WindowNotSupportedError,
    }

    impl std::error::Error for SurfaceError {}

    impl std::fmt::Display for SurfaceError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                SurfaceError::SurfaceCreationError(result) => {
                    write!(f, "SurfaceCreationError: {}", result)
                }
                SurfaceError::WindowNotSupportedError => write!(f, "WindowNotSupportedError"),
            }
        }
    }

    /// Get required instance extensions.
    /// This is windows specific.
    #[cfg(target_os = "windows")]
    pub fn required_extension_names() -> Vec<*const c_char> {
        use ash::extensions::khr::Win32Surface;
        vec![Surface::name().as_ptr(), Win32Surface::name().as_ptr()]
    }

    /// Get required instance extensions.
    /// This is linux specific.
    #[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
    pub fn required_extension_names() -> Vec<*const c_char> {
        use ash::extensions::khr::XlibSurface;
        vec![Surface::name().as_ptr(), XlibSurface::name().as_ptr()]
    }

    /// Get required instance extensions.
    /// This is macos specific.
    #[cfg(target_os = "macos")]
    pub fn required_extension_names() -> Vec<*const c_char> {
        use ash::extensions::mvk::MacOSSurface;
        vec![Surface::name().as_ptr(), MacOSSurface::name().as_ptr()]
    }

    /// Get required instance extensions.
    /// This is android specific.
    #[cfg(target_os = "android")]
    pub fn required_extension_names() -> Vec<*const c_char> {
        use ash::extensions::khr::AndroidSurface;
        vec![Surface::name().as_ptr(), AndroidSurface::name().as_ptr()]
    }

    /// Create the surface.
    /// This is windows specific.
    #[cfg(target_os = "windows")]
    pub unsafe fn create_surface<E: EntryV1_0, I: InstanceV1_0>(
        entry: &E,
        instance: &I,
        window: &Window,
    ) -> Result<vk::SurfaceKHR, SurfaceError> {
        use ash::extensions::khr::Win32Surface;

        log::debug!("Creating windows surface");
        match window.raw_window_handle() {
            RawWindowHandle::Windows(handle) => {
                let create_info = vk::Win32SurfaceCreateInfoKHR::builder()
                    .hinstance(handle.hinstance)
                    .hwnd(handle.hwnd);
                let surface_loader = Win32Surface::new(entry, instance);
                surface_loader
                    .create_win32_surface(&create_info, None)
                    .map_err(|e| SurfaceError::SurfaceCreationError(e))
            }
            _ => Err(SurfaceError::WindowNotSupportedError),
        }
    }

    /// Create the surface.
    /// This is linux specific.
    #[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
    pub unsafe fn create_surface<E: EntryV1_0, I: InstanceV1_0>(
        entry: &E,
        instance: &I,
        window: &Window,
    ) -> Result<vk::SurfaceKHR, SurfaceError> {
        use ash::extensions::khr::XlibSurface;
        use std::ffi::c_void;

        log::debug!("Creating linux surface");
        match window.raw_window_handle() {
            RawWindowHandle::Xlib(handle) => {
                let create_info = vk::XlibSurfaceCreateInfoKHR::builder()
                    .window(handle.window)
                    .dpy(handle.display as *mut *const c_void);
                let surface_loader = XlibSurface::new(entry, instance);
                surface_loader
                    .create_xlib_surface(&create_info, None)
                    .map_err(|e| SurfaceError::SurfaceCreationError(e))
            }
            _ => Err(SurfaceError::WindowNotSupportedError),
        }
    }

    /// Create the surface.
    /// This is macos specific.
    #[cfg(target_os = "macos")]
    pub unsafe fn create_surface<E: EntryV1_0, I: InstanceV1_0>(
        entry: &E,
        instance: &I,
        window: &Window,
    ) -> Result<vk::SurfaceKHR, SurfaceError> {
        use ash::extensions::mvk::MacOSSurface;

        log::debug!("Creating macos surface");
        match window.raw_window_handle() {
            RawWindowHandle::MacOS(handle) => {
                let create_info = vk::MacOSSurfaceCreateInfoMVK::builder().view(&*(handle.ns_view));
                let surface_loader = MacOSSurface::new(entry, instance);
                surface_loader
                    .create_mac_os_surface_mvk(&create_info, None)
                    .map_err(|e| SurfaceError::SurfaceCreationError(e))
            }
            _ => Err(SurfaceError::WindowNotSupportedError),
        }
    }

    /// Create the surface.
    /// This is android specific.
    #[cfg(target_os = "android")]
    pub unsafe fn create_surface<E: EntryV1_0, I: InstanceV1_0>(
        entry: &E,
        instance: &I,
        window: &Window,
    ) -> Result<vk::SurfaceKHR, SurfaceError> {
        use ash::extensions::khr::AndroidSurface;

        log::debug!("Creating android surface");
        match window.raw_window_handle() {
            RawWindowHandle::Android(handle) => {
                let create_info =
                    vk::AndroidSurfaceCreateInfoKHR::builder().window(handle.a_native_window);

                let surface_loader = AndroidSurface::new(entry, instance);
                surface_loader
                    .create_android_surface(&create_info, None)
                    .map_err(|e| SurfaceError::SurfaceCreationError(e))
            }
            _ => Err(SurfaceError::WindowNotSupportedError),
        }
    }
}
