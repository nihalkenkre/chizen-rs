#![allow(dead_code)]
#![allow(unused)]

#[macro_export]
macro_rules! aligned_size {
    ($size: expr, $alignment: expr) => {
        ($size + $alignment - 1) & !($alignment - 1)
    };
}

use std::{
    io::Write,
    sync::{Arc, atomic::AtomicBool},
};

use ash::vk::{self};
use winit::{
    application::ApplicationHandler,
    dpi::{PhysicalPosition, PhysicalSize},
    event::MouseScrollDelta,
    event_loop::{EventLoop, EventLoopProxy},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
};

use crate::vulkan_objects::AllocatorTrait;

mod display;
mod events;
mod renderer;
mod ui_imgui;
mod viewport;
mod vulkan_interface;
mod vulkan_objects;

#[derive(Default)]
enum DisplayMode {
    #[default]
    Viewport,
    Render,
}

#[derive(Default)]
struct Application<'instance_lifetime> {
    window: Option<winit::window::Window>,
    /// Contains the base class required by other vulkan operations. E.g instance, device, memory allocator etc etc
    vulkan_interface: Option<vulkan_interface::VulkanInterface<'instance_lifetime>>,
    /// Display the contents of the imported gltf scene.
    viewport: Option<viewport::Viewport>,
    /// Display the raytraced render of the 3d scene. 
    display: Option<display::Display>,
    /// GUI
    ui_imgui: Option<ui_imgui::UI>,
    /// Used to send user events around the application.
    event_proxy: Option<EventLoopProxy<events::UserEvent>>,
    /// The raytracer. Runs on a separate thread and writes to final_render_target
    renderer: Option<std::sync::Arc<std::sync::Mutex<renderer::Renderer>>>,
    /// Show Vieport or Render
    display_mode: DisplayMode,

    /// vulkan image which is written to by the raytracer, and is used as a sampled image by the display
    final_render_target: Option<vulkan_objects::ImageResource>,
    /// States required by the random number generator. 4 uints per render pixel. 
    /// Recreated for every run of the renderer to avoid repeating on the period.
    random_states: Option<vulkan_objects::BufferResource>,

    /// is set when render dimensions are changed on the UI
    recreate_render_resources: bool,
    /// True when raytracing in progress. Used to disable the UI.
    is_rendering: bool,
    /// True to stop the renderer. Due to user stopping the render or application quitting.
    should_stop_rendering: Arc<AtomicBool>,

    current_mouse_position: PhysicalPosition<f32>,
    last_mouse_position: PhysicalPosition<f32>,
    delta_mouse_position: PhysicalPosition<f32>,
    is_tracking_mouse: bool,
    monitor_size: PhysicalSize<u32>,
    zoom_level: f32,
}

impl Application<'static> {
    fn new(event_loop: &EventLoop<events::UserEvent>) -> Self {
        let event_proxy = Some(event_loop.create_proxy());

        Self {
            monitor_size: PhysicalSize {
                width: 1920,
                height: 1080,
            },
            event_proxy,
            zoom_level: 1f32,
            is_tracking_mouse: false,
            is_rendering: false,
            ..Default::default()
        }
    }
}

impl<'instance_lifetime> ApplicationHandler<events::UserEvent> for Application<'instance_lifetime> {
    fn user_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        event: events::UserEvent,
    ) {
        match event {
            events::UserEvent::FileOpen(file_path) => {
                let (gltf, buffers, images) = gltf::import(file_path).unwrap();

                let vulkan_interface = self.vulkan_interface.as_mut().unwrap();
                let viewport = self.viewport.as_mut().unwrap();

                // Create a scene for the viewport and renderer separately, since the requirements of the viewport and renderer would be different.
                viewport.create_scene(
                    &gltf,
                    &buffers,
                    &images,
                    &vulkan_interface.instance.instance,
                    &vulkan_interface.device.device,
                    &mut vulkan_interface.allocator,
                    &vulkan_interface.physical_device_data.queue_family_indices,
                    &vulkan_interface.debug_utils,
                    &mut vulkan_interface.data_helper,
                    &vulkan_interface.device.compute_queue,
                    vulkan_interface
                        .physical_device_data
                        .compute_queue_family_index,
                );

                let mut renderer = self.renderer.as_mut().unwrap().lock().unwrap();
                renderer.create_scene(
                    &gltf,
                    &buffers,
                    &images,
                    &vulkan_interface.instance.instance,
                    &vulkan_interface.device.device,
                    &mut vulkan_interface.allocator,
                    &vulkan_interface.physical_device_data.queue_family_indices,
                    vulkan_interface.device.compute_queue,
                    vulkan_interface.physical_device_data.raytracing_properties,
                    &ash::khr::ray_tracing_pipeline::Device::new(
                        &vulkan_interface.instance.instance,
                        &vulkan_interface.device.device,
                    ),
                    vulkan_interface
                        .physical_device_data
                        .acceleration_structure_properties
                        .min_acceleration_structure_scratch_offset_alignment,
                    &vulkan_interface.debug_utils,
                    &mut vulkan_interface.data_helper,
                    &viewport.common_scene_data(),
                );

                // Let the renderer scene know of the descriptor info for the final render target.
                renderer.update_final_render_target_desc(
                    &vulkan_interface.device.device,
                    self.final_render_target.as_ref().unwrap().descriptor_info,
                );

                // Show the camera names in the scene in the GUI.
                let ui_imgui = self.ui_imgui.as_mut().unwrap();
                ui_imgui.update_camera_names(viewport.camera_names());

                // Display the viewport from the first camera.
                self.display_mode = DisplayMode::Viewport;
            }
            events::UserEvent::ReloadShaders => {
                // Destroy the pipeline and create a new pipeline with the shaders on disk, on the viewport and renderer.
                let vulkan_interface = self.vulkan_interface.as_ref().unwrap();
                self.viewport.as_mut().unwrap().reload_shaders(
                    &vulkan_interface.device.device,
                    &vulkan_interface.device.graphics_queue,
                    &vulkan_interface.debug_utils,
                );

                self.renderer
                    .as_mut()
                    .unwrap()
                    .lock()
                    .unwrap()
                    .reload_shaders(
                        &vulkan_interface.device.device,
                        &vulkan_interface.device.compute_queue,
                        self.final_render_target.as_ref().unwrap().descriptor_info,
                        &vulkan_interface.debug_utils,
                    );
            }
            events::UserEvent::StartRender => {
                // Gather all resources, generate new resources if needed and spawn a new thread to run the raytracer.
                let vulkan_interface = self.vulkan_interface.as_mut().unwrap();
                let device = Arc::new(vulkan_interface.device.device.clone());

                unsafe {
                    device.device_wait_idle().unwrap();
                }

                let mut allocator = &mut vulkan_interface.allocator;
                let debug_utils = &vulkan_interface.debug_utils;
                let mut data_helper = &mut vulkan_interface.data_helper;

                let compute_queue = vulkan_interface.device.compute_queue.clone();
                let render_extent = self.ui_imgui.as_ref().unwrap().render_target_extent();

                print!("Generating random states...");
                std::io::stdout().flush().unwrap();

                let mut random_states = vec![];

                // initial states for the random number generator
                for r in 0..(4 * render_extent.width * render_extent.height) {
                    let mut rand: u32 = rand::random();

                    while rand < 128 {
                        rand = rand::random();
                    }

                    random_states.push(rand);
                }

                let staging_random_states = allocator.create_buffer_on_host_with_data(
                    &device,
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    &random_states,
                    &debug_utils,
                    None,
                    "staging random states",
                );

                let random_states = allocator.create_buffer_on_device(
                    &device,
                    vk::BufferUsageFlags::TRANSFER_DST,
                    (random_states.len() * size_of::<u32>()) as vk::DeviceSize,
                    &debug_utils,
                    None,
                    "random states",
                );
                data_helper.record_batch(&device);
                data_helper.copy_buffer_to_buffer(
                    &device,
                    staging_random_states.vk_buffer(),
                    random_states.vk_buffer(),
                    staging_random_states.data_size,
                );
                data_helper.submit_batch(&device, None, None, None);
                allocator.destroy_buffer_resource(&device, staging_random_states);
                println!("done");

                if self.recreate_render_resources {
                    print!("Recreating render resources...");
                    // destroy and create the final render target
                    std::io::stdout().flush().unwrap();

                    unsafe { vulkan_interface.device.device.device_wait_idle().unwrap() };

                    allocator.destroy_image_resource(
                        &vulkan_interface.device.device,
                        self.final_render_target.take().unwrap(),
                    );

                    let final_render_target = allocator.create_image(
                        &vulkan_interface.device.device,
                        &render_extent,
                        vk::Format::R32G32B32A32_SFLOAT,
                        vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                        &vulkan_interface.physical_device_data.queue_family_indices,
                        &vulkan_interface.debug_utils,
                        "final render target",
                    );

                    data_helper.record_batch(&device);
                    data_helper.change_image_layout(
                        &vulkan_interface.device.device,
                        vk::PipelineStageFlags2::TOP_OF_PIPE,
                        vk::AccessFlags2::empty(),
                        vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                        vk::AccessFlags2::empty(),
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::GENERAL,
                        vk::QUEUE_FAMILY_IGNORED,
                        vk::QUEUE_FAMILY_IGNORED,
                        vk::ImageAspectFlags::COLOR,
                        &final_render_target.image,
                    );
                    data_helper.submit_batch(&device, None, None, None);

                    let mut renderer = self.renderer.as_ref().unwrap().lock().unwrap();
                    renderer.recreate_accum_target(
                        &vulkan_interface.device.device,
                        &mut allocator,
                        &render_extent,
                        &vulkan_interface.physical_device_data.queue_family_indices,
                        &vulkan_interface.debug_utils,
                        &mut data_helper,
                    );

                    // let the renderer know of the latest final render target
                    renderer.update_final_render_target_desc(
                        &device,
                        final_render_target.descriptor_info,
                    );
                    // let the display know of the latest final render target
                    self.display
                        .as_ref()
                        .unwrap()
                        .update_final_render_target_desc(
                            &device,
                            final_render_target.descriptor_info,
                        );

                    self.final_render_target = Some(final_render_target);

                    println!("done");

                    self.recreate_render_resources = false;
                }

                let common_scene_data = self.viewport.as_ref().unwrap().common_scene_data();

                let final_render_target_desc =
                    self.final_render_target.as_ref().unwrap().descriptor_info;

                let renderer = self.renderer.as_mut().unwrap().clone();
                let should_stop = self.should_stop_rendering.clone();

                let imgui = self.ui_imgui.as_ref().unwrap();
                let render_mode = imgui.selected_render_mode as u32;
                let camera_index = imgui.selected_camera_index as u32;
                let sample_count = imgui.sample_count as u32;
                let bounce_count = imgui.bounce_count as u32;

                // Bombs away
                std::thread::spawn(move || {
                    renderer.lock().unwrap().start(
                        &device,
                        final_render_target_desc,
                        compute_queue,
                        render_extent,
                        sample_count,
                        render_mode,
                        camera_index,
                        bounce_count,
                        random_states.device_address,
                        common_scene_data,
                        should_stop,
                    );
                });

                self.random_states = Some(random_states);

                // let everyone know the render has started
                self.event_proxy
                    .as_ref()
                    .unwrap()
                    .send_event(events::UserEvent::RenderStarted)
                    .unwrap();
            }
            events::UserEvent::RenderStarted => {
                // everyone knows render has started
                self.ui_imgui.as_mut().unwrap().disable(true);
                self.display_mode = DisplayMode::Render;
                self.is_rendering = true;
            }
            events::UserEvent::RenderStopped => {
                // everyone knows render has stoppped
                self.ui_imgui.as_mut().unwrap().disable(false);

                let vulkan_interface = self.vulkan_interface.as_mut().unwrap();

                vulkan_interface.allocator.destroy_buffer_resource(
                    &vulkan_interface.device.device,
                    self.random_states.take().unwrap(),
                );
                self.is_rendering = false;
            }
            events::UserEvent::RenderExtentChanged(_) => {
                // bool to lazily recreate render resources just before rendering
                self.recreate_render_resources = true;
            }
        }
    }

    fn exiting(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        // cleanup
        self.should_stop_rendering
            .clone()
            .store(true, std::sync::atomic::Ordering::Relaxed);

        let mut vulkan_interface = self.vulkan_interface.take().unwrap();

        unsafe {
            vulkan_interface.device.device.device_wait_idle().unwrap();
        }

        self.renderer.as_mut().unwrap().lock().unwrap().destroy(
            &vulkan_interface.device.device,
            &mut vulkan_interface.allocator,
        );

        vulkan_interface.allocator.destroy_buffer_resource(
            &vulkan_interface.device.device,
            self.random_states.take().unwrap_or_default(),
        );

        self.ui_imgui
            .take()
            .unwrap()
            .destroy(&vulkan_interface.device.device);

        self.viewport.take().unwrap().destroy(
            &vulkan_interface.device.device,
            &mut vulkan_interface.allocator,
        );

        self.display.take().unwrap().destroy(
            &vulkan_interface.device.device,
            &mut vulkan_interface.allocator,
        );

        vulkan_interface.allocator.destroy_image_resource(
            &vulkan_interface.device.device,
            self.final_render_target.take().unwrap_or_default(),
        );

        vulkan_interface.destroy();
    }

    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        // first display of the window
        // initialize the required objects

        let window = event_loop
            .create_window(winit::window::WindowAttributes::default().with_title("Chizen-rs"))
            .unwrap();

        let extensions = ash_window::enumerate_required_extensions(
            window
                .display_handle()
                .unwrap()
                .display_handle()
                .unwrap()
                .as_raw(),
        )
        .unwrap();

        let mut vulkan_interface = vulkan_interface::VulkanInterface::new(
            extensions,
            window.display_handle().as_ref().unwrap().as_raw(),
            window.window_handle().as_ref().unwrap().as_raw(),
        );

        let max_frames_in_flight = vulkan_interface.swapchain.image_count as usize + 1;

        let final_render_target = vulkan_interface.allocator.create_image(
            &vulkan_interface.device.device,
            &vk::Extent2D::default().width(1920).height(1080),
            vk::Format::R32G32B32A32_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            &vulkan_interface.physical_device_data.queue_family_indices,
            &vulkan_interface.debug_utils,
            "final render target",
        );

        vulkan_interface
            .data_helper
            .record_batch(&vulkan_interface.device.device);
        vulkan_interface.data_helper.change_image_layout(
            &vulkan_interface.device.device,
            vk::PipelineStageFlags2::TOP_OF_PIPE,
            vk::AccessFlags2::empty(),
            vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            vk::AccessFlags2::empty(),
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::QUEUE_FAMILY_IGNORED,
            vk::QUEUE_FAMILY_IGNORED,
            vk::ImageAspectFlags::COLOR,
            &final_render_target.image,
        );
        vulkan_interface.data_helper.submit_batch(
            &vulkan_interface.device.device,
            None,
            None,
            None,
        );

        let display = display::Display::new(
            &vulkan_interface.device.device,
            &mut vulkan_interface.allocator,
            &vulkan_interface.physical_device_data.queue_family_indices,
            &vulkan_interface
                .surface_data
                .capabilities
                .surface_capabilities
                .current_extent,
            max_frames_in_flight,
            vulkan_interface.swapchain.image_count as usize,
            vulkan_interface
                .physical_device_data
                .graphics_queue_family_index,
            &vulkan_interface.debug_utils,
            &window,
            event_loop,
            &mut vulkan_interface.data_helper,
            &final_render_target,
            "display",
        );

        display.update_final_render_target_desc(
            &vulkan_interface.device.device,
            final_render_target.descriptor_info,
        );

        let viewport = viewport::Viewport::new(
            &vulkan_interface.device.device,
            &mut vulkan_interface.allocator,
            &vulkan_interface.physical_device_data.queue_family_indices,
            &vulkan_interface
                .surface_data
                .capabilities
                .surface_capabilities
                .current_extent,
            max_frames_in_flight,
            vulkan_interface.swapchain.image_count as usize,
            vulkan_interface
                .physical_device_data
                .graphics_queue_family_index,
            &vulkan_interface.debug_utils,
            &window,
            event_loop,
            &mut vulkan_interface.data_helper,
            "viewport",
        );

        let ui_imgui = ui_imgui::UI::new(
            &vulkan_interface.instance.instance,
            &vulkan_interface.physical_device_data.physical_device,
            &vulkan_interface.device.device,
            &vulkan_interface.device.graphics_queue,
            &window,
            vulkan_interface
                .physical_device_data
                .graphics_queue_family_index,
            self.event_proxy.clone().unwrap(),
            max_frames_in_flight,
        );

        let renderer = std::sync::Arc::new(std::sync::Mutex::new(renderer::Renderer::new(
            &vulkan_interface.device.device,
            &mut vulkan_interface.allocator,
            &vk::Extent2D::default().width(1920).height(1080),
            &vulkan_interface.physical_device_data.queue_family_indices,
            &vulkan_interface.device.compute_queue,
            vulkan_interface
                .physical_device_data
                .compute_queue_family_index,
            &vulkan_interface.debug_utils,
            &mut vulkan_interface.data_helper,
            self.event_proxy.clone().unwrap(),
            max_frames_in_flight,
        )));

        self.window = Some(window);
        self.vulkan_interface = Some(vulkan_interface);
        self.viewport = Some(viewport);
        self.display = Some(display);
        self.ui_imgui = Some(ui_imgui);
        self.renderer = Some(renderer);
        self.final_render_target = Some(final_render_target);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let ui_imgui = self.ui_imgui.as_mut().unwrap();
        ui_imgui.platform.handle_window_event(
            &mut ui_imgui.ctx,
            self.window.as_ref().unwrap(),
            &event,
        );

        match event {
            winit::event::WindowEvent::KeyboardInput {
                device_id: _,
                event,
                is_synthetic: _,
            } => match event.physical_key {
                winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Escape) => {
                    if let winit::event::ElementState::Pressed = event.state {
                        self.should_stop_rendering
                            .clone()
                            .store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                }
                _ => {}
            },
            winit::event::WindowEvent::CursorMoved {
                device_id: _,
                position,
            } => {
                if self
                    .ui_imgui
                    .as_ref()
                    .unwrap()
                    .ctx
                    .io()
                    .want_capture_mouse()
                {
                    return;
                }

                if self.is_tracking_mouse {
                    self.delta_mouse_position.x += ((self.last_mouse_position.x
                        - position.x as f32)
                        / self.monitor_size.width as f32)
                        * 2f32;
                    self.delta_mouse_position.y += ((self.last_mouse_position.y
                        - position.y as f32)
                        / self.monitor_size.height as f32)
                        * 2f32;

                    self.last_mouse_position = position.cast();
                } else {
                    self.current_mouse_position = position.cast();
                }
            }
            winit::event::WindowEvent::MouseWheel {
                device_id: _,
                delta,
                phase: _,
            } => {
                if let DisplayMode::Viewport = self.display_mode {
                    return;
                }

                // used to zoom into / out of the rendered image.
                if let MouseScrollDelta::LineDelta(_, y) = delta {
                    self.zoom_level = (self.zoom_level + y / 20f32).max(0.01f32);
                };
            }
            winit::event::WindowEvent::MouseInput {
                device_id: _,
                state,
                button,
            } => {
                if self
                    .ui_imgui
                    .as_ref()
                    .unwrap()
                    .ctx
                    .io()
                    .want_capture_mouse()
                {
                    return;
                }

                // the left mouse button is used to pan the render display image around the screen.
                // the middle mouse button is used to switch between viewport and render display.
                match button {
                    winit::event::MouseButton::Left => match state {
                        winit::event::ElementState::Pressed => {
                            if let DisplayMode::Viewport = self.display_mode {
                                return;
                            } else {
                                self.is_tracking_mouse = true;
                                self.last_mouse_position = self.current_mouse_position;
                            }
                        }
                        winit::event::ElementState::Released => {
                            if let DisplayMode::Viewport = self.display_mode {
                                return;
                            } else {
                                self.is_tracking_mouse = false;
                                self.last_mouse_position = [0f64, 0f64].into();
                            }
                        }
                    },
                    winit::event::MouseButton::Middle => {
                        if !self.is_rendering {
                            match self.display_mode {
                                DisplayMode::Viewport => {
                                    if let winit::event::ElementState::Pressed = state {
                                        self.display_mode = DisplayMode::Render;
                                    }
                                }

                                DisplayMode::Render => {
                                    if let winit::event::ElementState::Pressed = state {
                                        self.display_mode = DisplayMode::Viewport;
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            winit::event::WindowEvent::Resized(new_size) => {
                if new_size.height == 0 || new_size.width == 0 {
                    return;
                }

                // recreate the swapchain and depth texture
                let vulkan_interface = self.vulkan_interface.as_mut().unwrap();

                vulkan_interface.recreate_swapchain();
                self.viewport.as_mut().unwrap().recreate_depth_texture(
                    &vulkan_interface.device.device,
                    &mut vulkan_interface.allocator,
                    &vk::Extent2D::default()
                        .width(new_size.width)
                        .height(new_size.height),
                    &vulkan_interface.physical_device_data.queue_family_indices,
                    &vulkan_interface.debug_utils,
                    &mut vulkan_interface.data_helper,
                );
            }
            winit::event::WindowEvent::RedrawRequested => {
                if self.window.as_ref().unwrap().inner_size().width == 0
                    || self.window.as_ref().unwrap().inner_size().height == 0
                {
                    return;
                }

                // Draw the contents of the app depending on the state
                let vulkan_interface = self.vulkan_interface.as_mut().unwrap();
                let ui_imgui = self.ui_imgui.as_mut().unwrap();

                match self.display_mode {
                    DisplayMode::Viewport => {
                        self.viewport.as_mut().unwrap().render(
                            &vulkan_interface.device.device,
                            &vulkan_interface.swapchain,
                            &vulkan_interface.device.graphics_queue,
                            &vulkan_interface
                                .surface_data
                                .capabilities
                                .surface_capabilities
                                .current_extent,
                            ui_imgui,
                            ui_imgui.selected_camera_index as usize,
                            self.window.as_ref().unwrap(),
                            &mut vulkan_interface.data_helper,
                        );
                    }
                    DisplayMode::Render => {
                        self.display.as_mut().unwrap().render(
                            &vulkan_interface.device.device,
                            &vulkan_interface.swapchain,
                            &vulkan_interface.device.graphics_queue,
                            &vulkan_interface
                                .surface_data
                                .capabilities
                                .surface_capabilities
                                .current_extent,
                            ui_imgui,
                            self.window.as_ref().unwrap(),
                            &vulkan_interface.data_helper,
                            self.delta_mouse_position.into(),
                            self.zoom_level,
                        );
                    }
                }
                self.window.as_ref().unwrap().request_redraw();
            }
            winit::event::WindowEvent::CloseRequested => event_loop.exit(),
            _ => {}
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Hello, world!");

    let event_loop = EventLoop::<events::UserEvent>::with_user_event().build()?;
    let mut app = Application::new(&event_loop);
    event_loop.run_app(&mut app)?;

    println!("Bye, world!");
    Ok(())
}
