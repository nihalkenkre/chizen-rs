use std::path::PathBuf;

use ash::vk::{self};
use dear_file_browser::{FileDialog, FileDialogExt, FileDialogState};
use dear_imgui_ash::AshRenderer;
use dear_imgui_rs::Context;
use dear_imgui_winit::WinitPlatform;
use winit::event_loop::EventLoopProxy;

use crate::events;

/// Draws the ImGUI, stores the values.
pub struct UI {
    pub ctx: Context,
    pub platform: WinitPlatform,
    pub selected_render_mode: i32,
    pub selected_camera_index: i32,
    pub sample_count: i32,
    pub bounce_count: i32,

    camera_names: Vec<String>,

    /// renderer backend for the GUI
    renderer: AshRenderer,
    command_pool: vk::CommandPool,

    /// Direct + Indirect or Brute Force
    raytrace_modes: Vec<String>,
    disabled: bool,
    /// To be used when the scene is too big for the GPU. All path tracing will happen on the GPU, the results then passed on to the CPU for shading
    cpu_shading: bool,
    file_path: Option<PathBuf>,
    render_target_extent: [i32; 2],
    event_proxy: EventLoopProxy<events::UserEvent>,
    show_file_dialog: bool,
    file_browser_state: FileDialogState,
}

impl UI {
    pub fn new(
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
        device: &ash::Device,
        queue: &vk::Queue,
        window: &winit::window::Window,
        queue_family_index: u32,
        event_proxy: EventLoopProxy<events::UserEvent>,
        in_flight_frames: usize,
    ) -> Self {
        let mut ctx = Context::create();
        ctx.set_ini_filename(None::<String>).unwrap();
        let mut platform = WinitPlatform::new(&mut ctx);
        platform.attach_window(window, dear_imgui_winit::HiDpiMode::Default, &mut ctx);

        let command_pool;
        unsafe {
            command_pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(queue_family_index),
                    None,
                )
                .unwrap()
        }

        let dynamic_rendering = dear_imgui_ash::DynamicRendering {
            color_attachment_format: vk::Format::R8G8B8A8_SRGB,
            depth_attachment_format: Some(vk::Format::D32_SFLOAT),
        };

        let options = Some(dear_imgui_ash::Options {
            in_flight_frames,
            framebuffer_srgb: true,
            ..Default::default()
        });

        let renderer = AshRenderer::with_default_allocator(
            instance,
            *physical_device,
            device.clone(),
            *queue,
            command_pool,
            dynamic_rendering,
            &mut ctx,
            options,
        )
        .unwrap();

        let mut file_browser_state = FileDialogState::new(dear_file_browser::DialogMode::OpenFile);

        let render_modes = vec!["Direct + Indirect".to_owned(), "Brute Force".to_owned()];
        let camera_names = vec!["None".to_owned()];

        Self {
            ctx,
            platform,
            renderer,
            command_pool,
            disabled: false,
            cpu_shading: false,
            file_path: None,
            render_target_extent: [1920, 1080],
            sample_count: 256,
            bounce_count: 12,
            camera_names,
            selected_camera_index: 0,
            raytrace_modes: render_modes,
            selected_render_mode: 0,
            show_file_dialog: false,
            file_browser_state,
            event_proxy,
        }
    }

    pub fn render(&mut self, window: &winit::window::Window, command_buffer: &vk::CommandBuffer) {
        self.platform.prepare_frame(window, &mut self.ctx);
        let frame_time = 1000f32 / self.ctx.io().framerate();

        let ui = self.ctx.frame();
        ui.window("Awesome Panel").build(|| {
            let disable_token = ui.begin_disabled_with_cond(self.disabled);

            ui.text(format!("Frame time: {:.3} ms", frame_time));
            ui.checkbox("CPU Shading", &mut self.cpu_shading);

            if self.cpu_shading {
                self.file_browser_state.core.set_filters(vec![
                    dear_file_browser::FileFilter::from(("GLTF", &["gltf"][..])),
                ]);
            } else {
                self.file_browser_state.core.set_filters(vec![
                    dear_file_browser::FileFilter::from(("GLTF", &["glb", "gltf"][..])),
                ]);
            }

            ui.button("Load GLTF").then(|| {
                self.show_file_dialog = true;
            });

            if self.show_file_dialog {
                self.file_browser_state.open();
                if let Some(result) = ui.file_browser().show(&mut self.file_browser_state) {
                    match result {
                        Ok(selection) => {
                            let file_path = selection.paths[0].clone();
                            self.show_file_dialog = false;

                            self.event_proxy
                                .send_event(events::UserEvent::FileOpen(file_path.clone()))
                                .unwrap();

                            self.file_path = Some(file_path);
                        }
                        Err(_) => self.show_file_dialog = false,
                    }
                }
            }

            match ui.begin_modal_popup("Reload Scene") {
                Some(token) => {
                    ui.text("The loaded file must be an ASCII GLTF file...");
                    ui.separator();

                    ui.button("Ok").then(|| ui.close_current_popup());
                    std::mem::drop(token);
                }
                None => {}
            }

            ui.button("Reload Scene").then(|| match &self.file_path {
                Some(path) => match path.extension() {
                    Some(ext) => {
                        if self.cpu_shading && !ext.eq("gltf") {
                            ui.open_popup("Reload Scene");
                        } else {
                            let file_path = self.file_path.as_ref().unwrap();
                            self.event_proxy
                                .send_event(events::UserEvent::FileOpen(file_path.clone()))
                                .unwrap();
                        }
                    }
                    None => {}
                },
                None => {}
            });

            // ui.button("Reload Shaders").then(|| {
            //     self.event_proxy
            //         .send_event(events::UserEvent::ReloadShaders)
            //         .unwrap();
            // });

            if ui
                .input_int2("Render Dims", &mut self.render_target_extent)
                .build()
            {
                self.render_target_extent[0] = self.render_target_extent[0].clamp(1, 8192);
                self.render_target_extent[1] = self.render_target_extent[1].clamp(1, 8192);

                self.event_proxy
                    .send_event(events::UserEvent::RenderExtentChanged(
                        vk::Extent2D::default()
                            .width(self.render_target_extent[0] as u32)
                            .height(self.render_target_extent[1] as u32),
                    ))
                    .unwrap();
            }

            ui.drag_int("Num Samples", &mut self.sample_count).then(|| {
                // can use max(1) here but figured the `if` would save on read, write, math operations if not required.
                if self.sample_count <= 0 {
                    self.sample_count = 1
                }
            });

            ui.drag_int("Num Bounces", &mut self.bounce_count).then(|| {
                // can use max(1) here but figured the `if` would save on read, write, math operations if not required.
                if self.bounce_count < 0 {
                    self.bounce_count = 0
                }
            });

            if self.camera_names.len() > 0 {
                match ui.begin_combo(
                    "Cameras",
                    self.camera_names
                        .get(self.selected_camera_index as usize)
                        .unwrap(),
                ) {
                    Some(_) => {
                        for (index, camera_name) in self.camera_names.iter().enumerate() {
                            let is_selected = self.selected_camera_index == index as i32;
                            if ui.selectable(camera_name.to_owned() + "##" + &index.to_string()) {
                                self.selected_camera_index = index as i32;
                            }

                            if is_selected {
                                ui.set_item_default_focus();
                            }
                        }
                    }
                    None => {}
                }
            }

            match ui.begin_combo(
                "Raytrace Mode",
                self.raytrace_modes
                    .get(self.selected_render_mode as usize)
                    .unwrap(),
            ) {
                Some(_) => {
                    for (index, render_mode_name) in self.raytrace_modes.iter().enumerate() {
                        let is_selected = self.selected_render_mode == index as i32;
                        if ui.selectable(render_mode_name.to_owned() + "##" + &index.to_string()) {
                            self.selected_render_mode = index as i32;
                        }

                        if is_selected {
                            ui.set_item_default_focus();
                        }
                    }
                }
                None => {}
            };

            if ui.button("Render") {
                self.event_proxy
                    .send_event(events::UserEvent::StartRender)
                    .unwrap();
            }
        });

        self.platform.prepare_render_with_ui(ui, window);
        let draw_data = self.ctx.render();
        self.renderer.cmd_draw(*command_buffer, draw_data).unwrap();
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_command_pool(self.command_pool, None);
        }
    }

    pub fn update_camera_names(&mut self, camera_names: &[String]) {
        self.camera_names.clear();

        self.camera_names = camera_names.to_vec();
        self.selected_camera_index = 0;
    }

    pub fn disable(&mut self, disabled: bool) {
        self.disabled = disabled
    }

    pub fn render_target_extent(&self) -> vk::Extent2D {
        vk::Extent2D::default()
            .width(self.render_target_extent[0] as u32)
            .height(self.render_target_extent[1] as u32)
    }
}
