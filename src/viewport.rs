use std::u64;

use ash::vk;

mod scene;

use crate::{
    ui_imgui::{self},
    viewport::scene::SceneTrait,
    vulkan_objects::{self, AllocatorTrait},
};

#[derive(Debug, Default, Clone, Copy)]
pub struct VertexData {
    pub tangent: [f32; 4],
    pub normal: [f32; 4],
    pub uv_0: [f32; 2],
    /// Since all the primitives in the scene are merged into one and sent to the mesh shader, which processes them at random, this will
    /// help in correctly positioning the vertices and correctly shading them.
    /// This will not be needed in a scene with separate geometry since these can be updated at the primitive level
    pub model_matrix_material_index: [u32; 2],
}

#[derive(Clone)]
pub struct CommonSceneData {
    pub camera_instances_addr: vk::DeviceAddress,
    pub punctual_lights_addr: vk::DeviceAddress,
    pub mesh_emitters_addr: vk::DeviceAddress,
    pub materials_addr: vk::DeviceAddress,
    pub model_matrices_addr: vk::DeviceAddress,

    pub images_descriptor_info: Vec<vk::DescriptorImageInfo>,

    pub punctual_lights_count: u32,
    pub mesh_emitters_count: u32,
}

pub struct Viewport {
    frame_objects: vulkan_objects::FrameObjects,
    acquire_signal_semaphores: Vec<vk::Semaphore>,
    present_wait_semaphores: Vec<vk::Semaphore>,
    present_fences: Vec<vk::Fence>,
    max_frames_in_flight: usize,
    depth_texture: vulkan_objects::ImageResource,

    scene: scene::SceneType,
}

impl Viewport {
    pub fn new(
        device: &ash::Device,
        allocator: &mut vulkan_objects::AllocatorType,
        queue_family_indices: &[u32],
        extent: &vk::Extent2D,
        max_frames_in_flight: usize,
        swapchain_image_count: usize,
        frame_objects_queue_family_index: u32,
        debug_utils: &vulkan_objects::DebugUtils,
        window: &winit::window::Window,
        event_loop: &winit::event_loop::ActiveEventLoop,
        data_helper: &mut vulkan_objects::DataHelper,
        name: &str,
    ) -> Self {
        let mut acquire_signal_semaphores = vec![];
        let mut present_wait_semaphores = vec![];
        let mut present_fences = vec![];

        let sem_ci = vk::SemaphoreCreateInfo::default();
        let fnc_ci = vk::FenceCreateInfo::default();

        unsafe {
            for _ in 0..max_frames_in_flight {
                acquire_signal_semaphores.push(device.create_semaphore(&sem_ci, None).unwrap());
            }
            for _ in 0..swapchain_image_count {
                present_wait_semaphores.push(device.create_semaphore(&sem_ci, None).unwrap());
                present_fences.push(device.create_fence(&fnc_ci, None).unwrap());
            }
        }

        if cfg!(debug_assertions) {
            for (index, object) in present_wait_semaphores.iter().enumerate() {
                let name =
                    name.to_owned() + " present wait semaphore " + index.to_string().as_str();
                debug_utils.set_object_name(*object, name.as_str());
            }

            for (index, object) in acquire_signal_semaphores.iter().enumerate() {
                let name =
                    name.to_owned() + " acquire signal semaphore " + index.to_string().as_str();
                debug_utils.set_object_name(*object, name.as_str());
            }
        }

        let depth_texture = allocator.create_image(
            device,
            extent,
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            queue_family_indices,
            debug_utils,
            "depth texture",
        );

        data_helper.record_batch(device);
        data_helper.change_image_layout(
            device,
            vk::PipelineStageFlags2::TOP_OF_PIPE,
            vk::AccessFlags2::empty(),
            vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            vk::AccessFlags2::empty(),
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            vk::QUEUE_FAMILY_IGNORED,
            vk::QUEUE_FAMILY_IGNORED,
            vk::ImageAspectFlags::DEPTH,
            &depth_texture.image,
        );
        data_helper.submit_batch(device, None, None, None);

        let scene = scene::SceneType::None;

        Self {
            frame_objects: vulkan_objects::FrameObjects::new(
                device,
                frame_objects_queue_family_index,
                max_frames_in_flight,
                debug_utils,
            ),
            acquire_signal_semaphores,
            present_wait_semaphores,
            present_fences,
            max_frames_in_flight,
            depth_texture,
            scene,
        }
    }

    pub fn render(
        &mut self,
        device: &ash::Device,
        swapchain: &vulkan_objects::Swapchain,
        queue: &vk::Queue,
        extent: &vk::Extent2D,
        ui: &mut ui_imgui::UI,
        camera_index: usize,
        window: &winit::window::Window,
        data_helper: &mut vulkan_objects::DataHelper,
    ) {
        let cmd_buff = self.frame_objects.get_command_buffer();
        let frame_sem = self.frame_objects.get_semaphore();
        let frame_sem_value = self.frame_objects.get_frame_sem_value();
        let frame_in_flight = self.frame_objects.get_frame_in_flight();

        let values = [frame_sem_value];
        let acq_wait_sem = [*frame_sem];

        // wait for previous frame to get done before acquiring the presentation image index.
        let acq_wait_info = vk::SemaphoreWaitInfo::default()
            .values(&values)
            .semaphores(&acq_wait_sem);

        let acq_info = vk::AcquireNextImageInfoKHR::default()
            .device_mask(0x1)
            .timeout(u64::MAX)
            .semaphore(*self.acquire_signal_semaphores.get(frame_in_flight).unwrap())
            .swapchain(swapchain.swapchain);

        let image_index;

        unsafe {
            device.wait_semaphores(&acq_wait_info, u64::MAX).unwrap();
            match swapchain.loader.acquire_next_image2(&acq_info) {
                Ok(id) => {
                    if !id.1 {
                        image_index = id.0;
                    } else {
                        eprintln!("Swapchain is suboptimal");
                        return;
                    }
                }
                Err(err) => {
                    eprintln!("{err}");
                    return;
                }
            }
        }

        // change the presentation image layout for color attachment, clear the image, setup rendering info
        let cmd_buff_bi = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        let pre_render_image_memory_barriers = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(*swapchain.images.get(image_index as usize).unwrap())
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .level_count(1),
            )];
        let pre_render_dep_info =
            vk::DependencyInfo::default().image_memory_barriers(&pre_render_image_memory_barriers);

        let color_attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(*swapchain.image_views.get(image_index as usize).unwrap())
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.01, 0.01, 0.01, 1.0],
                },
            })];

        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.depth_texture.descriptor_info.image_view)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1f32,
                    ..Default::default()
                },
            });

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D::default().extent(*extent))
            .layer_count(1)
            .color_attachments(&color_attachments)
            .depth_attachment(&depth_attachment);

        let viewports = [vk::Viewport::default()
            .width(1920 as f32)
            .height(1080 as f32)
            .min_depth(0f32)
            .max_depth(1f32)];

        let scissors = [vk::Rect2D::default().extent(*extent)];

        // presentation image to present_src_khr
        let post_render_image_memory_barriers = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .image(swapchain.images[image_index as usize])
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .level_count(1),
            )];
        let post_render_dep_info =
            vk::DependencyInfo::default().image_memory_barriers(&post_render_image_memory_barriers);

        // wait for acquire signal semaphore to indicate that the image is acquired and ready to be drawn to.
        // wait for data_helper semaphore, which will signal the end of any transfer that this submit might be dependent on.
        let wait_sem_infos = [
            vk::SemaphoreSubmitInfo::default()
                .semaphore(
                    *self
                        .acquire_signal_semaphores
                        .get(frame_in_flight as usize)
                        .unwrap(),
                )
                .stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE),
            vk::SemaphoreSubmitInfo::default()
                .semaphore(data_helper.semaphore)
                .value(data_helper.semaphore_value)
                .stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE),
        ];

        let cmd_buff_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(*cmd_buff)];

        // signal the present wait semaphore, tell the presentation engine, the image is ready to present.
        // signal the FrameObjects semaphore, with an incremented value, we can wait on this value - enabled by calling next_frame() - at the start of the next render call.
        let sig_sem_infos = [
            vk::SemaphoreSubmitInfo::default()
                .semaphore(
                    *self
                        .present_wait_semaphores
                        .get(image_index as usize)
                        .unwrap(),
                )
                .stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE),
            vk::SemaphoreSubmitInfo::default()
                .semaphore(*frame_sem)
                .value(frame_sem_value + 1)
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT),
        ];

        let submit_infos = [vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait_sem_infos)
            .command_buffer_infos(&cmd_buff_infos)
            .signal_semaphore_infos(&sig_sem_infos)];

        let swapchains = [swapchain.swapchain];
        let image_indices = [image_index];
        let present_wait_semaphores = [*self
            .present_wait_semaphores
            .get(image_index as usize)
            .unwrap()];

        let present_info = vk::PresentInfoKHR::default()
            .swapchains(&swapchains)
            .image_indices(&image_indices)
            .wait_semaphores(&present_wait_semaphores);

        // This unsafe block has all the 'action' parts of the rendering sequence, they reference the data created above.
        unsafe {
            device
                .begin_command_buffer(*cmd_buff, &cmd_buff_bi)
                .unwrap();
            device.cmd_pipeline_barrier2(*cmd_buff, &pre_render_dep_info);
            device.cmd_begin_rendering(*cmd_buff, &rendering_info);

            device.cmd_set_scissor_with_count(*cmd_buff, &scissors);
            device.cmd_set_viewport_with_count(*cmd_buff, &viewports);
        }

        // All scene geometry is merged into one, to reduce the draw calls. Large scenes are crashing.
        // comment to scene.render is a workaround to load the scene and get the raytracer output.
        // self.scene.render(device, cmd_buff, camera_index);
        ui.render(window, cmd_buff);

        unsafe {
            device.cmd_end_rendering(*cmd_buff);
            device.cmd_pipeline_barrier2(*cmd_buff, &post_render_dep_info);
            device.end_command_buffer(*cmd_buff).unwrap();
            device
                .queue_submit2(*queue, &submit_infos, vk::Fence::null())
                .unwrap();

            // this because queue_present sometimes fails and unwrap would panic.
            // queue present failure might be due to window resize, and can be ignored
            match swapchain.loader.queue_present(*queue, &present_info) {
                Ok(_) => {}
                Err(err) => {
                    eprintln!("{err}");
                }
            }
        }

        self.frame_objects.next_frame();
    }

    pub fn create_scene(
        &mut self,
        gltf: &gltf::Document,
        buffers: &[gltf::buffer::Data],
        images: &[gltf::image::Data],
        instance: &ash::Instance,
        device: &ash::Device,
        allocator: &mut vulkan_objects::AllocatorType,
        queue_family_indices: &[u32],
        debug_utils: &vulkan_objects::DebugUtils,
        data_helper: &mut vulkan_objects::DataHelper,
        compute_queue: &vk::Queue,
        compute_queue_family_index: u32,
    ) {
        unsafe {
            device.device_wait_idle().unwrap();
        }

        // destroy the current scene
        std::mem::take(&mut self.scene).destroy(device, allocator);

        // create new scene
        self.scene = scene::SceneType::World(scene::Scene::new(
            gltf,
            buffers,
            images,
            instance,
            device,
            allocator,
            queue_family_indices,
            debug_utils,
            data_helper,
            compute_queue,
            compute_queue_family_index,
        ));
    }

    pub fn reload_shaders(
        &mut self,
        device: &ash::Device,
        queue: &vk::Queue,
        debug_utils: &vulkan_objects::DebugUtils,
    ) {
        self.scene.reload_shaders(device, queue, debug_utils);
    }

    pub fn recreate_depth_texture(
        &mut self,
        device: &ash::Device,
        allocator: &mut vulkan_objects::AllocatorType,
        extent: &vk::Extent2D,
        queue_family_indices: &[u32],
        debug_utils: &vulkan_objects::DebugUtils,
        data_helper: &mut vulkan_objects::DataHelper,
    ) {
        allocator.destroy_image_resource(device, std::mem::take(&mut self.depth_texture));

        self.depth_texture = allocator.create_image(
            device,
            extent,
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            queue_family_indices,
            debug_utils,
            "depth texture",
        );

        data_helper.record_batch(device);
        data_helper.change_image_layout(
            device,
            vk::PipelineStageFlags2::TOP_OF_PIPE,
            vk::AccessFlags2::empty(),
            vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            vk::AccessFlags2::empty(),
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            vk::QUEUE_FAMILY_IGNORED,
            vk::QUEUE_FAMILY_IGNORED,
            vk::ImageAspectFlags::DEPTH,
            &self.depth_texture.image,
        );
        data_helper.submit_batch(device, None, None, None);
    }

    pub fn destroy(self, device: &ash::Device, allocator: &mut vulkan_objects::AllocatorType) {
        unsafe {
            self.frame_objects.destroy(device);
            for semaphore in self.acquire_signal_semaphores {
                device.destroy_semaphore(semaphore, None);
            }

            for semaphore in self.present_wait_semaphores {
                device.destroy_semaphore(semaphore, None);
            }

            for fence in self.present_fences {
                device.destroy_fence(fence, None);
            }

            allocator.destroy_image_resource(device, self.depth_texture);
        }

        self.scene.destroy(device, allocator);
    }

    pub fn camera_names(&self) -> &[String] {
        &self.scene.camera_names()
    }

    pub fn common_scene_data(&self) -> CommonSceneData {
        self.scene.common_scene_data()
    }
}
